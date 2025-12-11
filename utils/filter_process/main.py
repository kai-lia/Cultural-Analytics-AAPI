import os
import re
import sys
import json
import gzip
import fasttext
import spacy
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict, Counter

from concurrent.futures import as_completed

from tqdm import tqdm
from datasets import load_dataset

from queue import Queue
import threading

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED




from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))



from dolma.core.data_types import Document

from utils.filter_process.pos_db import (
    build_group_lexicon,
    collect_all_modifiers,
)

from utils.filter_process.Fasttext.fasttext import (
    window_mask_sentence,
    load_fasttext_model,
)

from utils.filter_process.load_or_save import (
    init_db,
    save_progress,
    load_aapi_pkl,
)

from utils.filter_process.dolma_local import (
    AAPIKeywordsTagger,
    mix_aapi_doc,
    AAPITokenizer,
    predict_n_mix,
)

import duckdb

# ------------------------------------------------------------
# OUTPUT PATHS
# ------------------------------------------------------------

OUTPUT_TOKENIZED_DIR = Path("data/output/tokenized_c4")
OUTPUT_TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_PATH = Path("data/state/c4_progress.json")
PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)

LOCAL_C4_FOLDER = Path("/Users/kaionamartinson/Desktop/Cultural-Analytics/dolma/c4")


DB_PATH = Path("data/output/full_pipeline/ethnicity_pos.duckdb")

MODEL =  load_fasttext_model()
AAPI_KEYWORDS = load_aapi_pkl()

MAX_OUTSTANDING = 2000




import json



def db_writer(queue: Queue, stop_signal: object):
    con = duckdb.connect(str(DB_PATH))

    while True:
        item = queue.get()
        if item is stop_signal:
            break

        noun_hits, verb_hits, adj_hits = item
        save_pos_to_db(noun_hits, verb_hits, adj_hits, con)

    con.close()


def save_pos_to_db(noun_hits, verb_hits, adj_hits, con):
    # For each ethnicity in this document:
    ethnicities = set(noun_hits.keys()) | set(verb_hits.keys()) | set(adj_hits.keys())

    for eth in ethnicities:
        nouns = sorted(list(noun_hits.get(eth, set())))
        verbs = sorted(list(verb_hits.get(eth, set())))
        adjs  = sorted(list(adj_hits.get(eth, set())))

        if  not verbs and not adjs:
            continue

        nouns_json = json.dumps(nouns)
        verbs_json = json.dumps(verbs)
        adjs_json  = json.dumps(adjs)


        con.execute("""
            INSERT INTO ethnicity_sentence_modifiers
            (ethnicity, adjs, verbs, nouns, count, has_dup)
            VALUES (?, ?, ?, ?, 1, FALSE)
            ON CONFLICT (ethnicity, adjs, verbs, nouns)
            DO UPDATE SET
                count = ethnicity_sentence_modifiers.count + 1,
                has_dup = (ethnicity_sentence_modifiers.count + 1) > 1;
        """, [eth, adjs_json, verbs_json, nouns_json])



def process_single_doc(data, tagger, tokenizer):
    noun_hits = defaultdict(set)
    verb_hits = defaultdict(set)
    adj_hits  = defaultdict(set)

    mixed = predict_n_mix(data, tagger)
    if mixed is None:
        return None

    tokenized = tokenizer.tokenize(mixed)

    for sentence in tokenized.sents:
        tokens = [t.text for t in sentence]
        tokens_lower = [t.lower() for t in tokens]

        overlap = set(tokens_lower) & AAPI_KEYWORDS
        if not overlap:
            continue

        eth = list(overlap)[0]

        window_text = (
            window_mask_sentence(eth, tokens, tokens_lower)
            .replace("\n", " ")
            .strip()
        )

        if MODEL.predict(window_text)[0][0] == "__label__0":
            continue

        lex = build_group_lexicon(overlap)
        mods = collect_all_modifiers(sentence, lex)

        for e, m in mods.items():
            noun_hits[e].update(m["nouns"])
            verb_hits[e].update(m["verbs"])
            adj_hits[e].update(m["adjs"])

    return noun_hits, verb_hits, adj_hits



def run_loop(aapi_counter_pass):

    # Load shared objects ONCE — safe for threads
    tagger = AAPIKeywordsTagger(AAPI_KEYWORDS)
    tokenizer = AAPITokenizer(AAPI_KEYWORDS)

    ds = iter_local_c4_files(LOCAL_C4_FOLDER)

    # --- queue for results ---
    q = Queue()
    STOP = object()

    # --- start DB writer thread ---
    writer_thread = threading.Thread(target=db_writer, args=(q, STOP))
    writer_thread.start()

    # --- Thread pool for CPU-bound workers ---
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []

        for data in ds:
            futures.append(
                pool.submit(
                    process_single_doc,
                    data,
                    tagger,
                    tokenizer,
                )
            )
        if len(futures) >= MAX_OUTSTANDING:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                result = f.result()
                if result:
                    q.put(result)

        for f in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            result = f.result()
            if result is None:
                continue
            q.put(result)

    # Tell writer to stop
    q.put(STOP)
    writer_thread.join()


def iter_local_c4_files(folder_path):
    folder_path = Path(folder_path)

    files = sorted([
        *folder_path.glob("*.json.gz"),
        *folder_path.glob("*.jsonl.gz"),
    ])

    print(f"Found {len(files)} C4 files in {folder_path}")

    if not files:
        print("⚠ No matching C4 files found!")
        return

    for gz_file in files:
        print(f"Reading: {gz_file.name}")
        with gzip.open(gz_file, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


# ===================================================================
# MAIN
# ===================================================================

def main():
    aapi_counter_pass = Counter()
    init_db()
    run_loop(aapi_counter_pass)
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
