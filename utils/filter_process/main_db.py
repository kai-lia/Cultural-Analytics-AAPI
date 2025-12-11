import os
import re
import sys
import json
import gzip
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict, Counter

from tqdm import tqdm
from datasets import load_dataset

from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading



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
    fasttext_predict
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

MODEL_PATH = Path("utils/autotuned_fasttext_model.bin")
LOCAL_C4_FOLDER = Path("/Users/kaionamartinson/Desktop/Cultural-Analytics/dolma/c4")


DB_PATH = Path("data/output/full_pipeline/ethnicity_pos.duckdb")




import json

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




def db_writer(queue: Queue, stop_signal: object):
    con = duckdb.connect(str(DB_PATH))

    batch = []
    BATCH_SIZE_DB = 2000   # <- tune as needed

    while True:
        item = queue.get()
        if item is stop_signal:
            break

        batch.append(item)

        if len(batch) >= BATCH_SIZE_DB:
            con.execute("BEGIN;")
            for noun_hits, verb_hits, adj_hits in batch:
                save_pos_to_db(noun_hits, verb_hits, adj_hits, con)
            con.execute("COMMIT;")
            batch = []

    # flush remaining writes
    if batch:
        con.execute("BEGIN;")
        for noun_hits, verb_hits, adj_hits in batch:
            save_pos_to_db(noun_hits, verb_hits, adj_hits, con)
        con.execute("COMMIT;")

    con.close()


def process_doc(mixed, tokenizer, aapi_keywords, model):
    noun_hits = defaultdict(set)
    verb_hits = defaultdict(set)
    adj_hits  = defaultdict(set)

    doc = tokenizer.tokenize(mixed)

    window_texts = []
    sent_overlap_pairs = []

    # Collect sentences needing FT filtering
    for sentence in doc.sents:
        tokens = [t.text for t in sentence]
        tokens_lower = [t.lower() for t in tokens]

        overlap = set(tokens_lower) & aapi_keywords
        if not overlap:
            continue

        eth = list(overlap)[0]
        wt = window_mask_sentence(eth, tokens, tokens_lower)

        window_texts.append(wt)
        sent_overlap_pairs.append((sentence, overlap))

    # Batched FastText
    if window_texts:
        preds = fasttext_predict(model, window_texts, k=1)
    else:
        preds = []

    # Collect modifiers
    for (sentence, overlap), pred in zip(sent_overlap_pairs, preds):
        if pred == "__label__0":
            continue

        lex = build_group_lexicon(overlap)
        mods = collect_all_modifiers(sentence, lex)

        for e, m in mods.items():
            noun_hits[e].update(m["nouns"])
            verb_hits[e].update(m["verbs"])
            adj_hits[e].update(m["adjs"])

    return noun_hits, verb_hits, adj_hits


def run_loop(aapi_counter_pass):
    init_db()

    aapi_keywords = load_aapi_pkl()
    tagger = AAPIKeywordsTagger(aapi_keywords)
    tokenizer = AAPITokenizer(aapi_keywords)
    model = load_fasttext_model()

    ds = iter_local_c4_files(LOCAL_C4_FOLDER)

    # Queue + DB writer
    q = Queue()
    STOP = object()
    writer_thread = threading.Thread(target=db_writer, args=(q, STOP))
    writer_thread.start()

    # ThreadPool for processing docs
    MAX_WORKERS = 8
    futures = []
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for data in tqdm(ds, desc="Processing C4"):
            mixed = predict_n_mix(data, tagger)
            if mixed is None:
                continue

            # submit async job
            futures.append(pool.submit(
                process_doc,
                mixed,
                tokenizer,
                aapi_keywords,
                model
            ))

        # collect results
        for f in tqdm(as_completed(futures), total=len(futures), desc="Collecting"):
            result = f.result()
            if result:
                q.put(result)

            processed += 1

    # stop DB writer
    q.put(STOP)
    writer_thread.join()

    save_progress(processed)


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


def main():
    aapi_counter_pass = Counter()
    print("Starting AAPI extraction pipeline...")

    con = duckdb.connect(str(DB_PATH))   

    run_loop(aapi_counter_pass, con)    

    con.close()                        
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
