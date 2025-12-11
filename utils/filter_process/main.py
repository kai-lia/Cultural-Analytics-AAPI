import os
import re
import sys
import ujson as json
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

import threading
thread_local = threading.local()

import time



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
    fasttext_predict,
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
BATCH_SIZE = 500



import json


from utils.filter_process.dolma_local import AAPITokenizer

def get_tokenizer():
    if not hasattr(thread_local, "tokenizer"):
        thread_local.tokenizer = AAPITokenizer(AAPI_KEYWORDS)
    return thread_local.tokenizer



import re

MAX_TOKENS_BEFORE_SKIP = 120   # you can tune this

def simple_pre_split(text):
    """
    Fast pre-split to avoid giving massive or malformed lines to spaCy.
    """
    # crude sentence split on punctuation
    raw_sents = re.split(r"[.!?]+", text)

    for s in raw_sents:
        s = s.strip()
        if not s:
            continue

        # Skip "monster" sentences that cause 20–40 second spaCy freezes
        if len(s.split()) > MAX_TOKENS_BEFORE_SKIP:
            continue

        yield s






def db_writer(queue: Queue, stop_signal: object):
    con = duckdb.connect(str(DB_PATH))

    batch = []

    while True:
        item = queue.get()

        if item is stop_signal:
            break

        batch.append(item)
        #if len(batch) % 50 == 0:
        #    print("DB writer draining... queue now:", queue.qsize())

        if len(batch) >= BATCH_SIZE:

            for noun_hits, verb_hits, adj_hits in batch:
                save_pos_to_db(noun_hits, verb_hits, adj_hits, con)
            batch = []

    # flush remaining items
    if batch:
        con.execute("BEGIN;")
        for noun_hits, verb_hits, adj_hits in batch:
            save_pos_to_db(noun_hits, verb_hits, adj_hits, con)
        con.execute("COMMIT;")

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


def process_batch(batch, tagger, q):
    tokenizer = get_tokenizer()

    mixed_list = []
    for data in batch:
        mixed = predict_n_mix(data, tagger)
        if mixed:
            mixed_list.append(mixed)

    if not mixed_list:
        return
    

    
    #start = time.time()
    safe_texts = []
    safe_mixed = []

    for item in mixed_list:
        # item["text"] is your document
        for s in simple_pre_split(item["text"]):
            safe_texts.append(s)
            safe_mixed.append(item)

    if not safe_texts:
        return

    docs = tokenizer.nlp.pipe(safe_texts, batch_size=BATCH_SIZE_NLP)


    #print("spaCy batch time:", time.time() - start)

    for doc, mixed in zip(docs, mixed_list):
        noun_hits = defaultdict(set)
        verb_hits = defaultdict(set)
        adj_hits = defaultdict(set)

        # -----------------------------
        # A. Collect sentences + window texts
        # -----------------------------
        window_texts = []
        sent_overlap_pairs = []

        for sent in doc.sents:
            tokens = [t.text for t in sent]
            tokens_lower = [t.lower_ for t in sent]

            overlap = set(tokens_lower) & AAPI_KEYWORDS
            if not overlap:
                continue

            eth = list(overlap)[0]

            wt = window_mask_sentence(eth, tokens, tokens_lower)

            window_texts.append(wt)
            sent_overlap_pairs.append((sent, overlap))

        # -----------------------------
        # B. Batched FastText predict
        # -----------------------------
        if window_texts:
            preds = fasttext_predict(MODEL, window_texts, k=1)
        else:
            preds = []

        # -----------------------------
        # C. Process predictions
        # -----------------------------
        for (sent, overlap), pred in zip(sent_overlap_pairs, preds):
            if pred == "__label__0":
                continue

            lex = build_group_lexicon(overlap)
            mods = collect_all_modifiers(sent, lex)

            for e, m in mods.items():
                noun_hits[e].update(m["nouns"])
                verb_hits[e].update(m["verbs"])
                adj_hits[e].update(m["adjs"])

        # Send results to DB writer
        q.put((noun_hits, verb_hits, adj_hits))



BATCH_SIZE_NLP = 150   # good default






def run_loop(aapi_counter_pass):
    tagger = AAPIKeywordsTagger(AAPI_KEYWORDS)
    q = Queue()
    STOP = object()

    writer_thread = threading.Thread(target=db_writer, args=(q, STOP))
    writer_thread.start()

    ds = iter_local_c4_files(LOCAL_C4_FOLDER)
    batch = []
    futures = []

    pbar = tqdm(desc="Processing C4 batches", unit="batch", dynamic_ncols=True)


    with ThreadPoolExecutor(max_workers=2) as pool:
        for data in ds:
            batch.append(data)

            if len(batch) >= BATCH_SIZE_NLP:
                futures.append(
                    pool.submit(process_batch, list(batch), tagger, q)
                )
                batch.clear()
            pbar.update(5000)

        # Process final partial batch
        if batch:
            futures.append(
                pool.submit(process_batch, list(batch), tagger, q)
            )
            pbar.update(5000)

        for f in as_completed(futures):
            f.result()  
            qs = q.qsize()
           
            if qs > 2000:
                print(f"Queue size: {qs}")
                print("Queue growing, DB writer might be falling behind!")

            if qs > 10000:
                print(f"Queue size: {qs}")
                print("Queue extremely large — pipeline will start freezing!")
                    
        
    pbar.close()
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
                #t0 = time.time()
                
                try:
                    yield json.loads(line)
                except Exception:
                    continue
                #t1 = time.time()
                #if t1 - t0 > 0.01:   # any slow parse
                #    print(f"SLOW JSON LOAD: {t1 - t0:.4f}s line length={len(line)}")



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
