import os
import re
import json
import gzip
import pickle
import fasttext
import spacy
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict, Counter

from tqdm import tqdm
from datasets import load_dataset

from dolma.core.data_types import Document

from utils.filter_process.pos_db import (
    collect_adj,
    collect_verb,
    build_group_lexicon,
    ethnicity_modified_nouns,
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

        print(eth, nouns, verbs, adjs)

        con.execute("""
            INSERT INTO ethnicity_sentence_modifiers
            (ethnicity, adjs, verbs, nouns, count, has_dup)
            VALUES (?, ?, ?, ?, 1, FALSE)
            ON CONFLICT (ethnicity, adjs, verbs, nouns)
            DO UPDATE SET
                count = ethnicity_sentence_modifiers.count + 1,
                has_dup = (ethnicity_sentence_modifiers.count + 1) > 1;
        """, [eth, adjs_json, verbs_json, nouns_json])




def run_loop(aapi_counter_pass, con):
    traversed = 0

    # Setup
    init_db()
    aapi_keywords = load_aapi_pkl()
    tagger = AAPIKeywordsTagger(aapi_keywords)
    tokenizer = AAPITokenizer(aapi_keywords)
    ds = iter_local_c4_files(LOCAL_C4_FOLDER)
    model = load_fasttext_model()

    pbar = tqdm(
        total=None,
        desc="Processing C4 docs",
        mininterval=0.5,
        dynamic_ncols=True,
        leave=True,
    )

    for data in ds:
        pbar.update(1)
        mixed = predict_n_mix(data, tagger)

        if mixed is None:
            continue

        tokenized = tokenizer.tokenize(mixed)

        # ---------------------------------------------
        # DOC-LEVEL UNIQUE SETS
        # ---------------------------------------------
        doc_adj_hits = defaultdict(set)
        doc_verb_hits = defaultdict(set)
        doc_noun_hits = defaultdict(set)

        # ---------------------------------------------
        # SENTENCE LOOP
        # ---------------------------------------------
        for sentence in tokenized.sents:
            tokens = [t.text for t in sentence]
            tokens_lower = [t.lower() for t in tokens]

            # Ethnicity keyword present?
            overlap = set(tokens_lower) & aapi_keywords
            if not overlap:
                continue

            overlap_eth = list(overlap)[0]
            lex = build_group_lexicon(overlap)

            # Nouns
            noun = ethnicity_modified_nouns(sentence, lex)
            if noun and "noun" in noun:
                doc_noun_hits[overlap_eth].add(noun["noun"])

            # Fasttext filtering
            aapi_counter_pass[overlap_eth] += 1

            window_text = (
                window_mask_sentence(overlap_eth, tokens, tokens_lower)
                .replace("\n", " ")
                .strip()
            )

            if model.predict(window_text)[0][0] == "__label__0":
                continue

            aapi_counter_pass[overlap_eth] += 1

            # POS extraction
            adj_local = collect_adj(sentence, lex)
            verb_local = collect_verb(sentence, lex)

            for eth, adjs in adj_local.items():
                doc_adj_hits[eth].update(adjs)

            for eth, verbs in verb_local.items():
                doc_verb_hits[eth].update(verbs)
            sentence

        # END OF DOCUMENT — COMMIT MODIFIERS TO DB
        save_pos_to_db(
            noun_hits=doc_noun_hits,
            verb_hits=doc_verb_hits,
            adj_hits=doc_adj_hits, 
            con=con
        )

        traversed += 1

    pbar.close()
    save_progress(traversed)



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
        print(f"→ Reading: {gz_file.name}")
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
    print("Starting AAPI extraction pipeline...")

    con = duckdb.connect(str(DB_PATH))   # <-- OPEN connection once

    run_loop(aapi_counter_pass, con)     # <-- PASS connection

    con.close()                          # <-- CLOSE at the end
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
