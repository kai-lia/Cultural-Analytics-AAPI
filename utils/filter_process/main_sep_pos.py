import os
import re
import json
import gzip
import pickle
import fasttext
import spacy
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import Counter, defaultdict

from tqdm import tqdm
from datasets import load_dataset

from dolma.core.data_types import Document

from utils.filter_process.pos import (
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
    flush_all_to_db,
    save_artifacts_safely,
    load_ethnicity_dicts_from_db,
    save_progress,
    init_db,
    load_aapi_pkl,
)

from utils.filter_process.dolma_local import (
    AAPIKeywordsTagger,
    mix_aapi_doc,
    AAPITokenizer,
    predict_n_mix,
)


# ------------------------------------------------------------
# OUTPUT PATHS
# ------------------------------------------------------------

OUTPUT_TOKENIZED_DIR = Path("data/output/tokenized_c4")
OUTPUT_TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_PATH = Path("data/state/c4_progress.json")
PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("utils/autotuned_fasttext_model.bin")
LOCAL_C4_FOLDER = Path("/Users/kaionamartinson/Desktop/Cultural-Analytics/dolma/c4")


# ===================================================================
# MAIN PROCESSING LOOP
# ===================================================================

def run_loop(
    noun_heads_counter,
    aapi_counter_pass,
    verb_ethnicity_dict,
    adj_ethnicity_dict,
    out_dirname: Path = OUTPUT_TOKENIZED_DIR,
) -> None:

    traversed = 0

    # ---------- DB INIT ----------
    init_db()
    aapi_keywords = load_aapi_pkl()

    # Load existing counts (if any) to continue a previous run
    db_verb_dict, db_adj_dict = load_ethnicity_dicts_from_db()

    for eth, cnts in db_verb_dict.items():
        verb_ethnicity_dict.setdefault(eth, Counter()).update(cnts)

    for eth, cnts in db_adj_dict.items():
        adj_ethnicity_dict.setdefault(eth, Counter()).update(cnts)

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

    docs_in_shard = 0

    # ==========================================================
    #     DOCUMENT LOOP
    # ==========================================================
    for data in ds:
        pbar.update(1)

        mixed = predict_n_mix(data, tagger)
        if mixed is None:
            continue

        tokenized = tokenizer.tokenize(mixed)

        # ------------------------------------------------------
        # NEW: Per-DOCUMENT unique modifier sets
        # These ensure each modifier is counted ONCE PER DOC
        # ------------------------------------------------------
        doc_adj_hits = defaultdict(set)
        doc_verb_hits = defaultdict(set)
        doc_noun_hits = defaultdict(set)

        # ======================================================
        #     SENTENCE LOOP
        # ======================================================
        for sentence in tokenized.sents:

            tokens = [t.text for t in sentence]
            tokens_lower = [t.lower() for t in tokens]

            # Overlap with ethnicity keywords
            overlap = set(tokens_lower) & aapi_keywords
            if not overlap:
                continue

            overlap_eth = list(overlap)[0]

            lex = build_group_lexicon(overlap)

            # -------------------------
            # NOUN HEAD EXTRACTION
            # -------------------------
            noun = ethnicity_modified_nouns(sentence, lex)
            if noun and "noun" in noun:
                noun_head = noun["noun"]
                doc_noun_hits[overlap_eth].add(noun_head)

            # -------------------------
            # AAPI FASTTEXT FILTER
            # -------------------------
            aapi_counter_pass[overlap_eth] += 1

            window_text = (
                window_mask_sentence(overlap_eth, tokens, tokens_lower)
                .replace("\n", " ")
                .strip()
            )

            if model.predict(window_text)[0][0] == "__label__0":
                continue

            aapi_counter_pass[overlap_eth] += 1

            # -------------------------
            # ADJECTIVE EXTRACTION
            # -------------------------
            adj_local = collect_adj(sentence, lex)
            for eth, adjs in adj_local.items():
                doc_adj_hits[eth].update(adjs)

            # -------------------------
            # VERB EXTRACTION
            # -------------------------
            verb_local = collect_verb(sentence, lex)
            for eth, verbs in verb_local.items():
                doc_verb_hits[eth].update(verbs)

        # ======================================================
        #     END OF DOCUMENT: UPDATE MASTER COUNTERS
        # ======================================================
        for eth, adjs in doc_adj_hits.items():
            for adj in adjs:
                adj_ethnicity_dict[eth][adj] += 1

        for eth, verbs in doc_verb_hits.items():
            for verb in verbs:
                verb_ethnicity_dict[eth][verb] += 1

        for eth, nouns in doc_noun_hits.items():
            for noun in nouns:
                noun_heads_counter[noun] += 1

        # ======================================================
        #     BATCH FLUSH (every 10k docs)
        # ======================================================
        docs_in_shard += 1
        traversed += 1

        if docs_in_shard > 0 and docs_in_shard % 10000 == 0:

            save_artifacts_safely(aapi_counter_pass)

            flush_all_to_db(
                noun_heads_counter,
                verb_ethnicity_dict,
                adj_ethnicity_dict,
            )

            noun_heads_counter.clear()
            verb_ethnicity_dict.clear()
            adj_ethnicity_dict.clear()

    # ==========================================================
    # FINAL SAVE
    # ==========================================================
    pbar.close()

    save_artifacts_safely(aapi_counter_pass)

    flush_all_to_db(
        noun_heads_counter,
        verb_ethnicity_dict,
        adj_ethnicity_dict,
    )

    save_progress(traversed)



# ===================================================================
# ITERATOR FOR LOCAL C4 FILES
# ===================================================================

def iter_local_c4_files(folder_path):
    """Iterate through local C4 .json(.gz) files"""
    folder_path = Path(folder_path)

    files = sorted([
        *folder_path.glob("*.json.gz"),
        *folder_path.glob("*.jsonl.gz"),
    ])

    print(f"Found {len(files)} C4 files in {folder_path}")

    if not files:
        print("⚠ No matching C4 files found! Check extension.")
        return

    for gz_file in files:
        print(f"→ Reading: {gz_file.name}")
        with gzip.open(gz_file, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    yield data
                except Exception:
                    continue


# ===================================================================
# MAIN
# ===================================================================

def main():
    noun_heads_counter = Counter()
    aapi_counter_pass = Counter()
    verb_ethnicity_dict = defaultdict(Counter)
    adj_ethnicity_dict = defaultdict(Counter)

    print("Starting AAPI extraction pipeline...")

    run_loop(
        noun_heads_counter,
        aapi_counter_pass,
        verb_ethnicity_dict,
        adj_ethnicity_dict,
        out_dirname=OUTPUT_TOKENIZED_DIR,
    )

    print("Pipeline completed.")


if __name__ == "__main__":
    main()
