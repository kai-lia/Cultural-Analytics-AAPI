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

from utils.filter_process.pos import collect_adj, collect_verb, build_group_lexicon, ethnicity_modified_nouns

from utils.filter_process.Fasttext.fasttext import (
    window_mask_sentence,
    load_fasttext_model
)

from utils.filter_process.load_or_save import (
    flush_all_to_db,
    save_artifacts_safely,
    load_ethnicity_dicts_from_db,
    save_progress,
    init_db,
    load_aapi_pkl,
    load_model
)


from utils.filter_process.dolma import (
    AAPIKeywordsTagger,
    mix_aapi_doc,
    AAPITokenizer, 
    predict_n_mix
)






OUTPUT_TOKENIZED_DIR = Path("data/output/tokenized_c4")
OUTPUT_TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_PATH = Path("data/state/c4_progress.json")
PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "utils/autotuned_fasttext_model.bin"
LOCAL_C4_FOLDER = Path("/Users/kaionamartinson/Desktop/Cultural-Analytics/dolma/c4")





SHARD_SIZE = 500_000



def run_loop(
    noun_heads_counter,
    aapi_counter_pass,
    verb_ethnicity_dict,
    adj_ethnicity_dict,
    out_dirname: Path = OUTPUT_TOKENIZED_DIR,
) -> None:
    """
    ...
    """
    traversed = 0

    # --- DB setup ---
    init_db()
    aapi_keywords = load_aapi_pkl()

    # Optionally merge existing DB counts into in-memory dicts
    # so you keep accumulating across runs:
    db_verb_dict, db_adj_dict = load_ethnicity_dicts_from_db()

    # merge DB → in-memory
    for eth, cnts in db_verb_dict.items():
        verb_ethnicity_dict.setdefault(eth, Counter()).update(cnts)

    for eth, cnts in db_adj_dict.items():
        adj_ethnicity_dict.setdefault(eth, Counter()).update(cnts)

    # rest of your setup

    tagger = AAPIKeywordsTagger()
    tokenizer = AAPITokenizer()
    ds = iter_local_c4_files(LOCAL_C4_FOLDER)
    model = load_model()

    pbar = tqdm(
        total=None,
        desc="Processing C4 docs",
        mininterval=0.5,
        dynamic_ncols=True,
        leave=True,
    )

    docs_in_shard = 0

    for data in ds:
        pbar.update(1)
        mixed= predict_n_mix(data, tagger) # predicts and mixes
        
        if mixed is not None:
            tokenized = tokenizer.tokenize(mixed)

            for sentence in tokenized.sents:
                tokens = [t.text for t in sentence]
                tokens_lower = [t.lower() for t in tokens]

                overlap = set(tokens_lower) & aapi_keywords
                if not overlap:
                    continue
                overlap_eth = list(overlap)[0]

                noun_heads_counter[ethnicity_modified_nouns(sentence, overlap)] += 1
                aapi_counter_pass[overlap_eth] += 1

                window_text = (
                    window_mask_sentence(overlap_eth, tokens, tokens_lower)
                    .replace("\n", " ")
                    .strip()
                )
                if model.predict(window_text)[0][0] == "__label__0":
                    continue

                aapi_counter_pass[overlap_eth] += 1

                lex = build_group_lexicon(overlap)
                collect_verb(sentence, lex, verb_ethnicity_dict)
                collect_adj(sentence, lex, adj_ethnicity_dict)

            # you were already doing this every 1000 docs:
            if docs_in_shard > 0 and docs_in_shard % 10000 == 0:
                save_artifacts_safely(
                    aapi_counter_pass,
                )

                #
                flush_all_to_db(
                    noun_heads_counter,
                    verb_ethnicity_dict,
                    adj_ethnicity_dict,
                )

                # optionally clear in-memory to keep RAM low
                noun_heads_counter.clear()
                verb_ethnicity_dict.clear()
                adj_ethnicity_dict.clear()

            docs_in_shard += 1

            traversed += 1

    pbar.close()

    # final save at the end
    save_artifacts_safely(aapi_counter_pass)
    flush_all_to_db(
        noun_heads_counter,
        verb_ethnicity_dict,
        adj_ethnicity_dict,
    )

    save_progress(traversed)



import gzip
import json
from pathlib import Path


def iter_local_c4_files(folder_path):
    """get c4 and it"""
    folder_path = Path(folder_path)

    for gz_file in sorted(folder_path.glob("*.json.gz")):
        print(f"Reading: {gz_file}")

        with gzip.open(gz_file, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    yield data
                except Exception as e:
                    print(f"Skipping bad line: {e}")
                    continue
            print("ending reading this file, no more lines")





def main():
    """
    Run full ingestion + extraction loop
    """
    # In-memory counters
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
    print("🏁 Pipeline completed.")


if __name__ == "__main__":
    main()