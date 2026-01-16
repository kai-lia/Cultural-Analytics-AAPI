"""
Contrast Analysis: Ethnicity → Own Race vs Other Races

This script tests whether ethnicities align more strongly with
their own race descriptor than with other race descriptors.

Core question:
    When people talk about a race (e.g., "Asian"),
    are they primarily invoking specific ethnicities?

Pipeline:
1. Load ethnicity-level pseudo-documents (adjs / verbs)
2. Build race-level pseudo-documents
3. Compute cosine similarity:
       sim(ethnicity → own race)
       sim(ethnicity → other races)
4. Store long-format similarity values for boxplots
5. Run paired tests per ethnicity and globally

Output DuckDB tables:
- race_contrast_vals   (for boxplots)
- race_contrast_stats  (paired test results)
"""

import sys
import duckdb
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_rel

from utils.save_load import fetch_duck_df, save_duck_df


# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = Path("data/input/eval/ethnicity_clean.duckdb")
MODEL_NAME = "allenai/OLMo-1B-hf"

BATCH_SIZE = 8
MAX_LENGTH = 512


# ------------------------------------------------------------------
# OLMo LOADER (YOUR REQUIRED VERSION)
# ------------------------------------------------------------------


def load_olmo():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    model = AutoModel.from_pretrained(MODEL_NAME, dtype=dtype, device_map=device_map)
    model.eval()
    return tokenizer, model


def embed_texts(texts, tokenizer, model):
    embs = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1)

        embs.append(pooled.cpu().numpy())

    return np.vstack(embs)


# ------------------------------------------------------------------
# DATA PREP
# ------------------------------------------------------------------


def build_pseudodocs(df, pos_col):
    """
    pos_col: 'Top Adjs Log-Odds' or 'Top Verbs Log-Odds'
    """
    docs = {}
    for _, row in df.iterrows():
        terms = row[pos_col]
        if isinstance(terms, dict) and len(terms) > 0:
            docs[row["Ethnicity"]] = " ".join(terms.keys())
    return docs


def build_race_docs(eth_docs, eth_to_race):
    race_docs = defaultdict(list)
    for eth, text in eth_docs.items():
        race = eth_to_race.get(eth)
        if race:
            race_docs[race].append(text)

    return {r: " ".join(v) for r, v in race_docs.items()}


# ------------------------------------------------------------------
# CORE CONTRAST COMPUTATION
# ------------------------------------------------------------------


def compute_contrast(eth_docs, race_docs, eth_to_race, tokenizer, model, pos_label):
    eth_labels = list(eth_docs.keys())
    race_labels = list(race_docs.keys())

    eth_embs = embed_texts(list(eth_docs.values()), tokenizer, model)
    race_embs = embed_texts(list(race_docs.values()), tokenizer, model)

    eth_idx = {e: i for i, e in enumerate(eth_labels)}
    race_idx = {r: i for i, r in enumerate(race_labels)}

    rows = []

    for eth in eth_labels:
        own_race = eth_to_race.get(eth)
        if own_race not in race_idx:
            continue

        evec = eth_embs[eth_idx[eth]].reshape(1, -1)

        # own race
        rvec_own = race_embs[race_idx[own_race]].reshape(1, -1)
        sim_own = cosine_similarity(evec, rvec_own)[0, 0]

        rows.append(
            {
                "ethnicity": eth,
                "race": own_race,
                "race_type": "own",
                "pos": pos_label,
                "cosine_sim": sim_own,
            }
        )

        # other races
        for other_race in race_labels:
            if other_race == own_race:
                continue

            rvec_other = race_embs[race_idx[other_race]].reshape(1, -1)
            sim_other = cosine_similarity(evec, rvec_other)[0, 0]

            rows.append(
                {
                    "ethnicity": eth,
                    "race": other_race,
                    "race_type": "other",
                    "pos": pos_label,
                    "cosine_sim": sim_other,
                }
            )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------------


def compute_stats(df):
    stats = []

    for (eth, pos), g in df.groupby(["ethnicity", "pos"]):
        own = g[g["race_type"] == "own"]["cosine_sim"].values
        other = g[g["race_type"] == "other"]["cosine_sim"].values

        if len(own) == 0 or len(other) == 0:
            continue

        t, p = ttest_rel(np.repeat(own.mean(), len(other)), other)

        stats.append(
            {
                "ethnicity": eth,
                "pos": pos,
                "mean_own": own.mean(),
                "mean_other": other.mean(),
                "t_stat": t,
                "p_value": p,
                "n_other": len(other),
            }
        )

    return pd.DataFrame(stats)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------


def main():
    print("Loading ethnicity_log_odds...")
    df = fetch_duck_df(DB_PATH, "ethnicity_log_odds")

    # Build ethnicity → race map directly from table
    eth_to_race = (
        df[["Ethnicity", "Race"]]
        .dropna()
        .drop_duplicates()
        .set_index("Ethnicity")["Race"]
        .to_dict()
    )

    tokenizer, model = load_olmo()

    all_vals = []
    all_stats = []

    for pos, col in [
        ("adj", "Top Adjs Log-Odds"),
        ("verb", "Top Verbs Log-Odds"),
    ]:
        print(f"Running contrast for {pos}...")

        eth_docs = build_pseudodocs(df, col)
        race_docs = build_race_docs(eth_docs, eth_to_race)

        sim_df = compute_contrast(
            eth_docs, race_docs, eth_to_race, tokenizer, model, pos
        )

        stats_df = compute_stats(sim_df)

        all_vals.append(sim_df)
        all_stats.append(stats_df)

    sim_vals = pd.concat(all_vals, ignore_index=True)
    sim_stats = pd.concat(all_stats, ignore_index=True)

    print("Saving results to DuckDB...")

    save_duck_df(DB_PATH, sim_vals, "race_contrast_vals")
    save_duck_df(DB_PATH, sim_stats, "race_contrast_stats")

    print("Done.")
    print("Tables created:")
    print(" - race_contrast_vals   (for boxplots)")
    print(" - race_contrast_stats  (paired tests)")


if __name__ == "__main__":
    """
    Run with:
        python -m utils.eval.olmo.contrast
    """
    main()
