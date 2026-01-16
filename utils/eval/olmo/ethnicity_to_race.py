"""
Study 2B (Disaggregated):
Ethnicity descriptor terms → Race concepts

Produces per-term cosine similarities suitable for boxplots
and stores them in DuckDB table: sim_vals
"""

import sys
import torch
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from utils.save_load import fetch_duck_df, save_duck_df

# --------------------
# PATHS
# --------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DB_PATH = Path("data/input/eval/ethnicity_clean.duckdb")
OUT_TABLE = "sim_vals"
MODEL_NAME = "allenai/OLMo-1B-hf"


# --------------------
# MODEL LOADING (YOUR VERSION)
# --------------------
def load_olmo():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return tokenizer, model


# --------------------
# EMBEDDING
# --------------------
def embed_texts(texts, tokenizer, model, max_length=128):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

    return pooled.cpu().numpy()


# --------------------
# MAIN LOGIC
# --------------------
def main():
    print("Loading ethnicity_log_odds...")
    df = fetch_duck_df(DB_PATH, "ethnicity_log_odds")

    # Required columns check
    required = {
        "Ethnicity",
        "Race",
        "Top Adjs Log-Odds",
        "Top Verbs Log-Odds",
    }
    assert required.issubset(df.columns)

    tokenizer, model = load_olmo()

    rows = []

    # Precompute race embeddings
    print("Embedding race concepts...")
    race_texts = df[["Race"]].dropna().drop_duplicates()["Race"].tolist()
    race_embs = dict(
        zip(
            race_texts,
            embed_texts(race_texts, tokenizer, model),
        )
    )

    # Iterate per ethnicity
    for _, row in df.iterrows():
        eth = row["Ethnicity"]
        race = row["Race"]

        if not isinstance(race, str) or race not in race_embs:
            continue

        race_vec = race_embs[race].reshape(1, -1)

        for pos, col in [
            ("adj", "Top Adjs Log-Odds"),
            ("verb", "Top Verbs Log-Odds"),
        ]:
            terms = row[col]
            if not isinstance(terms, dict) or len(terms) == 0:
                continue

            term_list = list(terms.keys())
            term_embs = embed_texts(term_list, tokenizer, model)

            for term, vec in zip(term_list, term_embs):
                sim = cosine_similarity(vec.reshape(1, -1), race_vec)[0, 0]

                if np.isnan(sim):
                    continue

                rows.append(
                    {
                        "ethnicity": eth,
                        "race": race,
                        "pos": pos,
                        "term": term,
                        "cosine_sim": float(sim),
                    }
                )

    out_df = pd.DataFrame(rows)
    print(f"Saving {len(out_df)} rows to {OUT_TABLE}")

    save_duck_df(DB_PATH, out_df, OUT_TABLE)
    print("Done.")


if __name__ == "__main__":
    """python -m utils.eval.olmo.ethnicity_to_race"""
    main()
