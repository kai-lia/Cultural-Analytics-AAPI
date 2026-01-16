import sys
import math
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter

# from my code
from utils.eval.cleaning_data.clean_data import run as cleaning_pipeline
from utils.save_load import fetch_duck_df, save_duck_df


# setting to main
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# paths to data
DB_PATH = Path("data/input/eval/ethnicity_clean.duckdb")
OUT_CSV = Path("data/output/eval/top_logodds.csv")


# combining counts for log odds calc
def combine_counts(ethnicity_df, col):
    """takes a df and a column name (adjs, verbs, nouns) and creates a merged Counter"""
    merged = Counter()
    for ctr in ethnicity_df[col]:
        merged.update(ctr)

    print(f"global {col} counter created with {len(merged)} unique {col}")
    return merged


def top_k_log_odds_terms(log_odds_dict, k=0, min_score=0.0):
    """return top-k terms with log-odds >= min_score
    result: most distinct words per word group"""
    if not isinstance(log_odds_dict, dict):
        return {}
    # sorting and getting the top terms of log odds
    pos_scores = [v for v in log_odds_dict.values() if v > 0]
    if pos_scores:
        median_score = np.median(pos_scores)
    else:
        median_score = 0.0

    items = [
        (term, score) for term, score in log_odds_dict.items() if score >= median_score
    ]

    if k:
        items = sorted(items, key=lambda x: x[1], reverse=True)[:k]
    else:
        items = sorted(items, key=lambda x: x[1], reverse=True)

    return dict(items)


def weighted_log_odds(group_counter, global_counter, alpha=0.01, min_count=5):
    """Monroe et al. 2008 weighted log-odds with Dirichlet prior:
    counts_group: Counter of words in target group
    counts_global: Counter of words in all other groups
    """

    vocab = set(group_counter) | set(global_counter)
    V = len(vocab)

    # totals
    group_size = sum(group_counter.values())
    global_size = sum(global_counter.values())

    # alpha prior: proportional to total frequency
    alpha_vec = {w: alpha * global_counter.get(w, 0) for w in vocab}
    alpah_prior = sum(alpha_vec.values())

    scores = {}

    for word in vocab:
        # how many times word appears
        group_word_count = group_counter.get(word, 0)  # word appearance # in group
        global_word_count = global_counter.get(word, 0)  # word appearance # in global

        # frequency threshold
        if group_word_count < min_count:
            continue

        # posterior means
        grp_prb = (group_word_count + alpha_vec[word]) / (group_size + alpah_prior)
        glbl_prb = (global_word_count + alpha_vec[word]) / (global_size + alpah_prior)

        # weighted logodds
        delta = np.log(grp_prb / (1 - grp_prb)) - np.log(glbl_prb / (1 - glbl_prb))

        # variance gets rid of rare words
        var = 1 / (group_word_count + alpha_vec[word]) + 1 / (
            global_word_count + alpha_vec[word]
        )

        scores[word] = delta / math.sqrt(var)

    return scores


def export_top_log_odds_long(df, out_path):
    rows = []

    for _, row in df.iterrows():
        eth = row["Ethnicity"]

        for pos, col in [
            ("adj", "Top Adjs Log-Odds"),
            ("verb", "Top Verbs Log-Odds"),
        ]:
            top_terms = row.get(col, {})
            if not isinstance(top_terms, dict):
                continue

            for term, score in top_terms.items():
                rows.append(
                    {
                        "Ethnicity": eth,
                        "pos": pos,
                        "term": term,
                        "log_odds": score,
                    }
                )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")


def top_k_combined_terms(adj_log_odds, verb_log_odds, k=5):
    """
    Combine adjective + verb log-odds dictionaries,
    rank jointly, and return top-k terms (term, pos, score).
    """
    combined = []

    if isinstance(adj_log_odds, dict):
        combined.extend([(term, "adj", score) for term, score in adj_log_odds.items()])

    if isinstance(verb_log_odds, dict):
        combined.extend(
            [(term, "verb", score) for term, score in verb_log_odds.items()]
        )

    combined = sorted(combined, key=lambda x: x[2], reverse=True)[:k]
    return combined


def export_top5_combined(df, out_path):
    rows = []

    for _, row in df.iterrows():
        top5 = top_k_combined_terms(
            row.get("Adjs Log-Odds"),
            row.get("Verbs Log-Odds"),
            k=5,
        )

        rows.append(
            {
                "Ethnicity": row["Ethnicity"],
                "Top_5_Terms": ", ".join([t for t, _, _ in top5]),
                "Top_5_POS": ", ".join([p for _, p, _ in top5]),
                "Top_5_LogOdds": ", ".join([f"{s:.3f}" for _, _, s in top5]),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Saved combined top-5 CSV to {out_path}")


def max_adj_count_per_ethnicity(ethnicity_df, adj_col="Adjs", eth_col="Ethnicity"):
    """
    For each ethnicity, return the maximum adjective count.
    """
    rows = []

    for _, row in ethnicity_df.iterrows():
        eth = row[eth_col]
        adjs = row[adj_col]

        if isinstance(adjs, dict) and adjs:
            max_count = max(adjs.values())
        else:
            max_count = 0

        rows.append({"Ethnicity": eth, "Max_Adj_Count": max_count})

    return pd.DataFrame(rows)


def run():
    if not DB_PATH.exists():
        # making sure clean is there
        cleaning_pipeline()

    ethnicity_df = fetch_duck_df(DB_PATH, "ethnicity_clean")

    result = max_adj_count_per_ethnicity(ethnicity_df)

    print(result)

    # creating verb global, all verbs together
    global_verb_counter = combine_counts(ethnicity_df, "Verbs")
    # creating adj global, all adj together
    global_adj_counter = combine_counts(ethnicity_df, "Adjs")

    # getting regular log odds
    ethnicity_df["Verbs Log-Odds"] = ethnicity_df["Verbs"].apply(
        lambda x: weighted_log_odds(x, global_verb_counter)
    )
    ethnicity_df["Adjs Log-Odds"] = ethnicity_df["Adjs"].apply(
        lambda x: weighted_log_odds(x, global_adj_counter)
    )

    # top words for each ethnicity from log odds
    ethnicity_df["Top Adjs Log-Odds"] = ethnicity_df["Adjs Log-Odds"].apply(
        lambda x: top_k_log_odds_terms(x, 50)
    )
    ethnicity_df["Top Verbs Log-Odds"] = ethnicity_df["Verbs Log-Odds"].apply(
        lambda x: top_k_log_odds_terms(x, 50)
    )

    save_duck_df(DB_PATH, ethnicity_df, "ethnicity_log_odds")

    export_top_log_odds_long(ethnicity_df, OUT_CSV)

    TOP5_COMBINED_CSV = Path("data/output/eval/top5_combined_logodds.csv")

    export_top5_combined(ethnicity_df, TOP5_COMBINED_CSV)


if __name__ == "__main__":
    """run via the command python -m utils.eval.log_odds.log_odds"""
    run()
