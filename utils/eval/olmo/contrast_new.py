"""
Race Alignment Contrast Analysis

Tests whether ethnicities align more strongly with their own race category
than with other race categories.

Unit of analysis: ethnicity
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_1samp

DB_PATH = Path("data/input/eval/ethnicity_clean.duckdb")


# --------------------------------------------------
# LOAD SIMILARITY TABLE
# --------------------------------------------------


def load_sim_vals(db_path):
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(
        """
        SELECT
            ethnicity,
            race,
            pos,
            cosine_sim
        FROM sim_vals
        WHERE race IS NOT NULL
    """
    ).fetchdf()
    con.close()
    return df


# --------------------------------------------------
# BUILD CONTRASTS
# --------------------------------------------------


def compute_contrast(df, pos):
    """
    For each ethnicity:
      sim_own_race - mean(sim_other_races)
    """
    df = df[df["pos"] == pos]

    rows = []

    for eth, g in df.groupby("ethnicity"):
        own_race = g["race"].iloc[0]

        own_vals = g[g["race"] == own_race]["cosine_sim"].values
        other_vals = g[g["race"] != own_race]["cosine_sim"].values

        # require at least one comparison
        if len(own_vals) == 0 or len(other_vals) == 0:
            continue

        sim_own = float(np.mean(own_vals))
        sim_other = float(np.mean(other_vals))

        rows.append(
            {
                "ethnicity": eth,
                "own_race": own_race,
                "pos": pos,
                "sim_own_race": sim_own,
                "sim_other_races": sim_other,
                "diff": sim_own - sim_other,
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------


def save_contrast_table(db_path, df):
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS race_alignment_contrast")
    con.execute("CREATE TABLE race_alignment_contrast AS SELECT * FROM df")
    con.close()


# --------------------------------------------------
# MAIN
# --------------------------------------------------


def main():
    df = load_sim_vals(DB_PATH)

    all_results = []

    for pos in ["adj", "verb"]:
        print(f"Running race-alignment contrast for {pos}...")
        res = compute_contrast(df, pos)

        if len(res) == 0:
            print(f"⚠️ No valid rows for {pos}")
            continue

        # one-sample t-test across ethnicities
        t, p = ttest_1samp(res["diff"], 0.0)

        print(
            f"{pos.upper()} — mean diff = {res['diff'].mean():.4f}, "
            f"t = {t:.3f}, p = {p:.4g}, n = {len(res)}"
        )

        res["t_stat"] = t
        res["p_value"] = p
        res["n"] = len(res)

        all_results.append(res)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        save_contrast_table(DB_PATH, final_df)
        print("Saved table: race_alignment_contrast")

    print("Done.")


if __name__ == "__main__":
    main()
