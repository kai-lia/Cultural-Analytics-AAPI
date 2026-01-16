"""
Visualize ethnicity–category cosine similarities (Study 2A)

Produces boxplots of cosine similarity between ethnicity embeddings
and their parent race (or region) embeddings, stratified by POS.

Reads from DuckDB table: sim_vals
"""

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------
# CONFIG
# --------------------

DB_PATH = Path("data/input/eval/ethnicity_clean.duckdb")
TABLE = "sim_vals"
OUT_DIR = Path("data/output/eval/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# LOAD DATA
# --------------------


def load_sim_vals():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute(
        """
        SELECT
            ethnicity,
            category_label,
            category_type,
            pos,
            cosine_sim
        FROM sim_vals
        WHERE cosine_sim IS NOT NULL
        """
    ).fetchdf()
    con.close()
    return df


# --------------------
# PLOT
# --------------------


def plot_boxplots(df):
    """
    Boxplot of cosine similarity by category, split by POS.
    """

    # Only race for now (you can add region later)
    df = df[df["category_type"] == "race"]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="category_label", y="cosine_sim", hue="pos", showfliers=True)

    plt.xlabel("Race category")
    plt.ylabel("Cosine similarity")
    plt.title("Ethnicity–Race Similarity by Descriptor Type")
    plt.legend(title="POS")
    plt.tight_layout()

    out_path = OUT_DIR / "study2a_ethnicity_race_similarity_boxplot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved figure to {out_path}")


# --------------------
# MAIN
# --------------------


def main():
    print("Loading similarity values...")
    df = load_sim_vals()

    print(f"Loaded {len(df)} rows")
    print(df.head())

    print("Plotting...")
    plot_boxplots(df)

    print("Done.")


if __name__ == "__main__":
    """python -m utils.eval.olmo.visualize"""
    main()
