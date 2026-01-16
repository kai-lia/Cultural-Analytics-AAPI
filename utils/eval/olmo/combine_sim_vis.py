import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DB_PATH = Path("data/input/eval/ethnicity_clean.duckdb")
TABLE = "ethnicity_race_agg"


VIS_DIR = Path("data/output/eval/figures")
VIS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------
# load aggregated data
# --------------------
con = duckdb.connect(DB_PATH, read_only=True)

df = con.execute(
    f"""
    SELECT ethnicity, Region, pos, mean_sim, sd_sim, n_terms
    FROM {TABLE}
    """
).fetchdf()

con.close()

df["pos"] = df["pos"].str.lower()
sns.set_theme(style="whitegrid", palette="tab10")


def plot_region_summary(df, pos, title):
    sub = df[df["pos"] == pos]

    plt.figure(figsize=(6, 5))

    sns.boxplot(
        data=sub,
        x="Region",
        y="mean_sim",
        palette="tab10",
    )

    sns.stripplot(
        data=sub,
        x="Region",
        y="mean_sim",
        color="black",
        size=6,
        alpha=0.6,
    )

    plt.title(title, fontsize=16)
    plt.ylabel("Mean cosine similarity")
    plt.xlabel("")

    sns.despine()
    plt.tight_layout()
    # ---- save ----
    fname = f"{pos}_region_alignment.png"
    out_path = VIS_DIR / fname
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure → {out_path}")

    plt.show()


from scipy import stats
import numpy as np


def print_region_stats(df, pos):
    """
    One-sample t-tests (vs 0) computed separately for each Region.
    Uses ethnicity-level mean_sim values.
    """
    sub = df[df["pos"] == pos]

    print(f"\n[{pos.upper()} — region-level ethnicity stats]\n")

    for region, g in sub.groupby("Region"):
        x = g["mean_sim"].dropna().values
        n = len(x)

        if n < 2:
            print(f"{region}: n < 2 (skipping)")
            continue

        mean = x.mean()
        sd = x.std(ddof=1)
        se = sd / np.sqrt(n)

        t, p = stats.ttest_1samp(x, popmean=0.0)
        d = mean / sd if sd != 0 else float("nan")

        print(f"{region}")
        print(f"  B  = {mean:.4f}")
        print(f"  SE = {se:.4f}")
        print(f"  t  = {t:.2f}")
        print(f"  p  = {p:.3e}")
        print(f"  d  = {d:.3f}")
        print(f"  n  = {n} ethnicities\n")


def print_global_stats(df, pos):
    """
    Global one-sample t-test on ethnicity-level mean_sim values.
    Does NOT affect plotting.
    """
    sub = df[df["pos"] == pos]

    x = sub["mean_sim"].dropna().values
    n = len(x)

    mean = x.mean()
    sd = x.std(ddof=1)
    se = sd / np.sqrt(n)

    t, p = stats.ttest_1samp(x, popmean=0.0)

    # one-sample Cohen's d
    d = mean / sd

    print(f"\n[{pos.upper()} — ethnicity-level]")
    print(f"B  = {mean:.4f}")
    print(f"SE = {se:.4f}")
    print(f"t  = {t:.2f}")
    print(f"p  = {p:.3e}")
    print(f"d  = {d:.3f}")
    print(f"n  = {n} ethnicities")


plot_region_summary(
    df,
    pos="verb",
    title="Region-Level Alignment with Asian Race (Verbs)",
)
print_global_stats(df, pos="verb")

plot_region_summary(
    df,
    pos="adj",
    title="Region-Level Alignment with Asian Race (Adjectives)",
)
print_global_stats(df, pos="adj")


print_region_stats(df, pos="verb")


print_region_stats(df, pos="adj")
