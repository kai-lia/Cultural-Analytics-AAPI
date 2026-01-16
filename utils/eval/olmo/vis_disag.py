import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DB_PATH = Path("data/input/eval/ethnicity_clean.duckdb")


VIS_DIR = Path("data/output/eval/figures")
VIS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------
# load + join
# --------------------
con = duckdb.connect(DB_PATH, read_only=True)

df = con.execute(
    """
    SELECT
        s.ethnicity,
        e.Region,
        s.race,
        s.pos,
        s.cosine_sim
    FROM sim_vals AS s
    LEFT JOIN ethnicity_clean AS e
        ON s.ethnicity = e.ethnicity
    WHERE s.race = 'asian'
    AND e.Region IS NOT NULL
    """
).fetchdf()

con.close()


import numpy as np
from scipy import stats


import numpy as np
from scipy import stats


def pooled_region_stats(df, pos, region):
    sub = df[(df["pos"] == pos) & (df["Region"] == region)]["cosine_sim"].dropna()

    n = len(sub)
    mean = sub.mean()
    std = sub.std(ddof=1)
    se = std / np.sqrt(n)

    t_stat = mean / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    log_p = np.log(2) + stats.t.logsf(abs(t_stat), df=n - 1)
    d = mean / std

    return {
        "Region": region,
        "pos": pos,
        "B": round(mean, 4),
        "SE": round(se, 4),
        "t": round(t_stat, 2),
        "p": format_p_sci(log_p),
        "d": round(d, 3),
        "n": n,
    }


def format_p_sci(log_p, min_exp=-300):
    """
    Format log(p) as scientific notation.
    Handles underflow (log_p = -inf) safely.
    """
    if not np.isfinite(log_p):
        # numeric underflow → p is smaller than machine precision
        return f"< 1e{min_exp}"

    log10_p = log_p / np.log(10)
    exponent = int(np.floor(log10_p))
    mantissa = 10 ** (log10_p - exponent)

    return f"{mantissa:.2f}e{exponent}"


def region_stats(df, pos, region):
    sub = df[(df["pos"] == pos) & (df["Region"] == region)]["cosine_sim"].dropna()

    n = len(sub)
    mean = sub.mean()
    std = sub.std(ddof=1)
    se = std / np.sqrt(n)

    # one-sample t-test vs 0
    t_stat = mean / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    # Cohen's d
    d = mean / std

    return {
        "Region": region,
        "pos": pos,
        "B": round(mean, 4),
        "SE": round(se, 4),
        "t": round(t_stat, 2),
        "p": p_val,
        "d": round(d, 3),
        "n": n,
    }


df["pos"] = df["pos"].str.lower()

sns.set_theme(style="whitegrid", palette="tab10")


def plot_by_ethnicity_region(df, pos, title):
    sub = df[df["pos"] == pos]

    if sub.empty:
        print(f"No data for pos={pos}")
        return

    # mean-sort ethnicities (descending)
    order = (
        sub.groupby("ethnicity")["cosine_sim"].mean().sort_values(ascending=False).index
    )

    plt.figure(figsize=(max(10, len(order) * 0.4), 5))

    # region-colored boxes
    sns.boxplot(
        data=sub,
        x="ethnicity",
        y="cosine_sim",
        order=order,
        hue="Region",
        dodge=False,  # <-- critical
        width=0.6,
        palette="tab10",
        medianprops={"color": "black"},
        fliersize=0,
    )

    # neutral points
    sns.stripplot(
        data=sub,
        x="ethnicity",
        y="cosine_sim",
        order=order,
        color="black",
        jitter=True,
        size=2,
        alpha=0.35,
    )

    plt.title(title, fontsize=16)
    plt.ylabel("Cosine similarity")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")

    plt.legend(
        title="Region",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    sns.despine()
    plt.tight_layout()
    fname = f"{pos}_region_alignment.png"
    out_path = VIS_DIR / fname
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure → {out_path}")

    plt.show()

    plt.show()


def all_region_stats(df, pos):
    rows = []
    for region in sorted(df["Region"].dropna().unique()):
        stats_row = pooled_region_stats(df, pos=pos, region=region)
        rows.append(stats_row)
    return pd.DataFrame(rows)


# ---- verbs ----
plot_by_ethnicity_region(
    df,
    pos="verb",
    title="Ethnicity and Race Similarity (Verbs, Mean-Sorted)",
)

verb_region_stats = all_region_stats(df, pos="verb")
print("\n[VERB — region-level stats]")
print(verb_region_stats.to_string(index=False))


# ---- adjectives ----
plot_by_ethnicity_region(
    df,
    pos="adj",
    title="Ethnicity and Race Similarity (Adjectives, Mean-Sorted)",
)

adj_region_stats = all_region_stats(df, pos="adj")
print("\n[ADJ — region-level stats]")
print(adj_region_stats.to_string(index=False))
