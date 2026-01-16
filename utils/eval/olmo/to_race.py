import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = "data/input/eval/ethnicity_clean.duckdb"

# --------------------
# load data
# --------------------
con = duckdb.connect(DB_PATH, read_only=True)
df = con.execute(
    """
    SELECT ethnicity, race, region, pos, cosine_sim
    FROM sim_vals
    WHERE race = 'asian'
    """
).fetchdf()
con.close()

df["pos"] = df["pos"].str.lower()

sns.set_theme(style="whitegrid", palette="tab10")


def plot_by_ethnicity_region(
    df,
    pos,
    title,
):
    sub = df[df["pos"] == pos]

    if sub.empty:
        print(f"No data for pos={pos}")
        return

    # mean-sort ethnicities (descending)
    order = (
        sub.groupby("ethnicity")["cosine_sim"].mean().sort_values(ascending=False).index
    )

    plt.figure(figsize=(max(10, len(order) * 0.4), 5))

    # neutral boxes
    sns.boxplot(
        data=sub,
        x="ethnicity",
        y="cosine_sim",
        order=order,
        width=0.6,
        boxprops={"facecolor": "#E6E6E6"},
        medianprops={"color": "black"},
        fliersize=0,
    )

    # region-colored points
    sns.stripplot(
        data=sub,
        x="ethnicity",
        y="cosine_sim",
        order=order,
        hue="region",
        dodge=False,
        jitter=True,
        size=3,
        alpha=0.6,
        palette="tab10",
    )

    plt.title(title, fontsize=16)
    plt.ylabel("Cosine similarity")
    plt.xlabel("")
    plt.xticks(rotation=45, ha="right")

    # legend cleanup
    plt.legend(
        title="Region",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    sns.despine()
    plt.tight_layout()
    plt.show()


# --------------------
# figures
# --------------------
plot_by_ethnicity_region(
    df,
    pos="verb",
    title="Ethnicity → Asian Race Similarity (Verbs, Mean-Sorted by Ethnicity)",
)

plot_by_ethnicity_region(
    df,
    pos="adj",
    title="Ethnicity → Asian Race Similarity (Adjectives, Mean-Sorted by Ethnicity)",
)
