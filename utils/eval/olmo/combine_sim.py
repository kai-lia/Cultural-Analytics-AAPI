"""
Study 2C (Analytical Aggregation):
Ethnicity-level aggregation of term → race cosine similarities.

This script aggregates term-level cosine similarities from `sim_vals`
to the ethnicity level, producing one row per ethnicity × POS.

Unit of analysis: ethnicity
"""

import duckdb
import pandas as pd
from pathlib import Path

# --------------------
# PATHS
# --------------------
DB_PATH = Path("data/input/eval/ethnicity_clean.duckdb")
IN_TABLE = "sim_vals"
OUT_TABLE = "ethnicity_race_agg"


# --------------------
# MAIN
# --------------------
def main():
    print("Connecting to DuckDB...")
    con = duckdb.connect(DB_PATH)

    print("Loading term-level similarities...")
    df = con.execute(
        f"""
        SELECT
            s.ethnicity,
            e.Region,
            s.race,
            s.pos,
            s.cosine_sim
        FROM {IN_TABLE} AS s
        LEFT JOIN ethnicity_clean AS e
            ON s.ethnicity = e.ethnicity
        WHERE s.race = 'asian'
        """
    ).fetchdf()

    if df.empty:
        raise ValueError("No data loaded from sim_vals")

    df["pos"] = df["pos"].str.lower()

    print(f"Loaded {len(df)} term-level rows")

    # --------------------
    # ANALYTICAL AGGREGATION (OPTION 3)
    # --------------------
    print("Aggregating to ethnicity level...")

    agg_df = (
        df.groupby(["ethnicity", "Region", "pos"])
        .agg(
            mean_sim=("cosine_sim", "mean"),
            sd_sim=("cosine_sim", "std"),
            n_terms=("cosine_sim", "count"),
        )
        .reset_index()
    )

    print(f"Produced {len(agg_df)} ethnicity-level rows")

    # --------------------
    # SAVE
    # --------------------
    print(f"Saving aggregated table → {OUT_TABLE}")

    con.execute(f"DROP TABLE IF EXISTS {OUT_TABLE}")
    con.register("agg_df_view", agg_df)

    con.execute(
        f"""
        CREATE TABLE {OUT_TABLE} AS
        SELECT * FROM agg_df_view
        """
    )

    con.close()

    print("Done.")
    print(f"New table `{OUT_TABLE}` schema:")
    print(agg_df.dtypes)


# --------------------
# ENTRY POINT
# --------------------
if __name__ == "__main__":
    """
    Run with:
        python -m utils.eval.olmo.aggregate_ethnicity_to_race
    """
    main()
