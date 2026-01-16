import ast
import duckdb
import pandas as pd
from collections import Counter
from pathlib import Path


def fetch_duck_df(path, name):
    """load in ethncity data from duckdb"""
    print("Loading Data... ")
    with duckdb.connect(str(path), read_only=True) as con:
        df = con.execute(f"SELECT * FROM {name}").fetchdf()
    return df


def save_duck_df(path, df, name):
    """saving table in the db"""
    print("Saving df as duckdb ...")
    path.parent.mkdir(parents=True, exist_ok=True)

    counter_like_cols = [
        "Adjs",
        "Verbs",
        "Nouns",
        "Adjs Log-Odds",
        "Verbs Log-Odds",
        "Top Adjs Log-Odds",
        "Top Verbs Log-Odds",
    ]  # all columns containing counters

    for col in counter_like_cols:  # preserving counter format
        if col in df.columns:
            df[col] = df[col].apply(dict)

    con = duckdb.connect(str(path))
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {name} AS
        SELECT * FROM df
    """
    )
    con.close()
