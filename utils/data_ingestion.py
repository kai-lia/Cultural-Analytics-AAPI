import io
import requests
import gzip
import pandas as pd

from warcio.archiveiterator import ArchiveIterator


def get_ethnic_group_list():
    ethnic_df = pd.read_csv("data/DECENNIALDDHCB2020.T03004-Data.csv", header=1)
    ethnic_df["Ethnic Group"] = ethnic_df["Population Groups"].str.extract(
        r"(.*?) alone"
    )
    ethnic_df["Ethnic Group"] = ethnic_df["Ethnic Group"].replace(
        "Chinese, except Taiwanese", "Chinese"
    )
    ethnic_df = ethnic_df[~ethnic_df["Ethnic Group"].str.contains("Other", na=False)]
    ethnic_df = ethnic_df.dropna(subset=["Ethnic Group"])
    ethnic_group_list = sorted(ethnic_df["Ethnic Group"].unique().tolist())
    ethnic_group_list += ["AAPI", "Asian American", "Asian", "Pacific Islander"]

    return ethnic_group_list
