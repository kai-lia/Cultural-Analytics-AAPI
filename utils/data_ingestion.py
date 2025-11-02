import io
import gzip
import json
import requests
import pandas as pd
from datasets import load_dataset

from warcio.archiveiterator import ArchiveIterator

INGEST_BATCH_SIZE = 1000

ETHNIC_GROUP_LIST_FILE = "data/DECENNIALDDHCB2020.T03004-Data.csv"
DOLMA_INGESTION_CHECKPOINT_FILE = "dolma_checkpoint.json"


def load_checkpoint(checpoint_file: str):
    try:
        with open(checpoint_file, "r") as f:
            return json.load(f)["offset"]
    except FileNotFoundError:
        return 0


def save_checkpoint(checpoint_file: str, offset: int):
    with open(checpoint_file, "w") as f:
        json.dump({"offset": offset}, f)


def process_batches(dataset, batch_size, start_index=0):
    stream = dataset.skip(start_index)

    batch = []
    current_index = start_index

    for example in stream:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch, current_index
            batch = []
        current_index += 1

    if batch:
        yield batch, current_index


def get_ethnic_group_list() -> set[str]:
    ethnic_df = pd.read_csv(ETHNIC_GROUP_LIST_FILE, header=1)
    ethnic_df["Ethnic Group"] = ethnic_df["Population Groups"].str.extract(
        r"(.*?) alone"
    )
    ethnic_df["Ethnic Group"] = ethnic_df["Ethnic Group"].replace(
        "Chinese, except Taiwanese", "Chinese"
    )
    ethnic_df = ethnic_df[~ethnic_df["Ethnic Group"].str.contains("Other", na=False)]
    ethnic_df = ethnic_df.dropna(subset=["Ethnic Group"])
    ethnic_group_list = ethnic_df["Ethnic Group"].unique().tolist()
    ethnic_group_list += ["AAPI", "Asian American", "Asian", "Pacific Islander"]

    return set(ethnic_group_list)


def get_dolma_dataset(
    batch_size: int = INGEST_BATCH_SIZE, is_use_last_checkpoint: bool = True
):
    """
    @article{dolma,
    title = {{Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research}},
    author={
        Luca Soldaini and Rodney Kinney and Akshita Bhagia and Dustin Schwenk and David Atkinson and
        Russell Authur and Ben Bogin and Khyathi Chandu and Jennifer Dumas and Yanai Elazar and
        Valentin Hofmann and Ananya Harsh Jha and Sachin Kumar and Li Lucy and Xinxi Lyu and
        Nathan Lambert and Ian Magnusson and Jacob Morrison and Niklas Muennighoff and Aakanksha Naik and
        Crystal Nam and Matthew E. Peters and Abhilasha Ravichander and Kyle Richardson and Zejiang Shen and
        Emma Strubell and Nishant Subramani and Oyvind Tafjord and Pete Walsh and Luke Zettlemoyer and
        Noah A. Smith and Hannaneh Hajishirzi and Iz Beltagy and Dirk Groeneveld and Jesse Dodge and Kyle Lo
    },
    year = {2024},
    journal={arXiv preprint},
    }
    """
    start_index = 0
    if is_use_last_checkpoint:
        start_index = load_checkpoint(DOLMA_INGESTION_CHECKPOINT_FILE)

    ds = load_dataset(
        "allenai/dolma", "v1_7", streaming=True, split="train", trust_remote_code=True
    )

    for batch, curr_index in process_batches(ds, batch_size, start_index):
        save_checkpoint(DOLMA_INGESTION_CHECKPOINT_FILE, curr_index)
        yield batch
