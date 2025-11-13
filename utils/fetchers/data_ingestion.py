import json
import tqdm
import logging
import pandas as pd
from datasets import load_dataset

from warcio.archiveiterator import ArchiveIterator
from typing import (
    Iterable,
)
import itertools
import os

INGEST_BATCH_SIZE = 1000

ETHNIC_GROUP_LIST_FILE = "data/DECENNIALDDHCB2020.T03004-Data.csv"
DOLMA_INGESTION_CHECKPOINT_FILE = "dolma_checkpoint.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("cultural.analytics.util")


def _checkpoint_key(source_filter: str | None) -> str:
    return f"ALL" if not source_filter else f"source={source_filter}"


def load_checkpoint(checpoint_file: str, key: str) -> int:
    """
    Backward compatible:
    - If file contains {"offset": N}, use it only for key=="ALL"; others default to 0.
    - If file contains {"keys": {"ALL": N, "source=foo": M}}, use the matching key.
    """
    try:
        with open(checpoint_file, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "keys" in data and isinstance(data["keys"], dict):
            return int(data["keys"].get(key, 0))
        # legacy single-offset format
        if key == "ALL" and "offset" in data:
            return int(data["offset"])
        return 0
    except FileNotFoundError:
        return 0


def save_checkpoint(checpoint_file: str, key: str, offset: int) -> None:
    # Load existing (if any), migrate legacy format to multi-key
    payload = {"keys": {}}
    if os.path.exists(checpoint_file):
        try:
            with open(checpoint_file, "r") as f:
                data = json.load(f)
            if (
                isinstance(data, dict)
                and "keys" in data
                and isinstance(data["keys"], dict)
            ):
                payload = {"keys": data["keys"]}
            elif "offset" in data:
                # migrate legacy ALL offset
                payload = {"keys": {"ALL": int(data["offset"])}}
        except Exception:
            # If unreadable/corrupt, start fresh
            payload = {"keys": {}}

    payload["keys"][key] = int(offset)
    with open(checpoint_file, "w") as f:
        json.dump(payload, f)


def _skip_iterable(iterable: Iterable, n: int) -> Iterable:
    """Safe skip for any iterable/generator."""
    if n <= 0:
        return iterable
    return itertools.islice(iterable, n, None)


def process_batches(dataset: Iterable, batch_size: int, start_index: int = 0):
    """
    Works with either a Hugging Face streaming IterableDataset (has .skip)
    or any generic Iterable/generator.
    """
    if hasattr(dataset, "skip"):
        stream = dataset.skip(start_index)  # preserves HF streaming efficiency
    else:
        stream = _skip_iterable(dataset, start_index)

    batch = []
    current_index = start_index

    for example in tqdm.tqdm(
        stream,
        desc="Process batches",
        unit="batch",
    ):
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
    batch_size: int = INGEST_BATCH_SIZE,
    is_use_last_checkpoint: bool = True,
    source_filter: str | None = None,
    num_batch: float = float("inf"),
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
    ck_key = _checkpoint_key(source_filter)
    start_index = (
        load_checkpoint(DOLMA_INGESTION_CHECKPOINT_FILE, ck_key)
        if is_use_last_checkpoint
        else 0
    )

    if start_index:
        log.info(f"Resuming Dolma stream from offset {start_index:,} (key={ck_key})")
    else:
        log.info(f"Starting Dolma stream from offset 0 (key={ck_key})")

    ds = load_dataset(
        "allenai/dolma", "v1_7", streaming=True, split="train", trust_remote_code=True
    )

    if source_filter:
        log.info(f"Filtering stream: source == {source_filter!r}")
        # Prefer HF's built-in filter to preserve streaming ops like .skip()
        if hasattr(ds, "filter"):
            try:
                ds = ds.filter(
                    lambda ex: isinstance(ex, dict)
                    and ex.get("source") == source_filter
                )
            except Exception as e:
                log.warning(
                    f"HF filter failed ({e!r}); falling back to generator filter."
                )
                ds = (
                    ex
                    for ex in ds
                    if isinstance(ex, dict) and ex.get("source") == source_filter
                )
        else:
            ds = (
                ex
                for ex in ds
                if isinstance(ex, dict) and ex.get("source") == source_filter
            )
    curr_batch = 0
    for batch, curr_index in process_batches(
        ds, batch_size=batch_size, start_index=start_index
    ):
        # Save the checkpoint for the active filter key
        save_checkpoint(DOLMA_INGESTION_CHECKPOINT_FILE, ck_key, curr_index)
        yield batch

        curr_batch += 1
        if curr_batch >= num_batch:
            return
