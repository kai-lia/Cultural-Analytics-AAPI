import json
import os
import pytest
import pandas as pd

# Import your functions under test
from utils.data_ingestion import (
    load_checkpoint,
    save_checkpoint,
    process_batches,
    get_ethnic_group_list,
    get_dolma_dataset,
)

# -----------------------
# Helpers / Fakes
# -----------------------


class FakeStream:
    def __init__(self, data):
        self._data = data

    def __iter__(self):
        for x in self._data:
            yield x


class FakeDataset:
    """
    Mimics a streaming dataset with a .skip(n) method that returns an iterable.
    """

    def __init__(self, data):
        self.data = data
        self.last_skip = None

    def skip(self, n):
        self.last_skip = n
        return FakeStream(self.data[n:])


@pytest.fixture
def fake_ethnic_df():
    # Covers: rename "Chinese, except Taiwanese", drop "Other ...",
    # handle normal groups, and ignore NaNs.
    return pd.DataFrame(
        {
            "Population Groups": [
                "Chinese, except Taiwanese alone",
                "Filipino alone",
                "Other Asian alone",
                "Taiwanese alone",
                "Japanese alone",
                None,
                "Korean alone",
            ]
        }
    )


def test_get_ethnic_group_list_monkeypatched(monkeypatch, fake_ethnic_df):
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: fake_ethnic_df.copy())

    result = get_ethnic_group_list()
    extras = ["AAPI", "Asian American", "Asian", "Pacific Islander"]
    assert "Chinese" in result
    assert "Filipino" in result
    assert "Japanese" in result
    assert "Korean" in result
    assert "Taiwanese" in result
    assert not any("Other" in g for g in result)
    for e in extras:
        assert e in result


def test_load_checkpoint_missing_file_returns_zero(tmp_path):
    # File doesn't exist:
    path = tmp_path / "nope.json"
    assert load_checkpoint(str(path)) == 0


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "cp.json"
    save_checkpoint(str(path), 12345)
    assert path.exists()
    # read back
    assert load_checkpoint(str(path)) == 12345

    # sanity check of file content
    with open(path) as f:
        payload = json.load(f)
    assert payload == {"offset": 12345}


# -----------------------
# process_batches
# -----------------------


def test_process_batches_exact_multiple():
    data = [{"text": f"row{i}"} for i in range(10)]
    ds = FakeDataset(data)

    batches = list(process_batches(ds, batch_size=5, start_index=0))
    # Should produce two full batches
    assert len(batches) == 2

    # Each element is (batch, next_index)
    (b1, i1), (b2, i2) = batches
    assert [x["text"] for x in b1] == [f"row{i}" for i in range(5)]
    assert [x["text"] for x in b2] == [f"row{i}" for i in range(5, 10)]

    # next_index increments by batch size
    assert i1 == 4
    assert i2 == 9

    # .skip called with the right start_index
    assert ds.last_skip == 0


def test_process_batches_with_remainder_and_start_index():
    data = [{"text": f"row{i}"} for i in range(9)]
    ds = FakeDataset(data)

    # start_index=2, batch_size=4 -> batches: rows[2:6], rows[6:9]
    out = list(process_batches(ds, batch_size=4, start_index=2))
    assert len(out) == 2
    (b1, i1), (b2, i2) = out

    assert [x["text"] for x in b1] == ["row2", "row3", "row4", "row5"]
    assert i1 == 5

    assert [x["text"] for x in b2] == ["row6", "row7", "row8"]
    assert i2 == 9

    assert ds.last_skip == 2


# -----------------------
# get_dolma_dataset
# -----------------------


def test_get_dolma_dataset_starts_from_zero_and_saves_each_batch(monkeypatch):
    # Fake stream of 7 rows
    data = [{"text": f"row{i}"} for i in range(7)]
    fake_ds = FakeDataset(data)

    # Monkeypatch load_dataset to return our fake dataset
    def fake_load_dataset(*args, **kwargs):
        # Optional: assert the call looks right
        assert kwargs.get("streaming") is True
        assert kwargs.get("split") == "train"
        assert kwargs.get("trust_remote_code") is True
        return fake_ds

    monkeypatch.setattr("utils.data_ingestion.load_dataset", fake_load_dataset)

    # Start from checkpoint=0
    monkeypatch.setattr("utils.data_ingestion.load_checkpoint", lambda *_: 0)

    # Record all save_checkpoint calls
    saved = []

    def fake_save_checkpoint(path, offset):
        saved.append(offset)

    monkeypatch.setattr("utils.data_ingestion.save_checkpoint", fake_save_checkpoint)

    # Iterate the generator
    batches = list(get_dolma_dataset(batch_size=3, is_use_last_checkpoint=True))

    # Expect 3,3,1 rows
    assert len(batches) == 3
    assert [x["text"] for x in batches[0]] == ["row0", "row1", "row2"]
    assert [x["text"] for x in batches[1]] == ["row3", "row4", "row5"]
    assert [x["text"] for x in batches[2]] == ["row6"]

    assert saved == [2, 5, 7]

    # .skip called with start_index=0
    assert fake_ds.last_skip == 0


def test_get_dolma_dataset_resumes_from_checkpoint(monkeypatch):
    # 10 rows, resume from offset 6, batch size 4 -> batches: rows[6:10]
    data = [{"text": f"row{i}"} for i in range(10)]
    fake_ds = FakeDataset(data)

    monkeypatch.setattr("utils.data_ingestion.load_dataset", lambda *a, **k: fake_ds)
    monkeypatch.setattr("utils.data_ingestion.load_checkpoint", lambda *_: 6)

    saved = []
    monkeypatch.setattr(
        "utils.data_ingestion.save_checkpoint", lambda p, k, o: saved.append(o)
    )

    batches = list(get_dolma_dataset(batch_size=4, is_use_last_checkpoint=True))
    assert len(batches) == 1
    assert [x["text"] for x in batches[0]] == ["row6", "row7", "row8", "row9"]
    assert saved == [9]
    assert fake_ds.last_skip == 6
