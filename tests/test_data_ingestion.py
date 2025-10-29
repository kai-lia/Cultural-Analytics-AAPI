import pandas as pd
import pytest
from utils.data_ingestion import get_ethnic_group_list


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

    base = [g for g in result if g not in extras]
    assert base == sorted(set(base))
