
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.filter_process.dolma_local import AAPITokenizer, AAPIKeywordsTagger

AAPI_TERMS = {
    "asian_american",
    "sri_lankan",
}

tokenizer = AAPITokenizer({"sri_lankan", "asian_american"})



def test_multiword_ethnicity_detection():
    tokenizer = AAPITokenizer(AAPI_TERMS)

    doc = tokenizer.tokenize({"text": "Sri Lankan communities are underrepresented."})
    tokens = [t.text for t in doc]

    assert "sri_lankan" in tokens


def test_hyphenated_ethnicity_detection():
    tokenizer = AAPITokenizer(AAPI_TERMS)

    doc = tokenizer.tokenize({"text": "Asian-American activists organized the rally."})
    tokens = [t.text for t in doc]

    assert "asian_american" in tokens

