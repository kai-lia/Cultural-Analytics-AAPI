#!/usr/bin/env python3
"""
Dolma AAPI Keyword Filtering + Tokenization Pipeline

Modules included:
    - Progress save/load utilities
    - Shard writer utilities
    - AAPI keyword-based filtering tagger (Dolma BaseTagger)
    - Tokenizer that handles AAPI group terms as single tokens
"""

# ===========================
# Imports
# ===========================
import json
import pickle
import re
import dolma
import gzip
from pathlib import Path
from typing import Dict, Any, Optional

import spacy

from dolma import BaseTagger, add_tagger
from dolma.core.data_types import DocResult, Document, Span
from collections import Counter, defaultdict




def open_new_shard(out_dir: Path, shard_idx: int) -> gzip.GzipFile:
    """
    Create a new JSONL.GZ shard file to store mixed docs.
    Naming format: mixed.000000123.jsonl.gz
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"mixed.{shard_idx:09d}.jsonl.gz"
    return gzip.open(shard_path, "wt", encoding="utf-8")



def _find_pickle(pickle_path: Optional[str]) -> Path:
    """
    Resolve a pickle path into a concrete Path() object.
    """
    if pickle_path is None:
        raise ValueError("keyword_pickle must be provided!")

    path = Path(pickle_path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    return path



class AAPITokenizer:
    """
    spaCy tokenizer that keeps ethnicity terms as atomic tokens.
    e.g. "filipino" not broken into "fil" / "ipino"
    """

    def __init__(self, keyword_pickle: Optional[str] = None) -> None:
        super().__init__()

        # Load AAPI group names
        self.keyword_pickle = _find_pickle(keyword_pickle)
        with self.keyword_pickle.open("rb") as f:
            raw_terms = pickle.load(f)

        terms = [str(t).strip().lower() for t in list(raw_terms)]
        if not terms:
            raise ValueError(f"No terms found in {self.keyword_pickle}")

        self.aapi_groups_set = set(terms)

        # Load spaCy and apply special cases
        self.nlp = spacy.load("en_core_web_sm")
        for text in self.aapi_groups_set:
            self.nlp.tokenizer.add_special_case(text, [{"ORTH": text}])

    def tokenize(self, data: Dict[str, Any]) -> Any:
        """
        Return a spaCy doc from incoming JSON dict with a "text" field.
        """
        return self.nlp(data["text"])  # `.to_json()` intentionally removed


# ===========================
# AAPI Document Filter
# ===========================
def mix_aapi_doc(result: DocResult) -> Optional[Dict[str, Any]]:
    """
    Given a DocResult from AAPIKeywordsTagger:
        If score > 0 → return dict for output JSONL profiling
        Else → return None to discard
    """
    doc = result.doc
    spans = result.spans or []

    score = float(spans[0].score or 0.0) if spans else 0.0
    if score <= 0.0:
        return None

    return {
        "id": doc.id,
        "text": doc.text,
        # Optionally include metadata for debugging:
        # "aapi_score": score,
        # "aapi_spans": [
        #     {"start": s.start, "end": s.end, "type": s.type}
        #     for s in spans
        # ],
    }


@add_tagger("aapi_keywords_v1")
class AAPIKeywordsTagger(BaseTagger):
    """
    Tag documents that contain any AAPI-related keyword
    loaded from a pickle of ethnicity terms.
    """

    def __init__(self, keyword_pickle: Optional[str] = None) -> None:
        super().__init__()

        self.keyword_pickle = _find_pickle(keyword_pickle)

        with self.keyword_pickle.open("rb") as f:
            raw_terms = pickle.load(f)

        terms = [str(t).strip().lower() for t in list(raw_terms)]
        if not terms:
            raise ValueError(f"No AAPI terms found in: {self.keyword_pickle}")

        # Compile keyword regex
        pattern = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
        self.regex = re.compile(pattern, flags=re.IGNORECASE)

    def predict(self, doc: Document) -> DocResult:
        text = doc.text or ""
        matches = self.regex.findall(text)

        if not matches:
            return DocResult(doc=doc, spans=[Span(0, 0, "aapi_keyword", 0.0)])

        score = float(len({m.lower() for m in matches}))
        span = Span(0, len(text), "aapi_keyword", score)
        return DocResult(doc=doc, spans=[span])
