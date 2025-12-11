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
import threading
thread_local = threading.local()


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

def predict_n_mix(data, tagger): 
    
    doc = Document(
            id=data["id"],
            text=data["text"],
            source=data.get("source"),
        )

    tagged = tagger.predict(doc)
    mixed = mix_aapi_doc(tagged)

    return mixed



import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex


class AAPITokenizer:
    """
    spaCy tokenizer that keeps ethnicity terms as atomic tokens,
    including:
        - multiword: "central asian"
        - hyphenated: "central-asian"
        - mixed case: "Central-Asian"
    Everything is merged into a single token: "central_asian".
    """

    def __init__(self, aapi_terms: set | list) -> None:
        super().__init__()

        # Normalize ethnicity list:
        base_terms = {
            str(t).strip().lower().replace("-", " ")
            for t in aapi_terms
        }

        self.base_terms = base_terms  # debugging visibility
        self.aapi_groups_set = set(base_terms)

        #if the word is multi, make sure stick together
        self.multiword_map: dict[str, str] = {
            term: term.replace(" ", "_")
            for term in base_terms
            if " " in term       # only multiword terms
        }

        # ---------------------------------------------------
        # add merged versions to ethnicities set
        self.aapi_groups_set.update(self.multiword_map.values())

        # ---------------------------------------------------
        # 4. Build regex patterns for multiword/hyphenated forms
        # ---------------------------------------------------
        # This matches:
        #   "central asian"
        #   "central-asian"
        #   WITH ANY capitalization
        #
        # Pattern built:
        #   r"\bcentral[-\s]+asian\b"
        # ---------------------------------------------------
        patterns = []
        for term in self.multiword_map.keys():
            parts = term.split()
            if len(parts) == 2:
                w1, w2 = parts
                p = rf"\b{re.escape(w1)}[-\s]+{re.escape(w2)}\b"
                patterns.append(p)

        self.multiword_regex = (
            re.compile("|".join(patterns), flags=re.IGNORECASE)
            if patterns
            else None
        )

        # ---------------------------------------------------
        # 5. Load spaCy normally
        # ---------------------------------------------------
        
        self.nlp = spacy.load("en_core_web_sm")
        

        # Add special-case tokens ONLY for single-word ethnicities
        # (multiword are handled by pretokenizer)
        for text in self.aapi_groups_set:
            if " " not in text and "-" not in text:
                self.nlp.tokenizer.add_special_case(text, [{"ORTH": text}])


    # -------------------------------------------------------
    # PRETOKENIZER: Replace multiword ethnicities BEFORE spaCy
    # -------------------------------------------------------
    def preprocess(self, text: str) -> str:
        """
        Convert multi-word or hyphenated ethnicities to merged tokens.
        Example:
            "central asian"   → "central_asian"
            "central-asian"   → "central_asian"
            "Asian-American"  → "asian_american"
        """
        if not self.multiword_regex:
            return text

        def replacer(match):
            raw = match.group(0)         
            lower = raw.lower()
            norm = lower.replace("-", " ")  
            merged = self.multiword_map.get(norm)
            if merged:
                return merged
            # fallback (should rarely happen)
            return norm.replace(" ", "_")

        return self.multiword_regex.sub(replacer, text)


    # -------------------------------------------------------
    # TOKENIZE METHOD USED BY YOUR PIPELINE
    # -------------------------------------------------------
    def tokenize(self, data: Dict[str, Any]) -> Any:
        """
        1. Run pretokenization to merge ethnicity phrases.
        2. Tokenize with spaCy.
        """
        text = self.preprocess(data["text"])
        return self.nlp(text)
    
    def pipe_batch(self, data_list):
        """
        Accepts a list of dicts like {"id": ..., "text": ...}
        Returns a list of spaCy Docs in same order.
        """
        texts = [self.preprocess(d["text"]) for d in data_list]
        docs = list(self.nlp.pipe(texts, batch_size=50))
        return docs




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
    }


@add_tagger("aapi_keywords_v1")
class AAPIKeywordsTagger(BaseTagger):
    """
    Tag documents that contain any AAPI-related keyword
    loaded from a pickle of ethnicity terms.
    """

    def __init__(self,  aapi_terms: set | list) -> None:
        super().__init__()
        raw_terms = aapi_terms

        terms = [str(t).strip().lower() for t in list(raw_terms)]
        
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
