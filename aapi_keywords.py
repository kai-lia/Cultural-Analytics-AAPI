# aapi_taggers.py
import pickle
import re
from pathlib import Path

from dolma import BaseTagger, add_tagger
from dolma.core.data_types import DocResult, Document, Span


@add_tagger("aapi_keywords_v1")
class AAPIKeywordsTagger(BaseTagger):
    """
    Tags documents that contain any AAPI-related keyword loaded from a pickle.
    """

    def __init__(self, keyword_pickle: str | None = None) -> None:
        super().__init__()

        # Resolve default path relative to the project root
        if keyword_pickle is None:
            # aapi_taggers.py is at: CULTURAL-ANALYTICS-AAPI/aapi_taggers.py
            # so repo_root = that file's parent
            repo_root = Path(__file__).resolve().parent
            keyword_pickle = repo_root / "data" / "aapiGroups.pkl"

        self.keyword_pickle = Path(keyword_pickle)

        with self.keyword_pickle.open("rb") as f:
            raw_terms = pickle.load(f)

        terms = [str(t).lower() for t in raw_terms]
        pattern = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
        self.regex = re.compile(pattern, flags=re.IGNORECASE)

    def predict(self, doc: Document) -> DocResult:
        text = doc.text or ""

        matches = self.regex.findall(text)
        if not matches:
            return DocResult(doc=doc, spans=[])

        unique_matches = sorted({m.lower() for m in matches})

        span = Span(
            start=0,
            end=len(text),
            type="aapi_keyword",
            score=float(len(unique_matches)),
        )

        return DocResult(doc=doc, text=text, spans=[span])
