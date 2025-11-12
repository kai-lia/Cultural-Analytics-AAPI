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
    Also returns the document's original content with the tag.
    """

    def __init__(self, keyword_pickle: str | None = None) -> None:
        super().__init__()

        if keyword_pickle is None:
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
        unique_matches = sorted({m.lower() for m in matches})

        score = float(len(unique_matches))
        if score == 0.0:
            return DocResult(doc=doc, spans=[])

        
        # Build span kwargs compatible with older Dolma
        span_kwargs = dict(
            start=0,
            end=len(text),
            type="aapi_keyword",
            score=score,
        )


        span = Span(**span_kwargs)
        return DocResult(doc=doc, spans=[span])