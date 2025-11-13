# utils/dolma-related/aapi_keywords.py
import os
import pickle
import re
from pathlib import Path

from dolma import BaseTagger, add_tagger
from dolma.core.data_types import DocResult, Document, Span

def _find_pickle(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if p.exists():
            return p

    here = Path(__file__).resolve()
    repo_root = here.parents[2]

    # candidates (now including utils/data/)
    candidates = [
        repo_root / "data" / "aapiGroups.pkl",
        here.parent / "data" / "aapiGroups.pkl",
        repo_root / "utils" / "data" / "aapiGroups.pkl",  # 👈 add this line
        Path.cwd() / "data" / "aapiGroups.pkl",
    ]

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        "Could not find aapiGroups.pkl. Tried:\n  - " + "\n  - ".join(map(str, candidates))
    )


@add_tagger("aapi_keywords_v1")
class AAPIKeywordsTagger(BaseTagger):
    """
    Tags documents that contain any AAPI-related keyword loaded from a pickle.
    """

    def __init__(self, keyword_pickle: str | None = None) -> None:
        super().__init__()

        resolved = _find_pickle(keyword_pickle)
        if not resolved:
            here = Path(__file__).resolve()
            tried = [
                f"explicit: {keyword_pickle!r}",
                f"env AAPI_KEYWORDS_PICKLE={os.environ.get('AAPI_KEYWORDS_PICKLE', '')!r}",
                # show a few likely locations we probed
                str(here.parent / "data" / "aapiGroups.pkl"),
                str(here.parents[2] / "data" / "aapiGroups.pkl") if len(here.parents) >= 3 else "(no parents[2])",
                str(Path.cwd() / "data" / "aapiGroups.pkl"),
            ]
            raise FileNotFoundError(
                "Could not locate aapiGroups.pkl. Searched:\n  - " + "\n  - ".join(tried) +
                "\nTip: set env var AAPI_KEYWORDS_PICKLE=/abs/path/to/aapiGroups.pkl"
            )

        self.keyword_pickle = resolved

        with self.keyword_pickle.open("rb") as f:
            raw_terms = pickle.load(f)

        # normalize terms → list[str]
        terms = [str(t).strip().lower() for t in list(raw_terms)]
        # guard against empty list
        if not terms:
            raise ValueError(f"No terms found in {self.keyword_pickle}")

        # compile a single regex (case-insensitive), escaping each term
        pattern = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
        self.regex = re.compile(pattern, flags=re.IGNORECASE)

    def predict(self, doc: Document) -> DocResult:
        text = doc.text or ""
        matches = self.regex.findall(text)
        if not matches:
            # Return a dummy key so Dolma mixer won't treat it as "missing attributes"
            span = Span(start=0, end=0, type="aapi_keyword", score=0.0)
            return DocResult(doc=doc, spans=[span])
            # return DocResult(doc=doc, spans=[])


        # unique matches → score is count of unique AAPI terms present
        unique_matches = {m.lower() for m in matches}
        span = Span(
            start=0,
            end=len(text),
            type="aapi_keyword",
            score=float(len(unique_matches)),
        )
        return DocResult(doc=doc, spans=[span])
