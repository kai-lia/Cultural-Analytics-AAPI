#dolma
def save_progress(count: int) -> None:

    """
    Save how many C4 docs we've fully traversed/processed (for resume).
    """
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("w", encoding="utf-8") as f:
        json.dump({"c4_docs_done": count}, f)


def load_progress() -> int:
    """
    Load how many C4 docs we've previously processed. Returns 0 if none.
    """
    if PROGRESS_PATH.exists():
        with PROGRESS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("c4_docs_done", 0))
    return 0


def open_new_shard(out_dir: Path, shard_idx: int) -> gzip.GzipFile:
    """
    Open a new gzip'd JSONL shard for writing mixed docs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"mixed.{shard_idx:09d}.jsonl.gz"
    return gzip.open(shard_path, "wt", encoding="utf-8")

class AAPITokenizer:

    def __init__(self, keyword_pickle: Optional[str] = None) -> None:
        super().__init__()

        self.keyword_pickle = _find_pickle(keyword_pickle)
        with self.keyword_pickle.open("rb") as f:
            raw_terms = pickle.load(f)
        terms = [str(t).strip().lower() for t in list(raw_terms)]
        if not terms:
            raise ValueError(f"No terms found in {self.keyword_pickle}")

        self.aapi_groups_set = set(terms)

        self.nlp = spacy.load("en_core_web_sm")
        for token_text in self.aapi_groups_set:
            self.nlp.tokenizer.add_special_case(token_text, [{"ORTH": token_text}])

    def tokenize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.nlp(data["text"])  # .to_json() changed this
    

def mix_aapi_doc(result: DocResult) -> Optional[Dict[str, Any]]:
    """
    Given a DocResult from AAPIKeywordsTagger, return a JSON-serializable dict
    if score > 0, else return None (filter out the doc).
    """
    doc = result.doc
    spans = result.spans or []

    score = 0.0
    if spans:
        score = float(spans[0].score or 0.0)

    if score <= 0.0:
        return None

    return {
        "id": doc.id,
        "text": doc.text,
        # "source": getattr(doc, "source", None),
        # "aapi_score": score,
        # "aapi_spans": [
        #     {
        #         "start": s.start,
        #         "end": s.end,
        #         "type": s.type,
        #     }
        #     for s in spans
        # ],
    }


@add_tagger("aapi_keywords_v1")
class AAPIKeywordsTagger(BaseTagger):
    """
    Tags documents that contain any AAPI-related keyword loaded from a pickle.
    """

    def __init__(self, keyword_pickle: Optional[str] = None) -> None:
        super().__init__()

        self.keyword_pickle = _find_pickle(keyword_pickle)

        with self.keyword_pickle.open("rb") as f:
            raw_terms = pickle.load(f)

        terms = [str(t).strip().lower() for t in list(raw_terms)]
        if not terms:
            raise ValueError(f"No terms found in {self.keyword_pickle}")

        pattern = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
        self.regex = re.compile(pattern, flags=re.IGNORECASE)

    def predict(self, doc: Document) -> DocResult:
        text = doc.text or ""
        matches = self.regex.findall(text)
        if not matches:
            # no matches → score 0
            span = Span(start=0, end=0, type="aapi_keyword", score=0.0)
            return DocResult(doc=doc, spans=[span])

        # unique matches → score is count of unique AAPI terms present
        unique_matches = {m.lower() for m in matches}
        score = float(len(unique_matches))

        span = Span(
            start=0,
            end=len(text),
            type="aapi_keyword",
            score=score,
        )
        return DocResult(doc=doc, spans=[span])