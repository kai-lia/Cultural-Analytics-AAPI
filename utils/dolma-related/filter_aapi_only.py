import gzip
import json
from pathlib import Path

DOCS = Path("data/src/v0/c4-train.00009.jsonl.gz")
ATTRS = Path("data/output/tagged/c4-train.00009.jsonl.gz")
OUT = Path("data/output/mixed/documents/data/src/v0/c4-train.00009.jsonl.gz")
OUT.parent.mkdir(parents=True, exist_ok=True)

KEY = "aapi_keywords_v1__aapi_keywords_v1__aapi_keyword"

kept = 0
total = 0

with gzip.open(DOCS, "rt", encoding="utf-8") as docs_f, gzip.open(
    ATTRS, "rt", encoding="utf-8"
) as attrs_f, gzip.open(OUT, "wt", encoding="utf-8") as out_f:

    for doc_line, attr_line in zip(docs_f, attrs_f):
        total += 1
        doc = json.loads(doc_line)
        attr = json.loads(attr_line)

        if doc.get("id") != attr.get("id"):
            raise ValueError(
                f"ID mismatch at line {total}: " f"{doc.get('id')} != {attr.get('id')}"
            )

        attrs = attr.get("attributes", {})
        spans = attrs.get(KEY, [])

        score = 0.0
        if spans and spans[0] and len(spans[0]) >= 3:
            score = float(spans[0][2])

        if score > 0:
            out_f.write(json.dumps(doc) + "\n")
            kept += 1

print(f"Done. Kept {kept} / {total} docs with AAPI score > 0.")
print(f"Output written to: {OUT}")
