import gzip
import json
from pathlib import Path

DOCS = Path(
    "/Users/naycal/Documents/funsies/Cultural-Analytics-AAPI/data/src/v0/c4-train.00000.jsonl.gz"
)
ATTRS = Path(
    "/Users/naycal/Documents/funsies/Cultural-Analytics-AAPI/data/output/tagged/c4-train.00000.jsonl.gz"
)

KEY = "aapi_keywords_v1__aapi_keywords_v1__aapi_keyword"

with gzip.open(DOCS, "rt", encoding="utf-8") as docs_f, gzip.open(
    ATTRS, "rt", encoding="utf-8"
) as attrs_f:

    for i, (doc_line, attr_line) in enumerate(zip(docs_f, attrs_f), start=1):
        doc = json.loads(doc_line)
        attr = json.loads(attr_line)

        print(f"\n--- LINE {i} ---")
        print("doc.id   :", doc.get("id"))
        print("attr.id  :", attr.get("id"))

        if doc.get("id") != attr.get("id"):
            print("!! ID MISMATCH !!")
            break

        attrs = attr.get("attributes", {})
        print("attribute keys:", list(attrs.keys()))

        if KEY not in attrs:
            print(f"!! MISSING KEY {KEY!r} in attributes !!")
            break

        spans = attrs[KEY]
        print("raw spans:", spans)

        try:
            score = spans[0][2]
        except Exception as e:
            print("!! ERROR getting spans[0][2]:", e)
            break

        print("score:", score)
        print("passes filter (>0)?", score > 0)

        if i >= 5:  # stop after a few lines for sanity
            break
