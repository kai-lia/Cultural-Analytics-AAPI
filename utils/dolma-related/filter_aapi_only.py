import gzip, json

src_docs = "data/src/v0/c4-train.00000.jsonl.gz"
tagged = "data/output/tagged/c4-train.00000.jsonl.gz"
out = "data/output/mixed/aapi_only.jsonl.gz"

with gzip.open(src_docs, "rt") as d, gzip.open(tagged, "rt") as t, gzip.open(out, "wt") as o:
    for doc_line, tag_line in zip(d, t):
        doc = json.loads(doc_line)
        tag = json.loads(tag_line)
        spans = tag["attributes"].get("aapi_keywords_v1__aapi_keywords_v1__aapi_keyword", [])
        if spans and spans[0][2] > 0:
            doc["attributes"] = tag["attributes"]
            o.write(json.dumps(doc) + "\n")

print(f"✅ wrote filtered docs → {out}")
