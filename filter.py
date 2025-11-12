

# aapi_taggers.py
import pickle
import re
from pathlib import Path

import json
import gzip

# Paths
docs_path = "/Users/kaionamartinson/Desktop/Cultural-Analytics/Cultural-Analytics-AAPI/c4/v0/documents/c4-train.00000.jsonl.gz"
attrs_path = "/Users/kaionamartinson/Desktop/Cultural-Analytics/Cultural-Analytics-AAPI/attributes/aapi_test/c4-train.00000.jsonl.gz"
output_path = "/Users/kaionamartinson/Desktop/Cultural-Analytics/Cultural-Analytics-AAPI/filtered/aapi_sample.jsonl.gz"

# Load the filter logic
filt = Filter("aapi_keywords_v1__aapi_keywords_v1__aapi_keyword>0")

total, kept = 0, 0

with DocumentStream(docs_path, attrs_path) as stream, gzip.open(output_path, "wt", encoding="utf-8") as out:
    for doc in stream:
        total += 1
        if filt.keep(doc):
            kept += 1
            out.write(json.dumps(doc.to_json()) + "\n")

print(f"✅ Done. Kept {kept} of {total} total documents.")
print(f"Filtered output saved to: {output_path}")
