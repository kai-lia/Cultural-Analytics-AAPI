# Cultural-Analytics-AAPI

Fall 2025 Cultural Analytics Project

## Example: How to Run

### 1. Tag documents using the AAPI keyword tagger

dolma tag \
  --documents data/src/v0/c4-train.00000.jsonl.gz \
  --destination data/output/tagged \
  --taggers aapi_keywords_v1 \
  --tagger-modules utils/dolma-related/aapi_keywords.py \
  --ignore-existing false

### 2. Run a mixer pipeline

dolma -c configs/mixer.json mix