from datasets import load_dataset
import json
import gzip
from pathlib import Path

# Where to store the C4-style docs
out_dir = Path("data/c4")
out_dir.mkdir(parents=True, exist_ok=True)

# Use the Dolma sample (16.4 GB compressed total, across all sources)
ds = load_dataset("allenai/dolma", "v1_7", split="train", streaming=True)

SHARD_SIZE = 10_000    # docs per output .jsonl.gz
MAX_DOCS   = 100_000   # bump this if you have disk & want more

shard_idx = 0
count = 0
writer = None

for example in ds:
    # Keep only C4-sourced docs
    if example.get("source") != "c4":
        continue

    # Open a new shard file every SHARD_SIZE docs
    if writer is None or (count % SHARD_SIZE == 0):
        if writer is not None:
            writer.close()
        shard_path = out_dir / f"c4-train.{shard_idx:05d}.jsonl.gz"
        print(f"Opening new shard: {shard_path}")
        writer = gzip.open(shard_path, "wt", encoding="utf-8")
        shard_idx += 1

    # Minimal Dolma doc fields
    doc = {
        "id": example["id"],
        "text": example["text"],
        "source": example["source"],
    }
    writer.write(json.dumps(doc) + "\n")
    count += 1

    if count >= MAX_DOCS:
        break

if writer is not None:
    writer.close()

print(f"Exported {count} C4 documents into {shard_idx} shard(s) in {out_dir}")
