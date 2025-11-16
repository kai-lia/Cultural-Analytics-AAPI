from datasets import load_dataset
import json
import gzip
from pathlib import Path
from tqdm.auto import tqdm

# -----------------------------
# Config
# -----------------------------
out_dir = Path("data/c4/v1")
out_dir.mkdir(parents=True, exist_ok=True)

SHARD_SIZE = 10_000      # docs per output .jsonl.gz
MAX_DOCS   = 100_000     # total desired C4 docs (across *all* runs)

# -----------------------------
# Helper: count docs in existing shards
# -----------------------------
def count_docs_in_shard(path: Path) -> int:
    """Count lines (docs) in one .jsonl.gz shard."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return sum(1 for _ in f)

# -----------------------------
# Inspect existing shards
# -----------------------------
existing_shards = sorted(out_dir.glob("c4-train.*.jsonl.gz"))

total_existing_docs = 0
if existing_shards:
    print("Found existing shards:")
    for p in existing_shards:
        n = count_docs_in_shard(p)
        total_existing_docs += n
        print(f"  {p.name}: {n} docs")
else:
    print("No existing shards found; starting fresh.")

print(f"Total existing C4 docs: {total_existing_docs}")

# If we already have enough, just stop
if total_existing_docs >= MAX_DOCS:
    print(f"Already have >= MAX_DOCS ({MAX_DOCS}) docs. Nothing to do.")
    raise SystemExit

# -----------------------------
# Decide where to resume shards
# -----------------------------
writer = None
docs_in_current_shard = total_existing_docs % SHARD_SIZE

if existing_shards:
    last_shard = existing_shards[-1]
    last_idx = int(last_shard.stem.split(".")[-1])  # c4-train.00007 -> 7

    if docs_in_current_shard == 0:
        # All previous shards are full → next shard is new
        shard_idx = last_idx + 1
        writer = None
        print(f"Last shard ({last_shard.name}) is full. "
              f"Next shard will be index {shard_idx:05d}.")
    else:
        # Last shard is partial → append to it
        shard_idx = last_idx + 1  # next *new* shard after this one
        writer = gzip.open(last_shard, "at", encoding="utf-8")
        print(f"Resuming in partial shard {last_shard.name} "
              f"with {docs_in_current_shard} docs already.")
else:
    # No shards at all
    shard_idx = 0
    writer = None
    docs_in_current_shard = 0

# Global count of docs written so far
count = total_existing_docs

# -----------------------------
# Dataset streaming setup
# -----------------------------
ds = load_dataset("allenai/dolma", "v1_7", split="train", streaming=True)

# We’ll skip the first `total_existing_docs` C4 docs in the stream,
# since those are already on disk.
c4_seen = 0

pbar = tqdm(
    total=MAX_DOCS,
    initial=min(total_existing_docs, MAX_DOCS),
    desc="Exporting C4 docs"
)

try:
    for example in ds:
        if example.get("source") != "c4":
            continue

        c4_seen += 1

        # Skip C4 docs we've already exported in previous runs
        if c4_seen <= total_existing_docs:
            continue

        # Stop if we've reached the target
        if count >= MAX_DOCS:
            break

        # Open a new shard if needed
        if writer is None or docs_in_current_shard == SHARD_SIZE:
            if writer is not None:
                writer.close()
            shard_path = out_dir / f"c4-train.{shard_idx:05d}.jsonl.gz"
            print(f"Opening new shard: {shard_path}")
            writer = gzip.open(shard_path, "wt", encoding="utf-8")
            shard_idx += 1
            docs_in_current_shard = 0

        # Write the new C4 document
        doc = {
            "id": example["id"],
            "text": example["text"],
            "source": example["source"],
        }
        writer.write(json.dumps(doc) + "\n")

        count += 1
        docs_in_current_shard += 1
        pbar.update(1)

        if count >= MAX_DOCS:
            break

finally:
    if writer is not None:
        writer.close()
    pbar.close()

print(f"Exported {count} C4 documents total into {shard_idx} shard(s) in {out_dir}")
print("Final shard index (next to be used) would be", shard_idx)
