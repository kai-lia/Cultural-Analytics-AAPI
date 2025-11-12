#c4/v0/documents/c4-train.00000.jsonl.gz
#attributes/aapi_test/c4-train.00000.jsonl.gz
#test/filtered/tagged.gz



import gzip
import json
from pathlib import Path

# Ask user for file path in the terminal
path_str = input("Path to .jsonl.gz file: ").strip()
path = Path(path_str).expanduser()

if not path.is_file():
    print(f"File not found: {path}")
else:
    print(f"\nReading records with non-empty attributes from: {path}\n")
    try:
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                data = json.loads(line)
                # Only print if "attributes" exists and is not empty
                if data.get("attributes"):
                    print(json.dumps(data, indent=2))
                    print("-" * 40)
                    count += 1
        print(f"\nTotal lines read: {i}")
        print(f"Records with attributes: {count}")
    except Exception as e:
        print(f"Error reading file: {e}")
