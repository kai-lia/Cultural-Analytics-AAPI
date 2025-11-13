#data/c4/v0//c4-train.00000.jsonl.gz
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
    print(f"\nReading first 5 records from: {path}\n")
    #c4/v0/documents/c4-train.00000.jsonl.gz
    #attributes/aapi_test/c4-train.00000.jsonl.gz
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                print(json.dumps(data, indent=2))
                print("-" * 40)
                if i == 4:
                    break
                print(i)
                
    except Exception as e:
        print(f"Error reading file: {e}")
