#!/usr/bin/env python3
"""
SQLite DB for ethnicity lexicon aggregation + progress tracking.
Stores:
    - verb_counts
    - adj_counts
    - noun_heads
    - overall aggregated counts (optional artifacts)
"""

# Imports
import json
import pickle
import sqlite3
from collections import Counter
from pathlib import Path
import duckdb


# Paths
DB_PATH = Path("data/output/full_pipeline/ethnicity_pos.duckdb")

PROGRESS_PATH = Path("data/output/full_pipeline/aapi_progress.json")

AAPI_PKL_PATH = Path("data/input/aapiGroups.pkl")

ARTIFACTS_PATH = Path("data/output/full_pipeline/ethnicity_counts.pkl")



def load_aapi_pkl(): 
    """ loading my pkl and returning"""
    with AAPI_PKL_PATH .open("rb") as f:
        aapi_set  = pickle.load(f)
    # Safety check
    if not isinstance(aapi_set, set):
        aapi_set = set(aapi_set)

    print(f"loaded {len(aapi_set)} AAPI keywords")
    return aapi_set



import json

def save_sentence_modifiers(eth, noun_hits, verb_hits, adj_hits):
    nouns = sorted(list(noun_hits))
    verbs = sorted(list(verb_hits))
    adjs  = sorted(list(adj_hits))

    # Convert lists to JSON strings
    nouns_json = json.dumps(nouns)
    verbs_json = json.dumps(verbs)
    adjs_json  = json.dumps(adjs)

    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        INSERT INTO ethnicity_sentence_modifiers
        (ethnicity, nouns, verbs, adjs, count, has_dup)
        VALUES (?, ?, ?, ?, 1, FALSE)
        ON CONFLICT (ethnicity, nouns, verbs, adjs)
        DO UPDATE SET
            count = ethnicity_sentence_modifiers.count + 1,
            has_dup = (ethnicity_sentence_modifiers.count + 1) > 1;
    """, [eth, nouns_json, verbs_json, adjs_json])

    con.close()





# Progress Save / Load
def save_progress(count: int) -> None:
    """
    Save how many C4 docs we've fully traversed/processed.
    Allows pipeline to resume if interrupted.
    """
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("w", encoding="utf-8") as f:
        json.dump({"c4_docs_done": count}, f)


def load_progress() -> int:
    """
    Returns how many C4 docs were previously processed.
    Defaults to 0 if no progress file exists.
    """
    if PROGRESS_PATH.exists():
        with PROGRESS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("c4_docs_done", 0))
    return 0



def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE TABLE IF NOT EXISTS ethnicity_sentence_modifiers (
            ethnicity TEXT NOT NULL,
            adjs      TEXT NOT NULL,
            verbs     TEXT NOT NULL,
            nouns     TEXT NOT NULL,
            count     INTEGER NOT NULL,
            has_dup   BOOLEAN NOT NULL,
            PRIMARY KEY (ethnicity, adjs, verbs, nouns)
        );
    """)

    con.close()




# DB Insert (Accumulation)
def flush_all_to_db(
    noun_heads_counter,
    verb_ethnicity_dict,
    adj_ethnicity_dict,
):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # --- noun_heads_counter ---
    for noun, cnt in noun_heads_counter.items():
        cur.execute(
            """
            INSERT INTO noun_heads (noun_head, count)
            VALUES (?, ?)
            ON CONFLICT(noun_head)
            DO UPDATE SET count = noun_heads.count + excluded.count;
        """,
            (noun, cnt),
        )

    # --- verb_ethnicity_dict ---
    for ethnicity, verb_counter in verb_ethnicity_dict.items():
        for verb, cnt in verb_counter.items():
            cur.execute(
                """
                INSERT INTO verb_counts (ethnicity, verb, count)
                VALUES (?, ?, ?)
                ON CONFLICT(ethnicity, verb)
                DO UPDATE SET count = verb_counts.count + excluded.count;
            """,
                (ethnicity, verb, cnt),
            )

    # --- adj_ethnicity_dict ---
    for ethnicity, adj_counter in adj_ethnicity_dict.items():
        for adj, cnt in adj_counter.items():
            cur.execute(
                """
                INSERT INTO adj_counts (ethnicity, adjective, count)
                VALUES (?, ?, ?)
                ON CONFLICT(ethnicity, adjective)
                DO UPDATE SET count = adj_counts.count + excluded.count;
            """,
                (ethnicity, adj, cnt),
            )

    conn.commit()
    conn.close()


# DB Load into Dictionaries
def load_ethnicity_dicts_from_db():
    """
    Load verb_ethnicity_dict and adj_ethnicity_dict from DB
    into the same structure: dict[str, Counter]
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    verb_ethnicity_dict = {}
    adj_ethnicity_dict = {}

    # verbs
    cur.execute("SELECT ethnicity, verb, count FROM verb_counts")
    for ethnicity, verb, cnt in cur.fetchall():
        verb_ethnicity_dict.setdefault(ethnicity, Counter())[verb] = cnt

    # adjectives
    cur.execute("SELECT ethnicity, adjective, count FROM adj_counts")
    for ethnicity, adj, cnt in cur.fetchall():
        adj_ethnicity_dict.setdefault(ethnicity, Counter())[adj] = cnt

    conn.close()
    return verb_ethnicity_dict, adj_ethnicity_dict


# Artifact Save (Atomic Merge)
def save_artifacts_safely(aapi_counter_pass) -> None:
    """
    Merge current aapi_counter_pass into whatever is already on disk,
    then save atomically.
    """
    # Ensure it's a Counter
    if not isinstance(aapi_counter_pass, Counter):
        aapi_counter_pass = Counter(aapi_counter_pass)

    old_counter = Counter()
    if ARTIFACTS_PATH.exists():
        with ARTIFACTS_PATH.open("rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, Counter):
            old_counter = loaded

    # Merge: old + new
    merged = old_counter.copy()
    merged.update(aapi_counter_pass)

    # Atomic write (avoid corruption)
    tmp_path = ARTIFACTS_PATH.with_suffix(".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(merged, f)
    tmp_path.replace(ARTIFACTS_PATH)
