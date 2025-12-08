import sqlite3
from collections import Counter
from pathlib import Path

DB_PATH = Path(
    "/Users/kaionamartinson/Desktop/Cultural-Analytics/Cultural-Analytics-AAPI/data/ethnicity_lexicon.db"
)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS verb_counts (
            ethnicity TEXT NOT NULL,
            verb      TEXT NOT NULL,
            count     INTEGER NOT NULL,
            PRIMARY KEY (ethnicity, verb)
        );
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS adj_counts (
            ethnicity TEXT NOT NULL,
            adjective TEXT NOT NULL,
            count     INTEGER NOT NULL,
            PRIMARY KEY (ethnicity, adjective)
        );
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS noun_heads (
            noun_head TEXT PRIMARY KEY,
            count     INTEGER NOT NULL
        );
    """
    )

    conn.commit()
    conn.close()


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


def load_ethnicity_dicts_from_db():
    """
    Load full verb_ethnicity_dict and adj_ethnicity_dict from DB
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

ARTIFACTS_PATH = Path(
    "/Users/kaionamartinson/Desktop/Cultural-Analytics/Cultural-Analytics-AAPI/ethnicity_counts.pkl"
)


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

    # Atomic write
    tmp_path = ARTIFACTS_PATH.with_suffix(".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(merged, f)
    tmp_path.replace(ARTIFACTS_PATH)