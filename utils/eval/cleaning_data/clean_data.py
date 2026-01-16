import ast
import sys
import pickle
import duckdb
import pandas as pd
import spacy


from pathlib import Path
from collections import Counter
from utils.save_load import fetch_duck_df, save_duck_df


# setting path root for reading or writing data
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# writing and saving paths
AAPI_PKL_PATH = Path("data/input/aapiGroups.pkl")
TAX_PKL_PATH = Path("data/input/eval/taxonomy_structure.pkl")
DB_PATH = Path("data/input/eval/ethnicity_pos.duckdb")
OUT_DB_PATH = Path("data/input/eval/ethnicity_clean.duckdb")


# spacy type
NLP = spacy.load("en_core_web_sm")

ETHNICITY_ALIAS = {  # for the sake of another project these are separate in my data, am combining them
    "asian indian": "indian",
    "native hawaiian": "hawaiian",
}

DIRECTIONAL = {
    "north",
    "south",
    "east",
    "west",
    "northern",
    "northeast",
    "southern",
    "eastern",
    "western",
    "northward",
    "southward",
    "easternly",
    "westernly",
    "-",
    "mainland",
    "inner",
}


ETHNIC = {
    "white",
    "black",
    "latino",
    "tibetan",
    "nepali" "nb",
    "spanish",
    "mexican",
    "canadian",
    "mexican",
    "american",
    "caribbean",
    "african",
    "eyelid",
    "TRUE",
    "british",
    "australian",
    "timorese",
    "russian",
    "german",
    "mandopop",
    "sinhalese",
    "hispanic",
    "turkish",
    "dutch",
    "endangered",
    "european",
    "economic",
    "carolinian",
    "akaye",
    "somali",
    "latin",
    "iraqi",
    "eaff",
    "royal",
    "french",
    "irish",
    "imperial",
    "unified",
    "prime",
    "deputy",
    "surname",
    "militant",
    "former",
    "pastoral",
    "national",
    "military",
    "roman",
    "siberian",
    "yapese",
    "projecttarget",
    "pohnpeian",
    "cofa",
    "base",
    "nativehawaiian",
    "vast",
    "blue",
}


def load_pickle(path):
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def is_norp_token(token: str) -> bool:
    """
    Returns True if the token is tagged as NORP by NER.
    """
    if not isinstance(token, str) or not token.strip():
        return False

    doc = NLP(token)
    for ent in doc.ents:
        if ent.label_ == "NORP":
            return True
    return False


def clean_ethnicity(df):
    """formatting name and grouping together ethnicities"""

    def normalize_ethnicity(eth):
        """grouping ethnicities ex: hawaiian and native hawaiian"""
        if not isinstance(eth, str):
            return eth
        eth = eth.lower().strip()
        return ETHNICITY_ALIAS.get(eth, eth)

    # I formatted things as south_asian for tokenizing purposes, cleaning to south asian
    df["ethnicity"] = df["ethnicity"].str.replace("_", " ", regex=False)
    df["ethnicity"] = df["ethnicity"].apply(normalize_ethnicity)


def parse_list(x):
    """input df saved the lists as strings, reverting to lists
    input: list as string "[]"
    output: actual list []
    """
    if isinstance(x, list):  # is list, ignore
        return x
    if isinstance(x, str):  # revert to list if not list
        try:
            parsed = ast.literal_eval(x)  # takes list in string and makes into a list
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def add_taxonomy(df):
    """adding lables for hierarical taxonomy
    add lable columns "Race", "Region"

    taxonomy made in cleaning_data/create_taxomomy.py for more details
    """

    def get_classification(eth, taxonomy_struct):
        # if term is missing ret None, None
        if eth not in taxonomy_struct:
            return None, None

        values = taxonomy_struct[eth]

        # values may be [] or shorter/longer lists
        race = values[0] if len(values) > 0 else None
        region = values[1] if len(values) > 1 else None

        return race, region

    print("loading pickle...")
    taxonomy_structure = load_pickle(TAX_PKL_PATH)  # checkout

    df[["Race", "Region"]] = df["Ethnicity"].apply(
        lambda x: pd.Series(get_classification(x, taxonomy_structure))
    )
    return df


def sanitize_token(t):
    "was running into emoji issues, will clean here"
    if isinstance(t, str):
        return t.encode("utf-8", "ignore").decode("utf-8")
    return t


def contains_ethnicity(token, eth):
    t = token.lower().replace("_", "")
    e = eth.lower().replace("_", "")
    return e in t or "asian" in t or "pacific" in t


def format_to_counters(df, aapi):
    """data base is currently very unstructured
    input (sentance level parses):
       ethnicity adjs        verbs                nouns   count    has_dup
       chinese["cool"]	  ["found", "win"]	  ["people"]	1	  False

    out:
        Ethnicity	Adjs	                  Verbs	                 Nouns	               Race	   Region
            chinese	   Counter("lovely": 5, ...) Counter("eat": 10, ...) Counter("kid": 10, ..) Asian  East Asian
    """
    results = []
    # had some formating issues with str
    clean_ethnicity(df)  # remove whitespace

    df["adjs"] = df["adjs"].apply(parse_list)
    df["verbs"] = df["verbs"].apply(parse_list)
    df["nouns"] = df["nouns"].apply(parse_list)

    for eth, group in df.groupby("ethnicity"):
        adj_counter = Counter()
        verb_counter = Counter()
        noun_counter = Counter()

        for _, row in group.iterrows():
            # statments are aggregated by count because they appear multiple times
            row_count = row["count"]

            # add weighted adjective counts
            for adj in row["adjs"]:
                adj = sanitize_token(adj)
                if len(adj) <= 2 or adj in DIRECTIONAL or adj in ETHNIC:
                    continue
                if contains_ethnicity(adj, eth) or adj in aapi:
                    continue
                adj_counter[adj] += row_count

            # add weighted verb counts
            for verb in row["verbs"]:
                verb = sanitize_token(verb)
                if len(verb) <= 2 or verb in DIRECTIONAL or verb in ETHNIC:
                    continue
                if contains_ethnicity(verb, eth) or verb in aapi:
                    continue
                verb_counter[verb] += row_count

            # add weighted noun counts
            for noun in row["nouns"]:
                noun_counter[sanitize_token(noun)] += row_count

        results.append(
            {
                "Ethnicity": eth,
                "Adjs": adj_counter,
                "Verbs": verb_counter,
                "Nouns": noun_counter,
            }
        )
    ethnicity_df = pd.DataFrame(results)
    return ethnicity_df


def run():

    df = fetch_duck_df(DB_PATH, "ethnicity_sentence_modifiers")

    aapi_set = set(load_pickle(AAPI_PKL_PATH))

    print("Formatting and Cleaning ...")
    ethnicity_df = format_to_counters(df, aapi_set)
    print("Adding Taxonomy ...")
    ethnicity_df = add_taxonomy(ethnicity_df)

    save_duck_df(OUT_DB_PATH, ethnicity_df, "ethnicity_clean")


if __name__ == "__main__":
    """run via the command python -m utils.eval.cleaning_data.clean_data"""
    run()
