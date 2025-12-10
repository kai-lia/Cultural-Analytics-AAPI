import spacy
from collections import Counter

import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))



from filter_process.pos_db import (
    
    collect_all_modifiers, 
    build_group_lexicon, 
    collect_all_modifiers
)



nlp = spacy.load("en_core_web_sm")

GROUPS = {"filipino", "korean", "japanese", "chinese"}
GL = build_group_lexicon(GROUPS, nlp.vocab)


# ----
GROUPS = {"filipino", "korean", "japanese", "chinese"}
GL = build_group_lexicon(GROUPS)


# -----------------------------
# Helper to run one-line test
# -----------------------------
def run(text):
    doc = nlp(text)
    return collect_all_modifiers(doc, GL)


# -----------------------------
# NOUN EXTRACTION TESTS
# -----------------------------

def test_noun_amod_case():
    out = run("the filipino women protested.")
    assert out["filipino"]["nouns"] == {"woman"}


def test_noun_compound_case():
    out = run("chinese doctors treated patients.")
    assert out["chinese"]["nouns"] == {"doctor"}


def test_noun_child_case():
    out = run("the korean boy cheered.")

    # should catch "boy"
    assert "cheer" in out["korean"]["verbs"]


def test_noun_simple():
    out = run("chinese doctors respected the patient")
    # should catch "boy"
    assert "doctor" in out["chinese"]["nouns"]


# -----------------------------
# VERB EXTRACTION TESTS
# -----------------------------

def test_verb_subject_case():
    out = run("Filipinos celebrate every weekend.")
    assert out["filipino"]["verbs"] == {"celebrate"}


def test_verb_modified_subject_case():
    out = run("the chinese workers protested loudly.")
    assert out["chinese"]["verbs"] == {"protest"}


def test_verb_passive_agent_case():
    out = run("Rice is cooked by Filipinos.")
    assert out["filipino"]["verbs"] == {"cook"}


def test_verb_coordination():
    out = run("Koreans dance and sing during festivals.")
    assert out["korean"]["verbs"] == {"dance", "sing"}


# -----------------------------
# ADJECTIVE EXTRACTION TESTS
# -----------------------------

def test_adj_predicate_simple():
    out = run("Filipinos are hardworking.")
    assert out["filipino"]["adjs"] == {"hardworking"}


def test_adj_predicate_multiple():
    out = run("Japanese are polite and hardworking.")
    assert out["japanese"]["adjs"] == {"polite", "hardworking"}


def test_adj_directional_does_not_block():
    out = run("South Korean students are ambitious.")
    assert out["korean"]["adjs"] == {"ambitious"}


def test_adj_hyphenated():
    out = run("Filipinos are hard-working.")
    assert out["filipino"]["adjs"] == {"hardworking"}


def test_adj_with_intensifier():
    out = run("Japanese are very polite.")
    # intensifier removed → "polite" only
    assert out["japanese"]["adjs"] == {"polite"}


def test_adj_fragment_style():
    out = run("japanese super polite")
    assert out["japanese"]["adjs"] == {"polite"}


# -----------------------------
# FULL INTEGRATION TEST
# -----------------------------

def test_combined_noun_verb_adj():
    text = "The filipino workers are hardworking and protested yesterday."
    out = run(text)

    assert out["filipino"]["nouns"] == {"worker"}
    assert out["filipino"]["verbs"] == {"protest"}
    assert out["filipino"]["adjs"] == {"hardworking"}







def test_chinese_mutual_lies():
    out = run("Supposedly, Chinese partners expect mutual lies to get embedded in their business deals.")
    assert out["chinese"]["nouns"] == {"partners"}
    assert out["chinese"]["verbs"] == {"expect"}
    assert out["chinese"]["adjs"] == {}


def test_vietnamese_lunar_age():
    out = run("As birthdays are not celebrated on the actual date someone was born, Vietnamese people give their age as that of the lunar symbol of the year of their birth.")
    assert out["vietnamese"]["nouns"] == {"people"}
    assert out["vietnamese"]["verbs"] == {"give"}
    assert out["vietnamese"]["adjs"] == {}


def test_chinese_ancient_soups():
    out = run("Ancient Chinese people also enjoyed making similar soups, with researchers finding cookware filled with liquid and bone fragments dating over 2,000 years.")
    # Only keep true ethnicity-modifying adjectives (no gerunds like "making/dating/finding")
    assert out["chinese"]["nouns"] == {"people"}
    assert out["chinese"]["verbs"] == {"enjoy"}  # lemma handling
    assert out["chinese"]["adjs"] == {"ancient"}


def test_thai_billionaire_accurate():
    out = run("He’s a Thai billionaire no one’s ever heard of, which I think is still pretty accurate.")
    assert out["thai"]["nouns"] == {"billionaire"}
    # No subject-verb link grounded → empty verbs
    assert out["thai"]["verbs"] == set()
    # "accurate" is a narrative adjective describing the ethnic subject
    assert out["thai"]["adjs"] == set()






if __name__ == "__main__":
    test_combined_noun_verb_adj()
