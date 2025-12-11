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

GROUPS = {"filipino", "korean", "japanese", "chinese", "vietnamese", "thai", "maori", "indian"}
GL = build_group_lexicon(GROUPS, nlp.vocab)


# ----
GROUPS = {"filipino", "korean", "japanese", "chinese", "vietnamese", "thai", "maori", "indian"}
GL = build_group_lexicon(GROUPS)


# -----------------------------
# Helper to run one-line test
# -----------------------------
def run(text):
    doc = nlp(text)
    for tok in doc:
        print(
            f"TEXT={tok.text:12} | "
            f"LEMMA={tok.lemma_:12} | "
            f"POS={tok.pos_:6} | "
            f"TAG={tok.tag_:6} | "
            f"DEP={tok.dep_:10} | "
            f"HEAD={tok.head.text:12} | "
            f"ENT={tok.ent_type_ or '-'}"
        )




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
    out = run("the chinese are pretty people.")
    assert out["chinese"]["adjs"] == {"pretty"}
    assert out["chinese"]["nouns"] == {"people"}


def test_adj_predicate_multiple():
    out = run("Japanese are polite and pretty")
    assert out["japanese"]["adjs"] == {"polite", "pretty"}


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
    text = "The filipino workers are pretty and protested yesterday."
    out = run(text)

    assert out["filipino"]["nouns"] == {"worker"}
    assert out["filipino"]["verbs"] == {"protest"}
    assert out["filipino"]["adjs"] == {"pretty"}



def test_chinese_parents_boon():
    out = run("But for Chinese parents expecting a baby, the Year of the Sheep isn’t necessarily a boon.")
    assert out["chinese"]["nouns"] == {"parent"}
    assert out["chinese"]["verbs"] == {"expect"}
    assert out["chinese"]["adjs"] == set()



def test_indian_cricketer_case():
    out = run("The best friends will compete against telly celebs Anup Jalota, Dipika Kakar, former Indian cricketer S Sreesanth, Srishty Rode and more.")
    assert out["indian"]["nouns"] == {"cricketer"}
    assert out["indian"]["verbs"] == set()
    assert out["indian"]["adjs"] == {"former", "more"}


def test_maori_author_case():
    out = run('In the textbook you can read a short story, "Butterflies", by a Maori author.')
    assert out["maori"]["nouns"] == {"author"}
    assert out["maori"]["verbs"] == set()
    assert out["maori"]["adjs"] == set()



def test_chinese_people_soups():
    out = run("Ancient Chinese people also enjoyed making similar soups, with researchers finding cookware filled with liquid and bone fragments dating over 2,000 years.")
    assert out["chinese"]["nouns"] == {"people"}
    assert out["chinese"]["verbs"] == {"enjoy"}
    assert out["chinese"]["adjs"] == {"ancient"}


def test_chinese_mutual_lies():
    out = run("Supposedly, Chinese partners expect mutual lies to get embedded in their business deals.")
    assert out["chinese"]["nouns"] == {"partner"}
    assert out["chinese"]["verbs"] == {"expect"}
    assert out["chinese"]["adjs"] == set()


def test_vietnamese_lunar_age():
    out = run("As birthdays are not celebrated on the actual date someone was born, Vietnamese people give their age as that of the lunar symbol of the year of their birth.")
    assert out["vietnamese"]["nouns"] == {"people"}
    assert out["vietnamese"]["verbs"] == {"give"}
    assert out["vietnamese"]["adjs"] == set()


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


def new_test(): 
    out = run("Korean women are still worried about the latter concern to exfoliate.")
    print(out)

    out = run("Not so much the Indian accented guy who wants to help me fix my computer.")
    print(out)


    out = run("Defence analyst Lt General Abdul Qayyum (retd) said the Chinese nation had surprised the entire world by exhibiting extraordinary resilience, courage and guts by bravely facing and surmounting unprecedented challenges in the last 16 years.")
    print(out)

    out = run("They wanted to get out so Mr. Becker interested a group of Japanese businessmen in Hawaii who bought out the Pasadena group and put up more money.")
    print(out)

    out = run("Thus in order to regulate the growth of blood cancer in India, a number of groups have invested a lot in the segment and giving away nothing but the best not just the Indian patients but also the global patients who visit this country at much of the affordable cost at high quality.")
    print(out)




if __name__ == "__main__":
    new_test()
