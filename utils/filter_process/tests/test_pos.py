"""
Tests for NEW ethnicity extractor format:
    nouns, verbs, adjs → {ethnicity: {values}}
"""

import spacy
from collections import defaultdict
from utils.filter_process.pos_db import (
    build_group_lexicon,
    ethnicity_modified_nouns,
    collect_adj,
    collect_verb
)

nlp = spacy.load("en_core_web_sm")

GROUPS = {"filipino", "korean", "japanese", "chinese"}
GL = build_group_lexicon(GROUPS)


# ----------------------------
# Helper for running extractor
# ----------------------------

def run_extractors(text):
    doc = nlp(text)
    
    noun_hits = defaultdict(set)
    adj_hits  = defaultdict(set)
    verb_hits = defaultdict(set)

    for sent in doc.sents:
        nouns = ethnicity_modified_nouns(sent, GL)
        adjs  = collect_adj(sent, GL)
        verbs = collect_verb(sent, GL)

        # merge into aggregate sets
        for eth, vals in nouns.items():
            noun_hits[eth].update(vals)
        for eth, vals in adjs.items():
            adj_hits[eth].update(vals)
        for eth, vals in verbs.items():
            verb_hits[eth].update(vals)

    return noun_hits, adj_hits, verb_hits


# ----------------------------
# Tests
# ----------------------------

def test_simple_adj():
    text = "Filipinos are hardworking."
    nouns, adjs, verbs = run_extractors(text)

    assert adjs["filipino"] == {"hardworking"}
    assert nouns == {}
    assert verbs == {}


def test_simple_verb():
    text = "Filipinos protest loudly."
    nouns, adjs, verbs = run_extractors(text)

    assert verbs["filipino"] == {"protest"}


def test_modified_noun():
    text = "The Filipino workers protested."
    nouns, adjs, verbs = run_extractors(text)

    assert nouns["filipino"] == {"worker"}
    assert verbs["filipino"] == {"protest"}


def test_multiple_ethnicities():
    text = "Filipinos are hardworking and Japanese are polite."
    nouns, adjs, verbs = run_extractors(text)

    assert adjs["filipino"] == {"hardworking"}
    assert adjs["japanese"] == {"polite"}


def test_hyphenated_adj():
    text = "Filipinos are hard-working people."
    nouns, adjs, verbs = run_extractors(text)

    assert adjs["filipino"] == {"hard-working"}


def test_directional_modifier():
    text = "South Korean students are ambitious."
    nouns, adjs, verbs = run_extractors(text)

    # "south" should NOT get confused as ethnicity
    assert adjs["korean"] == {"ambitious"}


def test_passive_voice():
    text = "Rice was cooked by Chinese immigrants."
    nouns, adjs, verbs = run_extractors(text)

    assert verbs["chinese"] == {"cook"}


def test_noun_compound_modifiers():
    text = "Chinese restaurant workers protested."
    nouns, adjs, verbs = run_extractors(text)

    assert nouns["chinese"] == {"worker"}
    assert verbs["chinese"] == {"protest"}


def test_fragment_sentence():
    text = "Japanese hardworking always!"
    nouns, adjs, verbs = run_extractors(text)

    assert adjs["japanese"] == {"hardworking"}


def test_multi_adj_coordination():
    text = "Filipinos are smart and hardworking."
    nouns, adjs, verbs = run_extractors(text)

    assert adjs["filipino"] == {"smart", "hardworking"}


def test_unicode_punctuation():
    text = "Filipinos hardworking 💪 always!"
    nouns, adjs, verbs = run_extractors(text)

    assert adjs["filipino"] == {"hardworking"}


if __name__ == "__main__":
    # Manual debugging section
    print("Running manual debug mode...\n")

    import spacy
    from utils.filter_process.pos_db import (
        build_group_lexicon,
        ethnicity_modified_nouns,
        collect_adj,
        collect_verb
    )

    nlp = spacy.load("en_core_web_sm")
    GROUPS = {"filipino", "korean", "japanese", "chinese"}
    GL = build_group_lexicon(GROUPS)

    def debug(sentence):
        print("\n====================================")
        print("Sentence:", sentence)
        doc = nlp(sentence)

        print("\nTOKENS:")
        for tok in doc:
            print(
                f"{tok.text:10} | pos={tok.pos_:5} | dep={tok.dep_:10} | "
                f"head={tok.head.text:10} | lemma={tok.lemma_}"
            )

        # Run your extractors
        noun_hits = ethnicity_modified_nouns(doc, GL)
        adj_hits  = collect_adj(doc, GL)
        verb_hits = collect_verb(doc, GL)

        print("\nNOUN HITS:", noun_hits)
        print("ADJ HITS: ", adj_hits)
        print("VERB HITS:", verb_hits)
        print("====================================\n")


    # Try out sentences here:
    debug("Filipino workers protested loudly.")
    debug("South Korean boys are ambitious.")
    debug("A Turkic and central Asian ethnic group.")
    debug("Filipinos are hard-working and polite.")

