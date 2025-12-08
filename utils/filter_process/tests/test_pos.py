import spacy
from collections import Counter

import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))



from filter_process.pos import (
    build_group_lexicon,
    collect_adj,
    collect_verb,
    ethnicity_modified_nouns,
    build_adj_dep_matcher,
    tokenize_clean
)





nlp = spacy.load("en_core_web_sm")
matcher = build_adj_dep_matcher(nlp)

GROUPS = {"filipino", "korean", "japanese", "chinese"}
GL = build_group_lexicon(GROUPS, nlp.vocab)


def test_ethnicity_modified_nouns_amod():
    sent = nlp("the filipino women protested.")
    noun = ethnicity_modified_nouns(sent, GROUPS)
    print(noun)
    assert noun == "woman"


def test_ethnicity_modified_nouns_compound():
    sent = nlp("chinese doctors treated patients.")
    noun = ethnicity_modified_nouns(sent, GROUPS)
    print(noun)
    assert noun == "doctor"


def test_collect_adj_simple():
    sent = nlp(("filipinos are hardworking.").strip())
    d = {}
    
    collect_adj(sent, GL, d)
    print(d)
    assert "filipino" in d
    assert d == {"filipino": Counter({'hardworking': 1})}


def test_collect_adj_norp_block():
    sent = nlp("chinese people are chinese.")
    d = {}
    collect_adj(sent, GL, d)
    print(d)
    assert d == {}  # ethnicity adj should not be counted


def test_collect_verb_passive_subj():
    sent = nlp("rice is cooked by filipinos.")
    d = {}
    collect_verb(sent, GL, d)
    print(d)
    assert "filipino" in d
    assert d["filipino"]["cook"] == 1


def test_collect_adj_fallback_single_group():
    sent = nlp("japanese hardworking.")
    d = {}
    collect_adj(sent, GL, d)
    print(d)
    assert "japanese" in d
    assert d["japanese"]["hardworking"] == 1



def debug_print(sentence: str):
    doc = nlp(sentence)
    print("\nDEBUG —", sentence)
    for tok in doc:
        print(
            f"{tok.text:12} | pos={tok.pos_:5} | dep={tok.dep_:10} | "
            f"lem={tok.lemma_:12} | ent={tok.ent_type_}"
        )

def test_directional_adj_filtering():
    sent = nlp("South Korean men arrived.")
    noun = ethnicity_modified_nouns(sent, GROUPS)
    # directional adjective "South" should NOT block detection of "Korean"
    assert noun == "man"


def test_debug_inspect_tokens():
    # Use this when a sentence fails unexpectedly in other tests
    debug_print("South Korean men arrived.")
    debug_print("The filipino women protested.")
    debug_print("Japanese hardworking.")
    # No assertion — just prints diagnostics

def test_collect_adj_multiple_ethnicities():
    sent = nlp("Filipinos are hardworking and Japanese are polite.")
    d = {}
    collect_adj(sent, GL, d)
    assert d["filipino"]["hardworking"] == 1
    assert d["japanese"]["polite"] == 1


def test_collect_adj_hyphenated():
    sent = nlp("Filipinos are hard-working.")
    d = {}
    collect_adj(sent, GL, d)
    assert d["filipino"]["hard-working"] == 1

def test_collect_adj_directional_modifier():
    sent = nlp("South Korean students are ambitious.")
    d = {}
    collect_adj(sent, GL, d)
    assert d["korean"]["ambitious"] == 1

def test_collect_adj_multi_adj_coordination():
    sent = nlp("Filipinos are smart and hardworking.")
    d = {}
    collect_adj(sent, GL, d)
    assert d["filipino"]["smart"] == 1
    assert d["filipino"]["hardworking"] == 1


def test_collect_adj_fragment_style():
    sent = nlp("japanese super polite")
    d = {}
    collect_adj(sent, GL, d)
    assert d["japanese"]["polite"] == 1


def test_collect_adj_compound_noun_subject():
    sent = nlp("Chinese restaurant workers are efficient.")
    d = {}
    collect_adj(sent, GL, d)
    assert d["chinese"]["efficient"] == 1

def test_collect_adj_comma_interrupted():
    sent = nlp("Filipinos, hardworking, always help.")
    d = {}
    collect_adj(sent, GL, d)
    assert d["filipino"]["hardworking"] == 1


def test_collect_adj_social_media_punctuation():
    sent = nlp("Filipinos hardworking 💪 always!")
    d = {}
    collect_adj(sent, GL, d)
    assert d["filipino"]["hardworking"] == 1

def test_ethnicity_noun_modifiers_examples():
    tests = [
        ("the [ETHNICITY] bodyguards of wealthy CEOs", "bodyguard"),  # amod
        ("precarious dwelling situations of the [ETHNICITY] poor.", "poor"),  # compound
        ("the general welfare of the [ETHNICITY] consumers.", "consumer"),
        ("very dorky cut for a [ETHNICITY] fighter", "fighter"),
        ("a noted [ETHNICITY] archeologist discovers …", "archeologist"),
        ("the [ETHNICITY] rulers feared rebellion", "ruler"),
        ("of Aboriginal father Lloyd, and [ETHNICITY] mother Aiaga", "mother"),
        ("A [ETHNICITY] and East Asian slant …", "slant"),  # no noun after modifier
        ("an [ETHNICITY]-Australian’s ancestral home", "australian"),  # hyphen split
        ("Former [ETHNICITY] cricketers…", "cricketer"),
        ("Indulge in mouthwatering [ETHNICITY] hawker fare", "fare"),
        ("as a [ETHNICITY], you wouldn’t have much to worry", None),  # ethnicity as noun
        ("Native [ETHNICITY] Health Board…", "board"),  # multi-word compound
        ("[ETHNICITY] factory workers protested", "worker"),
        ("[ETHNICITY] delegation arrived", "delegation"),
        ("healthy [ETHNICITY] participants with prehypertension", "participant"),
        ("[ETHNICITY] migrants to the UK would…", "migrant"),
        ("the [ETHNICITY] community have worked…", "community"),
        ("vision of how to improve [ETHNICITY] society…", "society"),
    ]

    for text, expected in tests:
        sent = nlp(text.replace("[ETHNICITY]", "japanese"))
        result = ethnicity_modified_nouns(sent, GROUPS)
        print(text, "→", result)
        assert result == expected



def test_collect_adj_real_world_cases():
    nlp = spacy.load("en_core_web_sm")
    matcher = build_adj_dep_matcher(nlp)

    tests = [
        ("the Japanese poor are suffering.", {"japanese": {"poor": 1}}),
        ("very dorky cut for a Japanese fighter.", {"japanese": {"dorky": 1}}),
        ("the Japanese community are strong.", {"japanese": {"strong": 1}}),
        ("Japanese hardworking 💪 always!", {"japanese": {"hardworking": 1}}),
        ("A South Korean boy is ambitious.", {"korean": {"ambitious": 1}}),
        ("Japanese rocket scientists are brilliant.", {"japanese": {"brilliant": 1}}),
        ("Japanese are polite and hardworking.", {"japanese": {"polite": 1, "hardworking": 1}}),
    ]

    for text, expected in tests:
        doc = nlp(text)
        d = {}

        for sent in doc.sents:              # ⬅ IMPORTANT
            collect_adj(sent, GL, d)

        print(text, "→", d)

        for eth, adjectives in expected.items():
            assert eth in d
            for adj, cnt in adjectives.items():
                assert d[eth][adj] == cnt




def test_collect_verb_real_world_cases():
    nlp = spacy.load("en_core_web_sm")

    tests = [
        # Passive voice: ethnicity is AGENT of verb
        ("Rice is cooked by Filipinos.", {"filipino": {"cook": 1}}),

        # Passive with trailing PP
        ("The food was invented by Chinese immigrants.", {"chinese": {"invent": 1}}),

        # Active — subject is ethnicity group
        ("Filipinos celebrate together every weekend.", {"filipino": {"celebrate": 1}}),

        # Coordinated verbs
        ("Koreans dance and sing during festivals.", {"korean": {"dance": 1, "sing": 1}}),

        # Relative clause verbs
        # ("This is the man who taught Japanese students.", {"japanese": {}}),

        # Participial clause modifying ethnicity
        ("A group led by Japanese arrived early.", {"japanese": {"arrive": 1}}),

        # Single-group fallback heuristic (fragment)
        #("Chinese fighting always.", {"chinese": {"fight": 1}}),

        # Ethnicity noun deep in sutree
        ("The workers, mostly Korean, protested.", {"korean": {"protest": 1}}),

        # Verb phrase with auxiliaries
        ("Filipinos have been working tirelessly.", {"filipino": {"work": 1}}),

        # Negation case (counts verb; negation handled later if needed)
        ("Chinese will not surrender.", {"chinese": {"surrender": 1}}),
    ]

    for text, expected in tests:
        doc = nlp(text)
        d = {}

        # IMPORTANT: run per sentence to preserve dependency attachment
        for sent in doc.sents:
            collect_verb(sent, GL, d)

        print(text, "→", d)

        for eth, verbs in expected.items():
            assert eth in d
            for v, cnt in verbs.items():
                assert d[eth][v] == cnt








if __name__ == "__main__":
    test_ethnicity_noun_modifiers_examples()
