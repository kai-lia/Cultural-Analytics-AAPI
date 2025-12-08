"""
Bulk Ethnicity Attribute Runner
Validates extraction on multiple test sentences at once.
"""

import spacy
from pos import collect_adj, collect_verb, build_group_lexicon, ethnicity_modified_nouns

# -------------------------------------------------------
# Load your spaCy model + group lexicon
# -------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# use your real list
GROUPS = [
    "filipino"
]
GL = build_group_lexicon(GROUPS)

# -------------------------------------------------------
# Test sentences from your message
# -------------------------------------------------------
RAW_SENTENCES = [
    "the Southeast including the first [ETHNICITY] owned Cut Make Trim factory",
    "matter of time before the [ETHNICITY] buy their first Hollywood studio.",
    "for the older generations of [ETHNICITY] viewers, but the director is",
    "was even detained by the [ETHNICITY] police last month for reasons",
    "worn by university students and [ETHNICITY] officials to demonstrate Thai nationalist",
    "found that only 11% of [ETHNICITY] teenagers preferred domestic cartoons, compared",
    "My favourite [ETHNICITY] fast bowler with a spectacular",
    "is to assess whether the [ETHNICITY] voters are able to exercise"
]

raw = [

    
    "American automotive must keep up with hybrid technology, which was introduced by the [ETHNICITY].",
    "example of a 1920's faux [ETHNICITY] uke.",
    "issue on behalf of an [ETHNICITY] manufacturer that had received a",
    "years of rapid Latino and [ETHNICITY] growth.",
    "certain ethnic groups such as [ETHNICITY] or African Caribbean.",
    "It’s every [ETHNICITY]’s dream!",
    "More popular with [ETHNICITY] than overseas tourists, there wasn’t",
    "involved in a Swedish-[ETHNICITY] packaging venture .",
    "person voluntarily registers as an [ETHNICITY] under the Indian Act.",
    "because the [ETHNICITY] path is very hard.",
    "the whole [ETHNICITY] voters, and not the regional",
    "has been teaching [ETHNICITY] dance and culture",
    "serving alongside the [ETHNICITY] church to reach people",
    "Sixth Patriarch of [ETHNICITY] Cha’an Buddhism",
    "ease with which [ETHNICITY] workers tromp all over roofs",
    "every [ETHNICITY] child robbed of a future",
    "an advocacy tool for the [ETHNICITY] local authorities",
    "the [ETHNICITY] 3D printing experts are also",
    "Skilled [ETHNICITY] Workforce Act",
    "0.49% Native American, 1.07% [ETHNICITY],",
    "the [ETHNICITY] competitor it had acquired",
    "for [ETHNICITY] and European Cuisines.",
    "responsible [ETHNICITY] leaders.",
    "bean any [ETHNICITY] takeout around.",
    "The [ETHNICITY] bowlers took early wickets",
    "the most talked about [ETHNICITY] celebrities and events",
    "the dynamic changes of [ETHNICITY] society",
    "13 [ETHNICITY] Alone residents (0.504%)",
    "individual with [ETHNICITY] ethnicity",
    "As an [ETHNICITY] expat",
    "a [ETHNICITY] princess being conveyed",
    "customers 100% all [ETHNICITY],",
    "an eccentric [ETHNICITY] man.",
    "[ETHNICITY]-born singer Anggun",
    "for the hordes of [ETHNICITY] fans"
]






def run_tests(sentences):
    failures = []

    for sent in sentences:
        for eth in GROUPS:
            s = sent.replace("[ETHNICITY]", eth)
            doc = nlp(s)

            # Initialize combined result storage
            result = {}

            # ---- Collect features ----
            collect_adj(doc, GL, result)
            collect_verb(doc, GL, result)

            noun_attrs = ethnicity_modified_nouns(doc, GL)
            # Merge noun results if the group exists
            if eth in noun_attrs:
                result.setdefault(eth, {}).update(noun_attrs[eth])

            # ---- Print unified result ----
            # ---- Print unified result by category ----
            if eth in result and result[eth]:
                # Split by POS category
                noun_attrs = {w: c for w, c in result[eth].items() if doc.text.lower().find(w) != -1 and any(tok.lemma_.lower() == w and tok.pos_ in {"NOUN", "PROPN"} for tok in doc)}
                adj_attrs  = {w: c for w, c in result[eth].items() if any(tok.lemma_.lower() == w and tok.pos_ == "ADJ" for tok in doc)}
                verb_attrs = {w: c for w, c in result[eth].items() if any(tok.lemma_.lower() == w and tok.pos_ == "VERB" for tok in doc)}

                print(f"✔ {eth} in: {s}")
                print(f"   nouns: {noun_attrs or {}}")
                print(f"   adjectives: {adj_attrs or {}}")
                print(f"   verbs: {verb_attrs or {}}")

            else:
                failures.append((eth, s))
                print(f"✘ MISSING: {eth} in: {s}")

            print()


    print("\n======================================")
    print(" FAILURES SUMMARY")
    print("======================================")
    for eth, s in failures:
        print(f"❌ Missing: {eth} → {s}")

if __name__ == "__main__":
    run_tests(RAW_SENTENCES)
