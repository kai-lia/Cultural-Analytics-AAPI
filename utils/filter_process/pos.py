
# pos.py — AAPI relation extraction

from collections import Counter
from spacy.matcher import DependencyMatcher
from collections import Counter



STOP_ADJ = {"north", "south", "east", "west", "central"}

BAD_VERBS = {"be", "have", "do"}
BAD_LEMMAS = {"orient"}


def tokenize_clean(doc):
    return [
        t.lemma_.lower()
        for t in doc
        if t.is_alpha and not t.is_stop
    ]



def build_group_lexicon(groups, vocab=None):
    """
    Return dict mapping RAW STRING forms -> canonical group name.
    Includes plural fallback: koreans -> korean
    """
    lex = {}

    for g in groups:
        g_l = g.lower()
        lex[g_l] = g_l  # base form

        # plural dictionary fallback
        if not g_l.endswith("s"):
            lex[g_l + "s"] = g_l  # filipinos -> filipino

    return lex



STOP_ADJ = {"north", "south", "east", "west", "central"}

def is_directional(adj_str: str) -> bool:
    """Only operate on lowercase strings."""
    if not isinstance(adj_str, str):
        return False
    return adj_str in STOP_ADJ or any(s in adj_str for s in STOP_ADJ)



def get_ethnicity_cache(sentence, group_lexicon):
    eth_by_i = {}
    groups_in_sentence = set()
    for tok in sentence:
        # allow plural: Filipinos → filipino
        text_form = tok.text.lower()
        lemma_form = tok.lemma_.lower()

        eth = (group_lexicon.get(text_form) or
               group_lexicon.get(lemma_form))

        if eth:
            eth_by_i[tok.i] = eth
            groups_in_sentence.add(eth)
    return eth_by_i, groups_in_sentence


def get_subjects_for_pred(pred):
    subs = [c for c in pred.children if c.dep_ in {"nsubj", "nsubjpass"}]
    if subs:
        return subs

    head = pred.head
    visited = set()
    while head is not None and head not in visited:
        visited.add(head)
        subs = [c for c in head.children if c.dep_ in {"nsubj", "nsubjpass"}]
        if subs:
            return subs
        if head.dep_ in {"conj", "acomp", "attr"} or head.pos_ == "AUX":
            head = head.head
        else:
            break
    return []




from collections import defaultdict, Counter

def ethnicity_modified_nouns(doc, group_lexicon):
    """
    Extract nouns that are syntactically modified by an ethnicity.
    
    Returns:
        dict: {ethnicity: Counter({noun: count})}
    """
    results = defaultdict(Counter)

    for tok in doc:
        # Only detect ethnicity mentions
        eth = tok.lemma_.lower()
        if eth not in group_lexicon:
            continue

        group = group_lexicon[eth]
        head = tok.head

        # Two common syntactic cases:
        # 1) Ethnicity is a modifier of noun ("Filipino workers")
        if head.pos_ in {"NOUN", "PROPN"} and tok.dep_ in {"amod", "compound"}:
            results[group][head.lemma_.lower()] += 1

        # 2) Ethnicity as a compound in multi-token adjectives ("Filipino-American community")
        for child in tok.children:
            if child.pos_ in {"NOUN", "PROPN"} and child.dep_ in {"compound", "amod"}:
                results[group][child.lemma_.lower()] += 1

    return dict(results)


from spacy.matcher import DependencyMatcher

def build_verb_dep_matcher(nlp):
    matcher = DependencyMatcher(nlp.vocab)

    # Pattern: Subject → Verb
    # e.g. "Filipinos cook rice"
    pattern_active = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB"},
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "subj",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}},
        },
    ]

    # Pattern: Passive Agent "by" ethnicity
    # e.g. "rice is cooked by Filipinos"
    pattern_passive_agent = [
        {
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB"},
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "agent",
            "RIGHT_ATTRS": {"DEP": "agent"},
        },
    ]

    matcher.add("ACTIVE_VERB_SUBJ", [pattern_active])
    matcher.add("PASSIVE_AGENT_VERB", [pattern_passive_agent])

    return matcher


def collect_verb(doc, group_lexicon, d):
    """
    Extract verbs that an ethnicity or ethnicity-modified subject actively performs.
    Do NOT extract passive constructions or cases where ethnicity is object.
    """
    for tok in doc:
        eth = tok.lemma_.lower()
        if eth not in group_lexicon:
            continue

        group = group_lexicon[eth]
        head = tok.head

        # Case 1 — ethnicity modifies the grammatical subject of a verb
        # e.g. "Filipino workers took..."
        if head.pos_ in {"NOUN", "PROPN"} and tok.dep_ in {"amod", "compound"}:
            if head.dep_ == "nsubj" and head.head.pos_ == "VERB":
                d.setdefault(group, Counter())[head.head.lemma_.lower()] += 1

        # Case 2 — ethnicity itself is the grammatical SUBJECT of a verb
        # e.g. "Filipinos tromp..."
        if tok.dep_ == "nsubj" and tok.head.pos_ == "VERB":
            d.setdefault(group, Counter())[tok.head.lemma_.lower()] += 1

def build_adj_dep_matcher(nlp):
    matcher = DependencyMatcher(nlp.vocab)

    # Pattern 1 — predicate complement (Filipinos are hardworking)
    pattern_predicate = [
        {
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {"POS": "ADJ"},
        },
        {
            "LEFT_ID": "adj",
            "REL_OP": ">",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"POS": "VERB"},
        },
    ]

    # Pattern 2 — direct modifier (Chinese doctors)
    pattern_amod = [
        {
            "RIGHT_ID": "adj",
            "RIGHT_ATTRS": {"POS": "ADJ"},
        },
        {
            "LEFT_ID": "adj",
            "REL_OP": ">",
            "RIGHT_ID": "noun",
            "RIGHT_ATTRS": {"POS": "NOUN"},
        },
    ]

    matcher.add("ETHNICITY_ADJ", [pattern_predicate, pattern_amod])
    return matcher




from collections import Counter

# Words that never express a trait
INTENSIFIERS = {
    "super", "very", "really", "quite", "so", "too", "extremely", "highly",
    "fairly", "slightly", "pretty", "rather", "especially", "incredibly",
}

# Words describing category/quantity, not traits
CATEGORY_ADJ = {
    "ethnic", "certain", "such", "various", "many", "several", "some",
    "other", "only", "own", "different", "separate",
    "this", "that", "these", "those"  # demonstratives
}


def is_hyphenated_adj(tok):
    """Return hyphenated ADJ-'ADJ/VERB' form if present, else None.
    Example: 'hard-working' → keep single token
    """
    if tok.pos_ != "ADJ":
        return None
    nxt = tok.nbor(1) if tok.i + 1 < len(tok.doc) else None
    nxt2 = tok.nbor(2) if tok.i + 2 < len(tok.doc) else None
    if nxt and nxt.text == "-" and nxt2 and nxt2.pos_ in {"ADJ", "VERB"}:
        return f"{tok.text.lower()}-{nxt2.text.lower()}"
    return None


def is_taxonomy_listing(tok):
    """
    True if ethnicity is part of structural taxonomy, not a trait:
        'ethnic groups such as japanese or African Caribbean'
        '[ETH] or [ETH]'
    """
    # Coordinated ethnicity nouns/adjectives in a list
    if tok.dep_ == "conj" and tok.head.pos_ in {"NOUN", "ADJ"}:
        return True

    # Head noun subtree includes 'such' (taxonomy marker)
    subtree = [t.text.lower() for t in tok.head.subtree]
    if "such" in subtree and tok.head.pos_ == "NOUN":
        return True

    return False


def is_ethnicity_as_modifier(tok, group_lexicon):
    """
    If token is itself in group lexicon, ignore as attribute.
    e.g., 'japanese or African Caribbean'
    """
    return tok.lemma_.lower() in group_lexicon


def attribute_applies_to_ethnicity(adj_tok, eth_tok):
    """
    Ensure the ADJ is **syntactically modifying** the ethnicity.
    Allowed links:
      - adj → ethnicity via 'amod' (standard)
      - adj is 'acomp' with ethnicity as subject
    """
    # Standard adjective modifier: 'polite japanese people'
    if adj_tok.dep_ == "amod" and adj_tok.head == eth_tok.head:
        return True

    # Predicate adjective: 'Japanese people are polite'
    if adj_tok.dep_ in {"acomp", "attr"}:  # adjective complement
        # ethnicity must be grammatical subject
        for child in adj_tok.head.children:
            if child == eth_tok and child.dep_ in {"nsubj", "nsubjpass"}:
                return True

    return False


def collect_adj(doc, group_lexicon, d):
    """
    Extract adjectives describing ethnic groups.
    Adds: d[group][adj] += count
    """
    # Locate ethnicity mentions first
    ethnicity_tokens = [
        tok for tok in doc
        if tok.lemma_.lower() in group_lexicon
    ]

    if not ethnicity_tokens:
        return

    for eth in ethnicity_tokens:
        group = group_lexicon[eth.lemma_.lower()]
        d.setdefault(group, Counter())

        for tok in doc:

            # Skip the ethnicity token itself
            if tok == eth:
                continue

            # Skip intensifiers + category words
            lemma = tok.lemma_.lower()
            if lemma in INTENSIFIERS or lemma in CATEGORY_ADJ:
                continue

            # Skip other ethnicity tokens (don't treat as attribute)
            if is_ethnicity_as_modifier(tok, group_lexicon):
                continue

            # Skip enumeration/listing cases
            if is_taxonomy_listing(tok):
                continue

            # Hyphenated adjective?
            hyphen = is_hyphenated_adj(tok)
            if hyphen:
                # Only if hyphenated adj modifies this ethnicity
                if attribute_applies_to_ethnicity(tok, eth):
                    d[group][hyphen] += 1
                continue  # skip further processing

            # Only adjectives count
            if tok.pos_ != "ADJ":
                continue

            # Ensure **syntactic** link to the ethnicity
            if attribute_applies_to_ethnicity(tok, eth):
                d[group][lemma] += 1
