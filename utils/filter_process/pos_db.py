# ============================================================
# pos.py — unified extractor for ethnicity nouns / verbs / adjs
# ============================================================

import re
import unicodedata
from collections import defaultdict

# ------------------------------------------------------------
# STOPWORD SETS
# ------------------------------------------------------------

INTENSIFIERS = {
    "super", "very", "really", "quite", "so", "too", "extremely",
    "highly", "fairly", "slightly", "pretty", "rather", "especially",
    "incredibly"
}

CATEGORY_ADJ = {
    "ethnic", "certain", "such", "various", "many", "several",
    "some", "other", "only", "own", "different", "separate",
    "this", "that", "these", "those"
}

# ------------------------------------------------------------
# GROUP LEXICON
# ------------------------------------------------------------

def build_group_lexicon(groups, vocab=None):
    lex = {}
    for g in groups:
        g = g.lower()
        lex[g] = g
        if not g.endswith("s"):
            lex[g + "s"] = g     # Koreans → korean
    return lex


# ------------------------------------------------------------
# ADJECTIVE CLEANER
# ------------------------------------------------------------

def clean_adj_string(lemma: str, tok=None) -> str:
    """Normalize and preserve correct -ing form."""
    lemma_clean = unicodedata.normalize("NFKD", lemma).encode(
        "ascii", "ignore"
    ).decode("ascii").lower().strip()

    lemma_clean = re.sub(r"[^a-z\-]", "", lemma_clean)

    if tok is not None:
        text = tok.text.lower()
        if text.endswith("ing") and not lemma_clean.endswith("ing"):
            if lemma_clean.endswith("e"):
                lemma_clean = lemma_clean[:-1]     # hardworke → hardwork
            lemma_clean = lemma_clean + "ing"       # → hardworking

    return lemma_clean


# ------------------------------------------------------------
# TAXONOMY DETECTOR
# ------------------------------------------------------------

def is_taxonomy_listing(tok, group_lexicon):
    lemma = tok.lemma_.lower()

    if lemma not in group_lexicon:
        return False

    if tok.ent_type_ == "NORP":
        return True

    if tok.dep_ == "conj" and tok.head.lemma_.lower() in group_lexicon:
        return True

    subtree = [t.text.lower() for t in tok.head.subtree]
    if "such" in subtree and tok.head.pos_ == "NOUN":
        return True

    return False


# ------------------------------------------------------------
# HYPHENATED ADJECTIVE DETECTOR
# ------------------------------------------------------------

def is_hyphenated_adj(tok):
    doc = tok.doc
    i = tok.i
    if i + 2 >= len(doc):
        return None, None

    t1, t2, t3 = doc[i], doc[i+1], doc[i+2]

    if t2.text != "-":
        return None, None

    if not (t3.pos_ in {"ADJ", "VERB"} or t3.text.lower().endswith("ing")):
        return None, None

    if t1.pos_ not in {"ADJ", "ADV", "VERB"}:
        return None, None

    combined = f"{t1.text.lower()}{t3.text.lower()}"
    return combined, [i, i+1, i+2]


# ------------------------------------------------------------
# FIXED PREDICATE-ADJECTIVE DETECTOR (THE KEY FIX)
# ------------------------------------------------------------

def is_predicate_adj(tok, eth_tok):
    """
    Works for:
        Filipinos are hardworking.
        Japanese are polite and disciplined.
    spaCy parses "hardworking" as ROOT(VERB) with "are" as AUX child.
    """

    # Look for "are/is/was/were" as AUX child
    has_be_aux = any(
        c.pos_ == "AUX" and c.lemma_ == "be"
        for c in tok.children
    )
    if not has_be_aux:
        return False

    # Check ethnicity is subject of this token
    has_eth_subject = any(
        c == eth_tok and c.dep_ in {"nsubj", "nsubjpass"}
        for c in tok.children
    )

    return has_eth_subject


# ------------------------------------------------------------
# ADJECTIVE-LIKE DETECTOR
# ------------------------------------------------------------

def is_adjective_like(tok, eth_tok):
    text = tok.text.lower()

    if tok.pos_ == "ADJ":
        return True

    if text.endswith("ing"):
        # handled via predicate detector only
        return True

    if text.endswith("ed"):
        if tok.dep_ == "amod" and tok.head.pos_ == "NOUN":
            return True
        if tok.head.lemma_ == "be":
            return True
        return False

    return False


# ------------------------------------------------------------
# MAIN FUNCTION — UNIFIED EXTRACTION
# ------------------------------------------------------------

def collect_all_modifiers(doc, group_lexicon):
    out = defaultdict(lambda: {"nouns": set(), "verbs": set(), "adjs": set()})

    eth_tokens = [t for t in doc if t.lemma_.lower() in group_lexicon]
    if not eth_tokens:
        return out

    for eth_tok in eth_tokens:
        eth = group_lexicon[eth_tok.lemma_.lower()]
        head = eth_tok.head

        # =====================================================
        # NOUN EXTRACTION
        # =====================================================

        if head.pos_ in {"NOUN", "PROPN"} and eth_tok.dep_ in {"amod", "compound"}:
            out[eth]["nouns"].add(head.lemma_.lower())

        for child in eth_tok.children:
            if child.pos_ in {"NOUN", "PROPN"} and child.dep_ in {"compound", "amod"}:
                out[eth]["nouns"].add(child.lemma_.lower())

        # =====================================================
        # VERB EXTRACTION
        # =====================================================

        # "Filipino workers protested"
        if head.pos_ in {"NOUN", "PROPN"} and eth_tok.dep_ in {"amod", "compound"}:
            if head.dep_ == "nsubj" and head.head.pos_ == "VERB":
                out[eth]["verbs"].add(head.head.lemma_.lower())

        # ethnicity itself is subject: "Filipinos protested"
        if eth_tok.dep_ in {"nsubj", "nsubjpass"} and head.pos_ == "VERB":
            # avoid predicate adjectives misparsed as verbs
            if head.text.lower().endswith("ing") and any(
                c.pos_ == "AUX" and c.lemma_ == "be"
                for c in head.children
            ):
                pass
            else:
                out[eth]["verbs"].add(head.lemma_.lower())

                for conj in head.children:
                    if conj.dep_ == "conj" and conj.pos_ == "VERB":
                        out[eth]["verbs"].add(conj.lemma_.lower())

        # Passive agent: "cooked by Filipinos"
        if eth_tok.dep_ == "pobj" and eth_tok.head.dep_ == "agent":
            verb = eth_tok.head.head
            if verb.pos_ == "VERB":
                out[eth]["verbs"].add(verb.lemma_.lower())

        # =====================================================
        # ADJECTIVE EXTRACTION
        # =====================================================

        blocked = set()

        for tok in doc:
            if tok == eth_tok or tok.ent_type_ == "NORP":
                continue

            lemma = tok.lemma_.lower()

            if lemma in INTENSIFIERS or lemma in CATEGORY_ADJ:
                continue
            if lemma in group_lexicon:
                continue
            if is_taxonomy_listing(tok, group_lexicon):
                continue

            # --- hyphenated adjectives ---
            hyph, span = is_hyphenated_adj(tok)
            if hyph:
                blocked.update(span)

                pred = tok.head
                if pred.head.lemma_ == "be" and any(
                    c == eth_tok and c.dep_ in {"nsubj", "nsubjpass"}
                    for c in pred.head.children
                ):
                    out[eth]["adjs"].add(hyph)
                    continue

                if tok.head == eth_tok:
                    out[eth]["adjs"].add(hyph)
                    continue

                continue

            # --- normal adjectives ---
            if not is_adjective_like(tok, eth_tok):
                continue

            clean = clean_adj_string(lemma, tok)

            # Proper predicate handling (THE KEY FIX)
            if is_predicate_adj(tok, eth_tok):
                out[eth]["adjs"].add(clean)
                continue

            # coordinated adjectives: polite and hardworking
            if tok.dep_ == "conj" and tok.head.pos_ == "ADJ":
                if is_predicate_adj(tok.head, eth_tok):
                    out[eth]["adjs"].add(clean)
                    continue

            # attributive: hardworking Filipino workers
            if tok.dep_ == "amod" and tok.head == eth_tok.head:
                out[eth]["adjs"].add(clean)
                continue

            # fragment: "Japanese super polite"
            if eth_tok.i < tok.i:
                between = doc[eth_tok.i + 1 : tok.i]
                if not any(t.pos_ == "VERB" for t in between):
                    out[eth]["adjs"].add(clean)
                    continue

    return out





