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
    "highly", "fairly", "slightly", "rather", "especially",
    "incredibly, most"
}

CATEGORY_ADJ = {
    "ethnic", "certain", "such", "various", "many", "several",
    "some", "other", "only", "own", "different", "separate",
    "this", "that", "these", "those"
}

NEVER_ADJ_POS = {
    "DET", "AUX", "CCONJ", "SCONJ", "ADP", "PART", "INTJ",
    "SYMBOL", "NUM", "PUNCT", "X", "SPACE", "PRON"
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
    """Normalize and preserve correct form: "naïve" goes to "naive"""
    lemma_clean = unicodedata.normalize("NFKD", lemma).encode(
        "ascii", "ignore"
    ).decode("ascii").lower().strip()

    lemma_clean = re.sub(r"[^a-z\-]", "", lemma_clean)


    return lemma_clean


# ------------------------------------------------------------
# TAXONOMY DETECTOR
# ------------------------------------------------------------

def is_taxonomy_listing(tok, group_lexicon):
    lemma = tok.lemma_.lower()

    if lemma not in group_lexicon:
        return False

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
    
     # Left part must be an adjective-like token
    if t1.pos_ not in {"ADJ", "ADV"}:
        # Allow cases like "Filipino-American"
        if t1.pos_ == "PROPN" or t1.ent_type_ == "NORP":
            pass
        else:
            return None, None

    # Right part must be adjective-like
    if not (
        t3.pos_ in {"ADJ"}
        or t3.tag_ in {"VBG", "VBN"}  # good-looking, hard-working, well-trained
        or t3.text.lower().endswith("ing")  # fallback
    ):
        return None, None

    # Build a combined form like "hardworking"
    combined = (t1.text + t3.text).replace("-", "").lower()
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

    # Must be adjectival complement
    if tok.dep_ != "acomp":
        return False
    
    # Copula verb
    cop = tok.head
    if cop.lemma_ != "be":
        return False

    # Ethnicity must be subject of this same copula
    for child in cop.children:
        if child.dep_ in {"nsubj", "nsubjpass"}:
            # ethnicity may attach directly or via head
            if child == eth_tok or child.head == eth_tok:
                return True

    return False


# ------------------------------------------------------------
# ADJECTIVE-LIKE DETECTOR
# ------------------------------------------------------------

def is_adjective_like(tok, eth_tok):
    text = tok.text.lower()

    if tok.pos_ == "ADJ":
        return True


    if tok.pos_ == "VERB" and tok.tag_ == "VBG":
        # only accept if it modifies eth_tok itself (rare)
        if tok.dep_ == "amod" and tok.head == eth_tok:
            return True
        # or if it's a predicate adjective with ethnic subject
        if eth_tok.dep_ == "nsubj" and eth_tok.head == tok.head and tok.dep_ == "acomp":
            return True
        return False

    return False


# ------------------------------------------------------------
# MAIN FUNCTION — UNIFIED EXTRACTION
# ------------------------------------------------------------

def collect_all_modifiers(doc, group_lexicon):
    out = defaultdict(lambda: {"nouns": set(), "verbs": set(), "adjs": set()}) # formating for final output

    eth_tokens = [t for t in doc if t.lemma_.lower() in group_lexicon] # collecting ethnicities
    if not eth_tokens: # if no ethnicity, break
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

        # collecting cases like,  the chinese are people
        if eth_tok.dep_ == "nsubj" and eth_tok.head.lemma_ == "be":
            for child in eth_tok.head.children:
                if child.dep_ == "attr" and child.pos_ in {"NOUN", "PROPN"}:
                    out[eth]["nouns"].add(child.lemma_.lower())
        # =====================================================
        # VERB EXTRACTION
        # =====================================================

        # If ethnicity modifies a noun, and that noun is subject of a verb/copula
        if head.pos_ in {"NOUN", "PROPN"} and eth_tok.dep_ in {"amod", "compound"}:

            # Case: noun is subject of a verb ("Filipino workers protested")
            if head.dep_ == "nsubj" and head.head.pos_ == "VERB":
                main_verb = head.head
                out[eth]["verbs"].add(main_verb.lemma_.lower())

                # coordinated verbs: "protested and marched"
                for sib in main_verb.children:
                    if sib.dep_ == "conj" and sib.pos_ == "VERB":
                        out[eth]["verbs"].add(sib.lemma_.lower())

            # Case: noun is subject of BE copula ("are pretty", "are pretty and protested")
            if head.dep_ == "nsubj" and head.head.lemma_ == "be":
                copula = head.head

                # predicate adjectival complement: "pretty"
                for child in copula.children:
                    if child.dep_ == "acomp" and child.pos_ == "ADJ":
                        out[eth]["adjs"].add(child.lemma_.lower())

                        # coordinated adjectives: "pretty and hardworking"
                        for sib in child.children:
                            if sib.dep_ == "conj" and sib.pos_ == "ADJ":
                                out[eth]["adjs"].add(sib.lemma_.lower())

                        # coordinated verbs: "pretty and protested"
                        for sib in child.children:
                            if sib.dep_ == "conj" and sib.pos_ == "VERB":
                                out[eth]["verbs"].add(sib.lemma_.lower())


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
            if tok.pos_ in NEVER_ADJ_POS: # speeeeding up proceses
                continue
            #checking if ethnic term
            if tok == eth_tok or tok.ent_type_ == "NORP": # removes tems like chinese malay
                continue
            lemma = tok.lemma_.lower()
            if lemma in group_lexicon:
                continue
            if lemma in INTENSIFIERS or lemma in CATEGORY_ADJ:
                continue
            
           

            # hypenated dealings 
            if tok.i in blocked: # this is for my hypened case
                continue

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

            if is_taxonomy_listing(tok, group_lexicon):
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
            if tok.dep_ == "amod" and (tok.head == eth_tok.head or tok.head == eth_tok):
                out[eth]["adjs"].add(clean)
                continue

            # fragment: "Japanese super polite"
            if eth_tok.i < tok.i:
                between = doc[eth_tok.i + 1 : tok.i]
                if not any(t.pos_ in {"VERB", "NOUN"} for t in between):
                    out[eth]["adjs"].add(clean)
                    continue

    return out





