#!/usr/bin/env python3
"""
FastText Model Loader + Ethnicity Window Masking Utility

Used for:
    - Loading an autotuned FastText model
    - Masking ethnicity terms in sentences for classification tasks
"""

# Imports

import fasttext
from typing import List, Optional, Tuple, Union
from pathlib import Path


# Path to FastText Model
MODEL_PATH = Path("utils/filter_process/Fasttext/autotuned_fasttext_model.bin")


def load_fasttext_model() -> Optional[fasttext.FastText._FastText]:
    """
    Safely load the FastText model with clear error messages.
    Returns: model or None if loading failed.
    """
    try:
        model = fasttext.load_model(str(MODEL_PATH))
        print(f"FastText model loaded successfully from {MODEL_PATH}")
        return model

    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
    except ValueError as e:
        print(f"ValueError: {e}")
        print("    The model file may be corrupted or not a valid FastText model.")
    except Exception as e:
        print(f"Unexpected error while loading FastText model: {e}")

    return False


def window_mask_sentence(ethnicity_term, sent_tokens, tokens_lower, window=5):
    """
    sent_tokens: list of token strings (in order)
    aapi_groups_set: set of lowercase group terms
    window: number of tokens to keep on each side

    Returns: a masked window string or None if no ethnicity term found.
    """
    # find its index in the ordered list
    idx = tokens_lower.index(ethnicity_term)

    # mask token
    tokens_masked = sent_tokens[:]  # copy list
    tokens_masked[idx] = "[ETHNICITY]"

    # window boundaries
    start = max(0, idx - window)
    end = min(len(tokens_masked), idx + window + 1)
    # final join
    return " ".join(tokens_masked[start:end])


def fasttext_predict(model, text: Union[str, List[str]], k: int = 1):
    """
    Unified API that supports both single-string prediction
    and batched prediction over a list of strings.
    
    Returns:
        - If input is str → returns label string
        - If input is list[str] → returns list of label strings
    """
    if isinstance(text, str):
        labels, _ = model.predict(text, k=k)
        return labels[0]  # "__label__1" or "__label__0"

    # Batched
    labels_list, _ = model.predict(text, k=k)
    # Return ["__label__1", "__label__0", ...]
    return [labels[0] for labels in labels_list]
