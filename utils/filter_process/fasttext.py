
model_path = "utils/autotuned_fasttext_model.bin"

try:
    # Load the FastText model
    model = fasttext.load_model(model_path)
    print(f"FastText model loaded successfully from {model_path}")
except ValueError as e:
    print(
        f"Error loading model: {e}. It might be that the file is corrupted or not a valid FastText model."
    )
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


    


def load_model():
    try:
        # Load the FastText model
        model = fasttext.load_model(MODEL_PATH)
        return model
        print(f"FastText model loaded successfully from {MODEL_PATH}")
    except ValueError as e:
        print(
            f"Error loading model: {e}. It might be that the file is corrupted or not a valid FastText model."
        )
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def window_mask_sentence(ethnicity_term, sent_tokens, tokens_lower, window=5):
    """
    sent_tokens: list of token strings (in order)
    aapi_groups_set: set of lowercase group terms
    window: number of tokens to keep on each side

    Returns: a masked window string or None if no ethnicity term found.
    """

    # Find its index in the ordered list
    idx = tokens_lower.index(ethnicity_term)

    # Mask token
    tokens_masked = sent_tokens[:]  # copy list
    tokens_masked[idx] = "[ETHNICITY]"

    # Window boundaries
    start = max(0, idx - window)
    end = min(len(tokens_masked), idx + window + 1)
    # final join
    return " ".join(tokens_masked[start:end])