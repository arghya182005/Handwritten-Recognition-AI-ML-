"""
CTC greedy decoding: turns the CRNN's (1, time_steps, num_classes)
softmax output into text plus a confidence score.

The LAST class index (num_classes - 1) is reserved for the CTC
"blank" token. For this model that is index 79, which matches
vocab.json's "pad_token": 79 (char_to_num holds 79 real characters at
indices 0-78, and the model's final Dense layer outputs one extra
class beyond them for blank — 80 classes total).
"""

import numpy as np


def _ensure_probabilities(preds):
    """CTC models almost always end in a softmax, but guard against
    raw logits being passed in so the confidence score stays
    meaningful (argmax-based decoding is unaffected either way, since
    softmax is monotonic)."""
    sums = preds.sum(axis=-1)
    if np.allclose(sums, 1.0, atol=1e-2):
        return preds
    exp = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def ctc_greedy_decode(preds, charset):
    """
    preds: np.ndarray shaped (1, time_steps, num_classes) — raw
        model.predict() output.
    charset: list where charset[i] is the character for class i.
        (It's fine if charset also has an entry for the blank index —
        it's never read, since blank is filtered before the charset
        lookup happens.)

    Returns: (text, confidence)
    """
    probs = _ensure_probabilities(preds)

    pred_indices = np.argmax(probs, axis=-1)[0]   # (T,) predicted class per timestep
    pred_probs = np.max(probs, axis=-1)[0]         # (T,) confidence per timestep

    # *** FIX ***
    # The previous version computed `blank_index = len(charset)`,
    # which silently breaks if charset isn't exactly the real
    # characters with nothing else appended (e.g. it becomes wrong
    # the moment charset includes a placeholder slot for the blank
    # itself, or is any length other than num_classes - 1). Deriving
    # blank_index straight from the model's output width is
    # unambiguous and always correct for this "last class = blank"
    # convention, regardless of how charset happens to be built
    # elsewhere in the app.
    num_classes = preds.shape[-1]
    blank_index = num_classes - 1

    decoded_chars = []
    kept_probs = []
    last_idx = -1

    for idx, prob in zip(pred_indices, pred_probs):
        # Standard CTC greedy-decode rule: collapse consecutive
        # repeats (of ANY class, including blank) before filtering.
        if idx == last_idx:
            continue

        if idx != blank_index and 0 <= idx < len(charset):
            decoded_chars.append(charset[idx])
            kept_probs.append(prob)

        last_idx = idx

    text = "".join(decoded_chars)
    confidence = float(np.mean(kept_probs)) if kept_probs else 0.0

    return text, confidence
