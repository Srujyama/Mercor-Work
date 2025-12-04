import re

from datasets import load_dataset

GEMINI_NAME_PATTERN = r"gemini"
EXPECTED_GEMINI_NOT_WIN = 318  # This is what you just observed


def has_gemini(example):
    """True if either model_a or model_b contains 'gemini'."""
    return (
        re.search(GEMINI_NAME_PATTERN, example["model_a"], re.IGNORECASE) is not None
        or re.search(GEMINI_NAME_PATTERN, example["model_b"], re.IGNORECASE) is not None
    )


def gemini_not_winner(example):
    """
    Gemini is present AND Gemini did NOT win.

    We EXCLUDE only:
      - Gemini in model_a AND winner == 'model_a'
      - Gemini in model_b AND winner == 'model_b'

    Everything else (including 'tie', 'both_bad', or the *other* model winning)
    is counted as 'Gemini did not win'.
    """
    winner = example["winner"]

    model_a_is_gemini = (
        re.search(GEMINI_NAME_PATTERN, example["model_a"], re.IGNORECASE) is not None
    )
    model_b_is_gemini = (
        re.search(GEMINI_NAME_PATTERN, example["model_b"], re.IGNORECASE) is not None
    )

    # If Gemini isn't in either slot → don't count
    if not (model_a_is_gemini or model_b_is_gemini):
        return False

    # Cases where Gemini *wins* → do NOT count
    if model_a_is_gemini and winner == "model_a":
        return False
    if model_b_is_gemini and winner == "model_b":
        return False

    # All other cases → Gemini did NOT win
    return True


def test_gemini_not_winner_count():
    ds = load_dataset("lmarena-ai/arena-expert-5k")
    train = ds["train"]

    gemini_cases = train.filter(has_gemini)
    gemini_not_win = gemini_cases.filter(gemini_not_winner)

    count = len(gemini_not_win)
    assert count == EXPECTED_GEMINI_NOT_WIN, (
        f"Expected {EXPECTED_GEMINI_NOT_WIN} Gemini-not-winner cases, got {count}"
    )


# python3 -m pytest -q
