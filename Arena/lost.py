import re

from datasets import load_dataset

GEMINI_NAME_PATTERN = r"gemini"  # same as in hug.py
EXPECTED_GEMINI_LOST = 240


def has_gemini(example):
    return (
        re.search(GEMINI_NAME_PATTERN, example["model_a"], re.IGNORECASE) is not None
        or re.search(GEMINI_NAME_PATTERN, example["model_b"], re.IGNORECASE) is not None
    )


def gemini_lost(example):
    winner = example["winner"]
    if winner not in ["model_a", "model_b"]:
        return False

    model_a_is_gemini = (
        re.search(GEMINI_NAME_PATTERN, example["model_a"], re.IGNORECASE) is not None
    )
    model_b_is_gemini = (
        re.search(GEMINI_NAME_PATTERN, example["model_b"], re.IGNORECASE) is not None
    )

    if model_a_is_gemini and winner == "model_b":
        return True
    if model_b_is_gemini and winner == "model_a":
        return True

    return False


def test_gemini_lost_count():
    ds = load_dataset("lmarena-ai/arena-expert-5k")
    train = ds["train"]

    gemini_cases = train.filter(has_gemini)
    gemini_lost_cases = gemini_cases.filter(gemini_lost)

    assert len(gemini_lost_cases) == EXPECTED_GEMINI_LOST, (
        f"Expected {EXPECTED_GEMINI_LOST} Gemini-lost cases, "
        f"but got {len(gemini_lost_cases)}"
    )


# python3 -m pytest -q
