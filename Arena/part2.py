# phase2_gemini3_eval.py

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
import pandas as pd
from anthropic import Anthropic
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# 1. ENV + CONFIG
# ============================================================

load_dotenv()

GOOGLE_API_KEY_1 = os.getenv("GOOGLE_API_KEY_1")  # Gemini autorater
GOOGLE_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2")  # Gemini 3.0 generator
GPT_API_KEY = os.getenv("GPT_API_KEY")
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")

GEMINI_NAME_PATTERN = r"gemini"

AUTORATER_PROMPT = """I'm going to give you a prompt with two responses titled conversation_a and conversation_b.
Tell me which response you prefer. Output a one word answer (model_a or model_b) on the first line.
Output a short 1 sentence justification for why you prefer that response on the second line.

Prompt:
{prompt}

conversation_a:
{answer_a}

conversation_b:
{answer_b}
"""

# ============================================================
# 2. Helpers
# ============================================================


def extract_user_prompt(conversation: Any) -> str:
    """
    Extract all user-turn texts from a structured conversation.
    If the conversation is already a string, just return it as-is.
    """
    # Some arena-expert-5k rows use plain strings instead of structured turns.
    if isinstance(conversation, str):
        return conversation.strip()

    texts: List[str] = []
    for turn in conversation:
        # Defensive: skip non-dict turns
        if not isinstance(turn, dict):
            continue
        if turn.get("role") == "user":
            for item in turn.get("content", []):
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item["text"]))
    return "\n".join(texts).strip()


def extract_last_assistant_text(conversation: Any) -> str:
    """
    Extract the last assistant-turn text from a structured conversation.
    If the conversation is already a string, just return it as-is
    (we can't distinguish roles in that case).
    """
    if isinstance(conversation, str):
        return conversation.strip()

    last = None
    for turn in conversation:
        if isinstance(turn, dict) and turn.get("role") == "assistant":
            last = turn

    if last is None:
        return ""

    texts: List[str] = []
    for item in last.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            texts.append(str(item["text"]))
    return "\n".join(texts).strip()


def has_gemini(example: Dict[str, Any]) -> bool:
    """
    Return True if either model_a or model_b name includes 'gemini'.
    """
    return re.search(
        GEMINI_NAME_PATTERN, example["model_a"], re.IGNORECASE
    ) or re.search(GEMINI_NAME_PATTERN, example["model_b"], re.IGNORECASE)


def gemini_lost_strict(example: Dict[str, Any]) -> bool:
    """
    Keep matchups where:
      - The human winner is explicitly model_a or model_b (no ties / unknown)
      - Exactly one of the models' names matches GEMINI_NAME_PATTERN
        (i.e., Gemini vs non-Gemini only)

    Excludes:
      - Gemini vs Gemini matchups
      - Ties or non-binary winners
    """
    winner = example["winner"]

    # Exclude ties / null / anything not model_a or model_b
    if winner not in ["model_a", "model_b"]:
        return False

    a_is_gem = bool(re.search(GEMINI_NAME_PATTERN, example["model_a"], re.IGNORECASE))
    b_is_gem = bool(re.search(GEMINI_NAME_PATTERN, example["model_b"], re.IGNORECASE))

    # Exclude Gemini vs Gemini matchups entirely
    if a_is_gem and b_is_gem:
        return False

    # Keep matchups where exactly ONE side is Gemini
    return a_is_gem != b_is_gem


# ============================================================
# 3. Autoraters + generation
#    (match phase1 autorater setup)
# ============================================================

# Gemini autorater: gemini-3-pro-preview
if GOOGLE_API_KEY_1:
    genai.configure(api_key=GOOGLE_API_KEY_1)
    GEM_AUTORATER_MODEL = genai.GenerativeModel("gemini-3-pro-preview")
else:
    GEM_AUTORATER_MODEL = None


def call_gemini_autorater(prompt: str) -> str:
    if GEM_AUTORATER_MODEL is None:
        raise RuntimeError(
            "GOOGLE_API_KEY_1 not set or GEM_AUTORATER_MODEL not initialized"
        )
    resp = GEM_AUTORATER_MODEL.generate_content(prompt)
    return resp.text


# Gemini 3.0 generator (conversation_c)
def call_gemini_3_generation(prompt: str) -> str:
    if not GOOGLE_API_KEY_2:
        raise RuntimeError("GOOGLE_API_KEY_2 not set")
    # Reconfigure with generation key
    genai.configure(api_key=GOOGLE_API_KEY_2)
    model = genai.GenerativeModel("gemini-3-pro-preview")
    resp = model.generate_content(prompt)
    return resp.text


# Claude autorater: claude-sonnet-4-20250514
def call_claude_autorater(prompt: str) -> str:
    if not CLAUDE_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=CLAUDE_API_KEY)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    first = resp.content[0]
    return first.text if hasattr(first, "text") else str(first)


# GPT autorater: gpt-5
def call_gpt5_autorater(prompt: str) -> str:
    if not GPT_API_KEY:
        raise RuntimeError("GPT_API_KEY not set")
    client = OpenAI(api_key=GPT_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def parse_choice_and_justification(text: str) -> Tuple[str, str]:
    """
    Parse autorater response into (choice, justification).
    choice in {"model_a", "model_b", "unknown"}.
    """
    if not text:
        return "unknown", ""

    lines = text.strip().split("\n")

    first_line = lines[0].strip().lower() if lines else ""
    choice = "unknown"

    first_word = first_line.split()[0] if first_line else ""
    if first_word in ["model_a", "model_b"]:
        choice = first_word
    elif "model_a" in first_line:
        choice = "model_a"
    elif "model_b" in first_line:
        choice = "model_b"

    if len(lines) > 1:
        justification = " ".join(lines[1:]).strip()
    elif choice == "unknown":
        justification = text.strip()
    else:
        justification = ""

    return choice, justification


def parse_choice(text: str) -> str:
    """
    Thin wrapper when we only care about the choice, not justification.
    """
    choice, _ = parse_choice_and_justification(text)
    return choice


def win_rate(df: pd.DataFrame, col: str) -> float:
    """
    Win rate of Gemini 3.0 (conversation_c) according to autorater 'col'.
    We define model_a = Gemini 3.0 in AUTORATER_PROMPT.
    """
    v = df[df[col].isin(["model_a", "model_b"])]
    if len(v) == 0:
        return 0.0
    return float((v[col] == "model_a").mean())


def autorater_win_stats(df: pd.DataFrame, col: str) -> Dict[str, float]:
    """
    Compute win stats for Gemini 3.0 (model_a) for a given autorater column.
    Returns dict with counts and rates.
    """
    valid = df[df[col].isin(["model_a", "model_b"])]
    total = len(valid)
    if total == 0:
        return {"total_judged": 0, "wins": 0, "win_rate": 0.0}
    wins = int((valid[col] == "model_a").sum())
    return {
        "total_judged": int(total),
        "wins": wins,
        "win_rate": float(wins / total),
    }


# ============================================================
# 4. MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", action="store_true", help="Run only first 10 examples"
    )
    args = parser.parse_args()

    print("Loading dataset...")
    ds = load_dataset("lmarena-ai/arena-expert-5k")
    train = ds["train"]

    print("Filtering Gemini vs non-Gemini matchups (no ties, no Gemini-vs-Gemini)...")
    gemini_cases = train.filter(has_gemini)
    gemini_lost = gemini_cases.filter(gemini_lost_strict)
    total_strict = len(gemini_lost)
    print(f"Total Gemini vs non-Gemini examples (binary human winner): {total_strict}")

    if args.test:
        limit = min(10, total_strict)
        print(f"TEST MODE: limiting to first {limit} examples")
        gemini_lost = gemini_lost.select(range(limit))

    print("Generating Gemini 3.0 (conversation_c) responses...")
    new_responses: List[str] = []
    for ex in tqdm(gemini_lost):
        prompt = extract_user_prompt(ex["conversation_a"])
        try:
            new_resp = call_gemini_3_generation(prompt)
        except Exception as e:
            new_resp = f"ERROR: {e}"
        new_responses.append(new_resp)

    print("Comparing Gemini 3.0 vs original non-Gemini opponent...")
    rows: List[Dict[str, Any]] = []

    for ex, gen3 in tqdm(list(zip(gemini_lost, new_responses))):
        a_is_gem = re.search(GEMINI_NAME_PATTERN, ex["model_a"], re.IGNORECASE)
        b_is_gem = re.search(GEMINI_NAME_PATTERN, ex["model_b"], re.IGNORECASE)

        # opponent = the original *non-Gemini* model response
        if a_is_gem:
            opponent = extract_last_assistant_text(ex["conversation_b"])
            original_gem_side = "model_a"
        else:
            opponent = extract_last_assistant_text(ex["conversation_a"])
            original_gem_side = "model_b"

        prompt = extract_user_prompt(ex["conversation_a"])

        # conversation_c is Gemini 3.0, always mapped to model_a in the autorater prompt
        payload = AUTORATER_PROMPT.format(
            prompt=prompt,
            answer_a=gen3,
            answer_b=opponent,
        )

        try:
            g_raw = call_gemini_autorater(payload)
        except Exception as e:
            g_raw = f"ERROR: {e}"

        try:
            c_raw = call_claude_autorater(payload)
        except Exception as e:
            c_raw = f"ERROR: {e}"

        try:
            p_raw = call_gpt5_autorater(payload)
        except Exception as e:
            p_raw = f"ERROR: {e}"

        # Parse choices + one-sentence justifications
        g_choice, g_just = parse_choice_and_justification(g_raw)
        c_choice, c_just = parse_choice_and_justification(c_raw)
        p_choice, p_just = parse_choice_and_justification(p_raw)

        rows.append(
            {
                "id": ex["id"],
                "human_winner_original": ex["winner"],
                "original_gemini_side": original_gem_side,
                "original_model_a_name": ex["model_a"],
                "original_model_b_name": ex["model_b"],
                "gemini3_response": gen3,
                "opponent_response": opponent,
                # autorater choices
                "gemini_autorater_pref": g_choice,
                "claude_autorater_pref": c_choice,
                "gpt5_autorater_pref": p_choice,
                # one-sentence justifications
                "gemini_autorater_justification": g_just,
                "claude_autorater_justification": c_just,
                "gpt5_autorater_justification": p_just,
            }
        )

    df2 = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Compute metrics: how well did new Gemini do vs original
    # overall, on original losses, and on original wins
    # --------------------------------------------------------
    original_gemini_won_mask = (
        df2["human_winner_original"] == df2["original_gemini_side"]
    )
    original_gemini_lost_mask = ~original_gemini_won_mask

    num_examples = int(len(df2))
    num_orig_wins = int(original_gemini_won_mask.sum())
    num_orig_losses = int(original_gemini_lost_mask.sum())

    original_overall_win_rate = (
        float(num_orig_wins / num_examples) if num_examples > 0 else 0.0
    )
    original_loss_win_rate = 0.0  # in the loss subset, Gemini always lost by definition
    original_win_win_rate = 1.0 if num_orig_wins > 0 else 0.0  # in the win subset

    df_overall = df2
    df_losses = df2[original_gemini_lost_mask]
    df_wins = df2[original_gemini_won_mask]

    metrics = {
        "overall": {
            "num_examples": num_examples,
            "original_gemini_human_wins": num_orig_wins,
            "original_gemini_human_losses": num_orig_losses,
            "original_gemini_human_win_rate": original_overall_win_rate,
            "autoraters": {
                "gemini": autorater_win_stats(df_overall, "gemini_autorater_pref"),
                "claude": autorater_win_stats(df_overall, "claude_autorater_pref"),
                "gpt5": autorater_win_stats(df_overall, "gpt5_autorater_pref"),
            },
        },
        "original_loss_subset": {
            "num_examples": int(len(df_losses)),
            "original_gemini_human_win_rate": original_loss_win_rate,
            "autoraters": {
                "gemini": autorater_win_stats(df_losses, "gemini_autorater_pref"),
                "claude": autorater_win_stats(df_losses, "claude_autorater_pref"),
                "gpt5": autorater_win_stats(df_losses, "gpt5_autorater_pref"),
            },
        },
        "original_win_subset": {
            "num_examples": int(len(df_wins)),
            "original_gemini_human_win_rate": original_win_win_rate,
            "autoraters": {
                "gemini": autorater_win_stats(df_wins, "gemini_autorater_pref"),
                "claude": autorater_win_stats(df_wins, "claude_autorater_pref"),
                "gpt5": autorater_win_stats(df_wins, "gpt5_autorater_pref"),
            },
        },
    }

    # --------------------------------------------------------
    # Simple console summary using win_rate()
    # --------------------------------------------------------
    print(
        "\nGemini 3.0 win rates on Gemini-vs-non-Gemini set "
        "(conversation_c vs original opponent):"
    )
    g_wr = win_rate(df2, "gemini_autorater_pref")
    c_wr = win_rate(df2, "claude_autorater_pref")
    p_wr = win_rate(df2, "gpt5_autorater_pref")

    print(f"Original Gemini human overall win rate: {original_overall_win_rate:.3f}")
    print(f"Gemini autorater Gemini-3.0 win rate: {g_wr:.3f}")
    print(f"Claude autorater Gemini-3.0 win rate: {c_wr:.3f}")
    print(f"GPT-5 autorater Gemini-3.0 win rate: {p_wr:.3f}")

    # --------------------------------------------------------
    # Save CSV + metrics JSON
    # --------------------------------------------------------
    out_name = (
        "phase2_gemini3_vs_opponent_strict_TEST.csv"
        if args.test
        else "phase2_gemini3_vs_opponent_strict.csv"
    )
    df2.to_csv(out_name, index=False)
    print(f"\nSaved: {out_name}")

    metrics_file = (
        "phase2_gemini3_metrics_TEST.json"
        if args.test
        else "phase2_gemini3_metrics.json"
    )
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON: {metrics_file}")


if __name__ == "__main__":
    main()
