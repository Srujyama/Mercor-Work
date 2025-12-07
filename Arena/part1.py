import argparse
import json
import os
import re
from typing import Any, Dict, List

# Gemini
import google.generativeai as genai
import pandas as pd

# Claude (Anthropic)
from anthropic import Anthropic
from datasets import load_dataset
from dotenv import load_dotenv

# GPT (OpenAI)
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

GOOGLE_API_KEY_1 = os.getenv("GOOGLE_API_KEY_1")
GPT_API_KEY = os.getenv("GPT_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

GEMINI_NAME_PATTERN = r"gemini"

AUTORATER_PROMPT = """I'm going to give you a prompt with two responses titled conversation_a and conversation_b.
Tell me which response you prefer. Output a one word answer (model_a or model_b) on the first line.
Output a short 1 sentence justification for why you prefer that response on the second line.

conversation_a:
{answer_a}

conversation_b:
{answer_b}
"""

# ============================================================
# Normalize / stringify conversations
# ============================================================


def _normalize_conversation(conv: Any) -> List[Dict[str, Any]]:
    if conv is None:
        return []
    if isinstance(conv, (list, tuple)):
        return list(conv)
    try:
        return list(conv)
    except Exception:
        return []


def conversation_to_text(conv: Any) -> str:
    if isinstance(conv, str):
        return conv.strip()

    conv = _normalize_conversation(conv)
    out: List[str] = []

    for turn in conv:
        if isinstance(turn, dict):
            role = turn.get("role", "unknown")
            out.append(f"[{role}]")

            content = turn.get("content", [])
            try:
                content = list(content)
            except Exception:
                content = []

            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    out.append(str(item.get("text", "")))

            out.append("")  # blank line between turns

    return "\n".join(out).strip()


# ============================================================
# Filters to get "Gemini lost" cases
# ============================================================


def has_gemini(example):
    return re.search(
        GEMINI_NAME_PATTERN, example["model_a"], re.IGNORECASE
    ) or re.search(GEMINI_NAME_PATTERN, example["model_b"], re.IGNORECASE)


def gemini_lost_strict(example):
    winner = example["winner"]
    if winner not in ["model_a", "model_b"]:
        return False

    a_is_gem = bool(re.search(GEMINI_NAME_PATTERN, example["model_a"], re.IGNORECASE))
    b_is_gem = bool(re.search(GEMINI_NAME_PATTERN, example["model_b"], re.IGNORECASE))

    # Exclude Gemini vs Gemini head-to-head matchups
    if a_is_gem and b_is_gem:
        return False

    # Gemini lost: it’s Gemini on one side only, and the *other* side won
    return (a_is_gem and winner == "model_b") or (b_is_gem and winner == "model_a")


# ============================================================
# Autorater calls
# ============================================================

# Gemini
if GOOGLE_API_KEY_1:
    genai.configure(api_key=GOOGLE_API_KEY_1)
    GEM_MODEL = genai.GenerativeModel("gemini-3-pro-preview")
else:
    GEM_MODEL = None


def call_gemini(prompt: str) -> str:
    if GEM_MODEL is None:
        raise RuntimeError("GOOGLE_API_KEY_1 not set or GEM_MODEL not initialized")
    resp = GEM_MODEL.generate_content(prompt)
    return resp.text


# Claude (Sonnet 4)
def call_claude(prompt: str) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",  # Claude Sonnet 4
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    # resp.content is a list of TextBlock / content blocks; extract the first text block
    # The SDK returns a .content attribute holding blocks; the first block's .text is the reply
    first = resp.content[0]
    return first.text if hasattr(first, "text") else str(first)


# GPT (OpenAI)
def call_gpt(prompt: str) -> str:
    if not GPT_API_KEY:
        raise RuntimeError("GPT_API_KEY not set")
    client = OpenAI(api_key=GPT_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-5",  # GPT-5
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ============================================================
# Parse autorater result into model_a/model_b/unknown + justification
# ============================================================


def parse_choice_and_justification(text: str) -> tuple[str, str]:
    """
    Parse the autorater response to extract choice and justification.
    Returns: (choice, justification)
    """
    if not text:
        return "unknown", ""

    lines = text.strip().split("\n")

    # Get the first line for choice
    first_line = lines[0].strip().lower() if lines else ""
    choice = "unknown"

    # Check if first word is model_a or model_b
    first_word = first_line.split()[0] if first_line else ""
    if first_word in ["model_a", "model_b"]:
        choice = first_word
    elif "model_a" in first_line:
        choice = "model_a"
    elif "model_b" in first_line:
        choice = "model_b"

    # Get justification (second line or remainder of text)
    justification = ""
    if len(lines) > 1:
        justification = " ".join(lines[1:]).strip()
    elif choice == "unknown":
        # If we couldn't parse a clear choice, keep the whole text as justification
        justification = text.strip()

    return choice, justification


# ============================================================
# Calculate alignment with human preferences
# ============================================================


def calculate_alignment(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the percentage agreement between each autorater and human preferences.
    """
    total = len(df)

    # Only consider cases where both autorater and human made a clear choice
    gemini_agreements = sum(
        (df["gemini_pref"] == df["human_winner"])
        & (df["gemini_pref"].isin(["model_a", "model_b"]))
        & (df["human_winner"].isin(["model_a", "model_b"]))
    )
    gemini_valid = sum(
        (df["gemini_pref"].isin(["model_a", "model_b"]))
        & (df["human_winner"].isin(["model_a", "model_b"]))
    )

    claude_agreements = sum(
        (df["claude_pref"] == df["human_winner"])
        & (df["claude_pref"].isin(["model_a", "model_b"]))
        & (df["human_winner"].isin(["model_a", "model_b"]))
    )
    claude_valid = sum(
        (df["claude_pref"].isin(["model_a", "model_b"]))
        & (df["human_winner"].isin(["model_a", "model_b"]))
    )

    gpt_agreements = sum(
        (df["gpt_pref"] == df["human_winner"])
        & (df["gpt_pref"].isin(["model_a", "model_b"]))
        & (df["human_winner"].isin(["model_a", "model_b"]))
    )
    gpt_valid = sum(
        (df["gpt_pref"].isin(["model_a", "model_b"]))
        & (df["human_winner"].isin(["model_a", "model_b"]))
    )

    return {
        "gemini_alignment": (gemini_agreements / gemini_valid * 100)
        if gemini_valid > 0
        else 0,
        "claude_alignment": (claude_agreements / claude_valid * 100)
        if claude_valid > 0
        else 0,
        "gpt_alignment": (gpt_agreements / gpt_valid * 100) if gpt_valid > 0 else 0,
        "gemini_valid_count": gemini_valid,
        "claude_valid_count": claude_valid,
        "gpt_valid_count": gpt_valid,
    }


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run only 10 examples")
    args = parser.parse_args()

    print("Loading dataset...")
    ds = load_dataset("lmarena-ai/arena-expert-5k")["train"]

    print("Filtering...")
    gem_cases = ds.filter(has_gemini)
    lost = gem_cases.filter(gemini_lost_strict)

    total = len(lost)
    print(f"Count: {total}")

    if args.test:
        print("TEST MODE: limiting to first 10 examples")
        lost = lost.select(range(min(10, len(lost))))

    rows = []
    log_path = "phase1_autorater_debug_log.jsonl"
    log_f = open(log_path, "w", encoding="utf-8")
    print(f"Logging payloads to {log_path}")

    for idx, ex in enumerate(tqdm(lost)):
        ex_id = ex["id"]
        a_conv = conversation_to_text(ex["conversation_a"])
        b_conv = conversation_to_text(ex["conversation_b"])

        payload = AUTORATER_PROMPT.format(
            answer_a=a_conv,
            answer_b=b_conv,
        )

        if idx < 3:
            print("\n=== DEBUG EXAMPLE", idx, "ID:", ex_id, "===")
            print(payload[:1200])

        try:
            gem_raw = call_gemini(payload)
        except Exception as e:
            gem_raw = f"ERROR: {e}"

        try:
            cla_raw = call_claude(payload)
        except Exception as e:
            cla_raw = f"ERROR: {e}"

        try:
            gpt_raw = call_gpt(payload)
        except Exception as e:
            gpt_raw = f"ERROR: {e}"

        gem_choice, gem_just = parse_choice_and_justification(gem_raw)
        cla_choice, cla_just = parse_choice_and_justification(cla_raw)
        gpt_choice, gpt_just = parse_choice_and_justification(gpt_raw)

        log_f.write(
            json.dumps(
                {
                    "id": ex_id,
                    "human_winner": ex["winner"],
                    "payload": payload,
                    "gemini_raw": gem_raw,
                    "claude_raw": cla_raw,
                    "gpt_raw": gpt_raw,
                    "gemini_choice": gem_choice,
                    "gemini_justification": gem_just,
                    "claude_choice": cla_choice,
                    "claude_justification": cla_just,
                    "gpt_choice": gpt_choice,
                    "gpt_justification": gpt_just,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        rows.append(
            {
                "id": ex_id,
                "human_winner": ex["winner"],
                "model_a_name": ex["model_a"],
                "model_b_name": ex["model_b"],
                "gemini_pref": gem_choice,
                "gemini_justification": gem_just,
                "claude_pref": cla_choice,
                "claude_justification": cla_just,
                "gpt_pref": gpt_choice,
                "gpt_justification": gpt_just,
            }
        )

    log_f.close()
    print(f"Saved log to {log_path}")

    df = pd.DataFrame(rows)
    out_csv = (
        "phase1_full_conversation_autorater_results_TEST.csv"
        if args.test
        else "phase1_full_conversation_autorater_results.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    # Calculate and display alignment statistics
    print("\n" + "=" * 60)
    print("ALIGNMENT WITH HUMAN PREFERENCES")
    print("=" * 60)

    alignment_stats = calculate_alignment(df)

    print(
        f"\nGemini Alignment: {alignment_stats['gemini_alignment']:.2f}% "
        f"({alignment_stats['gemini_valid_count']} valid comparisons)"
    )
    print(
        f"Claude Alignment: {alignment_stats['claude_alignment']:.2f}% "
        f"({alignment_stats['claude_valid_count']} valid comparisons)"
    )
    print(
        f"GPT Alignment: {alignment_stats['gpt_alignment']:.2f}% "
        f"({alignment_stats['gpt_valid_count']} valid comparisons)"
    )

    # Save alignment stats to a separate file
    alignment_file = (
        "phase1_alignment_stats_TEST.json"
        if args.test
        else "phase1_alignment_stats.json"
    )
    with open(alignment_file, "w") as f:
        json.dump(alignment_stats, f, indent=2)
    print(f"\nSaved alignment statistics to {alignment_file}")

    # Determine best aligned model
    best_model = max(
        [
            ("Gemini", alignment_stats["gemini_alignment"]),
            ("Claude", alignment_stats["claude_alignment"]),
            ("GPT", alignment_stats["gpt_alignment"]),
        ],
        key=lambda x: x[1],
    )
    print(
        f"\nBest aligned autorater: {best_model[0]} with {best_model[1]:.2f}% agreement"
    )


if __name__ == "__main__":
    main()
