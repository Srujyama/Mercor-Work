# phase2_gemini3_eval.py

import os
import re
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

import google.generativeai as genai
from openai import OpenAI
import anthropic


# ============================================================
# 1. ENV + CONFIG
# ============================================================

load_dotenv()

GOOGLE_API_KEY_1 = os.getenv("GOOGLE_API_KEY_1")  # Gemini autorater
GOOGLE_API_KEY_2 = os.getenv("GOOGLE_API_KEY_2")  # Gemini 3.0 generator
GPT_API_KEY      = os.getenv("GPT_API_KEY")
CLAUDE_API_KEY   = os.getenv("CLAUDE_API_KEY")

GEMINI_NAME_PATTERN = r"gemini"

AUTORATER_PROMPT = """I'm going to give you a prompt with two responses titled conversation_a and conversation_b. Tell me which response you prefer. Output a one word answer (model_a or model_b). Output a short 1 sentence justification for why you prefer that response.

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

def extract_user_prompt(conversation):
    texts = []
    for turn in conversation:
        if turn.get("role") == "user":
            for item in turn.get("content", []):
                if item.get("type") == "text":
                    texts.append(item["text"])
    return "\n".join(texts)


def extract_last_assistant_text(conversation):
    last = None
    for turn in conversation:
        if turn.get("role") == "assistant":
            last = turn
    if last is None:
        return ""
    texts = []
    for item in last.get("content", []):
        if item.get("type") == "text":
            texts.append(item["text"])
    return "\n".join(texts)


def has_gemini(example):
    return (
        re.search(GEMINI_NAME_PATTERN, example["model_a"], re.IGNORECASE) or
        re.search(GEMINI_NAME_PATTERN, example["model_b"], re.IGNORECASE)
    )


def gemini_lost_strict(example):
    winner = example["winner"]
    if winner not in ["model_a", "model_b"]:
        return False
    a_is = re.search(GEMINI_NAME_PATTERN, example["model_a"], re.IGNORECASE)
    b_is = re.search(GEMINI_NAME_PATTERN, example["model_b"], re.IGNORECASE)
    if a_is and winner == "model_b":
        return True
    if b_is and winner == "model_a":
        return True
    return False


# ============================================================
# 3. Autoraters + generation
# ============================================================

def call_gemini_autorater(prompt: str) -> str:
    genai.configure(api_key=GOOGLE_API_KEY_1)
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model.generate_content(prompt).text


def call_gemini_3_generation(prompt: str) -> str:
    genai.configure(api_key=GOOGLE_API_KEY_2)
    model = genai.GenerativeModel("gemini-3.0-pro")
    return model.generate_content(prompt).text


def call_claude_autorater(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    resp = client.messages.create(
        model="claude-4.5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in resp.content if hasattr(block, "text")
    )


def call_gpt5_autorater(prompt: str) -> str:
    client = OpenAI(api_key=GPT_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def parse_choice(text: str) -> str:
    text = text.strip().lower()
    if text.startswith("model_a"):
        return "model_a"
    if text.startswith("model_b"):
        return "model_b"
    if "model_a" in text:
        return "model_a"
    if "model_b" in text:
        return "model_b"
    return "unknown"


def win_rate(df, col):
    v = df[df[col].isin(["model_a", "model_b"])]
    return (v[col] == "model_a").mean()


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("Loading dataset...")
    ds = load_dataset("lmarena-ai/arena-expert-5k")
    train = ds["train"]

    print("Filtering Gemini strict losses (180)...")
    gemini_cases = train.filter(has_gemini)
    gemini_lost = gemini_cases.filter(gemini_lost_strict)
    assert len(gemini_lost) == 180

    print("Generating Gemini 3.0 responses...")
    new_responses = []
    for ex in tqdm(gemini_lost):
        prompt = extract_user_prompt(ex["conversation_a"])
        try:
            new_resp = call_gemini_3_generation(prompt)
        except Exception as e:
            new_resp = f"error: {e}"
        new_responses.append(new_resp)

    print("Comparing Gemini 3.0 vs opponent...")
    rows = []

    for ex, gen3 in tqdm(list(zip(gemini_lost, new_responses))):
        a_is = re.search(GEMINI_NAME_PATTERN, ex["model_a"], re.IGNORECASE)
        b_is = re.search(GEMINI_NAME_PATTERN, ex["model_b"], re.IGNORECASE)

        if a_is:
            opponent = extract_last_assistant_text(ex["conversation_b"])
        else:
            opponent = extract_last_assistant_text(ex["conversation_a"])

        prompt = extract_user_prompt(ex["conversation_a"])

        payload = AUTORATER_PROMPT.format(
            prompt=prompt,
            answer_a=gen3,
            answer_b=opponent,
        )

        try: g_raw = call_gemini_autorater(payload)
        except Exception as e: g_raw = f"model_a error {e}"

        try: c_raw = call_claude_autorater(payload)
        except Exception as e: c_raw = f"model_b error {e}"

        try: p_raw = call_gpt5_autorater(payload)
        except Exception as e: p_raw = f"model_a error {e}"

        rows.append({
            "id": ex["id"],
            "gemini3": gen3,
            "opponent": opponent,
            "gemini_autorater_pref": parse_choice(g_raw),
            "claude_autorater_pref": parse_choice(c_raw),
            "gpt5_autorater_pref": parse_choice(p_raw),
        })

    df2 = pd.DataFrame(rows)

    print("\nGemini 3.0 win rates:")
    print("Gemini autorater:", win_rate(df2, "gemini_autorater_pref"))
    print("Claude autorater:", win_rate(df2, "claude_autorater_pref"))
    print("GPT-5 autorater:", win_rate(df2, "gpt5_autorater_pref"))

    df2.to_csv("phase2_gemini3_vs_opponent_strict.csv", index=False)
    print("Saved: phase2_gemini3_vs_opponent_strict.csv")


if __name__ == "__main__":
    main()
