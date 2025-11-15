#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# ===============================
# Config
# ===============================
PROMPT_COL = "Prompt"
COL_OPENAI = "GPT 5 - Reasoning (High)"

# ===============================
# Setup
# ===============================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
if not OPENAI_API_KEY:
    raise SystemExit("Missing required env var: OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


class TransientError(Exception):
    pass


def _validate_model(c: OpenAI, model: str):
    """Ping the model once to verify access."""
    try:
        c.responses.create(model=model, input="ping", max_output_tokens=16)
    except Exception as e:
        raise SystemExit(
            f"OPENAI_MODEL='{model}' is not available to your API key.\n"
            f"Try 'gpt-5' or another allowed model.\n"
            f"Original error: {e}"
        )


_validate_model(client, OPENAI_MODEL)


# ===============================
# Extract Text Utility
# ===============================
def _extract_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return text
    collected = []
    if hasattr(resp, "output") and resp.output:
        for item in resp.output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    t = getattr(part, "text", None)
                    if t:
                        collected.append(t)
    if collected:
        return "\n".join(collected).strip()
    for attr in ("response", "data", "content"):
        node = getattr(resp, attr, None)
        if isinstance(node, list):
            for n in node:
                t = getattr(n, "text", None)
                if t:
                    collected.append(t)
        elif hasattr(node, "text"):
            t = getattr(node, "text", None)
            if t:
                collected.append(t)
    return "\n".join(collected).strip() if collected else ""


# ===============================
# Call OpenAI + Continue if Truncated
# ===============================
@retry(
    wait=wait_exponential(multiplier=1, min=1, max=16),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(TransientError),
)
def call_openai(
    prompt: str,
    max_tokens: int,
    row_idx: int | None = None,
    max_continuations: int = 3,
    debug_dump: bool = True,
) -> str:
    combined = []
    for seg in range(max_continuations + 1):
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt if seg == 0 else "Continue.",
                reasoning={"effort": "high"},
                max_output_tokens=max_tokens,
            )
        except Exception as e:
            raise TransientError(str(e))

        part = _extract_text(resp)
        if part:
            combined.append(part)

        status = getattr(resp, "status", None)
        reason = None
        if hasattr(resp, "incomplete_details") and resp.incomplete_details:
            reason = getattr(resp.incomplete_details, "reason", None)
        if status != "incomplete" or reason != "max_output_tokens":
            break  # Done or truncated for another reason

    final_text = "\n".join([t for t in combined if t]).strip()

    if not final_text and debug_dump:
        try:
            raw = (
                resp.model_dump()
                if hasattr(resp, "model_dump")
                else getattr(resp, "__dict__", {})
            )
            Path("openai_debug").mkdir(exist_ok=True)
            fname = (
                Path("openai_debug")
                / f"row_{row_idx if row_idx is not None else 'unknown'}.json"
            )
            with open(fname, "w") as f:
                json.dump(raw, f, indent=2, default=str)
            return f"(no text found; raw response dumped to {fname})"
        except Exception:
            return ""
    return final_text


# ===============================
# CSV Processing
# ===============================
def process_csv(
    input_csv: str,
    output_csv: str,
    start_row: int,
    max_rows: int | None,
    overwrite: bool,
    openai_max_tokens: int,
    continuations: int,
    dry_run: bool,
):
    df = pd.read_csv(input_csv, dtype={PROMPT_COL: "string"})

    if PROMPT_COL not in df.columns:
        raise SystemExit(f"Missing required column in CSV: {PROMPT_COL}")

    if COL_OPENAI not in df.columns:
        print(f"[init] Creating missing column: {COL_OPENAI}")
        df[COL_OPENAI] = pd.Series([None] * len(df), dtype="string")
    else:
        df[COL_OPENAI] = df[COL_OPENAI].astype("string")

    total = len(df)
    begin = max(0, start_row)
    end = total if max_rows is None else min(total, begin + max_rows)
    print(f"[plan] Processing rows {begin}..{end - 1} (count={end - begin})")
    print(
        f"[plan] Writing to column: '{COL_OPENAI}'  | overwrite={overwrite} | tokens={openai_max_tokens}"
    )

    wrote_any = False

    for idx in tqdm(range(begin, end), desc="Processing rows"):
        prompt_val = df.at[idx, PROMPT_COL]
        prompt = prompt_val if pd.notna(prompt_val) else ""
        prompt = str(prompt).strip()
        if not prompt:
            continue

        existing = df.at[idx, COL_OPENAI]
        exists = pd.notna(existing) and str(existing).strip() != ""
        if exists and not overwrite:
            continue

        try:
            out = call_openai(
                prompt=prompt,
                max_tokens=openai_max_tokens,
                row_idx=idx,
                max_continuations=continuations,
                debug_dump=True,
            )
        except Exception as e:
            print(f"\n[row {idx}] OpenAI error: {e}")
            continue

        df.at[idx, COL_OPENAI] = str(out)
        wrote_any = True
        print(f"\n[row {idx}] wrote {len(out)} chars to '{COL_OPENAI}'")

    if dry_run:
        print("[done] Dry run — not writing file.")
    else:
        df.to_csv(output_csv, index=False)
        print(f"[done] Wrote: {output_csv}")
        if not wrote_any:
            print(
                "[note] No rows were updated. Use --no-overwrite false (default overwrites), check prompts, or increase tokens/continuations."
            )


# ===============================
# CLI
# ===============================
def main():
    ap = argparse.ArgumentParser(
        description="Write OpenAI output into 'GPT 5 - Reasoning (High)' without system nudge."
    )
    ap.add_argument("input_csv", help="Path to input CSV")
    ap.add_argument(
        "-o", "--output-csv", default=None, help="Output CSV (default: <input>.out.csv)"
    )
    ap.add_argument(
        "--start-row", type=int, default=0, help="0-based starting row index"
    )
    ap.add_argument(
        "--max-rows", type=int, default=None, help="Process only N rows from start-row"
    )
    ap.add_argument(
        "--no-overwrite", action="store_true", help="Do not overwrite existing values"
    )
    ap.add_argument(
        "--openai-max-tokens",
        type=int,
        default=16384,
        help="Max output tokens per chunk",
    )
    ap.add_argument(
        "--continuations",
        type=int,
        default=3,
        help="Auto-continue up to N extra chunks on token cutoff",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Don’t write the file; just show what would happen",
    )
    args = ap.parse_args()

    output_csv = args.output_csv or args.input_csv.replace(".csv", ".out.csv")
    process_csv(
        input_csv=args.input_csv,
        output_csv=output_csv,
        start_row=args.start_row,
        max_rows=args.max_rows,
        overwrite=not args.no_overwrite,
        openai_max_tokens=args.openai_max_tokens,
        continuations=args.continuations,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
