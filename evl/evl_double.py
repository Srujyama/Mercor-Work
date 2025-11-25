#!/usr/bin/env python3
# pip install --upgrade google-genai python-dotenv requests tqdm

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

# ----------------------------
# Environment & Constants
# ----------------------------
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")
AIRTABLE_VIEW_EVALSET = os.getenv("AIRTABLE_VIEW_EVALSET")
AIRTABLE_VIEW_TRAINING = os.getenv("AIRTABLE_VIEW_TRAINING")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Output fields
EVAL_OUTPUT_FIELD = "Gemini 3.0 model responses - Eval"
TRAINING_OUTPUT_FIELD = "Gemini 3.0 model responses - Tasks"

# Relevant Airtable fields
PROMPT_FIELD = "Consolidated Prompt - 10/25"
REQUESTED_OUTPUTS_FIELD = "Requested Outputs"

AIRTABLE_API = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}"
AIRTABLE_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json",
}


# ----------------------------
# Airtable helpers
# ----------------------------
def airtable_list_records(view_id: str, page_size: int = 100) -> List[Dict[str, Any]]:
    records = []
    params = {"pageSize": page_size, "view": view_id}
    while True:
        resp = requests.get(
            AIRTABLE_API, headers=AIRTABLE_HEADERS, params=params, timeout=60
        )
        if resp.status_code == 429:
            time.sleep(1.5)
            continue
        resp.raise_for_status()
        data = resp.json()
        records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break
        params["offset"] = offset
        time.sleep(0.2)
    return records


def airtable_update_record(
    record_id: str, fields: Dict[str, Any], max_retries: int = 5
) -> None:
    """PATCH update a single record with retries on 429/5xx."""
    url = f"{AIRTABLE_API}/{record_id}"
    payload = {"fields": fields}
    for attempt in range(1, max_retries + 1):
        resp = requests.patch(
            url, headers=AIRTABLE_HEADERS, data=json.dumps(payload), timeout=60
        )
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 * attempt, 8))
            continue
        resp.raise_for_status()
        return


# ----------------------------
# Skip logic and prompt helpers
# ----------------------------
def get_prompt(fields: Dict[str, Any]) -> Optional[str]:
    """Return ONLY the 'Consolidated Prompt - 10/25' value (or None if missing/blank)."""
    v = fields.get(PROMPT_FIELD)
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def should_skip_by_output(fields: Dict[str, Any]) -> bool:
    """
    Skip if 'Requested Outputs' contains 'image' or 'file' (case-insensitive).
    Supports string, list, or Airtable multi-select dicts.
    """
    v = fields.get(REQUESTED_OUTPUTS_FIELD)
    if v is None:
        return False

    def contains_disallowed(s: str) -> bool:
        s = s.lower()
        return ("image" in s) or ("file" in s)

    if isinstance(v, str):
        return contains_disallowed(v)

    if isinstance(v, list):
        for item in v:
            if isinstance(item, str) and contains_disallowed(item):
                return True
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and contains_disallowed(name):
                    return True
        return False

    if isinstance(v, dict):
        name = v.get("name")
        if isinstance(name, str) and contains_disallowed(name):
            return True
        for vv in v.values():
            if isinstance(vv, str) and contains_disallowed(vv):
                return True

    return False


# ----------------------------
# Gemini helpers
# ----------------------------
def build_genai_client(api_key: str, api_version: str = "v1beta") -> genai.Client:
    return genai.Client(
        api_key=api_key, http_options=types.HttpOptions(api_version=api_version)
    )


def call_gemini(
    client: genai.Client,
    model_name: str,
    prompt: str,
    system_instructions: Optional[str] = None,
    max_retries: int = 3,
) -> Tuple[str, Optional[str]]:
    """Call Gemini and return (text, served_model)."""
    guard_rails = (
        "Important: Produce a plain text answer only. "
        "Do not include or reference images. "
        "If the task requested images or files, ignore that and provide a text-only alternative."
    )
    sys_part = f"{system_instructions}\n\n" if system_instructions else ""
    composed = f"{guard_rails}\n\n{sys_part}{prompt}"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(model=model_name, contents=composed)
            text = getattr(resp, "text", None)
            if not text:
                cands = getattr(resp, "candidates", [])
                if cands and getattr(cands[0], "content", None):
                    parts = cands[0].content.parts or []
                    text = "\n".join(
                        [
                            getattr(p, "text", "")
                            for p in parts
                            if getattr(p, "text", "")
                        ]
                    )
            text = (text or "").strip()
            served_model = getattr(resp, "model", None)
            return text, served_model
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


# ----------------------------
# Simple token bucket limiter for Airtable writes
# ----------------------------
class RateLimiter:
    def __init__(self, rate_per_sec: float, capacity: Optional[int] = None):
        self.rate = float(rate_per_sec)
        self.capacity = int(capacity or max(1, int(rate_per_sec)))
        self.tokens = self.capacity
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                refill = (now - self.last) * self.rate
                if refill > 0:
                    self.tokens = min(self.capacity, self.tokens + refill)
                    self.last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time.sleep(0.01)


# ----------------------------
# Core processing
# ----------------------------
def process_view(
    client: genai.Client,
    model_name: str,
    view_id: str,
    output_field: str,
    overwrite: bool = False,
    dry_run: bool = False,
    limit: Optional[int] = None,
    workers: int = 6,
    airtable_rps: float = 4.0,
) -> Dict[str, int]:
    records = airtable_list_records(view_id)
    if limit is not None:
        records = records[:limit]
    total = len(records)
    print(f"Total records in this view: {total}")

    jobs = []  # (rec_id, prompt)
    already_filled = skipped_no_prompt = skipped_output_tasks = 0

    for rec in records:
        rec_id = rec.get("id")
        fields = rec.get("fields", {})

        if should_skip_by_output(fields):
            skipped_output_tasks += 1
            continue

        existing = fields.get(output_field)
        if existing and not overwrite:
            already_filled += 1
            continue

        prompt = get_prompt(fields)
        if not prompt:
            skipped_no_prompt += 1
            continue

        jobs.append((rec_id, prompt))

    if dry_run:
        print(f"[DRY RUN] Would generate for {len(jobs)} tasks.")
        return {
            "processed": len(jobs),
            "skipped_no_prompt": skipped_no_prompt,
            "skipped_output_tasks": skipped_output_tasks,
            "already_filled": already_filled,
            "errors": 0,
            "total_seen": total,
        }

    gen_results: Dict[str, Tuple[str, Optional[str], Optional[Exception]]] = {}
    errors = 0

    def _gen_task(rec_id: str, prompt: str) -> Tuple[str, str, Optional[str]]:
        text, served = call_gemini(client, model_name, prompt)
        return rec_id, text, served

    # Parallel Gemini calls
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_gen_task, rid, pr) for rid, pr in jobs]
        with tqdm(
            total=len(futures), desc=f"Generating ({view_id})", unit="task"
        ) as pbar:
            for fut in as_completed(futures):
                try:
                    rec_id, text, served = fut.result()
                    gen_results[rec_id] = (text, served, None)
                    pbar.set_postfix_str(f"✅ {rec_id[:6]} gen")
                except Exception as e:
                    gen_results[f"unknown-{len(gen_results)}"] = ("", None, e)
                    errors += 1
                    pbar.set_postfix_str("❌ gen err")
                finally:
                    pbar.update(1)

    # Airtable writes (rate-limited)
    limiter = RateLimiter(rate_per_sec=float(airtable_rps))
    processed = write_errors = 0

    with tqdm(
        total=len(gen_results), desc=f"Writing ({view_id})", unit="row"
    ) as pbar_w:
        for rec_id, (text, served, err) in gen_results.items():
            if err is not None:
                write_errors += 1
                pbar_w.set_postfix_str(f"❌ {rec_id[:6]} gen_err")
                pbar_w.update(1)
                continue
            try:
                limiter.acquire()
                airtable_update_record(rec_id, {output_field: text})
                processed += 1
                tag = f"{served}" if served else "ok"
                pbar_w.set_postfix_str(f"⬆️ {rec_id[:6]} {tag}")
            except Exception:
                write_errors += 1
                pbar_w.set_postfix_str(f"❌ {rec_id[:6]} write_err")
            finally:
                pbar_w.update(1)

    errors += write_errors
    return {
        "processed": processed,
        "skipped_no_prompt": skipped_no_prompt,
        "skipped_output_tasks": skipped_output_tasks,
        "already_filled": already_filled,
        "errors": errors,
        "total_seen": total,
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Parallel Gemini 3.0 Airtable generator."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing responses."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Skip API calls and Airtable writes."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit rows per view (for testing)."
    )
    parser.add_argument(
        "--only", choices=["eval", "training"], help="Process only one view."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GEMINI_MODEL", "gemini-3-pro-preview"),
        help="Gemini model to use (default gemini-3-pro-preview).",
    )
    parser.add_argument(
        "--api-version",
        type=str,
        default=os.getenv("GENAI_API_VERSION", "v1beta"),
        help="GenAI API version (default v1beta).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("EVL_WORKERS", "6")),
        help="Parallel Gemini calls (default 6).",
    )
    parser.add_argument(
        "--airtable-rps",
        type=float,
        default=float(os.getenv("AIRTABLE_RPS", "4")),
        help="Max Airtable writes/sec (default 4).",
    )
    args = parser.parse_args()

    if not GOOGLE_API_KEY:
        raise SystemExit("Missing GOOGLE_API_KEY or GEMINI_API_KEY.")

    client = build_genai_client(GOOGLE_API_KEY, api_version=args.api_version)
    summaries: Dict[str, Dict[str, int]] = {}

    if args.only in (None, "eval"):
        print("\n=== Processing EVALSET view ===")
        summaries["eval"] = process_view(
            client,
            args.model,
            AIRTABLE_VIEW_EVALSET,
            EVAL_OUTPUT_FIELD,
            args.overwrite,
            args.dry_run,
            args.limit,
            args.workers,
            args.airtable_rps,
        )
        print(json.dumps(summaries["eval"], indent=2))

    if args.only in (None, "training"):
        print("\n=== Processing TRAINING view ===")
        summaries["training"] = process_view(
            client,
            args.model,
            AIRTABLE_VIEW_TRAINING,
            TRAINING_OUTPUT_FIELD,
            args.overwrite,
            args.dry_run,
            args.limit,
            args.workers,
            args.airtable_rps,
        )
        print(json.dumps(summaries["training"], indent=2))

    print("\n=== Done ===")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
