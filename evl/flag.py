#!/usr/bin/env python3
# pip install python-dotenv requests tqdm

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ------------ ENV ------------
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")

# New general view env var from your .env
AIRTABLE_VIEW_GENERAL = os.getenv("AIRTABLE_VIEW_General")

AIRTABLE_API_ROOT = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json",
}

# ------------ CONFIG ------------
# Single Gemini autorater score field used for flagging
GEMINI_SCORE_FIELD = "GPT5 Autorater - Gemini 3.0 Response Score"

LOW_SCORING_CHECKBOX = "Low-Scoring Tasks"  # existing checkbox column
THRESHOLD = 20.0


# ------------ Airtable helpers ------------
def list_records(view_id: str, page_size: int = 100) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    params = {"pageSize": page_size, "view": view_id}
    while True:
        resp = requests.get(
            AIRTABLE_API_ROOT, headers=HEADERS, params=params, timeout=60
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
        time.sleep(0.1)
    return records


def update_record(record_id: str, fields: Dict[str, Any], max_retries: int = 5) -> None:
    url = f"{AIRTABLE_API_ROOT}/{record_id}"
    payload = {"fields": fields}
    for attempt in range(1, max_retries + 1):
        resp = requests.patch(
            url, headers=HEADERS, data=json.dumps(payload), timeout=60
        )
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 * attempt, 8))
            continue
        resp.raise_for_status()
        return


def to_number(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        try:
            return float(str(x).strip())
        except Exception:
            return None


# ------------ Core ------------
def process_view(
    view_name: str,
    view_id: str,
    score_field: str,
    dry_run: bool,
    airtable_rps: float,
    limit: Optional[int],
    overwrite: bool,
) -> Dict[str, int]:
    recs = list_records(view_id)
    if limit is not None:
        recs = recs[:limit]

    total = len(recs)
    print(f"Total records in {view_name.upper()} view: {total}")

    flagged = 0
    cleared = 0
    missing_score = 0
    unchanged = 0
    errors = 0

    delay = 1.0 / max(airtable_rps, 0.1)

    it = tqdm(recs, desc=f"Marking low-scoring ({view_name})", unit="row")
    for rec in it:
        rec_id = rec.get("id")
        fields = rec.get("fields", {})
        score_raw = fields.get(score_field)
        score = to_number(score_raw)

        if score is None:
            missing_score += 1
            it.set_postfix_str("skip: no score")
            continue

        # Only this Gemini field is used for flagging
        should_check = score <= THRESHOLD
        existing = bool(fields.get(LOW_SCORING_CHECKBOX, False))

        # Decide whether we need to write
        will_write = False
        payload: Dict[str, Any] = {}

        if should_check:
            if not existing:
                payload[LOW_SCORING_CHECKBOX] = True
                will_write = True
        else:
            if overwrite and existing:
                payload[LOW_SCORING_CHECKBOX] = False
                will_write = True

        if not will_write:
            unchanged += 1
            it.set_postfix_str("unchanged")
            continue

        if dry_run:
            if should_check:
                flagged += 1
                it.set_postfix_str(f"DRY: check {rec_id[:6]}")
            else:
                cleared += 1
                it.set_postfix_str(f"DRY: uncheck {rec_id[:6]}")
            continue

        try:
            update_record(rec_id, payload)
            if should_check:
                flagged += 1
                it.set_postfix_str(f"✅ check {rec_id[:6]}")
            else:
                cleared += 1
                it.set_postfix_str(f"✅ uncheck {rec_id[:6]}")
            time.sleep(delay)
        except Exception as e:
            errors += 1
            it.set_postfix_str(f"❌ {type(e).__name__}: {e}")

    return {
        "total_seen": total,
        "checked": flagged,
        "unchecked": cleared,
        "missing_score": missing_score,
        "unchanged": unchanged,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check 'Low-Scoring Tasks' when "
            "'Gemini Autorater - Gemini 3.0 Response Score' ≤ 20."
        )
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan only — do not write to Airtable."
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows.")
    parser.add_argument(
        "--airtable-rps",
        type=float,
        default=float(os.getenv("AIRTABLE_RPS", "4")),
        help="Max Airtable writes per second (default 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If score > 20, uncheck the box when currently checked.",
    )
    args = parser.parse_args()

    # Env checks
    missing = [
        name
        for name, val in [
            ("AIRTABLE_API_KEY", AIRTABLE_API_KEY),
            ("AIRTABLE_BASE_ID", AIRTABLE_BASE_ID),
            ("AIRTABLE_TABLE", AIRTABLE_TABLE),
            ("AIRTABLE_VIEW_General", AIRTABLE_VIEW_GENERAL),
        ]
        if not val
    ]
    if missing:
        raise SystemExit(f"Missing required env vars: {', '.join(missing)}")

    summaries: Dict[str, Dict[str, int]] = {}

    print(f"\n=== Processing GENERAL view ===")
    res = process_view(
        view_name="general",
        view_id=AIRTABLE_VIEW_GENERAL,
        score_field=GEMINI_SCORE_FIELD,
        dry_run=args.dry_run,
        airtable_rps=args.airtable_rps,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    summaries["general"] = res
    print(json.dumps(res, indent=2))

    print("\n=== Done ===")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
