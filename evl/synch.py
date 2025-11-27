#!/usr/bin/env python3
# pip install python-dotenv requests

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")

AIRTABLE_VIEW_EVALSET = os.getenv("AIRTABLE_VIEW_EVALSET")
AIRTABLE_VIEW_TRAINING = os.getenv("AIRTABLE_VIEW_TRAINING")
AIRTABLE_VIEW_CUJ = os.getenv("AIRTABLE_VIEW_CUJ")

if not AIRTABLE_API_KEY or not AIRTABLE_BASE_ID or not AIRTABLE_TABLE:
    raise SystemExit(
        "Missing AIRTABLE_API_KEY, AIRTABLE_BASE_ID, or AIRTABLE_TABLE in .env"
    )

if not AIRTABLE_VIEW_EVALSET or not AIRTABLE_VIEW_TRAINING or not AIRTABLE_VIEW_CUJ:
    raise SystemExit(
        "Missing one of AIRTABLE_VIEW_EVALSET, AIRTABLE_VIEW_TRAINING, AIRTABLE_VIEW_CUJ in .env"
    )

AIRTABLE_API = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json",
}

# Target (general) fields
TARGET_FIELDS = {
    "gemini_score": "Gemini Autorater - Gemini 3.0 Response Score",
    "gemini_summary": "Gemini Autorater - Gemini 3.0 Response Summary",
    "gpt5_score": "GPT5 Autorater - Gemini 3.0 Response Score",
    "gpt5_summary": "GPT5 Autorater - Gemini 3.0 Response Summary",
}


def list_records(view_id: str, page_size: int = 100) -> List[Dict[str, Any]]:
    """Fetch all records from a given view with pagination."""
    records: List[Dict[str, Any]] = []
    params = {"pageSize": page_size, "view": view_id}
    while True:
        resp = requests.get(AIRTABLE_API, headers=HEADERS, params=params, timeout=60)
        if resp.status_code == 429:
            print("[AIRTABLE] 429 rate limit, sleeping...")
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


def batch_update(records_payload: List[Dict[str, Any]], max_retries: int = 5) -> None:
    """Batch update up to 10 records at a time."""
    if not records_payload:
        return

    payload = {"records": records_payload}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.patch(
                AIRTABLE_API, headers=HEADERS, data=json.dumps(payload), timeout=60
            )
        except Exception as e:
            print(f"[AIRTABLE] Exception on batch PATCH: {e}")
            if attempt == max_retries:
                raise
            time.sleep(min(2 * attempt, 8))
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            print(
                f"[AIRTABLE] {resp.status_code} on batch PATCH, attempt {attempt}, retrying..."
            )
            if attempt == max_retries:
                resp.raise_for_status()
            time.sleep(min(2 * attempt, 8))
            continue

        resp.raise_for_status()
        return


def value_is_empty(v: Any) -> bool:
    """Treat None, empty string, and empty list as 'empty'."""
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, list) and len(v) == 0:
        return True
    return False


def build_updates_for_view(
    records: List[Dict[str, Any]],
    view_name_suffix: str,
) -> List[Dict[str, Any]]:
    """
    For a given view (Eval / Training / CUJ suffix), build Airtable update payloads.

    Copies:
      Gemini Autorater - Gemini 3.0 Response Score - [view]   -> Gemini Autorater - Gemini 3.0 Response Score
      Gemini Autorater - Gemini 3.0 Response Summary - [view] -> Gemini Autorater - Gemini 3.0 Response Summary
      GPT5 Autorater - Gemini 3.0 Response Score - [view]     -> GPT5 Autorater - Gemini 3.0 Response Score
      GPT5 Autorater - Gemini 3.0 Response Summary - [view]   -> GPT5 Autorater - Gemini 3.0 Response Summary

    Only when:
      - Source field (with suffix) is non-empty
      - Target field is empty or missing
    """
    updates: List[Dict[str, Any]] = []

    # Build source field names for this view
    source_fields = {
        "gemini_score": f"Gemini Autorater - Gemini 3.0 Response Score - {view_name_suffix}",
        "gemini_summary": f"Gemini Autorater - Gemini 3.0 Response Summary - {view_name_suffix}",
        "gpt5_score": f"GPT5 Autorater - Gemini 3.0 Response Score - {view_name_suffix}",
        "gpt5_summary": f"GPT5 Autorater - Gemini 3.0 Response Summary - {view_name_suffix}",
    }

    for rec in records:
        rec_id = rec.get("id")
        fields = rec.get("fields", {})

        new_fields: Dict[str, Any] = {}

        for key, src_field in source_fields.items():
            dst_field = TARGET_FIELDS[key]

            src_val = fields.get(src_field)
            dst_val = fields.get(dst_field)

            # Skip if source is empty or missing
            if value_is_empty(src_val):
                continue

            # Skip if target already has data
            if not value_is_empty(dst_val):
                continue

            # Copy the value
            new_fields[dst_field] = src_val

        if new_fields:
            updates.append({"id": rec_id, "fields": new_fields})

    return updates


def process_view(view_id: str, view_suffix: str) -> None:
    """
    Process one view:
      1. Fetch records
      2. Build updates
      3. Send in batches of 10
    """
    print(f"\n=== Processing view '{view_suffix}' (view_id={view_id}) ===")
    records = list_records(view_id)
    print(f"[INFO] Retrieved {len(records)} records from view '{view_suffix}'")

    updates = build_updates_for_view(records, view_suffix)
    print(
        f"[INFO] {len(updates)} records need autorater field copying for '{view_suffix}'"
    )

    # Send in batches of 10
    batch_size = 10
    for i in range(0, len(updates), batch_size):
        batch = updates[i : i + batch_size]
        batch_update(batch)
        print(
            f"[INFO] Updated {len(batch)} records (batch {i // batch_size + 1}) in '{view_suffix}'"
        )

    print(f"[DONE] Finished view '{view_suffix}'")


def main():
    # Order: Eval, Training, CUJ
    process_view(AIRTABLE_VIEW_EVALSET, "Eval")
    process_view(AIRTABLE_VIEW_TRAINING, "Training")
    process_view(AIRTABLE_VIEW_CUJ, "CUJ")


if __name__ == "__main__":
    main()
