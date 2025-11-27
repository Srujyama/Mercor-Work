### NEVER GOT THIS FINISHED AND NEVER RAN THIS
# THE PRINTED OUT JSONS ARE NOT IN A READBLE FORMAT FOR HUMANS BUT IT WORKS FOR CHATGPT

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# 1. Load environment variables from .env
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE = os.getenv("AIRTABLE_TABLE")
AIRTABLE_VIEW_GENERAL = os.getenv("AIRTABLE_VIEW_General")

if not all([AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE, AIRTABLE_VIEW_GENERAL]):
    raise RuntimeError("Missing one or more required env vars. Check your .env file.")

AIRTABLE_API_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE}"
HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_API_KEY}",
    "Content-Type": "application/json",
}

# These are the two columns we will clean
TARGET_FIELDS = [
    "Gemini Autorater - Gemini 3.0 Response Summary",
    "GPT5 Autorater - Gemini 3.0 Response Summary",
]

# Max lengths for text fields (tweak as you like)
MAX_JUSTIFICATION_LEN = 600
MAX_REASONING_LEN = 600


def fetch_all_records() -> List[Dict[str, Any]]:
    """
    Fetch all records from the Airtable table for the General view.
    Handles pagination using offset.
    """
    records: List[Dict[str, Any]] = []
    params = {"view": AIRTABLE_VIEW_GENERAL}
    offset: Optional[str] = None

    while True:
        if offset:
            params["offset"] = offset
        resp = requests.get(AIRTABLE_API_URL, headers=HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        records.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            break

    return records


def shorten_text(text: str, max_len: int) -> str:
    """
    Shorten text to a maximum length, attempting to cut at sentence boundaries.
    """
    text = text.strip()
    if len(text) <= max_len:
        return text

    # Try to cut at the last period before the limit
    truncated = text[:max_len]
    last_period = truncated.rfind(".")
    if last_period > 50:  # avoid cutting if the period is too early
        truncated = truncated[: last_period + 1]
    else:
        truncated = truncated.rstrip()
        truncated += "..."

    return truncated


def clean_json_list(json_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean a list of criterion objects:
    - Normalize whitespace
    - Truncate very long text fields
    - Keep structure intact
    """
    cleaned: List[Dict[str, Any]] = []

    for obj in json_list:
        if not isinstance(obj, dict):
            cleaned.append(obj)
            continue

        new_obj = dict(obj)  # shallow copy

        # Normalize and shorten big text fields
        for key in ["description", "justification", "reasoning", "sources"]:
            if key in new_obj and isinstance(new_obj[key], str):
                value = " ".join(new_obj[key].split())  # collapse whitespace

                if key == "justification":
                    value = shorten_text(value, MAX_JUSTIFICATION_LEN)
                elif key == "reasoning":
                    value = shorten_text(value, MAX_REASONING_LEN)

                new_obj[key] = value

        cleaned.append(new_obj)

    return cleaned


def process_field_value(value: Any, record_id: str, field_name: str) -> Optional[str]:
    """
    If the cell contains JSON (string), parse it, clean it, and return a
    new pretty-printed JSON string. If no update needed, return None.
    """
    if not value:
        return None

    if not isinstance(value, str):
        print(f"[{record_id}] {field_name}: Value is not a string, skipping.")
        return None

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        print(f"[{record_id}] {field_name}: Failed to parse JSON, skipping.")
        return None

    if not isinstance(parsed, list):
        print(f"[{record_id}] {field_name}: Parsed JSON is not a list, skipping.")
        return None

    # Clean the JSON list deterministically (no LLM)
    print(f"[{record_id}] {field_name}: Cleaning {len(parsed)} criterion entries...")
    cleaned_list = clean_json_list(parsed)

    pretty_str = json.dumps(cleaned_list, indent=2, ensure_ascii=False)

    # Avoid rewriting if identical
    if pretty_str == value:
        return None

    return pretty_str


def update_airtable_records(updates: List[Dict[str, Any]]):
    """
    Send batched updates to Airtable (max 10 updates per request).
    """
    BATCH_SIZE = 10
    for i in range(0, len(updates), BATCH_SIZE):
        batch = updates[i : i + BATCH_SIZE]
        payload = {"records": batch}
        resp = requests.patch(
            AIRTABLE_API_URL, headers=HEADERS, data=json.dumps(payload)
        )
        if resp.status_code >= 400:
            print("Error updating batch:", resp.status_code, resp.text)
            resp.raise_for_status()
        time.sleep(0.2)  # small delay to be nice to the API


def main():
    print("Fetching records from Airtable (General view)...")
    records = fetch_all_records()
    print(f"Fetched {len(records)} records.")

    updates: List[Dict[str, Any]] = []

    for rec in records:
        record_id = rec["id"]
        fields = rec.get("fields", {})
        new_fields: Dict[str, Any] = {}

        for field_name in TARGET_FIELDS:
            if field_name not in fields:
                continue

            new_value = process_field_value(fields[field_name], record_id, field_name)
            if new_value is not None and new_value != fields[field_name]:
                new_fields[field_name] = new_value

        if new_fields:
            updates.append({"id": record_id, "fields": new_fields})

    if not updates:
        print("No records need updating.")
        return

    print(f"Updating {len(updates)} records in Airtable...")
    update_airtable_records(updates)
    print("Done!")


if __name__ == "__main__":
    main()
