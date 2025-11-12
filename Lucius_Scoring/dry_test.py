import os
import sys

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
AIRTABLE_VIEW_NAME = os.getenv("AIRTABLE_VIEW_NAME")

TIMEOUT = 45


def fail(msg: str):
    print(f"❌ {msg}")
    sys.exit(1)


def warn(msg: str):
    print(f"⚠️  {msg}")


# --- Grounded current time test ---
def test_grounded_time():
    if not GEMINI_API_KEY:
        warn("GEMINI_API_KEY not set; skipping grounded time test.")
        return
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        tool = types.Tool(
            google_search=types.GoogleSearch()
        )  # Grounding tool  :contentReference[oaicite:6]{index=6}
        cfg = types.GenerateContentConfig(tools=[tool])

        prompt = "What is the current UTC date and time? Return one ISO-8601 value."
        resp = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt, config=cfg
        )
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            fail("Grounded time call returned empty text.")
        print("✅ Gemini grounded time:", text)

        # Optional: see grounding metadata if provided
        meta = getattr(resp, "grounding_metadata", None)
        if meta:
            print("Grounding metadata present (sources may be cited).")
    except Exception as e:
        fail(f"Gemini grounded time failed: {e}")


# --- Airtable columns (metadata API -> fallback to sample records) ---
def airtable_headers(json=True):
    h = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    if json:
        h["Content-Type"] = "application/json"
    return h


def airtable_meta_tables():
    url = f"https://api.airtable.com/v0/meta/bases/{AIRTABLE_BASE_ID}/tables"
    return requests.get(url, headers=airtable_headers(json=False), timeout=TIMEOUT)


def airtable_meta_table_detail(table_id: str):
    url = f"https://api.airtable.com/v0/meta/bases/{AIRTABLE_BASE_ID}/tables/{table_id}"
    return requests.get(url, headers=airtable_headers(json=False), timeout=TIMEOUT)


def airtable_records_sample():
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{requests.utils.quote(AIRTABLE_TABLE_NAME)}"
    params = {"pageSize": 5}
    if AIRTABLE_VIEW_NAME:
        params["view"] = AIRTABLE_VIEW_NAME
    return requests.get(
        url, headers=airtable_headers(json=False), params=params, timeout=TIMEOUT
    )


def test_airtable_columns():
    if not (AIRTABLE_API_KEY and AIRTABLE_BASE_ID and AIRTABLE_TABLE_NAME):
        warn("Airtable env vars not fully set; skipping Airtable test.")
        return

    try:
        r = airtable_meta_tables()
        if r.status_code == 200:
            tables = r.json().get("tables", [])
            table = next(
                (t for t in tables if t.get("name") == AIRTABLE_TABLE_NAME), None
            )
            if not table:
                warn("Table not found in Metadata list—trying detailed lookup anyway.")
            table_id = table["id"] if table else None
            if not table_id and tables:
                for t in tables:
                    if t.get("name") == AIRTABLE_TABLE_NAME:
                        table_id = t.get("id")
                        break
            if table_id:
                r2 = airtable_meta_table_detail(table_id)
                if r2.status_code == 200:
                    fields = r2.json().get("fields", [])
                    print("✅ Airtable Metadata API. Columns (name : type):")
                    for f in fields:
                        print(f" - {f.get('name')} : {f.get('type')}")
                    return
                else:
                    warn(
                        f"Metadata table detail failed ({r2.status_code}). Body: {r2.text[:200]}"
                    )
            else:
                warn("Could not resolve table_id from Metadata API.")
        else:
            warn(f"Metadata API not available ({r.status_code}). Body: {r.text[:200]}")
    except Exception as e:
        warn(f"Metadata API error: {e}")

    # Fallback to sample records
    try:
        r = airtable_records_sample()
        if r.status_code != 200:
            fail(f"Airtable records request failed ({r.status_code}): {r.text[:200]}")
        records = r.json().get("records", [])
        if not records:
            print("✅ Airtable reachable, but no records in this view/table.")
            return
        colnames = set()
        for rec in records:
            colnames.update((rec.get("fields") or {}).keys())
        print("✅ Airtable fallback. Inferred columns:")
        for name in sorted(colnames):
            print(f" - {name}")
    except Exception as e:
        fail(f"Airtable fallback error: {e}")


if __name__ == "__main__":
    print("=== DRY TEST (Grounded) START ===")
    test_grounded_time()
    test_airtable_columns()
    print("=== DRY TEST (Grounded) DONE ===")
