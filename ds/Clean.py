#!/usr/bin/env python3
"""
Fill missing Gemini 3.0 model responses with '-'
------------------------------------------------

If "Gemini 3.0 model responses" is empty, null, or blank → set it to "-".
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from pyairtable import Api

# ----------------------- Load .env -----------------------
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE")
AIRTABLE_VIEW_COMBINED = os.getenv("AIRTABLE_VIEW_COMBINED")  # optional

if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY in .env")

FIELD_NAME = "Gemini 3.0 model responses"
PLACEHOLDER = "-"

# ----------------------- Logging -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ----------------------- Helpers -----------------------
def is_blank(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, list) and len(val) == 0:
        return True
    return False


# ----------------------- Main -----------------------
def main():
    logger.info("🚀 Starting placeholder fill for Gemini 3.0 model responses")

    api = Api(AIRTABLE_API_KEY)
    table = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

    # Fetch full records (no fields filter to avoid 422 errors)
    fetch_kwargs = {}
    if AIRTABLE_VIEW_COMBINED:
        fetch_kwargs["view"] = AIRTABLE_VIEW_COMBINED

    records = table.all(**fetch_kwargs)
    logger.info(f"Fetched {len(records)} records")

    updated = 0
    skipped = 0

    for rec in records:
        rec_id = rec["id"]
        fields = rec.get("fields", {})

        val = fields.get(FIELD_NAME)

        if is_blank(val):
            try:
                table.update(rec_id, {FIELD_NAME: PLACEHOLDER})
                updated += 1
                logger.info(f"Updated record {rec_id}")
            except Exception as e:
                logger.error(f"Failed to update {rec_id}: {e}")
        else:
            skipped += 1

    logger.info(f"✅ Done. Updated: {updated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
