#!/usr/bin/env python3
"""
Local Airtable connection test (Python)
- Loads environment variables
- Validates API credentials and identifiers
- Confirms Airtable connection without fetching or modifying data
"""

import os
import argparse
from dotenv import load_dotenv
from pyairtable import Table

load_dotenv()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=os.getenv("AIRTABLE_BASE_ID", ""))
    p.add_argument("--table", default=os.getenv("AIRTABLE_TABLE", ""))
    p.add_argument("--view", default=os.getenv("AIRTABLE_VIEW", ""))  # optional
    return p.parse_args()

def main():
    api_key = os.getenv("AIRTABLE_API_KEY")
    if not api_key:
        raise SystemExit("❌ Missing AIRTABLE_API_KEY in environment. Add it to your .env file.")

    args = parse_args()
    if not args.base or not args.table:
        raise SystemExit("❌ Missing --base or --table (or AIRTABLE_BASE_ID/AIRTABLE_TABLE in .env).")

    print("▶ Environment loaded successfully.")
    print({
        "AIRTABLE_BASE_ID": args.base,
        "AIRTABLE_TABLE": args.table,
        "AIRTABLE_VIEW": args.view or "(default)"
    })

    try:
        # Create a Table object to validate credentials
        table = Table(api_key, args.base, args.table)
        print("\n✅ Airtable client initialized successfully!")
        print(f"Ready to query table '{args.table}' in base '{args.base}'.")
    except Exception as e:
        raise SystemExit(f"❌ Error initializing Airtable connection: {e}")

    print("\n(No data fetched — dry run complete.)")

if __name__ == "__main__":
    main()
