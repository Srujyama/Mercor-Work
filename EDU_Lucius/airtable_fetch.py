#!/usr/bin/env python3
"""
Reads from 'Rubric JSON', transforms each criterion, and writes results into 'IA Modified Rubric'.
Transformation:
- Move any text inside parentheses in 'sources' → append to first item in 'criterion_type'
- Clear 'sources'
Writes only to the target field (never touches others).

Usage:
  python update_ia_modified.py                 # Dry run (pretty output by default)
  python update_ia_modified.py --apply         # Apply updates to Airtable (pretty)
  python update_ia_modified.py --format compact
  python update_ia_modified.py --apply --overwriteTarget

.env must include:
  AIRTABLE_API_KEY
  AIRTABLE_BASE_ID
  AIRTABLE_TABLE
  (optional) AIRTABLE_VIEW
"""

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from pyairtable import Api

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

load_dotenv()
LOG = logging.getLogger("rubric-json-to-ia-modified")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=os.getenv("AIRTABLE_BASE_ID", ""), help="Base ID")
    p.add_argument(
        "--table", default=os.getenv("AIRTABLE_TABLE", ""), help="Table ID or name"
    )
    p.add_argument(
        "--view", default=os.getenv("AIRTABLE_VIEW", ""), help="Optional view name"
    )
    p.add_argument("--sourceField", default="Rubric JSON", help="Source field name")
    p.add_argument(
        "--targetField", default="IA Modified Rubric", help="Target field name"
    )
    p.add_argument(
        "--maxRecords", type=int, default=None, help="Optional limit on records"
    )
    p.add_argument("--apply", action="store_true", help="Persist updates to Airtable")
    p.add_argument(
        "--overwriteTarget",
        action="store_true",
        help="Allow overwriting non-empty target",
    )
    p.add_argument(
        "--format",
        choices=["pretty", "compact"],
        default="pretty",
        help="Formatting for transformed JSON written to target",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


PARENS_RE = re.compile(r"\(([^)]+)\)")


def extract_labels(s: Any) -> List[str]:
    """Return labels found in parentheses."""
    if not isinstance(s, str) or not s:
        return []
    return [m.strip() for m in PARENS_RE.findall(s) if m.strip()]


def append_labels_to_first_type(ct_val: Any, labels: List[str]) -> Any:
    """Append ' (label1, label2)' to the first string of criterion_type."""
    if not labels:
        return ct_val
    if ct_val is None:
        lst = []
    elif isinstance(ct_val, list):
        lst = list(ct_val)
    else:
        lst = [str(ct_val)]

    if not lst:
        lst = [f"({', '.join(labels)})"]
        return lst

    first = str(lst[0])
    suffix = f" ({', '.join(labels)})"
    if not first.endswith(suffix):
        lst[0] = first + suffix
    return lst


def transform_criterion_obj(obj: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Move labels from sources → append to criterion_type; clear sources."""
    sources = obj.get("sources", "")
    labels = extract_labels(sources)
    if not labels:
        return False, obj

    new_obj = dict(obj)
    new_obj["criterion_type"] = append_labels_to_first_type(
        new_obj.get("criterion_type"), labels
    )
    new_obj["sources"] = ""
    changed = (new_obj.get("criterion_type") != obj.get("criterion_type")) or bool(
        sources
    )
    return changed, new_obj


def _dump_json_clean(data: Any, fmt: str) -> str:
    """Return JSON as a clean string according to fmt ('pretty' or 'compact')."""
    if fmt == "pretty":
        # Keep natural key order, readable indentation, and UTF-8 characters.
        return json.dumps(data, ensure_ascii=False, indent=2)
    # Compact (single line) like before.
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def transform_rubric_payload(payload: Any, fmt: str) -> Tuple[bool, Any]:
    """Handle JSON payloads (string or parsed), returning (changed, result_or_str)."""
    original_text = None
    parsed = payload
    if isinstance(payload, str):
        original_text = payload
        try:
            parsed = json.loads(payload)
        except Exception:
            return False, payload

    changed_any = False

    def handle_dict(d: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        changed = False
        out = dict(d)
        for k, v in d.items():
            if isinstance(v, dict) and k.lower().startswith("criterion "):
                c_changed, c_new = transform_criterion_obj(v)
                if c_changed:
                    changed = True
                    out[k] = c_new
        return changed, out

    if isinstance(parsed, dict):
        c, new = handle_dict(parsed)
        changed_any |= c
        result = new
    elif isinstance(parsed, list):
        new_list = []
        for elt in parsed:
            if isinstance(elt, dict):
                c, new_d = handle_dict(elt)
                changed_any |= c
                new_list.append(new_d)
            else:
                new_list.append(elt)
        result = new_list
    else:
        return False, payload

    if original_text is not None:
        new_text = _dump_json_clean(result, fmt)
        if changed_any and new_text != original_text:
            return True, new_text
        return False, original_text
    return changed_any, result


def iterate_records(table, view: str, limit: int | None):
    """Yield individual record dicts (compatible with pyairtable>=3.x)."""
    params = {}
    if view:
        params["view"] = view
    seen = 0
    try:
        iterator = table.iterate(**params)
        for page in iterator:
            if isinstance(page, dict):  # single record
                yield page
                seen += 1
            elif isinstance(page, list):  # list of records
                for rec in page:
                    yield rec
                    seen += 1
            if limit is not None and seen >= limit:
                return
    except Exception:
        records = table.all(**params)
        for rec in records[:limit] if limit is not None else records:
            yield rec


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )

    api_key = os.getenv("AIRTABLE_API_KEY")
    if not api_key:
        LOG.error("❌ Missing AIRTABLE_API_KEY in environment (.env).")
        sys.exit(1)
    if not args.base or not args.table:
        LOG.error("❌ Missing --base or --table (or env vars).")
        sys.exit(1)

    LOG.info("▶ Identifiers")
    LOG.info(
        {
            "base": args.base,
            "table": args.table,
            "view": args.view or "(default)",
            "sourceField": args.sourceField,
            "targetField": args.targetField,
            "format": args.format,
        }
    )

    api = Api(api_key)
    table = api.table(args.base, args.table)

    planned, skipped = [], []
    LOG.info("\n▶ Scanning records…")
    for rec in iterate_records(table, args.view, args.maxRecords):
        rid = rec.get("id")
        fields = rec.get("fields", {}) or {}
        src_val = fields.get(args.sourceField)
        tgt_val = fields.get(args.targetField)

        if not src_val or (isinstance(src_val, str) and not src_val.strip()):
            skipped.append({"id": rid, "reason": "empty source"})
            continue

        if not args.overwriteTarget and isinstance(tgt_val, str) and tgt_val.strip():
            skipped.append({"id": rid, "reason": "target already populated"})
            continue

        changed, transformed = transform_rubric_payload(src_val, args.format)
        if not changed:
            skipped.append({"id": rid, "reason": "no change needed or invalid JSON"})
            continue

        planned.append({"id": rid, "update": {args.targetField: transformed}})

    LOG.info(
        json.dumps(
            {"plannedCount": len(planned), "skipped": len(skipped)},
            ensure_ascii=False,
            indent=2,
        )
    )

    if not planned:
        LOG.info("\n✅ Nothing to write.")
        return

    if not args.apply:
        LOG.info(
            "\nℹ️ Dry run complete. Re-run with --apply to write ONLY to the target field."
        )
        return

    LOG.info("\n▶ Writing updates to target field…")
    ok, fail = 0, 0
    for p in planned:
        try:
            table.update(p["id"], p["update"])
            ok += 1
        except Exception as e:
            fail += 1
            LOG.error(f"Failed to update {p['id']}: {e}")

    LOG.info(f"\n✅ Done. Applied: {ok}, Failed: {fail}")


if __name__ == "__main__":
    main()
