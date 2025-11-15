import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
from dotenv import load_dotenv

# =========================
# Setup & env
# =========================
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE_ID")
AIRTABLE_VIEW_ID = os.getenv("AIRTABLE_VIEW_ID")

# New: the two scoring-summary columns (override via .env if needed)
GEMINI_SUMMARY_FIELD = os.getenv(
    "GEMINI_SUMMARY_FIELD", "[GPT5 graded] Gemini Response Scoring Summary"
)
GPT5_SUMMARY_FIELD = os.getenv(
    "GPT5_SUMMARY_FIELD", "[GPT5 graded] GPT5 Response Scoring Summary"
)

# Optional: comma-separated alternate keys for autorater & human values (in case schema drifts)
# Defaults are sensible, but you can override in .env like: AUTORATER_KEYS="autorater_rating,autorating"
_AUTORATER_KEYS = [
    s.strip()
    for s in os.getenv(
        "AUTORATER_KEYS",
        "autorater_rating, autorating, auto_rating, model_rating, gpt_rating, gemini_rating",
    ).split(",")
    if s.strip()
]
_HUMAN_KEYS = [
    s.strip()
    for s in os.getenv("HUMAN_KEYS", "human_rating, human").split(",")
    if s.strip()
]

if not all([AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID, AIRTABLE_VIEW_ID]):
    raise SystemExit(
        "Missing required .env variables: AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID, AIRTABLE_VIEW_ID"
    )

BASE_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
HEADERS = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}


# =========================
# Airtable fetch
# =========================
def fetch_all_records_for_view(
    view_id: str, page_size: int = 100, fields: Optional[List[str]] = None
) -> Iterable[Dict[str, Any]]:
    """Fetch all records for a given Airtable view ID (pagination handled)."""
    params: Any
    if fields:
        # Airtable expects 'fields[]' repeated for each field
        params = [
            ("pageSize", page_size),
            ("view", view_id),
            *[("fields[]", f) for f in fields],
        ]
    else:
        params = {"pageSize": page_size, "view": view_id}

    offset = None
    while True:
        if offset:
            if isinstance(params, list):
                params = [p for p in params if p[0] != "offset"] + [("offset", offset)]
            else:
                params["offset"] = offset
        resp = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Airtable API error {resp.status_code}: {resp.text}")
        data = resp.json()
        for rec in data.get("records", []):
            yield rec
        offset = data.get("offset")
        if not offset:
            break
        time.sleep(0.2)  # gentle pacing


# =========================
# JSON / parsing helpers
# =========================
def _strip_code_fences(s: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fencing if present."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)


def _json_load_maybe(s: Any) -> Any:
    """Load JSON if string; otherwise return the value unchanged (or None on error)."""
    if isinstance(s, str):
        s = _strip_code_fences(s)
        try:
            return json.loads(s)
        except Exception:
            return None
    return s


def _to_bool(x: Any) -> Optional[bool]:
    """Normalize common truthy/falsey forms to bool."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        if x == 1:
            return True
        if x == 0:
            return False
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "yes", "y", "1", "pass", "passed"}:
            return True
        if s in {"false", "no", "n", "0", "fail", "failed"}:
            return False
    return None


def _iter_items_from_any_rubric_shape(r: Any) -> List[Dict[str, Any]]:
    """
    Yield inner dicts for each criterion, handling shapes like:
      - list[ {"criterion N": {...}}, {"criterion M": {...}}, ... ]
      - list[ {...}, ... ]
      - dict with 'criteria' or 'rubric' -> (list or dict)
      - dict[str, dict] (criteria keyed by name)
    """
    if r is None:
        return []
    data = _json_load_maybe(r)
    if data is None:
        data = r

    # unwrap common container keys
    if isinstance(data, dict):
        for key in ("criteria", "rubric"):
            if key in data:
                data = data[key]
                break

    out: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for x in data:
            if not isinstance(x, dict):
                continue
            # Handle {"criterion N": {...}} wrapper
            if len(x) == 1:
                inner = next(iter(x.values()))
                if isinstance(inner, dict):
                    out.append(inner)
                    continue
            # Already the inner dict
            out.append(x)
        return out

    if isinstance(data, dict):
        # criteria keyed by name
        return [v for v in data.values() if isinstance(v, dict)]

    return []


# =========================
# Extraction helpers (match human vs autorater in a single item)
# =========================
def _get_first_present(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _extract_match_flag(
    item: Dict[str, Any],
    human_keys: List[str],
    autorater_keys: List[str],
) -> Tuple[Optional[bool], Optional[str]]:
    """
    Return (is_match, skip_reason).
      - is_match: True/False if both values parsed; None if cannot parse.
      - skip_reason: 'missing_keys' | 'unparseable_bool' | None
    """
    human_raw = _get_first_present(item, human_keys)
    auto_raw = _get_first_present(item, autorater_keys)

    if human_raw is None or auto_raw is None:
        return None, "missing_keys"

    human = _to_bool(human_raw)
    auto = _to_bool(auto_raw)

    if human is None or auto is None:
        return None, "unparseable_bool"

    return human == auto, None


def iter_matches_from_summary_field(
    field_value: Any,
    human_keys: List[str] = _HUMAN_KEYS,
    autorater_keys: List[str] = _AUTORATER_KEYS,
) -> Iterable[Tuple[Optional[bool], Optional[str]]]:
    """
    Iterate over criteria in a summary field value, yielding (is_match, skip_reason).
    """
    for item in _iter_items_from_any_rubric_shape(field_value):
        is_match, reason = _extract_match_flag(item, human_keys, autorater_keys)
        yield is_match, reason


# =========================
# Core aggregation (two columns)
# =========================
def aggregate_matches_for_view(
    view_id: str,
    gemini_field: str,
    gpt5_field: str,
) -> Dict[str, Any]:
    """
    For each record, read the two summary columns.
    For each criterion inside those columns, check whether autorating == human_rating.
    Aggregate totals per column.
    """
    # Counters
    totals = {
        "gemini_total_criteria": 0,
        "gemini_matches": 0,
        "gemini_mismatches": 0,
        "gpt5_total_criteria": 0,
        "gpt5_matches": 0,
        "gpt5_mismatches": 0,
    }
    skipped = {
        "gemini_missing_keys": 0,
        "gemini_unparseable_bool": 0,
        "gpt5_missing_keys": 0,
        "gpt5_unparseable_bool": 0,
    }
    records_with_any_field = 0

    fields_to_pull = [gemini_field, gpt5_field]
    records = fetch_all_records_for_view(view_id, fields=fields_to_pull)
    for idx, rec in enumerate(records):
        f = rec.get("fields", {})
        gem_val = f.get(gemini_field)
        gpt_val = f.get(gpt5_field)

        if gem_val is None and gpt_val is None:
            continue
        records_with_any_field += 1

        # GEMINI column
        if gem_val is not None:
            for is_match, reason in iter_matches_from_summary_field(gem_val):
                if reason is None:
                    totals["gemini_total_criteria"] += 1
                    if is_match:
                        totals["gemini_matches"] += 1
                    else:
                        totals["gemini_mismatches"] += 1
                elif reason == "missing_keys":
                    skipped["gemini_missing_keys"] += 1
                elif reason == "unparseable_bool":
                    skipped["gemini_unparseable_bool"] += 1

        # GPT-5 column
        if gpt_val is not None:
            for is_match, reason in iter_matches_from_summary_field(gpt_val):
                if reason is None:
                    totals["gpt5_total_criteria"] += 1
                    if is_match:
                        totals["gpt5_matches"] += 1
                    else:
                        totals["gpt5_mismatches"] += 1
                elif reason == "missing_keys":
                    skipped["gpt5_missing_keys"] += 1
                elif reason == "unparseable_bool":
                    skipped["gpt5_unparseable_bool"] += 1

        if idx < 2:
            logging.info(
                f"Sample record {idx}: "
                f"GEM parsed so far={totals['gemini_total_criteria']}, "
                f"GPT5 parsed so far={totals['gpt5_total_criteria']}"
            )

    # Percentages
    gem_tot = totals["gemini_total_criteria"]
    gpt_tot = totals["gpt5_total_criteria"]

    pct_gem_match = (totals["gemini_matches"] / gem_tot * 100) if gem_tot else None
    pct_gem_mismatch = (
        (totals["gemini_mismatches"] / gem_tot * 100) if gem_tot else None
    )

    pct_gpt_match = (totals["gpt5_matches"] / gpt_tot * 100) if gpt_tot else None
    pct_gpt_mismatch = (totals["gpt5_mismatches"] / gpt_tot * 100) if gpt_tot else None

    out = {
        "view_id": view_id,
        "records_with_any_summary_field": records_with_any_field,
        # GEMINI field stats
        "gemini_field": gemini_field,
        "gemini_total_criteria": gem_tot,
        "gemini_matches": totals["gemini_matches"],
        "gemini_mismatches": totals["gemini_mismatches"],
        "gemini_pct_matches": pct_gem_match,
        "gemini_pct_mismatches": pct_gem_mismatch,
        "gemini_skipped_missing_keys": skipped["gemini_missing_keys"],
        "gemini_skipped_unparseable_bool": skipped["gemini_unparseable_bool"],
        # GPT-5 field stats
        "gpt5_field": gpt5_field,
        "gpt5_total_criteria": gpt_tot,
        "gpt5_matches": totals["gpt5_matches"],
        "gpt5_mismatches": totals["gpt5_mismatches"],
        "gpt5_pct_matches": pct_gpt_match,
        "gpt5_pct_mismatches": pct_gpt_mismatch,
        "gpt5_skipped_missing_keys": skipped["gpt5_missing_keys"],
        "gpt5_skipped_unparseable_bool": skipped["gpt5_unparseable_bool"],
    }

    logging.info(
        f"[{view_id}] "
        f"GEMINI: total={gem_tot}, matches={totals['gemini_matches']} ({pct_gem_match:.2f}% if total), "
        f"mismatches={totals['gemini_mismatches']} ({pct_gem_mismatch:.2f}% if total); "
        f"GPT5: total={gpt_tot}, matches={totals['gpt5_matches']} ({pct_gpt_match:.2f}% if total), "
        f"mismatches={totals['gpt5_mismatches']} ({pct_gpt_mismatch:.2f}% if total)."
    )
    return out


# =========================
# Visualization (compare columns)
# =========================
def plot_two_bar(stats: Dict[str, Any], outdir: str = ".") -> None:
    labels = ["Gemini (match %)", "GPT-5 (match %)"]
    values = [
        stats.get("gemini_pct_matches") or 0.0,
        stats.get("gpt5_pct_matches") or 0.0,
    ]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values)
    for b, v in zip(bars, values):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.1f}%",
            ha="center",
            va="bottom",
        )
    plt.ylabel("Match Rate (%)")
    plt.title(f"Autorating vs Human — {stats['view_id']}")
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"autorater_match_chart__{stats['view_id']}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    logging.info(f"Saved chart -> {out_path}")


# =========================
# Main
# =========================
def _none_to_empty(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: ("" if v is None else v) for k, v in d.items()}


def main():
    stats = aggregate_matches_for_view(
        AIRTABLE_VIEW_ID, GEMINI_SUMMARY_FIELD, GPT5_SUMMARY_FIELD
    )

    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)

    # CSV
    csv_path = os.path.join(outdir, f"autorater_match_summary__{AIRTABLE_VIEW_ID}.csv")
    pd.DataFrame([_none_to_empty(stats)]).to_csv(csv_path, index=False)
    logging.info(f"Saved CSV -> {csv_path}")

    # Chart
    plot_two_bar(stats, outdir=outdir)

    # Console summary
    print(pd.DataFrame([stats]).to_string(index=False))


if __name__ == "__main__":
    main()
