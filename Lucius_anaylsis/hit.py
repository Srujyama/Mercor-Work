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
RUBRIC_FIELD = os.getenv("RUBRIC_FIELD", "Rubric JSON")

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
    view_id: str, page_size: int = 100
) -> Iterable[Dict[str, Any]]:
    """Fetch all records for a given Airtable view ID (pagination handled)."""
    params = {"pageSize": page_size, "view": view_id}
    offset = None
    while True:
        if offset:
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
    """Load JSON if string; otherwise return the value unchanged."""
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
        if s in {"true", "yes", "y", "1"}:
            return True
        if s in {"false", "no", "n", "0"}:
            return False
    return None


def _iter_items_from_any_rubric_shape(r: Any) -> List[Dict[str, Any]]:
    """
    Yield inner dicts for each criterion, handling these shapes:
      - list[ {"criterion N": {...}}, {"criterion M": {...}}, ... ]
      - list[ {...}, ... ]
      - dict with 'criteria' or 'rubric' -> (list or dict)
      - dict[str, dict]
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
# Extraction (STRICT + SKIP missing)
# =========================
def _extract_criterion_strict(
    item: Dict[str, Any],
) -> Optional[Tuple[bool, bool, bool]]:
    """
    STRICT extraction: only these exact keys are considered.
      - human:  'human_rating'
      - gemini: 'gemini_as_autorater_rating'
      - gpt:    'gpt_as_autorater_rating'
    If any of these keys are missing, skip the item (return None).
    """
    required_keys = {
        "human_rating",
        "gemini_as_autorater_rating",
        "gpt_as_autorater_rating",
    }
    if not required_keys.issubset(item.keys()):
        return None

    human = _to_bool(item.get("human_rating"))
    gemini = _to_bool(item.get("gemini_as_autorater_rating"))
    gpt = _to_bool(item.get("gpt_as_autorater_rating"))

    # Skip if any are None (invalid or unparseable)
    if human is None or gemini is None or gpt is None:
        return None

    return human, gemini, gpt


def iter_criteria_from_rubric(
    rubric: Any,
) -> Iterable[Tuple[bool, bool, bool]]:
    """Iterate (human, gemini, gpt) tuples from the rubric JSON (strict keys only)."""
    for item in _iter_items_from_any_rubric_shape(rubric):
        triple = _extract_criterion_strict(item)
        if triple:
            yield triple


# =========================
# Core aggregation
# =========================
def aggregate_misalignment_for_view(view_id: str) -> Dict[str, Any]:
    """
    Aggregate over ALL criteria across ALL tasks in the view:
      - Only includes criteria that have human, gemini, and gpt ratings present.
      - Misalignment counts:
          * +1 if autorater present and (autorater != human)
    """
    total_criteria = 0
    mis_gemini = 0
    mis_gpt = 0
    records_with_rubric = 0
    records_missing_rubric = 0

    records = list(fetch_all_records_for_view(view_id))
    logging.info(f"Fetched {len(records)} records for view ID '{view_id}'.")

    for idx, rec in enumerate(records):
        fields = rec.get("fields", {})
        rubric_raw = fields.get(RUBRIC_FIELD)
        if rubric_raw is None:
            records_missing_rubric += 1
            continue
        records_with_rubric += 1

        parsed_here = 0
        for human, gemini, gpt in iter_criteria_from_rubric(rubric_raw):
            total_criteria += 1
            parsed_here += 1
            if human != gemini:
                mis_gemini += 1
            if human != gpt:
                mis_gpt += 1

        if idx < 2:
            logging.info(f"Sample record {idx}: criteria_parsed={parsed_here}")

    pct_gemini = (mis_gemini / total_criteria * 100) if total_criteria else math.nan
    pct_gpt = (mis_gpt / total_criteria * 100) if total_criteria else math.nan

    logging.info(
        f"[{view_id}] total_criteria={total_criteria} | "
        f"mis_gemini={mis_gemini} ({pct_gemini:.2f}%) | "
        f"mis_gpt={mis_gpt} ({pct_gpt:.2f}%) | "
        f"records_with_rubric={records_with_rubric} | records_missing_rubric={records_missing_rubric}"
    )

    return {
        "view_id": view_id,
        "total_criteria": total_criteria,
        "misaligned_gemini": mis_gemini,
        "misaligned_gpt": mis_gpt,
        "pct_misaligned_gemini": pct_gemini,
        "pct_misaligned_gpt": pct_gpt,
        "records_with_rubric": records_with_rubric,
        "records_missing_rubric": records_missing_rubric,
    }


# =========================
# Visualization (single view)
# =========================
def plot_two_bar(stats: Dict[str, Any], outdir: str = ".") -> None:
    labels = ["Human vs Gemini", "Human vs GPT"]
    values = [stats["pct_misaligned_gemini"], stats["pct_misaligned_gpt"]]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Misalignment (%)")
    plt.title(f"Autorater Misalignment — View {stats['view_id']}")
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"misalignment_chart__{stats['view_id']}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    logging.info(f"Saved chart -> {out_path}")


# =========================
# Main
# =========================
def main():
    stats = aggregate_misalignment_for_view(AIRTABLE_VIEW_ID)

    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)

    # CSV
    csv_path = os.path.join(outdir, f"misalignment_summary__{AIRTABLE_VIEW_ID}.csv")
    pd.DataFrame([stats]).to_csv(csv_path, index=False)
    logging.info(f"Saved CSV -> {csv_path}")

    # Chart
    plot_two_bar(stats, outdir=outdir)

    # Console summary
    print(pd.DataFrame([stats]).to_string(index=False))


if __name__ == "__main__":
    main()
