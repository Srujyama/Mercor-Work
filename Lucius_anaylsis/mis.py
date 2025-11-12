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

# Optional key overrides if your field names ever change
HUMAN_KEY_ENV = os.getenv("HUMAN_KEY")  # e.g., human_rating
GEMINI_KEY_ENV = os.getenv("GEMINI_KEY")  # e.g., gemini_as_autorater_rating
GPT_KEY_ENV = os.getenv("GPT_KEY")  # e.g., gpt_as_autorater_rating


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


def _find_key_ci(
    d: Dict[str, Any], candidates: List[str], fallback_contains: List[str]
) -> Optional[str]:
    """Find a key in dict `d` by exact/ci match, else by 'contains' heuristics."""
    dl = {k.lower(): k for k in d.keys()}
    for cand in candidates:
        if cand.lower() in dl:
            return dl[cand.lower()]
    for k in d.keys():
        kl = k.lower()
        if any(sub in kl for sub in fallback_contains):
            return k
    return None


def _map_criterion_keys(
    item: Dict[str, Any],
) -> Tuple[Optional[bool], Optional[bool], Optional[bool]]:
    """
    Given the inner criterion dict, extract (human, gemini, gpt) booleans.
    Uses env overrides if provided; else fuzzy key search.
    """
    # human
    if HUMAN_KEY_ENV and HUMAN_KEY_ENV in item:
        human = _to_bool(item.get(HUMAN_KEY_ENV))
    else:
        hk = _find_key_ci(
            item,
            candidates=["human_rating", "human", "human_score"],
            fallback_contains=["human"],
        )
        human = _to_bool(item.get(hk)) if hk else None

    # gemini
    if GEMINI_KEY_ENV and GEMINI_KEY_ENV in item:
        gemini = _to_bool(item.get(GEMINI_KEY_ENV))
    else:
        gk = _find_key_ci(
            item,
            candidates=[
                "gemini_as_autorater_rating",
                "gemini_autorater",
                "gemini_as_autorater",
                "gemini",
            ],
            fallback_contains=["gemini"],
        )
        gemini = _to_bool(item.get(gk)) if gk else None

    # gpt
    if GPT_KEY_ENV and GPT_KEY_ENV in item:
        gpt = _to_bool(item.get(GPT_KEY_ENV))
    else:
        pk = _find_key_ci(
            item,
            candidates=[
                "gpt_as_autorater_rating",
                "gpt_autorater",
                "gpt_as_autorater",
                "gpt",
            ],
            fallback_contains=["gpt"],
        )
        gpt = _to_bool(item.get(pk)) if pk else None

    return human, gemini, gpt


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


def iter_criteria_from_rubric(
    rubric: Any,
) -> Iterable[Tuple[Optional[bool], Optional[bool], Optional[bool]]]:
    """Iterate (human, gemini, gpt) tuples from the rubric JSON."""
    for item in _iter_items_from_any_rubric_shape(rubric):
        human, gemini, gpt = _map_criterion_keys(item)
        if human is None:
            continue
        yield (human, gemini, gpt)


# =========================
# Core aggregation
# =========================
def aggregate_misalignment_for_view(view_id: str) -> Dict[str, Any]:
    """
    Aggregate over ALL criteria across ALL tasks in the view:
      - Denominator = total # of criteria with a valid human rating (across the view)
      - Misalignment counts:
          * +1 if autorater present and (autorater != human)
          * If autorater is missing on a criterion, skip that comparison.
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
        for human, gemini, gpt in iter_criteria_from_rubric(rubric_raw) or []:
            total_criteria += 1
            parsed_here += 1
            if gemini is not None and (human != gemini):
                mis_gemini += 1
            if gpt is not None and (human != gpt):
                mis_gpt += 1

        # Light sampling to help with debugging
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
