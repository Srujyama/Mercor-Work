#!/usr/bin/env python3
"""
Rubric Weight Distribution Pie Chart

Goal:
    Go criterion by criterion across all tasks (rows) and count how many rubric
    entries are:
        - "Primary objective(s)"
        - "Not primary objective"

    Then, make a pie chart with the percentage out of 100 for each.

Env-configurable:
    - AIRTABLE_API_KEY
    - AIRTABLE_BASE_ID          (default: appgeueGlH9mCUTvu)
    - AIRTABLE_TABLE            (default: tblfy3EPxl1PHvKV7)
    - AIRTABLE_VIEW_EVALSET     (default: viwAby4j1DDolKolH)
    - RUBRIC_JSON_FIELD         (default: "Rubric JSON")

Output:
    - rubric_primary_vs_notprimary_pie.png
"""

import json
import logging
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pyairtable import Api

# ----------------------- Load .env -----------------------
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "appgeueGlH9mCUTvu")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE", "tblfy3EPxl1PHvKV7")

AIRTABLE_VIEW_EVALSET = os.getenv("AIRTABLE_VIEW_EVALSET", "viwAby4j1DDolKolH")

# Name of the rubric JSON field in Airtable
RUBRIC_JSON_FIELD = os.getenv("RUBRIC_JSON_FIELD", "Rubric JSON")

if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY in .env")

# ----------------------- Logging -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------- Config --------------------------
WEIGHT_PRIMARY = "Primary objective(s)"
WEIGHT_NOT_PRIMARY = "Not primary objective"


# ----------------------- Airtable fetch ------------------
def fetch_airtable_dataframe() -> pd.DataFrame:
    """
    Fetch records from Airtable using the configured EVALSET view and
    return as a pandas DataFrame.
    """
    logger.info("Using Airtable view (evalset): %s", AIRTABLE_VIEW_EVALSET)
    api = Api(AIRTABLE_API_KEY)
    table = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

    records = table.all(view=AIRTABLE_VIEW_EVALSET)

    rows: List[Dict[str, Any]] = []
    for rec in records:
        rows.append(rec.get("fields", {}))

    df = pd.DataFrame(rows)
    logger.info(
        "Fetched %d records from view '%s'",
        len(df),
        AIRTABLE_VIEW_EVALSET,
    )
    logger.info("Available columns from Airtable: %s", df.columns.tolist())
    if RUBRIC_JSON_FIELD not in df.columns:
        logger.warning(
            "Rubric JSON field %r not found in columns. "
            "You may need to adjust RUBRIC_JSON_FIELD.",
            RUBRIC_JSON_FIELD,
        )
    else:
        sample = df[RUBRIC_JSON_FIELD].dropna().head(3).tolist()
        logger.info(
            "Sample values from %r (first 3 non-null): %r",
            RUBRIC_JSON_FIELD,
            sample,
        )
    return df


# ----------------------- JSON helpers --------------------
def _safe_json(val: Any) -> Any:
    """
    Safely normalize a value into JSON-like Python structures.

    - If it's already a list/dict, return as-is.
    - If it's a non-empty string, try json.loads.
    - Otherwise, return None.
    """
    if isinstance(val, (list, dict)):
        return val

    if isinstance(val, str) and val.strip():
        try:
            return json.loads(val)
        except Exception:
            logger.debug("Failed to json.loads rubric value: %r", val)
            return None

    return None


def extract_weights_from_rubric(rubric_json: Any) -> List[str]:
    """
    Given a JSON value from the rubric field, extract all 'weight' strings
    for each criterion.

    Actual structure (from Airtable):
        [
          {
            "criterion 1": {
              "description": "...",
              "weight": "...",
              ...
            }
          },
          {
            "criterion 2": {
              "description": "...",
              "weight": "...",
              ...
            }
          },
          ...
        ]

    So for each element (a dict), we take its single value (the inner dict),
    and read 'weight' from there.
    """
    out: List[str] = []

    if rubric_json is None:
        return out

    if not isinstance(rubric_json, list):
        logger.debug("Rubric JSON is not a list: %r", type(rubric_json))
        return out

    for crit_wrapper in rubric_json:
        # Each crit_wrapper should be something like {"criterion 1": {...}}
        if not isinstance(crit_wrapper, dict):
            continue

        if not crit_wrapper:
            continue

        # Take the first (and usually only) value from the wrapper dict
        inner = next(iter(crit_wrapper.values()))
        if not isinstance(inner, dict):
            continue

        weight = inner.get("weight")
        if isinstance(weight, str):
            out.append(weight)

    return out


# ----------------------- Counting ------------------------
def count_primary_vs_not_primary(df: pd.DataFrame) -> Dict[str, int]:
    """
    Iterate over all rows and all rubric criteria and count how many
    are primary vs not primary.

    Returns a dict:
        {
          "Primary objective(s)": count,
          "Not primary objective": count,
        }
    """
    counts = {
        WEIGHT_PRIMARY: 0,
        WEIGHT_NOT_PRIMARY: 0,
    }

    if RUBRIC_JSON_FIELD not in df.columns:
        logger.error(
            "Rubric JSON field %r not in dataframe columns. "
            "No counting will be performed.",
            RUBRIC_JSON_FIELD,
        )
        return counts

    total_criteria = 0

    for _, row in df.iterrows():
        rubric_raw = row.get(RUBRIC_JSON_FIELD)
        rubric_json = _safe_json(rubric_raw)
        weights = extract_weights_from_rubric(rubric_json)

        for w in weights:
            if w == WEIGHT_PRIMARY:
                counts[WEIGHT_PRIMARY] += 1
            elif w == WEIGHT_NOT_PRIMARY:
                counts[WEIGHT_NOT_PRIMARY] += 1

        total_criteria += len(weights)

    logger.info("Total rubric criteria (all weights): %d", total_criteria)
    logger.info(
        "Counts → Primary: %d, Not primary: %d",
        counts[WEIGHT_PRIMARY],
        counts[WEIGHT_NOT_PRIMARY],
    )
    return counts


# ----------------------- Plotting ------------------------
def plot_primary_vs_not_primary_pie(counts: Dict[str, int], output_path: str):
    """
    Create a pie chart showing the percentage of rubric criteria that are
    primary vs not primary.
    """
    primary_count = counts.get(WEIGHT_PRIMARY, 0)
    not_primary_count = counts.get(WEIGHT_NOT_PRIMARY, 0)
    total = primary_count + not_primary_count

    if total == 0:
        logger.warning(
            "No primary/not-primary rubric criteria found. Skipping pie chart."
        )
        return

    labels = ["Primary objective(s)", "Not primary objective"]
    sizes = [primary_count, not_primary_count]

    # Colors (roughly consistent with your other charts)
    primary_color = (0.397, 0.743, 0.567)
    not_primary_color = (0.378, 0.555, 0.936)
    colors = [primary_color, not_primary_color]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        textprops={"fontsize": 10},
    )

    ax.set_title(
        "Rubric Criteria by Weight\n"
        "Primary vs Not primary (percentage of all rubric criteria)"
    )

    # Equal aspect ratio → pie is a circle.
    ax.axis("equal")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved rubric primary vs not-primary pie chart to %s", output_path)


# ----------------------- Main ---------------------------
def main():
    df = fetch_airtable_dataframe()
    counts = count_primary_vs_not_primary(df)
    plot_primary_vs_not_primary_pie(counts, "rubric_primary_vs_notprimary_pie.png")
    logger.info("Done.")


if __name__ == "__main__":
    main()
