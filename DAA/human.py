#!/usr/bin/env python3
"""
Overall Agreement Chart: Gemini 3.0 Autorater vs Gemini 2.5 Autorater
(on Gemini 2.5 responses)

- No separation by domain (education vs HLE) or criterion type.
  We combine:
    - All domains (but still restricted to education + HLE rows)
    - All criterion types

- X-axis: Autorater
    - Gemini 3.0 Autorater
    - Gemini 2.5 Autorater

- Y-axis: Pass Rate (%), out of 100
    - A criterion "passes" if autorating == human_rating (both booleans)
    - For each autorater, pass rate = (#pass / total) * 100

- Data sources (per record):
    - Gemini 3.0 Autorater - Gemini 2.5 Response Summary
    - Gemini 2.5 Autorater - Gemini 2.5 Response Summary

- Output:
    - overall_agreement_gemini_autoraters_on_2_5.png
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from pyairtable import Api

# Try to use scipy for t-based CI; fall back to z=1.96 if unavailable
try:
    from scipy import stats

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ----------------------- Load .env -----------------------
load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "appgeueGlH9mCUTvu")
AIRTABLE_TABLE_ID = os.getenv("AIRTABLE_TABLE", "tblfy3EPxl1PHvKV7")

# Fixed evalset view (per instructions)
AIRTABLE_VIEW_EVALSET = os.getenv("AIRTABLE_VIEW_EVALSET", "viwAby4j1DDolKolH")

if not AIRTABLE_API_KEY:
    raise RuntimeError("Missing AIRTABLE_API_KEY in .env")

# ----------------------- Logging -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------- Config --------------------------
VALID_DOMAINS = ["education", "hle"]

# Airtable fields (two columns → two bars), UPDATED
GEMINI3_AUTORATER_COL = "Gemini 3.0 Autorater - Gemini 2.5 Response Summary"
GEMINI25_AUTORATER_COL = "Gemini 2.5 Autorater - Gemini 2.5 Response Summary"

AUTORATER_G3 = "Gemini 3.0 Autorater"
AUTORATER_G25 = "Gemini 2.5 Autorater"


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

    rows = []
    for rec in records:
        rows.append(rec.get("fields", {}))

    df = pd.DataFrame(rows)
    logger.info(
        "Fetched %d records from view '%s'",
        len(df),
        AIRTABLE_VIEW_EVALSET,
    )
    logger.info("Available columns from Airtable: %s", df.columns.tolist())
    return df


# ----------------------- JSON helpers --------------------
def _safe_json(s: Any) -> Any:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_pass_flags(summary_json: Any) -> List[bool]:
    """
    Given a JSON value from a '...Response Summary' field, return a flat list of
    pass flags (True/False) for all criterion instances, regardless of type.

    Expected structure: list of objects like:
        {
          "autorating": true/false,
          "human_rating": true/false,
          "criterion_type": [...],
          ...
        }

    A criterion "passes" if autorating == human_rating (both present and bool).
    We do NOT separate by criterion_type; all criteria are pooled.
    """
    out: List[bool] = []

    if summary_json is None:
        return out

    if not isinstance(summary_json, list):
        # If structure is different, bail gracefully
        return out

    for crit in summary_json:
        if not isinstance(crit, dict):
            continue

        autorating = crit.get("autorating", None)
        human_rating = crit.get("human_rating", None)

        # We require both to be explicitly True/False
        if not isinstance(autorating, bool) or not isinstance(human_rating, bool):
            continue

        passed = autorating == human_rating
        out.append(passed)

    return out


# ----------------------- Stats helpers -------------------
def compute_mean_and_ci(
    values: List[float], alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute mean and 95% confidence interval half-width for a 1D numeric list.

    Returns:
        (mean, ci_half_width)

    If n <= 1, CI is 0.0.
    """
    if not values:
        return float("nan"), float("nan")

    vals = pd.Series(values, dtype=float)
    n = len(vals)
    mean = vals.mean()

    if n == 1:
        return mean, 0.0

    sd = vals.std(ddof=1)
    se = sd / math.sqrt(n)

    if HAS_SCIPY:
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    else:
        t_crit = 1.96

    ci_half_width = t_crit * se
    return mean, ci_half_width


def compute_overall_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall pass rate and 95% CI for each autorater, combining:
      - All (education + HLE) records
      - All criterion types

    Returns a DataFrame with columns: model, mean, ci, n
    """
    rows: List[Dict[str, Any]] = []

    # Collect 0/1 values per autorater
    data: Dict[str, List[float]] = {
        AUTORATER_G3: [],
        AUTORATER_G25: [],
    }

    for _, r in df.iterrows():
        g3_summary_raw = r.get(GEMINI3_AUTORATER_COL)
        g25_summary_raw = r.get(GEMINI25_AUTORATER_COL)

        g3_summary = _safe_json(g3_summary_raw)
        g25_summary = _safe_json(g25_summary_raw)

        # Gemini 3.0 Autorater
        for passed in extract_pass_flags(g3_summary):
            data[AUTORATER_G3].append(1.0 if passed else 0.0)

        # Gemini 2.5 Autorater
        for passed in extract_pass_flags(g25_summary):
            data[AUTORATER_G25].append(1.0 if passed else 0.0)

    for model_name, vals in data.items():
        if not vals:
            logger.warning("No criterion data for '%s'", model_name)
            continue

        mean_0_1, ci_0_1 = compute_mean_and_ci(vals)
        mean_pct = mean_0_1 * 100.0
        ci_pct = ci_0_1 * 100.0

        rows.append(
            {
                "model": model_name,
                "mean": mean_pct,
                "ci": ci_pct,
                "n": len(vals),
            }
        )

    logger.info("Computed overall stats rows: %d", len(rows))
    return pd.DataFrame(rows)


# ----------------------- Plotting ------------------------
def plot_overall_stats(stats_df: pd.DataFrame, output_path: str):
    """
    Plot overall agreement chart with x-axis = autorater,
    and two bars: Gemini 3.0 Autorater, Gemini 2.5 Autorater.
    """
    if stats_df.empty:
        logger.warning("No stats to plot. Skipping.")
        return

    models = [AUTORATER_G3, AUTORATER_G25]
    stats_df = stats_df.set_index("model").reindex(models)

    means = stats_df["mean"].tolist()
    cis = stats_df["ci"].tolist()

    x = range(len(models))

    # Colors
    g3_color = (0.397, 0.743, 0.567)
    g25_color = (0.378, 0.555, 0.936)
    colors = [g3_color, g25_color]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        x,
        means,
        yerr=cis,
        capsize=5,
        color=colors,
    )

    # Label bars with percentages above CI
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        ci = cis[idx] if idx < len(cis) and not pd.isna(cis[idx]) else 0.0
        y = height + ci + 1.0
        if y > 100:
            y = 99.0

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("Agreement Rate Autorater vs. Human (%)")
    ax.set_xlabel("Autorater")
    ax.set_ylim(0, 100)
    ax.set_title("Agreement Autorater vs. Human on Gemini 2.5 responses")

    # 95% CI box
    ax.text(
        0.99,
        0.02,
        "95% CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.8,
        ),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved overall agreement chart to %s", output_path)


# ----------------------- Main ---------------------------
def main():
    df = fetch_airtable_dataframe()  # use ALL rows from the evalset view
    stats_df = compute_overall_stats(df)
    plot_overall_stats(stats_df, "overall_agreement_gemini_autoraters_on_2_5.png")
    logger.info("Done.")


if __name__ == "__main__":
    main()
