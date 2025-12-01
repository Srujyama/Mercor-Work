#!/usr/bin/env python3
"""
Primary vs Not Primary Agreement Chart:
Gemini 3.0 Autorater vs Gemini 2.5 Autorater (on Gemini 2.5 responses)

- View: AIRTABLE_VIEW_EVALSET (default: viwAby4j1DDolKolH)
- Columns:
    - Gemini 3.0 Autorater - Gemini 2.5 Response Summary
    - Gemini 2.5 Autorater - Gemini 2.5 Response Summary

- We split criteria into:
    - "Primary objective(s)"
    - "Not primary objective"

- For each group and each autorater, we compute:
    - Agreement pass rate = (#passes / total) * 100
      where a "pass" is:

      * If autorating & human_rating are both booleans:
            pass = (autorating == human_rating)
      * Else if only autorating is a boolean:
            pass = autorating
      * Else: ignore that criterion

- Output:
    - primary_vs_notprimary_gemini_autoraters_on_2_5.png
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

# Fixed evalset view
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
# NEW: use autorater-on-2.5 columns
GEMINI3_AUTORATER_COL = "Gemini 3.0 Autorater - Gemini 2.5 Response Summary"
GEMINI25_AUTORATER_COL = "Gemini 2.5 Autorater - Gemini 2.5 Response Summary"

# Values in the "weight" field
WEIGHT_PRIMARY = "Primary objective(s)"
WEIGHT_NOT_PRIMARY = "Not primary objective"

# Groups (x-axis order)
GROUP_LABELS = [WEIGHT_PRIMARY, WEIGHT_NOT_PRIMARY]

# Model/autorater display names
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


def extract_pass_flags_with_weight(summary_json: Any) -> List[Tuple[bool, str]]:
    """
    Given a JSON value from a '...Response Summary' field, return a list of
    (pass_bool, weight_str) for all criterion instances.

    Logic for pass_bool:
      - If both autorating and human_rating are booleans:
            pass = (autorating == human_rating)
      - Else if only autorating is a boolean and human_rating missing:
            pass = autorating
      - Otherwise: skip

    weight_str is taken from the 'weight' field if it's a string, else "".
    """
    out: List[Tuple[bool, str]] = []

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
        weight = crit.get("weight", "")

        if not isinstance(weight, str):
            weight = ""

        passed: bool | None = None

        # Case 1: full agreement check available
        if isinstance(autorating, bool) and isinstance(human_rating, bool):
            passed = autorating == human_rating

        # Case 2: only autorating present -> treat it directly as pass/fail
        elif isinstance(autorating, bool) and human_rating is None:
            passed = autorating

        if passed is None:
            continue

        out.append((passed, weight))

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


def compute_grouped_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pass rates and CIs separately for:

        - Primary objective(s)
        - Not primary objective

    For each group and each autorater (Gemini 3.0 Autorater, Gemini 2.5 Autorater).

    Returns a DataFrame with columns:
        group_label, model, mean, ci, n
    """
    rows: List[Dict[str, Any]] = []

    # Structure:
    # data[group_label][autorater] = [0/1, 0/1, ...]
    data: Dict[str, Dict[str, List[float]]] = {
        WEIGHT_PRIMARY: {AUTORATER_G3: [], AUTORATER_G25: []},
        WEIGHT_NOT_PRIMARY: {AUTORATER_G3: [], AUTORATER_G25: []},
    }

    for _, r in df.iterrows():
        g3_summary_raw = r.get(GEMINI3_AUTORATER_COL)
        g25_summary_raw = r.get(GEMINI25_AUTORATER_COL)

        g3_summary = _safe_json(g3_summary_raw)
        g25_summary = _safe_json(g25_summary_raw)

        # Gemini 3.0 Autorater
        for passed, weight in extract_pass_flags_with_weight(g3_summary):
            if weight in data:
                data[weight][AUTORATER_G3].append(1.0 if passed else 0.0)

        # Gemini 2.5 Autorater
        for passed, weight in extract_pass_flags_with_weight(g25_summary):
            if weight in data:
                data[weight][AUTORATER_G25].append(1.0 if passed else 0.0)

    for group_label in GROUP_LABELS:
        group_dict = data.get(group_label, {})
        for autorater_name in [AUTORATER_G3, AUTORATER_G25]:
            vals = group_dict.get(autorater_name, [])
            logger.info(
                "Group %s / Autorater %s: %d criterion instances",
                group_label,
                autorater_name,
                len(vals),
            )

            if not vals:
                # If a particular combination has no data, skip
                continue

            mean_0_1, ci_0_1 = compute_mean_and_ci(vals)
            mean_pct = mean_0_1 * 100.0
            ci_pct = ci_0_1 * 100.0

            rows.append(
                {
                    "group_label": group_label,
                    "model": autorater_name,
                    "mean": mean_pct,
                    "ci": ci_pct,
                    "n": len(vals),
                }
            )

    logger.info("Computed grouped stats rows: %d", len(rows))
    return pd.DataFrame(rows)


# ----------------------- Plotting ------------------------
def plot_grouped_stats(stats_df: pd.DataFrame, output_path: str):
    """
    Plot grouped agreement chart:

    X-axis = [Primary objective(s), Not primary objective]
    For each x:
        - Gemini 3.0 Autorater bar
        - Gemini 2.5 Autorater bar

    Legend shows which color is which autorater.
    """
    if stats_df.empty:
        logger.warning("No stats to plot. Skipping.")
        return

    # Pivot to get 2D structure: index=group_label, columns=model
    mean_pivot = stats_df.pivot(index="group_label", columns="model", values="mean")
    ci_pivot = stats_df.pivot(index="group_label", columns="model", values="ci")

    # Ensure consistent order of groups
    group_labels = [g for g in GROUP_LABELS if g in mean_pivot.index]
    mean_pivot = mean_pivot.reindex(group_labels)
    ci_pivot = ci_pivot.reindex(group_labels)

    models = [AUTORATER_G3, AUTORATER_G25]

    x = range(len(group_labels))
    width = 0.35

    # Colors for legend
    g3_color = (0.397, 0.743, 0.567)
    g25_color = (0.378, 0.555, 0.936)

    g3_means = mean_pivot.get(AUTORATER_G3)
    g25_means = mean_pivot.get(AUTORATER_G25)
    g3_ci = ci_pivot.get(AUTORATER_G3)
    g25_ci = ci_pivot.get(AUTORATER_G25)

    fig, ax = plt.subplots(figsize=(8, 5))

    g3_bars = None
    if g3_means is not None:
        g3_bars = ax.bar(
            [i - width / 2 for i in x],
            g3_means,
            width,
            yerr=g3_ci if g3_ci is not None else None,
            capsize=5,
            label=AUTORATER_G3,
            color=g3_color,
        )

    g25_bars = None
    if g25_means is not None:
        g25_bars = ax.bar(
            [i + width / 2 for i in x],
            g25_means,
            width,
            yerr=g25_ci if g25_ci is not None else None,
            capsize=5,
            label=AUTORATER_G25,
            color=g25_color,
        )

    # Label bars with percentages above CI
    def add_value_labels(bar_container, ci_series):
        if bar_container is None or ci_series is None:
            return
        ci_list = list(ci_series.values)

        for idx, bar in enumerate(bar_container):
            height = bar.get_height()
            if pd.isna(height):
                continue
            ci = ci_list[idx] if idx < len(ci_list) and pd.notna(ci_list[idx]) else 0.0
            y = height + ci + 1.0
            if y > 100:
                y = 99.0

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="black",
            )

    add_value_labels(g3_bars, g3_ci)
    add_value_labels(g25_bars, g25_ci)

    ax.set_xticks(list(x))
    ax.set_xticklabels(group_labels, rotation=10)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_xlabel("Criterion Weight")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Agreement by Criterion Weight – Gemini 3.0 Autorater vs Gemini 2.5 Autorater\n"
        "(on Gemini 2.5 responses, Primary vs Not primary objectives)"
    )

    # Legend key
    # Legend key: place it outside the plot on the right
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.0,
        title="Autorater",
    )

    plt.tight_layout()

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

    logger.info("Saved grouped primary vs not-primary chart to %s", output_path)


# ----------------------- Main ---------------------------
def main():
    df = fetch_airtable_dataframe()
    stats_df = compute_grouped_stats(df)
    logger.info("Final grouped stats dataframe:\n%s", stats_df)
    plot_grouped_stats(
        stats_df,
        "primary_vs_notprimary_gemini_autoraters_on_2_5.png",
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
