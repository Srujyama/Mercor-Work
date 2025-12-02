#!/usr/bin/env python3
"""
Overall Pass Rate by Weight: GPT Autoraters on Gemini Responses

- Columns:
    - GPT Autorater - Gemini 2.5 Response Summary
    - GPT5 Autorater - Gemini 3.0 Response Summary

- For each criterion in those summaries:
    - Look at "weight" (e.g., "Primary objective(s)", "Not primary objective")
    - Look at "autorating" (true/false)

- A criterion "passes" if:
    - autorating == true

- We group by:
    - autorater (GPT Autorater vs GPT5 Autorater)
    - weight ("Primary objective(s)" vs "Not primary objective")

- Pass rate = (# with autorating == true / total criteria for that weight & model) * 100

- Output:
    - overall_passrate_gpt_autoraters_by_weight.png
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
    format="%(asctime)s [%(LEVELNAME)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------- Config --------------------------
VALID_DOMAINS = ["education", "hle"]

# Airtable fields (two columns → two models), UPDATED
GPT_AUTORATER_COL = "GPT Autorater - Gemini 2.5 Response Summary"
GPT5_AUTORATER_COL = "GPT5 Autorater - Gemini 3.0 Response Summary"

AUTORATER_GPT = "Gemini 2.5 response"
AUTORATER_GPT5 = "Gemini 3.0 response"

# Weight categories (expected)
WEIGHT_PRIMARY = "Primary objective(s)"
WEIGHT_NOT_PRIMARY = "Not primary objective"
KNOWN_WEIGHTS = [WEIGHT_PRIMARY, WEIGHT_NOT_PRIMARY]


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


def extract_pass_by_weight(summary_json: Any) -> List[Tuple[str, bool]]:
    """
    Given a JSON value from a '...Response Summary' field, return a list of
    (weight, passed) tuples for all criterion instances.

    Expected structure: list of objects like:
        {
          "autorating": true/false,
          "weight": "Primary objective(s)" | "Not primary objective",
          ...
        }

    A criterion "passes" if autorating == true (we ignore human_rating here).
    Only weight values in KNOWN_WEIGHTS are included.
    """
    out: List[Tuple[str, bool]] = []

    if summary_json is None:
        return out

    if not isinstance(summary_json, list):
        # If structure is different, bail gracefully
        return out

    for crit in summary_json:
        if not isinstance(crit, dict):
            continue

        autorating = crit.get("autorating", None)
        weight = crit.get("weight", None)

        if weight not in KNOWN_WEIGHTS:
            # Skip unexpected weights; adjust if you want to bucket as "Other"
            continue

        if not isinstance(autorating, bool):
            continue

        passed = bool(autorating)  # True means pass, False means fail
        out.append((weight, passed))

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


def compute_stats_by_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pass rate and 95% CI for each autorater and weight.

    Groups:
      - autorater: AUTORATER_GPT vs AUTORATER_GPT5
      - weight: Primary objective(s), Not primary objective

    Pass definition:
      - autorating == true

    Returns a DataFrame with columns: model, weight, mean, ci, n
    """
    rows: List[Dict[str, Any]] = []

    # Collect 0/1 values per autorater and weight
    data: Dict[str, Dict[str, List[float]]] = {
        AUTORATER_GPT: {w: [] for w in KNOWN_WEIGHTS},
        AUTORATER_GPT5: {w: [] for w in KNOWN_WEIGHTS},
    }

    for _, r in df.iterrows():
        gpt_summary_raw = r.get(GPT_AUTORATER_COL)
        gpt5_summary_raw = r.get(GPT5_AUTORATER_COL)

        gpt_summary = _safe_json(gpt_summary_raw)
        gpt5_summary = _safe_json(gpt5_summary_raw)

        # GPT Autorater (on 2.5 responses)
        for weight, passed in extract_pass_by_weight(gpt_summary):
            data[AUTORATER_GPT][weight].append(1.0 if passed else 0.0)

        # GPT5 Autorater (on 3.0 responses)
        for weight, passed in extract_pass_by_weight(gpt5_summary):
            data[AUTORATER_GPT5][weight].append(1.0 if passed else 0.0)

    for model_name, per_weight in data.items():
        for weight in KNOWN_WEIGHTS:
            vals = per_weight.get(weight, [])
            if not vals:
                logger.warning(
                    "No criterion data for '%s' / weight '%s'", model_name, weight
                )
                continue

            mean_0_1, ci_0_1 = compute_mean_and_ci(vals)
            mean_pct = mean_0_1 * 100.0
            ci_pct = ci_0_1 * 100.0

            rows.append(
                {
                    "model": model_name,
                    "weight": weight,
                    "mean": mean_pct,
                    "ci": ci_pct,
                    "n": len(vals),
                }
            )

    logger.info("Computed stats rows: %d", len(rows))
    return pd.DataFrame(rows)


# ----------------------- Plotting ------------------------
def plot_stats_by_weight(stats_df: pd.DataFrame, output_path: str):
    """
    Plot success rates with:
      - x-axis: weight ("Primary objective(s)", "Not primary objective")
      - Two bars per weight: Gemini 2.5 response, Gemini 3.0 response
    """
    if stats_df.empty:
        logger.warning("No stats to plot. Skipping.")
        return

    weights = KNOWN_WEIGHTS
    models = [AUTORATER_GPT, AUTORATER_GPT5]

    # Pivot to convenient structure: (model, weight) -> stats
    stats_idx = stats_df.set_index(["weight", "model"])

    means = {m: [] for m in models}
    cis = {m: [] for m in models}
    ns = {m: [] for m in models}

    for w in weights:
        for m in models:
            row = stats_idx.loc[(w, m)] if (w, m) in stats_idx.index else None
            if row is None or row.isna().all():
                means[m].append(float("nan"))
                cis[m].append(float("nan"))
                ns[m].append(0)
            else:
                means[m].append(row["mean"])
                cis[m].append(row["ci"])
                ns[m].append(row["n"])

    x = range(len(weights))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    # Colors
    gpt_color = (0.397, 0.743, 0.567)
    gpt5_color = (0.378, 0.555, 0.936)

    bars_gpt = ax.bar(
        [xx - width / 2 for xx in x],
        means[AUTORATER_GPT],
        width,
        yerr=cis[AUTORATER_GPT],
        capsize=5,
        label=AUTORATER_GPT,
        color=gpt_color,
    )

    bars_gpt5 = ax.bar(
        [xx + width / 2 for xx in x],
        means[AUTORATER_GPT5],
        width,
        yerr=cis[AUTORATER_GPT5],
        capsize=5,
        label=AUTORATER_GPT5,
        color=gpt5_color,
    )

    # Label bars with percentages above CI
    def label_bars(bars, model_name):
        model_means = means[model_name]
        model_cis = cis[model_name]
        for idx, bar in enumerate(bars):
            height = model_means[idx]
            ci_val = (
                model_cis[idx]
                if idx < len(model_cis) and not pd.isna(model_cis[idx])
                else 0.0
            )
            if pd.isna(height):
                continue
            y = height + ci_val + 1.0
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

    label_bars(bars_gpt, AUTORATER_GPT)
    label_bars(bars_gpt5, AUTORATER_GPT5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(weights, rotation=0)  # horizontal labels
    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Weight category")
    ax.set_ylim(0, 100)

    # Title + subtitle
    ax.set_title("Using GPT-5 Autorater", fontsize=10, pad=10)
    fig.suptitle("Success Rates by Weight Category", fontsize=14, y=0.98)

    ax.legend()

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

    logger.info("Saved success rate chart to %s", output_path)


# ----------------------- Main ---------------------------
def main():
    df = fetch_airtable_dataframe()  # use ALL rows from the evalset view
    stats_df = compute_stats_by_weight(df)
    plot_stats_by_weight(stats_df, "overall_passrate_gpt_autoraters_by_weight.png")
    logger.info("Done.")


if __name__ == "__main__":
    main()
