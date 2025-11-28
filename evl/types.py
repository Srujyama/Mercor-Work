#!/usr/bin/env python3
"""
Criterion-Type Agreement Charts: GPT-5 vs Gemini 3.0

- X-axis: Criterion Type
    - Extraction (recall)
    - Reasoning
    - Compliance
    - Style

- Y-axis: Pass Rate (%), out of 100
    - A criterion "passes" if autorating == True (regardless of human_rating)
    - For each criterion_type, pass rate = (#pass / total) * 100
    - y-axis visually capped at 0–80

- Data sources (per record):
    - Gemini Autorater - Gemini 3.0 Response Summary
    - GPT5 Autorater - Gemini 3.0 Response Summary

- Grouped by domain (education / HLE)

- Outputs:
    - criterion_types_education.png
    - criterion_types_hle.png
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
AIRTABLE_VIEW_SORT = os.getenv("AIRTABLE_VIEW_Sort", "viwbeMBz0CI6eBY5e")

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

CRITERION_TYPES = ["Extraction (recall)", "Reasoning", "Compliance", "Style"]

# Airtable fields
GEMINI_SUMMARY_COL = "Gemini Autorater - Gemini 3.0 Response Summary"
GPT5_SUMMARY_COL = "GPT5 Autorater - Gemini 3.0 Response Summary"


# ----------------------- Airtable fetch ------------------
def fetch_airtable_dataframe() -> pd.DataFrame:
    """
    Fetch records from Airtable using the configured view and
    return as a pandas DataFrame.
    """
    logger.info("Using Airtable view: %s", AIRTABLE_VIEW_SORT)
    api = Api(AIRTABLE_API_KEY)
    table = api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_ID)

    records = table.all(view=AIRTABLE_VIEW_SORT)

    rows = []
    for rec in records:
        rows.append(rec.get("fields", {}))

    df = pd.DataFrame(rows)
    logger.info("Fetched %d records from view '%s'", len(df), AIRTABLE_VIEW_SORT)
    logger.info("Available columns from Airtable: %s", df.columns.tolist())
    return df


# ----------------------- Domain inference ----------------
def infer_domain_column(df: pd.DataFrame) -> str:
    """
    Infer the best-guess domain column:
      - Domain1 > Domain Name > Domain
    """
    cols = list(df.columns)
    domain_candidates = ["Domain1", "Domain Name", "Domain"]
    for name in domain_candidates:
        if name in df.columns:
            logger.info("Using domain column: %r", name)
            return name

    raise KeyError(
        "Could not infer a domain column (no Domain1/Domain Name/Domain). "
        f"Available columns: {cols}"
    )


def normalize_domain(value: Any) -> str | None:
    """
    Map arbitrary domain text to 'education' or 'hle' based on substrings.
    Returns None if it doesn't look like either.
    """
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None

    if "hle" in v:
        return "hle"
    if "educ" in v:
        return "education"

    return None


def clean_and_filter(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Infer domain column, normalize domain values, and keep only
    domains we care about.
    Returns:
        df_filtered, domain_norm_col
    """
    domain_col = infer_domain_column(df)

    df = df.copy()
    df["_domain_norm"] = df[domain_col].apply(normalize_domain)

    before = len(df)
    df = df[df["_domain_norm"].isin(VALID_DOMAINS)]
    after = len(df)

    logger.info(
        "After domain normalization & filtering, %d → %d records remain",
        before,
        after,
    )

    return df, "_domain_norm"


# ----------------------- JSON helpers --------------------
def _safe_json(s: Any) -> Any:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_criterion_passes(
    summary_json: Any,
) -> List[Tuple[str, bool]]:
    """
    Given a JSON value from a '...Response Summary' field, return a list of
    (criterion_type, pass_bool) for all criterion instances.

    Expected structure: list of objects like:
        {
          "autorating": true/false,
          "human_rating": true/false,  # ignored for pass/fail
          "criterion_type": ["Extraction (recall)", "Reasoning", ...],
          ...
        }

    A criterion "passes" if autorating is True.
    We return one entry per (criterion instance, type in its list).
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
        if not isinstance(autorating, bool):
            # If autorating is missing or non-boolean, skip this criterion
            continue

        passed = autorating is True

        c_types = crit.get("criterion_type", [])
        if isinstance(c_types, str):
            c_types = [c_types]
        if not isinstance(c_types, list):
            continue

        for ct in c_types:
            if not isinstance(ct, str):
                continue
            if ct not in CRITERION_TYPES:
                # Ignore unknown types
                continue
            out.append((ct, passed))

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


def compute_stats_for_criterion_types(
    df: pd.DataFrame,
    domain: str,
    domain_col_norm: str,
) -> pd.DataFrame:
    """
    For a given domain, compute pass rate (out of 100) and 95% CI for each
    (criterion_type, model), where model ∈ {Gemini, GPT-5}.

    Pass definition: autorating == True.
    """
    # Filter records to this domain
    sub = df[df[domain_col_norm] == domain].copy()
    logger.info("Domain '%s': %d records for criterion-type stats", domain, len(sub))

    rows: List[Dict[str, Any]] = []

    # Data[model_name][criterion_type] = list of 0/1
    data: Dict[str, Dict[str, List[float]]] = {
        "Gemini 3.0": {ct: [] for ct in CRITERION_TYPES},
        "GPT-5": {ct: [] for ct in CRITERION_TYPES},
    }

    # Iterate records and parse summaries
    for _, r in sub.iterrows():
        g_summary_raw = r.get(GEMINI_SUMMARY_COL)
        p_summary_raw = r.get(GPT5_SUMMARY_COL)

        g_summary = _safe_json(g_summary_raw)
        p_summary = _safe_json(p_summary_raw)

        # Gemini
        for ct, passed in extract_criterion_passes(g_summary):
            data["Gemini 3.0"][ct].append(1.0 if passed else 0.0)

        # GPT-5
        for ct, passed in extract_criterion_passes(p_summary):
            data["GPT-5"][ct].append(1.0 if passed else 0.0)

    # Aggregate per (model, criterion_type)
    for model_name, by_type in data.items():
        for ct in CRITERION_TYPES:
            vals = by_type.get(ct, [])
            if not vals:
                continue

            mean_0_1, ci_0_1 = compute_mean_and_ci(vals)
            mean_pct = mean_0_1 * 100.0
            ci_pct = ci_0_1 * 100.0

            rows.append(
                {
                    "criterion_type": ct,
                    "model": model_name,
                    "mean": mean_pct,
                    "ci": ci_pct,
                    "n": len(vals),
                }
            )

    logger.info("Domain '%s': %d stats rows for criterion types", domain, len(rows))
    return pd.DataFrame(rows)


# ----------------------- Plotting ------------------------
def _plot_bars_with_ci(
    ax,
    mean_pivot: pd.DataFrame,
    ci_pivot: pd.DataFrame,
    index_labels: List[str],
    x_label: str,
    title: str,
):
    """
    Draw GPT-5 and Gemini 3.0 bars with CI and labels.
    Y-axis: 0–80, but mean values are % out of 100.
    """
    x = range(len(index_labels))
    width = 0.35

    gpt5_means = mean_pivot.get("GPT-5")
    gemini_means = mean_pivot.get("Gemini 3.0")
    gpt5_ci = ci_pivot.get("GPT-5")
    gemini_ci = ci_pivot.get("Gemini 3.0")

    # Custom colors (RGB 0–1)
    gpt5_color = (0.397, 0.743, 0.567)
    gemini_color = (0.378, 0.555, 0.936)

    # Bars
    gpt5_bars = None
    if gpt5_means is not None:
        gpt5_bars = ax.bar(
            [i - width / 2 for i in x],
            gpt5_means,
            width,
            yerr=gpt5_ci if gpt5_ci is not None else None,
            capsize=5,
            label="GPT-5",
            color=gpt5_color,
        )

    gemini_bars = None
    if gemini_means is not None:
        gemini_bars = ax.bar(
            [i + width / 2 for i in x],
            gemini_means,
            width,
            yerr=gemini_ci if gemini_ci is not None else None,
            capsize=5,
            label="Gemini 3.0",
            color=gemini_color,
        )

    # Labels ABOVE error bars
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
            # y-axis is capped at 80
            if y > 80:
                y = 79.0

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

    add_value_labels(gpt5_bars, gpt5_ci)
    add_value_labels(gemini_bars, gemini_ci)

    # Axes & styling
    ax.set_xticks(list(x))
    ax.set_xticklabels(index_labels, rotation=15)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 80)
    ax.set_title(title)

    # Legend WITHOUT title (just Gemini / GPT-5)
    ax.legend(loc="upper right")

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


def plot_criterion_type_stats(stats_df: pd.DataFrame, domain: str, output_path: str):
    """
    Plot criterion-type pass-rate chart for a given domain.
    """
    if stats_df.empty:
        logger.warning("No stats to plot for domain '%s'. Skipping.", domain)
        return

    mean_pivot = stats_df.pivot(index="criterion_type", columns="model", values="mean")
    ci_pivot = stats_df.pivot(index="criterion_type", columns="model", values="ci")

    # Ensure order of criterion types
    ordered_labels = CRITERION_TYPES
    mean_pivot = mean_pivot.reindex(ordered_labels)
    ci_pivot = ci_pivot.reindex(ordered_labels)

    fig, ax = plt.subplots(figsize=(9, 6))

    if domain == "education":
        title = "Criterion-Type Pass Rate – Education"
    elif domain == "hle":
        title = "Criterion-Type Pass Rate – HLE"
    else:
        title = f"Criterion-Type Pass Rate – {domain.capitalize()}"

    _plot_bars_with_ci(
        ax,
        mean_pivot,
        ci_pivot,
        ordered_labels,
        x_label="Criterion Type",
        title=title,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved criterion-type chart for domain '%s' to %s", domain, output_path)


# ----------------------- Main ---------------------------
def main():
    df = fetch_airtable_dataframe()

    df_clean, dom_norm_col = clean_and_filter(df)

    logger.info(
        "Normalized domain values: %s",
        sorted(df_clean[dom_norm_col].dropna().unique()),
    )

    # Education chart
    edu_stats = compute_stats_for_criterion_types(df_clean, "education", dom_norm_col)
    plot_criterion_type_stats(edu_stats, "education", "criterion_types_education.png")

    # HLE chart
    hle_stats = compute_stats_for_criterion_types(df_clean, "hle", dom_norm_col)
    plot_criterion_type_stats(hle_stats, "hle", "criterion_types_hle.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()
