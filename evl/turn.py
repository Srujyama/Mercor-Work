#!/usr/bin/env python3
"""
Comparison Bar Charts: GPT-5 vs Gemini 3.0 (Single-turn vs Multi-turn)

- Reads data from Airtable (view: AIRTABLE_VIEW_Sort)
- Auto-detects:
    - domain column (prefers Domain1, Domain Name, Domain)
    - interaction column (prefers Interaction Type)
    - Gemini score column: Gemini Autorater - Gemini 3.0 Response Score
    - GPT5 score column:   GPT5 Autorater - Gemini 3.0 Response Score
- Infers domain values (education / HLE) from text
- Groups by Interaction Type ('Single-turn', 'Multi-turn')
- Computes mean (%) and 95% confidence intervals
- Outputs:
    - education_chart.png
    - hle_chart.png
"""

import logging
import math
import os

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
VALID_INTERACTIONS = ["single-turn", "multi-turn"]


# ----------------------- Helpers -------------------------
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


def infer_columns(df: pd.DataFrame):
    """
    Infer the best-guess column names for:
      - domain
      - interaction type
      - Gemini score (3.0)
      - GPT5 score (3.0)
    using heuristics tailored to your schema.
    """
    cols = list(df.columns)

    # --- Domain column: prefer Domain1 > Domain Name > Domain
    domain_candidates = ["Domain1", "Domain Name", "Domain"]
    domain_col = None
    for name in domain_candidates:
        if name in df.columns:
            domain_col = name
            break

    # --- Interaction column: prefer 'Interaction Type'
    interaction_col = None
    if "Interaction Type" in df.columns:
        interaction_col = "Interaction Type"
    else:
        # Fallback: any column containing 'interaction' or 'turn'
        for c in cols:
            lc = c.lower()
            if "interaction" in lc or "turn" in lc:
                interaction_col = c
                break

    # --- Gemini score: prefer exact 3.0 field
    gemini_col = None
    if "Gemini Autorater - Gemini 3.0 Response Score" in df.columns:
        gemini_col = "Gemini Autorater - Gemini 3.0 Response Score"
    else:
        # fallback: anything that looks like a Gemini autorater score
        for c in cols:
            lc = c.lower()
            if "gemini autorater" in lc and "response score" in lc:
                gemini_col = c
                break

    # --- GPT5 score: prefer exact 3.0 field
    gpt5_col = None
    if "GPT5 Autorater - Gemini 3.0 Response Score" in df.columns:
        gpt5_col = "GPT5 Autorater - Gemini 3.0 Response Score"
    else:
        for c in cols:
            lc = c.lower()
            if "gpt5 autorater" in lc and "response score" in lc:
                gpt5_col = c
                break

    logger.info("Inferred columns:")
    logger.info("  Domain column      → %r", domain_col)
    logger.info("  Interaction column → %r", interaction_col)
    logger.info("  Gemini score col   → %r", gemini_col)
    logger.info("  GPT5 score col     → %r", gpt5_col)

    if domain_col is None:
        raise KeyError(
            "Could not infer a domain column (no Domain1/Domain Name/Domain). "
            f"Available columns: {cols}"
        )
    if interaction_col is None:
        raise KeyError(
            "Could not infer an interaction column "
            "(no 'Interaction Type' or column with 'interaction'/'turn'). "
            f"Available columns: {cols}"
        )
    if gemini_col is None and gpt5_col is None:
        raise KeyError(
            "Could not infer Gemini/GPT5 3.0 score columns. "
            "Looked for 'Gemini Autorater - Gemini 3.0 Response Score' and "
            "'GPT5 Autorater - Gemini 3.0 Response Score'. "
            f"Available columns: {cols}"
        )

    return domain_col, interaction_col, gemini_col, gpt5_col


def normalize_domain(value: str) -> str | None:
    """
    Map arbitrary domain text to 'education' or 'hle' based on substrings.
    Returns None if it doesn't look like either.
    """
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None

    # Add more heuristics here if needed
    if "hle" in v:
        return "hle"
    if "educ" in v:
        return "education"

    return None


def normalize_interaction(value: str) -> str | None:
    """
    Normalize interaction type into 'single-turn' or 'multi-turn',
    based on substring matches.
    """
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None

    if "single" in v:
        return "single-turn"
    if "multi" in v:
        return "multi-turn"

    return None


def clean_and_filter(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, str | None, str | None]:
    """
    Infer relevant columns, normalize domain/interaction values,
    and ensure numeric percentages.

    Returns:
        df_filtered, domain_col, interaction_col, gemini_col, gpt5_col
    """
    domain_col, interaction_col, gemini_col, gpt5_col = infer_columns(df)

    df = df.copy()

    # Normalize domain/interaction into canonical labels
    df["_domain_norm"] = df[domain_col].apply(normalize_domain)
    df["_interaction_norm"] = df[interaction_col].apply(normalize_interaction)

    before = len(df)
    df = df[df["_domain_norm"].isin(VALID_DOMAINS)]
    df = df[df["_interaction_norm"].isin(VALID_INTERACTIONS)]
    after = len(df)

    logger.info(
        "After domain/interaction normalization & filtering, %d → %d records remain",
        before,
        after,
    )

    # Coerce score columns to numeric (floats)
    for col_name in [gemini_col, gpt5_col]:
        if col_name is None:
            continue
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        else:
            logger.warning(
                "Score column '%s' is missing; its values will be NaN.", col_name
            )
            df[col_name] = float("nan")

    score_cols_present = [c for c in [gemini_col, gpt5_col] if c is not None]
    if score_cols_present:
        before_drop = len(df)
        df = df.dropna(subset=score_cols_present, how="all")
        logger.info(
            "After dropping rows with all NaN scores, %d → %d records remain",
            before_drop,
            len(df),
        )

    return df, "_domain_norm", "_interaction_norm", gemini_col, gpt5_col


def compute_mean_and_ci(values: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """
    Compute mean and 95% confidence interval half-width for a 1D numeric Series.

    Returns:
        (mean, ci_half_width)

    If n <= 1, CI is 0.0.
    """
    vals = values.dropna().astype(float)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
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


def compute_stats_for_domain(
    df: pd.DataFrame,
    domain: str,
    domain_col_norm: str,
    interaction_col_norm: str,
    gemini_col: str | None,
    gpt5_col: str | None,
) -> pd.DataFrame:
    """
    For a given domain ('education' or 'hle'), compute mean (%) and 95% CI
    for each (Interaction Type, model).

    Returns a DataFrame with:
        - interaction_type
        - model (GPT-5 / Gemini)
        - mean
        - ci
    """
    sub = df[df[domain_col_norm] == domain].copy()
    logger.info("For domain '%s', used %d records for aggregation.", domain, len(sub))

    rows = []
    model_map: dict[str, str] = {}
    if gemini_col is not None:
        model_map[gemini_col] = "Gemini 3.0"
    if gpt5_col is not None:
        model_map[gpt5_col] = "GPT-5"

    for interaction in VALID_INTERACTIONS:
        inter_df = sub[sub[interaction_col_norm] == interaction]

        for col, model_name in model_map.items():
            if col not in inter_df.columns:
                continue
            vals = inter_df[col].dropna()
            if len(vals) == 0:
                continue

            mean, ci = compute_mean_and_ci(vals)
            rows.append(
                {
                    "interaction_type": interaction.title(),  # “Single-Turn”, “Multi-Turn”
                    "model": model_name,
                    "mean": mean,
                    "ci": ci,
                    "n": len(vals),
                }
            )

    logger.info("Domain '%s' stats rows: %d", domain, len(rows))
    return pd.DataFrame(rows)


def plot_domain_stats(stats_df: pd.DataFrame, domain: str, output_path: str):
    """
    Given stats_df with columns [interaction_type, model, mean, ci],
    produce a bar chart and save to output_path.

    Visual choices:
      - y-axis fixed to 0–40%
      - x-axis label: 'Interaction Type'
      - bottom-right label box: '95% CI'
      - GPT-5 color:  (0.397, 0.743, 0.567)
      - Gemini color: (0.378, 0.555, 0.936)
    """
    if stats_df.empty:
        logger.warning("No stats to plot for domain '%s'. Skipping.", domain)
        return

    # Pivot for easier plotting
    mean_pivot = stats_df.pivot(
        index="interaction_type", columns="model", values="mean"
    )
    ci_pivot = stats_df.pivot(index="interaction_type", columns="model", values="ci")

    ordered_labels = [i.title() for i in VALID_INTERACTIONS]
    mean_pivot = mean_pivot.reindex(ordered_labels)
    ci_pivot = ci_pivot.reindex(ordered_labels)

    x = range(len(ordered_labels))
    width = 0.35

    gpt5_means = mean_pivot.get("GPT-5")
    gemini_means = mean_pivot.get("Gemini 3.0")
    gpt5_ci = ci_pivot.get("GPT-5")
    gemini_ci = ci_pivot.get("Gemini 3.0")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Custom colors (RGB in [0,1])
    gpt5_color = (0.397, 0.743, 0.567)
    gemini_color = (0.378, 0.555, 0.936)

    # GPT-5 bars
    if gpt5_means is not None:
        ax.bar(
            [i - width / 2 for i in x],
            gpt5_means,
            width,
            yerr=gpt5_ci if gpt5_ci is not None else None,
            capsize=5,
            label="GPT-5",
            color=gpt5_color,
        )

    # Gemini bars
    if gemini_means is not None:
        ax.bar(
            [i + width / 2 for i in x],
            gemini_means,
            width,
            yerr=gemini_ci if gemini_ci is not None else None,
            capsize=5,
            label="Gemini 3.0",
            color=gemini_color,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered_labels)

    # Axis labels and limits
    ax.set_xlabel("Interaction Type")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 40)  # fixed to 40%

    # Title
    if domain == "education":
        title = "Single-Turn vs Multi-Turn – Education"
    elif domain == "hle":
        title = "Single-Turn vs Multi-Turn – HLE"
    else:
        title = f"Single-Turn vs Multi-Turn – {domain}"
    ax.set_title(title)

    # Legend with title "Graded by"
    legend = ax.legend(loc="upper right", title="Graded by")
    plt.setp(legend.get_title(), fontweight="bold")

    # Small label box in bottom right: "95% CI"
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

    logger.info("Saved chart for domain '%s' to %s", domain, output_path)


# ----------------------- Main -------------------------
def main():
    df = fetch_airtable_dataframe()

    df_clean, dom_norm_col, inter_norm_col, gemini_col, gpt5_col = clean_and_filter(df)

    # Debug: see what domain/interaction values we have after normalization
    logger.info(
        "Normalized domain values: %s", sorted(df_clean[dom_norm_col].dropna().unique())
    )
    logger.info(
        "Normalized interaction values: %s",
        sorted(df_clean[inter_norm_col].dropna().unique()),
    )

    # Education chart
    edu_stats = compute_stats_for_domain(
        df_clean, "education", dom_norm_col, inter_norm_col, gemini_col, gpt5_col
    )
    plot_domain_stats(edu_stats, "education", "education_chart.png")

    # HLE chart
    hle_stats = compute_stats_for_domain(
        df_clean, "hle", dom_norm_col, inter_norm_col, gemini_col, gpt5_col
    )
    plot_domain_stats(hle_stats, "hle", "hle_chart.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()
