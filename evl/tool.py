#!/usr/bin/env python3
"""
Comparison Bar Charts: GPT-5 vs Gemini 3.0

Charts produced:
1. Interaction Type (Education)                -> education_chart.png
2. Interaction Type (HLE)                      -> hle_chart.png
3. Requested Outputs (Education)               -> requested_outputs_education.png
4. Requested Outputs (HLE)                     -> requested_outputs_hle.png
5. Tool Use (Education)                        -> tool_use_education.png
6. Tool Use (HLE)                              -> tool_use_hle.png
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

REQUESTED_OUTPUTS_FIELD = "Requested Outputs"
OUTPUT_GROUPS = ["File / Image", "Text / LaTeX / Code"]

TOOL_USE_FIELD = "Tool Use"
TOOL_GROUPS = ["No Tool Use", "Tool Use (Checked)"]


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


# ----------------------- Column inference ----------------
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
    domain_col = None    # noqa: N806
    for name in domain_candidates:
        if name in df.columns:
            domain_col = name
            break

    # --- Interaction column: prefer 'Interaction Type'
    interaction_col = None   # noqa: N806
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
    gemini_col = None   # noqa: N806
    if "Gemini Autorater - Gemini 3.0 Response Score" in df.columns:
        gemini_col = "Gemini Autorater - Gemini 3.0 Response Score"
    else:
        for c in cols:
            lc = c.lower()
            if "gemini autorater" in lc and "response score" in lc:
                gemini_col = c
                break

    # --- GPT5 score: prefer exact 3.0 field
    gpt5_col = None     # noqa: N806
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


# ----------------------- Normalizers ---------------------
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


# ----------------------- Clean & filter ------------------
def clean_and_filter(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, str, str | None, str | None]:
    """
    Infer relevant columns, normalize domain/interaction values,
    and ensure numeric percentages.

    Returns:
        df_filtered, domain_col_norm, interaction_col_norm, gemini_col, gpt5_col
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


# ----------------------- Stats helpers -------------------
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


# ----------------------- Requested Outputs helpers -------
def _has_any_tag(req, target_tags: list[str]) -> bool:
    """
    Return True if Requested Outputs contains ANY of the target tags
    (case-insensitive, substring match).
    Handles str / list / dict formats.
    """
    if req is None:
        return False

    targets = [t.lower() for t in target_tags]

    def match(s: str) -> bool:
        s = s.lower()
        return any(t in s for t in targets)

    # String
    if isinstance(req, str):
        return match(req)

    # List of strings or dicts
    if isinstance(req, list):
        for item in req:
            if isinstance(item, str) and match(item):
                return True
            if isinstance(item, dict):
                name = item.get("name")
                if isinstance(name, str) and match(name):
                    return True
        return False

    # Dict
    if isinstance(req, dict):
        name = req.get("name")
        if isinstance(name, str) and match(name):
            return True
        for v in req.values():
            if isinstance(v, str) and match(v):
                return True

    return False


def compute_stats_for_requested_outputs(
    df: pd.DataFrame,
    domain: str,
    domain_col_norm: str,
    gemini_col: str | None,
    gpt5_col: str | None,
) -> pd.DataFrame:
    """
    For a given domain, compute mean (%) and 95% CI for each Requested Output group:
        - "File / Image"
        - "Text / LaTeX / Code"

    A task can contribute to both groups if it contains both types.
    """
    if REQUESTED_OUTPUTS_FIELD not in df.columns:
        logger.warning(
            "Requested Outputs field %r not found in dataframe; skipping outputs chart.",
            REQUESTED_OUTPUTS_FIELD,
        )
        return pd.DataFrame()

    sub = df[df[domain_col_norm] == domain].copy()
    logger.info(
        "[Outputs] For domain '%s', %d records before outputs filtering",
        domain,
        len(sub),
    )

    file_image_mask = sub[REQUESTED_OUTPUTS_FIELD].apply(
        lambda v: _has_any_tag(v, ["file", "image"])
    )
    text_code_mask = sub[REQUESTED_OUTPUTS_FIELD].apply(
        lambda v: _has_any_tag(v, ["text", "latex", "code"])
    )

    groups = {
        "File / Image": file_image_mask,
        "Text / LaTeX / Code": text_code_mask,
    }

    rows = []
    model_map: dict[str, str] = {}
    if gemini_col is not None:
        model_map[gemini_col] = "Gemini 3.0"
    if gpt5_col is not None:
        model_map[gpt5_col] = "GPT-5"

    for group_label, mask in groups.items():
        group_df = sub[mask].copy()
        logger.info(
            "[Outputs] Domain '%s', group '%s' → %d records",
            domain,
            group_label,
            len(group_df),
        )

        for col, model_name in model_map.items():
            if col not in group_df.columns:
                continue
            vals = group_df[col].dropna()
            if len(vals) == 0:
                continue

            mean, ci = compute_mean_and_ci(vals)
            rows.append(
                {
                    "output_group": group_label,
                    "model": model_name,
                    "mean": mean,
                    "ci": ci,
                    "n": len(vals),
                }
            )

    logger.info(
        "[Outputs] Domain '%s' total stats rows: %d", domain, len(rows)
    )
    return pd.DataFrame(rows)


# ----------------------- Tool Use helpers ----------------
def compute_stats_for_tool_use(
    df: pd.DataFrame,
    domain: str,
    domain_col_norm: str,
    gemini_col: str | None,
    gpt5_col: str | None,
) -> pd.DataFrame:
    """
    For a given domain, compute mean (%) and 95% CI for Tool Use groups:
        - "No Tool Use"       -> Tool Use unchecked / False / missing
        - "Tool Use (Checked)" -> Tool Use checked / True
    """
    if TOOL_USE_FIELD not in df.columns:
        logger.warning(
            "Tool Use field %r not found in dataframe; skipping tool-use chart.",
            TOOL_USE_FIELD,
        )
        return pd.DataFrame()

    sub = df[df[domain_col_norm] == domain].copy()
    logger.info(
        "[Tool Use] For domain '%s', %d records before tool-use filtering",
        domain,
        len(sub),
    )

    # Airtable checkbox: True (checked), False or NaN (unchecked)
    tool_series = sub[TOOL_USE_FIELD]

    checked_mask = tool_series.fillna(False).astype(bool)
    unchecked_mask = ~checked_mask

    groups = {
        "No Tool Use": unchecked_mask,
        "Tool Use (Checked)": checked_mask,
    }

    rows = []
    model_map: dict[str, str] = {}
    if gemini_col is not None:
        model_map[gemini_col] = "Gemini 3.0"
    if gpt5_col is not None:
        model_map[gpt5_col] = "GPT-5"

    for group_label, mask in groups.items():
        group_df = sub[mask].copy()
        logger.info(
            "[Tool Use] Domain '%s', group '%s' → %d records",
            domain,
            group_label,
            len(group_df),
        )

        for col, model_name in model_map.items():
            if col not in group_df.columns:
                continue
            vals = group_df[col].dropna()
            if len(vals) == 0:
                continue

            mean, ci = compute_mean_and_ci(vals)
            rows.append(
                {
                    "tool_group": group_label,
                    "model": model_name,
                    "mean": mean,
                    "ci": ci,
                    "n": len(vals),
                }
            )

    logger.info(
        "[Tool Use] Domain '%s' total stats rows: %d", domain, len(rows)
    )
    return pd.DataFrame(rows)


# ----------------------- Plotting: shared style ----------
def _plot_bars_with_ci(
    ax,
    mean_pivot: pd.DataFrame,
    ci_pivot: pd.DataFrame,
    index_labels: list[str],
    x_label: str,
    title: str,
):
    """
    Internal helper that:
      - draws GPT-5 and Gemini 3.0 bars with CI
      - y-limits 0–40
      - adds labels above error bars
      - adds legend and '95% CI' box
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
            y = height + ci + 0.8
            if y > 40:
                y = 39.5

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
    ax.set_xticklabels(index_labels, rotation=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 40)
    ax.set_title(title)

    legend = ax.legend(loc="upper right", title="Graded by")
    plt.setp(legend.get_title(), fontweight="bold")

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


# ----------------------- Plot: interaction type ----------
def plot_domain_stats(stats_df: pd.DataFrame, domain: str, output_path: str):
    """
    Interaction Type chart (Single-Turn vs Multi-Turn).
    """
    if stats_df.empty:
        logger.warning("No stats to plot for domain '%s'. Skipping.", domain)
        return

    mean_pivot = stats_df.pivot(
        index="interaction_type", columns="model", values="mean"
    )
    ci_pivot = stats_df.pivot(
        index="interaction_type", columns="model", values="ci"
    )

    ordered_labels = [i.title() for i in VALID_INTERACTIONS]
    mean_pivot = mean_pivot.reindex(ordered_labels)
    ci_pivot = ci_pivot.reindex(ordered_labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    if domain == "education":
        title = "Single-Turn vs Multi-Turn – Education"
    elif domain == "hle":
        title = "Single-Turn vs Multi-Turn – HLE"
    else:
        title = f"Single-Turn vs Multi-Turn – {domain}"

    _plot_bars_with_ci(
        ax,
        mean_pivot,
        ci_pivot,
        ordered_labels,
        x_label="Interaction Type",
        title=title,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved chart for domain '%s' to %s", domain, output_path)


# ----------------------- Plot: requested outputs ---------
def plot_outputs_stats(stats_df: pd.DataFrame, domain: str, output_path: str):
    """
    Requested Outputs chart:
        - File / Image
        - Text / LaTeX / Code
    """
    if stats_df.empty:
        logger.warning(
            "[Outputs] No stats to plot for domain '%s'. Skipping.", domain
        )
        return

    mean_pivot = stats_df.pivot(
        index="output_group", columns="model", values="mean"
    )
    ci_pivot = stats_df.pivot(
        index="output_group", columns="model", values="ci"
    )

    ordered_labels = OUTPUT_GROUPS
    mean_pivot = mean_pivot.reindex(ordered_labels)
    ci_pivot = ci_pivot.reindex(ordered_labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    if domain == "education":
        title = "Requested Outputs – Education"
    elif domain == "hle":
        title = "Requested Outputs – HLE"
    else:
        title = f"Requested Outputs – {domain.capitalize()}"

    _plot_bars_with_ci(
        ax,
        mean_pivot,
        ci_pivot,
        ordered_labels,
        x_label="Requested Outputs",
        title=title,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("[Outputs] Saved chart for domain '%s' to %s", domain, output_path)


# ----------------------- Plot: tool use ------------------
def plot_tool_use_stats(stats_df: pd.DataFrame, domain: str, output_path: str):
    """
    Tool Use chart:
        - No Tool Use
        - Tool Use (Checked)
    """
    if stats_df.empty:
        logger.warning(
            "[Tool Use] No stats to plot for domain '%s'. Skipping.", domain
        )
        return

    mean_pivot = stats_df.pivot(
        index="tool_group", columns="model", values="mean"
    )
    ci_pivot = stats_df.pivot(
        index="tool_group", columns="model", values="ci"
    )

    ordered_labels = TOOL_GROUPS
    mean_pivot = mean_pivot.reindex(ordered_labels)
    ci_pivot = ci_pivot.reindex(ordered_labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    if domain == "education":
        title = "Tool Use – Education"
    elif domain == "hle":
        title = "Tool Use – HLE"
    else:
        title = f"Tool Use – {domain.capitalize()}"

    _plot_bars_with_ci(
        ax,
        mean_pivot,
        ci_pivot,
        ordered_labels,
        x_label="Tool Use",
        title=title,
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("[Tool Use] Saved chart for domain '%s' to %s", domain, output_path)


# ----------------------- Main ---------------------------
def main():
    df = fetch_airtable_dataframe()

    df_clean, dom_norm_col, inter_norm_col, gemini_col, gpt5_col = clean_and_filter(df)

    logger.info(
        "Normalized domain values: %s",
        sorted(df_clean[dom_norm_col].dropna().unique()),
    )
    logger.info(
        "Normalized interaction values: %s",
        sorted(df_clean[inter_norm_col].dropna().unique()),
    )

    # Interaction-type charts
    edu_stats = compute_stats_for_domain(
        df_clean, "education", dom_norm_col, inter_norm_col, gemini_col, gpt5_col
    )
    plot_domain_stats(edu_stats, "education", "education_chart.png")

    hle_stats = compute_stats_for_domain(
        df_clean, "hle", dom_norm_col, inter_norm_col, gemini_col, gpt5_col
    )
    plot_domain_stats(hle_stats, "hle", "hle_chart.png")

    # Requested Outputs charts
    edu_out_stats = compute_stats_for_requested_outputs(
        df_clean, "education", dom_norm_col, gemini_col, gpt5_col
    )
    plot_outputs_stats(
        edu_out_stats, "education", "requested_outputs_education.png"
    )

    hle_out_stats = compute_stats_for_requested_outputs(
        df_clean, "hle", dom_norm_col, gemini_col, gpt5_col
    )
    plot_outputs_stats(
        hle_out_stats, "hle", "requested_outputs_hle.png"
    )

    # Tool Use charts
    edu_tool_stats = compute_stats_for_tool_use(
        df_clean, "education", dom_norm_col, gemini_col, gpt5_col
    )
    plot_tool_use_stats(
        edu_tool_stats, "education", "tool_use_education.png"
    )

    hle_tool_stats = compute_stats_for_tool_use(
        df_clean, "hle", dom_norm_col, gemini_col, gpt5_col
    )
    plot_tool_use_stats(
        hle_tool_stats, "hle", "tool_use_hle.png"
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
