#!/usr/bin/env python3
"""
Averages two Airtable numeric fields across all records in a given view.

Environment variables required (use a .env file or export them):
- AIRTABLE_API_KEY
- AIRTABLE_BASE_ID
- AIRTABLE_TABLE_ID
- AIRTABLE_VIEW_ID  (can also be a view NAME; Airtable accepts either)

Usage:
  python airtable_avg.py

Outputs the average of the two target columns across all records that have numeric values.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

TARGET_FIELDS = [
    "GPT5 Autorater - Gemini Response Score",
    "GPT5 Autorater - GPT5 Response Score",
]

API_URL_TEMPLATE = "https://api.airtable.com/v0/{base_id}/{table_id}"


class AirtableError(Exception):
    pass


def _is_number(x: Any) -> bool:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return True
    if isinstance(x, str):
        try:
            float(x.strip())
            return True
        except Exception:
            return False
    return False


def _to_float(x: Any) -> float:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise ValueError(f"Not a number: {x!r}")


def fetch_all_records(
    api_key: str, base_id: str, table_id: str, view_id_or_name: str
) -> List[Dict[str, Any]]:
    """Fetch all records from Airtable table/view, handling pagination."""
    url = API_URL_TEMPLATE.format(base_id=base_id, table_id=table_id)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params = {"pageSize": 100, "view": view_id_or_name}

    records: List[Dict[str, Any]] = []
    attempts = 0
    offset: str | None = None

    while True:
        if offset:
            params["offset"] = offset
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
        except requests.RequestException as e:
            raise AirtableError(f"Network error calling Airtable: {e}")

        if resp.status_code == 429:
            # Rate limited; back off and retry a few times.
            retry_after = int(resp.headers.get("Retry-After", "1"))
            attempts += 1
            if attempts > 6:
                raise AirtableError(
                    "Hit Airtable rate limits repeatedly. Try again later."
                )
            time.sleep(retry_after or (2**attempts) * 0.5)
            continue

        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise AirtableError(f"Airtable API error {resp.status_code}: {detail}")

        data = resp.json()
        batch = data.get("records", [])
        records.extend(batch)
        offset = data.get("offset")
        if not offset:
            break

    return records


def t_critical_95(df: int) -> float:
    """Two-tailed 95% t critical (alpha=0.05), returning a conservative value.
    Falls back to 1.96 (normal) for large df or if df < 1.
    Values from standard t-table.
    """
    if df <= 0:
        return 1.96
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
        40: 2.021,
        60: 2.000,
        120: 1.980,
    }
    if df in table:
        return table[df]
    # Choose the next lowest df (more conservative) or normal if very large
    keys = sorted(table.keys())
    if df > keys[-1]:
        return 1.96
    lower = max(k for k in keys if k <= df)
    return table[lower]


def compute_stats(
    records: List[Dict[str, Any]], fields: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute stats per field needed for 95% CI error bars.

    Returns a mapping per field:
      {
        'count': n,
        'mean': mean,
        'stdev': sample_std_dev,
        'stderr': stdev / sqrt(n),
        'ci95_halfwidth': t*stderr,
        'ci95_low': mean - t*stderr,
        'ci95_high': mean + t*stderr,
      }
    Non-numeric/blank values are ignored.
    """
    import math

    out: Dict[str, Dict[str, float]] = {}
    for f in fields:
        # collect numeric values for this field
        vals: List[float] = []
        for rec in records:
            v = rec.get("fields", {}).get(f)
            if _is_number(v):
                vals.append(_to_float(v))

        n = len(vals)
        if n == 0:
            out[f] = {
                "count": 0.0,
                "mean": float("nan"),
                "stdev": float("nan"),
                "stderr": float("nan"),
                "ci95_halfwidth": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
            }
            continue

        mean = sum(vals) / n
        stdev = float("nan")
        if n >= 2:
            mu = mean
            var = sum((x - mu) ** 2 for x in vals) / (n - 1)  # sample variance (ddof=1)
            stdev = math.sqrt(var)
        else:
            stdev = float("nan")  # undefined with a single point

        stderr = stdev / math.sqrt(n) if n >= 2 else float("nan")
        t = t_critical_95(n - 1)
        half = t * stderr if n >= 2 else float("nan")
        ci_low = mean - half if n >= 2 else float("nan")
        ci_high = mean + half if n >= 2 else float("nan")

        out[f] = {
            "count": float(n),
            "mean": mean,
            "stdev": stdev,
            "stderr": stderr,
            "ci95_halfwidth": half,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }
    return out


def main() -> int:
    load_dotenv()

    api_key = os.getenv("AIRTABLE_API_KEY")
    base_id = os.getenv("AIRTABLE_BASE_ID")
    table_id = os.getenv("AIRTABLE_TABLE_ID")
    view_id = os.getenv("AIRTABLE_VIEW_ID")

    missing = [
        k
        for k, v in [
            ("AIRTABLE_API_KEY", api_key),
            ("AIRTABLE_BASE_ID", base_id),
            ("AIRTABLE_TABLE_ID", table_id),
            ("AIRTABLE_VIEW_ID", view_id),
        ]
        if not v
    ]

    if missing:
        print(
            "Missing required environment variables: " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "Create a .env file with those keys or export them in your shell.",
            file=sys.stderr,
        )
        return 2

    records = fetch_all_records(api_key, base_id, table_id, view_id)
    stats = compute_stats(records, TARGET_FIELDS)

    def fmt(x: float) -> str:
        return "nan" if (x != x) else f"{x:.6f}"

    print(f"Airtable stats (view: {view_id})\n")

    # Print each field separately
    for f in TARGET_FIELDS:
        s = stats[f]
        print(f"Field: {f}")
        print(f"  n           : {int(s['count'])}")
        print(f"  mean        : {fmt(s['mean'])}")
        print(f"  stdev (ddof=1): {fmt(s['stdev'])}")
        print(f"  stderr      : {fmt(s['stderr'])}")
        print(f"  95% CI half : {fmt(s['ci95_halfwidth'])}")
        print(f"  95% CI low  : {fmt(s['ci95_low'])}")
        print(f"  95% CI high : {fmt(s['ci95_high'])}")
        print()

    # Also emit a compact CSV for copy/paste
    print("CSV:field,count,mean,stdev,stderr,ci95_halfwidth,ci95_low,ci95_high")
    for f in TARGET_FIELDS:
        s = stats[f]
        row = [
            f,
            str(int(s["count"])),
            fmt(s["mean"]),
            fmt(s["stdev"]),
            fmt(s["stderr"]),
            fmt(s["ci95_halfwidth"]),
            fmt(s["ci95_low"]),
            fmt(s["ci95_high"]),
        ]
        print(",".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
