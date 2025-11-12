# make_one_chart.py
# Usage: python make_one_chart.py

import os

import matplotlib.pyplot as plt

# <-- Edit these as needed -->
views = [
    {"title": "Text-Only", "total_criteria": 642, "mis_gemini": 107, "mis_gpt": 104},
    {
        "title": "Image Output",
        "total_criteria": 1454,
        "mis_gemini": 436,
        "mis_gpt": 319,
    },
    {"title": "Image Input", "total_criteria": 1546, "mis_gemini": 249, "mis_gpt": 233},
]

# Build arrays
labels = [v["title"] for v in views]
gemini_pct = [
    (v["mis_gemini"] / v["total_criteria"] * 100) if v["total_criteria"] else 0.0
    for v in views
]
gpt_pct = [
    (v["mis_gpt"] / v["total_criteria"] * 100) if v["total_criteria"] else 0.0
    for v in views
]

# Figure
plt.figure(figsize=(12, 7))
x = list(range(len(labels)))
bar_width = 0.38

bars1 = plt.bar(
    [i - bar_width / 2 for i in x], gemini_pct, width=bar_width, label="Human vs Gemini"
)
bars2 = plt.bar(
    [i + bar_width / 2 for i in x], gpt_pct, width=bar_width, label="Human vs GPT"
)

# Annotate percentages on bars
max_val = max(max(gemini_pct), max(gpt_pct)) if gemini_pct and gpt_pct else 0.0
for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        plt.text(
            b.get_x() + b.get_width() / 2,
            h + max_val * 0.03,
            f"{h:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

# Title and axis labels
plt.title("Misalignment: Autorater vs Human Rating (%)", pad=14)
plt.ylabel("Misalignment (%)")

# Group names with per-group totals underneath (two-line tick labels)
xtick_labels = [
    f"{v['title']}\nTotal Criteria Count: {v['total_criteria']}" for v in views
]
plt.xticks(x, xtick_labels, fontsize=11)

# Cosmetics
plt.grid(axis="y", linestyle="--", alpha=0.45)
plt.legend()
plt.tight_layout()

# Save
os.makedirs("nice_charts", exist_ok=True)
out_path = os.path.join("nice_charts", "misalignment_autorater_vs_human.png")
plt.savefig(out_path, dpi=240, bbox_inches="tight")
print(f"Saved: {out_path}")
