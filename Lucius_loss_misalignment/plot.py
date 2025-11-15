import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

# Data from your logs
data = [
    {
        "Category": "Text-Only",
        "View ID": "viwsTeXMlR6jWAF4m",
        "Total Criteria": 642,
        "Misalignment (Human vs Gemini) %": 16.510903,
        "Misalignment (Human vs GPT) %": 33.800623,
    },
    {
        "Category": "Image Output",
        "View ID": "viwFHSBIdstYr9Euw",
        "Total Criteria": 1447,
        "Misalignment (Human vs Gemini) %": 22.667588,
        "Misalignment (Human vs GPT) %": 41.828255,
    },
    {
        "Category": "Image Input",
        "View ID": "viwxyetBpKlK14AAs",
        "Total Criteria": 1547,
        "Misalignment (Human vs Gemini) %": 15.12605,
        "Misalignment (Human vs GPT) %": 30.833872,
    },
    {
        "Category": "IA Rubrics",
        "View ID": "viwcQkSgsonSQBjed",
        "Total Criteria": 466,
        "Misalignment (Human vs Gemini) %": 22.103004,
        "Misalignment (Human vs GPT) %": 39.914163,
    },
]

df = pd.DataFrame(data)

# Build grouped bar chart
categories = df["Category"].tolist()
gem_vals = df["Misalignment (Human vs Gemini) %"].tolist()
gpt_vals = df["Misalignment (Human vs GPT) %"].tolist()
totals = df["Total Criteria"].tolist()

x = np.arange(len(categories), dtype=float)
width = 0.35

# Nice, readable sizing and typography
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

# White background
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True, facecolor="white")
ax.set_facecolor("white")

# Bars
bars_gem = ax.bar(x - width / 2, gem_vals, width, label="Human vs Gemini")
bars_gpt = ax.bar(x + width / 2, gpt_vals, width, label="Human vs GPT")

# Title & labels
ax.set_title("Misalignment: Autorater vs Human Rating")
ax.set_ylabel("Misalignment (%)")

# X tick labels with total n appended on a second line
xtick_labels = [f"{cat}\n(n={tot:,})" for cat, tot in zip(categories, totals)]
ax.set_xticks(x, xtick_labels)

# Y as percent with clean limits
upper = max(max(gem_vals), max(gpt_vals))
ax.set_ylim(0, upper * 1.25)
ax.yaxis.set_major_formatter(PercentFormatter(100))

# Subtle grid and simplified spines
ax.grid(axis="y", linestyle="--", alpha=0.4)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Legend boxed off cleanly in the upper-right area
ax.legend(
    title="Comparison",
    loc="upper right",
    frameon=True,
    framealpha=1.0,
    edgecolor="lightgray",
    facecolor="white",
    bbox_to_anchor=(0.98, 0.95),  # a little higher, away from bars
)

# Raise the plot region slightly so legend has room above bars
plt.subplots_adjust(top=0.85, bottom=0.12, right=0.92, left=0.08)


# Helper to add value labels on each bar with slight outline for readability
def autolabel(barcontainer):
    for rect in barcontainer:
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )


autolabel(bars_gem)
autolabel(bars_gpt)

# Save & show
fig.savefig(
    "/Users/srujanyamali/Downloads/autorater_misalignment_by_view_white.png",
    dpi=200,
    facecolor="white",
)

# Adjust layout so legend fits neatly
plt.subplots_adjust(top=0.85, bottom=0.12, right=0.92, left=0.08)


plt.show()
