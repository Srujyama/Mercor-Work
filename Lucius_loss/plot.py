import os

import matplotlib.pyplot as plt
import numpy as np

# View labels (from your Airtable view names)
views = ["IA Rubrics", "Text Only", "Image Input", "Image Output"]

# Means and 95% CI half-widths
gemini_means = np.array([37.058466, 45.838414, 40.813575, 37.982768])
gemini_ci = np.array([5.396711, 5.827432, 3.665894, 3.945343])

gpt_means = np.array([46.170807, 64.788265, 46.128300, 57.949396])
gpt_ci = np.array([6.020952, 7.045979, 4.633096, 4.005788])

x = np.arange(len(views))
width = 0.36

# Colors (RGB 0–1)
gemini_color = (0.378, 0.555, 0.936)
gpt_color = (0.397, 0.743, 0.567)

fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
ax.set_facecolor("white")

# Bars + error bars (95% CI)
bars1 = ax.bar(
    x - width / 2,
    gemini_means,
    width,
    yerr=gemini_ci,
    capsize=5,
    label="Gemini-2-5-pro",
    color=gemini_color,
    edgecolor="black",
    linewidth=0.5,
)

bars2 = ax.bar(
    x + width / 2,
    gpt_means,
    width,
    yerr=gpt_ci,
    capsize=5,
    label="GPT-5",
    color=gpt_color,
    edgecolor="black",
    linewidth=0.5,
)

# Add raw percentage text above each bar
for i in range(len(views)):
    # Gemini
    ax.text(
        x[i] - width / 2,
        gemini_means[i] + gemini_ci[i] + 1.5,  # slightly above CI
        f"{gemini_means[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    # GPT
    ax.text(
        x[i] + width / 2,
        gpt_means[i] + gpt_ci[i] + 1.5,
        f"{gpt_means[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# Titles / labels
ax.set_title("Loss Analysis Gemini vs GPT")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(views, rotation=0, ha="center")

# Legend (key) top-right
ax.legend(loc="upper right", frameon=True)

# Small note about the error bars
ax.text(
    0.01,
    0.02,
    "Error bars show 95% confidence interval",
    transform=ax.transAxes,
    fontsize=10,
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9),
)

# Tight layout to avoid overflow
plt.tight_layout()

# Save to Downloads folder
plt.savefig(
    os.path.expanduser("~/Downloads/loss_analysis_gemini_vs_gpt.png"),
    dpi=200,
    bbox_inches="tight",
)

plt.show()
