"""Schematic of the comparison of abstention rules (Fig. 2)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
def box(x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10, fontweight="bold", color="#222")
    for i, l in enumerate(lines): ax.text(x + 1.2, y + h - 6.6 - 3.4 * i, l, ha="left", va="top", fontsize=8.2, color="#222")
def arrow(x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#555"))
box(1, 4, 21, 38, "Data and missingness", ["Development: 2017–2020", "Three cohorts as observed:", "  internal test, 2021–2023,", "  2005–2014 (free-day sleep", "  never collected)", "13 simulated scenarios:", "  MCAR, MNAR (0.6/0.1, 0.9/0.05),", "  outcome-dependent,", "  input absent with shift"], "#eef4fa", "#1b6ca8")
box(26, 4, 20, 38, "Base model (fixed)", ["Fuzzy additive model", "  (schema of admissible", "  values, bounded terms)", "Sign-constrained linear LR", "Spline LR", "", "Threshold fixed on", "development out-of-fold", "predictions (sens. ≥ 0.80)"], "#fbf3e6", "#e09f3e")
box(50, 4, 22, 38, "Abstention rule", ["1 Worst-case bounds", "  (all admissible completions)", "2 Imputation spread", "  (20 imputations, 90/10 rule)", "3 Pattern submodels", "4 Point-score selective", "  (band around threshold)", "5 Complete cases", "Output: refer / not / abstain"], "#eef7ee", "#2e8b57")
box(76, 4, 23, 38, "Comparison", ["Coverage (refer / not refer)", "Accuracy at matched", "  coverage", "Risk–coverage curve, AURC", "Refer-on-abstain per 1,000:", "  referred, missed,", "  questionnaires avoided", "Net benefit at 0.05, 0.10", "Bootstrap intervals"], "#f3eef8", "#7a4fa3")
for a, b in ((22.8, 25.2), (46.8, 49.2), (72.8, 75.2)): arrow(a, b, 23)
plt.tight_layout(); plt.savefig("/home/claude/results/fig_design.png", dpi=200); print("ok")
