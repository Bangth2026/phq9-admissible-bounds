"""Export the main-text figures as separate high-resolution files (PDF vector and 600-dpi TIFF) for submission."""
import json, subprocess
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "/home/claude/sub7/figures/"
# Fig. 1: flow (re-run script with extra outputs)
src = open("/home/claude/src/fig_flow.py").read().replace('fig.savefig("/home/claude/results/fig_flow.png", dpi=300)',
      f'fig.savefig("/home/claude/results/fig_flow.png", dpi=300); fig.savefig("{OUT}Figure_1.pdf"); fig.savefig("{OUT}Figure_1.tif", dpi=600, pil_kwargs={{"compression": "tiff_lzw"}})')
exec(src)
plt.close("all")
# Fig. 2: design
src = open("/home/claude/src/fig_design.py").read().replace('plt.savefig("/home/claude/results/fig_design.png", dpi=200)',
      f'plt.savefig("/home/claude/results/fig_design.png", dpi=300); plt.savefig("{OUT}Figure_2.pdf"); plt.savefig("{OUT}Figure_2.tif", dpi=600, pil_kwargs={{"compression": "tiff_lzw"}})')
exec(src)
plt.close("all")
# Fig. 3: risk-coverage curves from the stored curves
d = json.load(open("/home/claude/results/secondary/fig_riskcov4.json"))
GRID = np.linspace(0.05, 1.0, 20)
LAB = {"wc": "Worst-case bounds", "mi": "Imputation spread", "ps": "Pattern submodels", "sel": "Point-score selective"}
COL = {"wc": "#c0504d", "mi": "#4f81bd", "ps": "#2e8b57", "sel": "#7f7f7f"}
STY = {"wc": "-", "mi": "--", "ps": "-.", "sel": ":"}
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
for ax, (lab, rules) in zip(axes, d.items()):
    for r in ("wc", "mi", "ps", "sel"):
        if r not in rules: continue
        ba = np.array(rules[r]["curve"]); ax.plot(100 * GRID, ba, STY[r], color=COL[r], lw=1.8, label=LAB[r])
        oc = rules[r]["own"]
        if 0.05 <= oc <= 1.0: ax.plot(100 * oc, np.interp(oc, GRID, ba), "o", mfc="white", mec=COL[r], ms=6)
    ax.set_title(lab, fontsize=9.5); ax.set_xlabel("Participants decided (%)"); ax.grid(alpha=0.3)
axes[0].set_ylabel("Balanced accuracy among decided"); axes[0].legend(fontsize=7.5, loc="lower left")
plt.tight_layout(); plt.savefig("/home/claude/results/fig_riskcov4.png", dpi=300)
plt.savefig(OUT + "Figure_3.pdf"); plt.savefig(OUT + "Figure_3.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("figures exported")
