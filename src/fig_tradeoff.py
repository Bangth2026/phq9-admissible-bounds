"""Fig. 4: trade-off between cases missed and questionnaires avoided per 1,000 participants under refer-on-abstain,
for each abstention rule with the fuzzy additive base model, in six scenarios."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RV = "/home/claude/results/secondary/"
A4 = json.load(open(RV + "sec4.json")); A5 = json.load(open(RV + "sec5.json"))
SC = [("natural|external_pooled", "2005–2014, as observed"), ("natural|temporal", "2021–2023, as observed"), ("mcar30", "30% MCAR"),
      ("mnar_strong|srh", "MNAR 0.9/0.05: self-rated health"), ("outcome_dep", "Outcome-dependent"), ("shift|srh|1.0", "Self-rated health absent, γ = 1")]
LAB = {"wc": "Worst-case bounds", "mi": "Imputation spread", "ps": "Pattern submodels", "sel": "Point-score selective", "cc": "Complete cases"}
COL = {"wc": "#c0504d", "mi": "#4f81bd", "ps": "#2e8b57", "sel": "#7f7f7f", "cc": "#8e6bb0"}
MRK = {"wc": "o", "mi": "s", "ps": "^", "sel": "D", "cc": "v"}
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (key, lab) in zip(axes.ravel(), SC):
    F = A4[key]["fuzzy"]; sel = A5[key]["fuzzy"]["sel"] if key.startswith("natural") else A5[key]["fuzzy"]
    pts = {r: (F[r]["per1000_avoided"], F[r]["per1000_missed"]) for r in ("wc", "mi", "ps", "cc") if r in F}
    pts["sel"] = (sel["per1000_avoided"], sel["per1000_missed"])
    for r, (x, yv) in pts.items():
        ax.scatter(x, yv, s=60, marker=MRK[r], color=COL[r], label=LAB[r], zorder=3)
    ax.set_title(f"{lab} (prevalence {100 * A4[key]['prev']:.1f}%)", fontsize=9.5)
    ax.set_xlabel("Questionnaires avoided per 1,000"); ax.set_ylabel("Cases missed per 1,000"); ax.grid(alpha=0.3)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, frameon=False)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig("/home/claude/results/fig_tradeoff.png", dpi=300)
import os; os.makedirs("/home/claude/sub8/figures", exist_ok=True)
plt.savefig("/home/claude/sub8/figures/Figure_4.pdf"); plt.savefig("/home/claude/sub8/figures/Figure_4.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("ok")
