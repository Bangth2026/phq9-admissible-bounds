"""Risk-coverage curves for three abstention rules (third set of secondary analyses)."""
import json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/home/claude/results/"; RV = R + "secondary/"
X = json.load(open(RV + "sec3.json"))["mi_abstention"]
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
col = {"bounds": ("#1b6ca8", "Admissible-value bounds (proposed)", "-"), "linear": ("#d1495b", "Exact linear bounds (sign-constrained LR)", "--"),
       "mi": ("#2e8b57", "Spread of multiple imputations (LR, MI)", ":")}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
for k, (key, ttl) in enumerate([("natural|external_pooled", "2005–2014, free-day sleep absent by design"), ("mcar30", "Internal test, 30% MCAR"),
                                ("mnar|srh_poor", "Internal test, self-rated health MNAR")]):
    for m, (c, lab, ls) in col.items():
        cur = X[key][m]["curve"]; ax[k].plot([p["coverage"] * 100 for p in cur], [p["bacc"] for p in cur], color=c, ls=ls, lw=1.4, label=lab)
        o = X[key][m]["own"]; ax[k].plot(o["coverage"] * 100, (o["sens"] + o["spec"]) / 2, "o", color=c, ms=6, mfc="white", mew=1.5)
    ax[k].set_title(ttl, fontsize=8.5); ax[k].set_xlabel("Participants decided (%)"); ax[k].set_ylabel("Balanced accuracy among decided")
    if k == 0: ax[k].legend(fontsize=6.8, loc="lower left")
fig.tight_layout(); fig.savefig(R + "fig_riskcov.png", dpi=300); plt.close(fig)
print("ok")
