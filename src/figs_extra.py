import json, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from experiment import LRModel, GBTModel, BASE  # noqa
from scipy.special import expit
R = "/home/claude/results/"
S = pickle.load(open(R + "models.pkl", "rb")); m = S["models"]["Proposed (full)"]; test = S["test"]
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
# ---- Figure 1: architecture
fig, ax = plt.subplots(figsize=(11, 3.8)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
boxes = [(1, 22, 17, 14, "Observation layer\n14 harmonised NHANES\nvariables (no PHQ-9 items)\nadmissible ranges, units"),
         (21, 22, 18, 14, f"Concept layer\n{len(m.cnames)} linguistic concepts\ntrapezoidal / set memberships\n(Ruspini partitions)"),
         (42, 22, 18, 14, "Rule layer\nsingle-concept rules (one\nreference per variable), optional\npairs; sign of beta constrained\nby direction priors"),
         (63, 22, 17, 14, "Inference\nlogit p = b0 + sum beta_r mu_r\nmissing inputs -> exact bounds\nover admissible values"),
         (83, 22, 16, 14, "Output\npoint probability,\n[lower, upper] bounds,\nrefer / no-refer / abstain")]
for x, y, w, h, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc="#eef4fa", ec="#1b6ca8", lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.9)
for x0, x1 in [(18.5, 20.5), (39.5, 41.5), (60.5, 62.5), (80.5, 82.5)]:
    ax.annotate("", (x1, 29), (x0, 29), arrowprops=dict(arrowstyle="->", lw=1.1))
ax.add_patch(FancyBboxPatch((21, 2), 39, 11, boxstyle="round,pad=0.4", fc="#fbf3e6", ec="#e09f3e", lw=1))
ax.text(40.5, 7.5, "Provenance layer\nconcept sources (WHO BMI, sleep consensus, FPL cut-points)\ndirection priors estimated on NHANES 2015-2016 (not used for fitting)", ha="center", va="center", fontsize=7.6)
ax.add_patch(FancyBboxPatch((63, 2), 36, 11, boxstyle="round,pad=0.4", fc="#fbf3e6", ec="#e09f3e", lw=1))
ax.text(81, 7.5, "Executable trace (JSON)\ninputs, memberships, fired rules,\nlog-odds contributions, bounds", ha="center", va="center", fontsize=7.6)
for x in (30, 51):
    ax.annotate("", (x, 21.5), (x, 13.6), arrowprops=dict(arrowstyle="->", lw=0.9, color="#e09f3e"))
ax.annotate("", (81, 13.6), (81, 21.5), arrowprops=dict(arrowstyle="->", lw=0.9, color="#e09f3e"))
fig.tight_layout(); fig.savefig(R + "fig_framework.png", dpi=300); plt.close(fig)

# ---- rule base table (sorted by |beta| * mean firing on development data)
dev = S["dev"]
from fofs import prepare
_, M = m._M(dev); A = m._rule_mu(m._impute_mu(M))
rows = [dict(rule=m.rule_text(k), beta=m.beta[k], odds_ratio=float(np.exp(m.beta[k])), mean_firing=float(A[:, k].mean()),
             impact=float(abs(m.beta[k]) * A[:, k].mean()), sign=int(m.signs[k])) for k in range(len(m.rules))]
rb = pd.DataFrame(rows).sort_values("impact", ascending=False)
rb.to_csv(R + "rulebase_table.csv", index=False); print(rb.round(3).to_string()); print("intercept", m.b0)

# ---- example explanation (one positive with complete inputs)
zl, zu = m.bounds(test); s = m.score(test); thr = json.load(open(R + "main_results.json"))["Proposed (full)"]["threshold"]
cand = np.where((test.y == 1) & (zl == zu) & (s > thr) & (s < 0.5))[0]
nf = np.array([sum(abs(r.get("logit_contribution", 0)) > 1e-9 for r in m.explain(test.iloc[c][BASE])["rules"]) for c in cand])
i = int(cand[np.argmax(nf)])
t = m.explain(test.iloc[i][BASE])
fired = [r for r in t["rules"] if abs(r.get("logit_contribution", 0)) > 1e-9]
fired.sort(key=lambda r: r["logit_contribution"])
fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(fired) + 1.2))
ax.barh(range(len(fired)), [r["logit_contribution"] for r in fired],
        color=["#d1495b" if r["logit_contribution"] > 0 else "#2e8b57" for r in fired])
ax.set_yticks(range(len(fired))); ax.set_yticklabels([f'{r["rule"]}  (mu={r["firing_lower"]:.2f})' for r in fired], fontsize=7.5)
ax.axvline(0, c="k", lw=0.6); ax.set_xlabel("Contribution to log-odds (beta x firing)")
ax.set_title(f"Intercept {t['intercept']:.2f}; total log-odds {t['intercept'] + sum(r['logit_contribution'] for r in fired):.2f}; p = {t['prob_lower']:.3f} (threshold {thr:.3f})", fontsize=8)
fig.tight_layout(); fig.savefig(R + "fig_explanation.png", dpi=300); plt.close(fig)
json.dump(dict(index=i, trace=t, threshold=thr), open(R + "example_case.json", "w"), indent=1)
print(json.dumps({k: t[k] for k in ("inputs", "missing_inputs", "prob_lower")}, indent=0))
# example with a missing input
cand2 = np.where((test.y == 1) & (zl < zu) & test.pir.isna().values)[0]
j = int(cand2[0]); t2 = m.explain(test.iloc[j][BASE])
json.dump(dict(index=j, trace=t2, point=float(s[j])), open(R + "example_missing.json", "w"), indent=1)
print("missing example", t2["missing_inputs"], t2["prob_lower"], t2["prob_upper"], s[j])
