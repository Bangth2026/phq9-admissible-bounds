"""Figures for the revised manuscript: architecture (knowledge schema), decision curves, missing-data stress tests
with the added comparators, and an illustration of Algorithm 1 (exact bounds) on one missing variable."""
import json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from rev_common import *  # noqa
from exact import ExactBounds
from fofs import membership, prepare

S = pickle.load(open(R + "models.pkl", "rb")); m = S["models"]["Proposed (full)"]; test = S["test"]
RM = json.load(open(RV + "rev_models.json")); RX = json.load(open(RV + "rev_missing.json"))
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})

# ---------------------------------------------------------------- Fig. 2 architecture
fig, ax = plt.subplots(figsize=(11, 4.1)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 42)
boxes = [(1, 24, 17, 14, "Observation layer\n14 harmonised NHANES\nvariables (no PHQ-9 items),\nadmissible ranges, units"),
         (21, 24, 18, 14, f"Concept layer\n{len(m.cnames)} linguistic concepts,\ntrapezoidal / set memberships\n(Ruspini partitions)"),
         (42, 24, 18, 14, "Rule layer\nsingle-concept rules (one\nreference per variable);\nsign of β constrained by\ndirection priors"),
         (63, 24, 17, 14, "Inference\nlogit p = β0 + Σ βr μr\nAlgorithm 1: exact bounds\nover admissible values"),
         (83, 24, 16, 14, "Decision\nrefer / do not refer /\nabstain; Algorithm 2:\nwhich input to ask next")]
for x, y, w, h, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc="#eef4fa", ec="#1b6ca8", lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.9)
for x0, x1 in [(18.5, 20.5), (39.5, 41.5), (60.5, 62.5), (80.5, 82.5)]:
    ax.annotate("", (x1, 31), (x0, 31), arrowprops=dict(arrowstyle="->", lw=1.1))
ax.add_patch(FancyBboxPatch((1, 2), 59, 12, boxstyle="round,pad=0.4", fc="#fbf3e6", ec="#e09f3e", lw=1))
ax.text(30.5, 8, "Provenance layer and knowledge schema file (JSON-LD, SKOS concept scheme)\ncut-point sources (WHO BMI, sleep consensus, federal poverty level);\ndirection priors estimated on NHANES 2015–2016 (never used for fitting or testing)",
        ha="center", va="center", fontsize=7.4)
ax.add_patch(FancyBboxPatch((63, 2), 36, 12, boxstyle="round,pad=0.4", fc="#fbf3e6", ec="#e09f3e", lw=1))
ax.text(81, 8, "Executable trace (JSON)\ninputs, memberships, fired rules,\nlog-odds contributions, bounds,\nnext input to ask", ha="center", va="center", fontsize=7.4)
for x in (10, 30, 51):
    ax.annotate("", (x, 23.5), (x, 14.6), arrowprops=dict(arrowstyle="->", lw=0.9, color="#e09f3e"))
ax.annotate("", (81, 14.6), (81, 23.5), arrowprops=dict(arrowstyle="->", lw=0.9, color="#e09f3e"))
fig.tight_layout(); fig.savefig(R + "fig_framework.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- decision curves
col = {"Proposed (full)": "#1b6ca8", "Logistic regression": "#d1495b", "Gradient-boosted trees": "#2e8b57",
       "Explainable boosting machine (GA2M)": "#8d6a9f"}
lab = {"Proposed (full)": "Proposed", "Logistic regression": "Logistic regression", "Gradient-boosted trees": "Gradient-boosted trees",
       "Explainable boosting machine (GA2M)": "EBM (GA2M)"}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.9))
for k, (sn, ttl) in enumerate([("internal_test", "Internal test (2017–2020)"), ("temporal", "Temporal (2021–2023)"),
                               ("external_pooled", "External (2005–2014)")]):
    d = RM["dca"][sn]; t = np.array(d["thresholds"])
    ax[k].plot(t, d["treat_all"], color="grey", lw=1, ls="--", label="Refer everyone")
    ax[k].axhline(0, color="k", lw=0.7, label="Refer no one")
    for nm in col:
        ax[k].plot(t, d[nm], color=col[nm], lw=1.4, label=lab[nm])
    ax[k].plot(t, d["Proposed (full) (recalibrated)"], color=col["Proposed (full)"], lw=1.0, ls=":", label="Proposed, recalibrated")
    ax[k].set_ylim(-0.02, max(d["treat_all"][0], 0.14) + 0.01); ax[k].set_title(ttl)
    ax[k].set_xlabel("Threshold probability"); ax[k].set_ylabel("Net benefit")
    if k == 0: ax[k].legend(fontsize=6.8)
fig.tight_layout(); fig.savefig(R + "fig_dca.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- missing-data stress tests
st = pd.DataFrame(RX["stress"]); mc = st[st.scenario == "MCAR"].copy(); mc["lv"] = mc.level.astype(int)
show = [("Proposed (full)", "Proposed (point score)", "#1b6ca8", "-"), ("No interval missing handling", "Same rules, median imputation", "#6c757d", "-"),
        ("Logistic regression", "LR, indicators", "#d1495b", "-"), ("LR, median imputation without indicators", "LR, no indicators", "#d1495b", "--"),
        ("LR, multiple imputation", "LR, multiple imputation", "#d1495b", ":"), ("Gradient-boosted trees", "GBT", "#2e8b57", "-"),
        ("Explainable boosting machine (GA2M)", "EBM (GA2M)", "#8d6a9f", "-"), ("Version 1 (conjunctive rules)", "Version 1", "#e09f3e", "-"),
        ("Sign-constrained logistic regression", "Sign-constrained LR", "#a23b72", "--"), ("Monotone gradient-boosted trees", "Monotone GBT", "#2e8b57", "--")]
S2 = json.load(open(RV + "sec2_missing.json")); s2 = pd.DataFrame(S2["stress"]); s2 = s2[(s2.scenario == "MCAR")].copy(); s2["lv"] = s2.level.astype(int)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
for nm, lb, c, ls in show:
    d = mc[mc.model == nm].sort_values("lv"); ax[0].plot(d.lv, d.AUROC, marker="o", ms=3, lw=1.2, color=c, ls=ls, label=lb)
d = s2[(s2.model == "LR with restricted cubic splines") & (s2.kind == "point")].sort_values("lv")
ax[0].plot(d.lv, d.AUROC, marker="o", ms=3, lw=1.2, color="#f28e2b", ls="-.", label="LR, restricted cubic splines")
ax[0].set_xlabel("Predictor values masked at random (%)"); ax[0].set_ylabel("AUROC"); ax[0].legend(fontsize=6.0, ncol=2)
for nm, lb, c, ls in [("Proposed (full)", "Proposed, admissible bounds", "#1b6ca8", "-"), ("Box missing bounds", "Proposed, box bounds", "#b5838d", "-"),
                      ("Sign-constrained logistic regression", "Sign-constrained LR, exact linear bounds", "#d1495b", "--"),
                      ("Abstain if any input missing", "Abstain if any input missing", "#444444", ":")]:
    d = mc[mc.model == nm].sort_values("lv"); ax[1].plot(d.lv, 100 * d.Coverage, marker="o", ms=3, lw=1.2, color=c, ls=ls, label=lb)
d = s2[(s2.model == "LR with restricted cubic splines") & (s2.kind == "bounds")].sort_values("lv")
ax[1].plot(d.lv, 100 * d.Coverage, marker="o", ms=3, lw=1.2, color="#f28e2b", ls="-.", label="Spline LR, grid bounds")
ax[1].set_xlabel("Predictor values masked at random (%)"); ax[1].set_ylabel("Participants with a decision (%)"); ax[1].legend(fontsize=6.6)
fig.tight_layout(); fig.savefig(R + "fig_missing.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- Algorithm 1 illustration (version-1 rule base, one missing variable)
v1 = S["models"]["Version 1 (conjunctive rules)"]; eb = ExactBounds(v1)
cc = test[test[[f for f in BASE if f not in ("age", "female")]].notna().all(1)].reset_index(drop=True)
best = None
for var in ("bmi", "sleep_we", "sleep_wd", "pir", "sedentary"):
    for i in range(min(len(cc), 600)):
        row = cc.iloc[[i]].copy(); row[var] = np.nan
        el, eu = eb.bounds(row, return_eta=True); gl, gu = v1.bounds(row)
        lg = lambda p: np.log(p / (1 - p))
        gap = max(lg(gl[0]) - el[0], eu[0] - lg(gu[0]))
        if best is None or gap > best[0]: best = (gap, var, i)
gap, var, i = best
row = cc.iloc[[i]].copy(); row[var] = np.nan
lo, hi = VARIABLES[var][2]; xs = np.linspace(lo, hi, 4001)
Rr = pd.concat([row] * len(xs), ignore_index=True); Rr[var] = xs
_, Mm = v1._M(Rr); eta = v1.b0 + v1._rule_mu(Mm) @ v1.beta
grid = v1._grid(var); Rg = pd.concat([row] * len(grid), ignore_index=True); Rg[var] = grid
_, Mg = v1._M(Rg); eg = v1.b0 + v1._rule_mu(Mg) @ v1.beta
M0 = v1._mu(prepare(row))[0]
rk = np.array([any(v1.concepts[v1.cnames[j]]["variable"] == var for j in r) for r in v1.rules])
bp = eb._breaks(var, np.where(rk)[0], M0)
Rb = pd.concat([row] * len(bp), ignore_index=True); Rb[var] = bp
_, Mb = v1._M(Rb); ebp = v1.b0 + v1._rule_mu(Mb) @ v1.beta
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.plot(xs, eta, color="#1b6ca8", lw=1.2, label="η as a function of the missing input")
ax.plot(grid, eg, "s", ms=3, color="#e09f3e", label="41-point grid + trapezoid corners")
ax.plot(bp, ebp, "o", ms=4, mfc="none", color="#d1495b", label="Algorithm 1 candidate set")
ax.axhline(ebp.max(), color="#d1495b", lw=0.7, ls=":"); ax.axhline(eg.max(), color="#e09f3e", lw=0.7, ls=":")
ax.set_xlabel(f"{var} ({VARIABLES[var][3]})"); ax.set_ylabel("Linear predictor η"); ax.legend(fontsize=7)
ax.set_title(f"Version-1 rule base, one participant with {var} missing: grid misses the maximum by {gap:.4f} log-odds", fontsize=8)
fig.tight_layout(); fig.savefig(R + "fig_alg1.png", dpi=300); plt.close(fig)
json.dump(dict(var=var, gap=float(gap), n_candidates=int(len(bp)), n_grid=int(len(grid))), open(RV + "alg1_example.json", "w"))
print("fig_alg1", var, gap, len(bp), len(grid))

# ---------------------------------------------------------------- Fig. 8 trace with numeric labels
ex = json.load(open(R + "example_case.json")); t = ex["trace"]; thr0 = ex["threshold"]
fired = sorted([r for r in t["rules"] if abs(r.get("logit_contribution", 0)) > 1e-9], key=lambda r: r["logit_contribution"])
fig, ax = plt.subplots(figsize=(7.8, 0.36 * len(fired) + 1.3))
vals = [r["logit_contribution"] for r in fired]
ax.barh(range(len(fired)), vals, color=["#d1495b" if v > 0 else "#2e8b57" for v in vals], hatch=None)
for k, v in enumerate(vals):
    ax.text(v + (0.02 if v > 0 else -0.02), k, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", fontsize=7)
ax.set_yticks(range(len(fired))); ax.set_yticklabels([f'{r["rule"]}  (μ = {r["firing_lower"]:.2f})' for r in fired], fontsize=7.5)
ax.axvline(0, c="k", lw=0.6); ax.set_xlabel("Contribution to log-odds, β × firing strength (+ raises, − lowers)")
lo_, hi_ = min(vals), max(vals); ax.set_xlim(lo_ - 0.25, hi_ + 0.3)
tot = t["intercept"] + sum(vals)
ax.set_title(f"Intercept {t['intercept']:.2f} + sum of bars {sum(vals):+.2f} = log-odds {tot:.2f}; p = {t['prob_lower']:.3f} (threshold {thr0:.3f})", fontsize=8)
fig.tight_layout(); fig.savefig(R + "fig_explanation.png", dpi=300); plt.close(fig)
print("fig8 ok")
