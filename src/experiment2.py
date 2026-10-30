"""Secondary analyses: missing-data stress tests, explanation fidelity, sensitivity analyses,
subgroups, descriptive table, figures."""
import json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from experiment import (BASE, DOMAINS, PRIORS, SEED, LRModel, GBTModel, FOFS, fofs_factory, cv_scores,
                        pick_threshold, platt, metrics, VARIANTS)
from fofs import reconstruct_from_trace, prepare

warnings.filterwarnings("ignore")
R = "/home/claude/results/"
S = pickle.load(open(R + "models.pkl", "rb"))
models, preds, dev, test, L, POOL = S["models"], S["preds"], S["dev"], S["test"], S["L"], S["POOL"]
main = json.load(open(R + "main_results.json"))
out = {}
FULL = "Proposed (full)"
full = models[FULL]
thr = {nm: main[nm]["threshold"] for nm in models}

# ---------------------------------------------------------------- missing-data stress tests
MASKABLE = [f for f in BASE if f not in ("age", "female")]
cmp = [FULL, "Box missing bounds", "No interval missing handling", "Logistic regression", "Gradient-boosted trees", "Version 1 (conjunctive rules)"]


def eval_masked(df, nm):
    m = models[nm]; zl, zu = m.bounds(df); s = m.score(df); y = df.y.values
    r = {"AUROC": roc_auc_score(y, s)}
    t = thr[nm]
    if isinstance(m, FOFS) and m.interval_missing:
        pos, neg = zl >= t, zu < t; ab = ~(pos | neg)
        r.update(Coverage=1 - ab.mean(), Sens_refer=((pos | ab) & (y == 1)).sum() / y.sum(),
                 Spec_refer=(neg & (y == 0)).sum() / (y == 0).sum(),
                 Sens_covered=(pos & (y == 1)).sum() / max(((y == 1) & ~ab).sum(), 1),
                 Spec_covered=(neg & (y == 0)).sum() / max(((y == 0) & ~ab).sum(), 1),
                 MeanWidth=float(np.mean(zu - zl)))
    else:
        pred = s >= t
        r.update(Coverage=1.0, Sens_refer=(pred & (y == 1)).sum() / y.sum(),
                 Spec_refer=(~pred & (y == 0)).sum() / (y == 0).sum())
    return r


rows = []
rng = np.random.default_rng(SEED)
for rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for rep in range(10 if rate > 0 else 1):
        d = test.copy()
        mask = rng.random((len(d), len(MASKABLE))) < rate
        for j, f in enumerate(MASKABLE):
            d.loc[mask[:, j], f] = np.nan
        for nm in cmp:
            rows.append(dict(scenario="MCAR", level=rate, rep=rep, model=nm, **eval_masked(d, nm)))
for dom, vars_ in list(DOMAINS.items()) + [("Self-rated health", ["srh"])]:
    d = test.copy(); d[vars_] = np.nan
    for nm in cmp:
        rows.append(dict(scenario="Domain removed", level=dom, rep=0, model=nm, **eval_masked(d, nm)))
miss = pd.DataFrame(rows)
miss.to_csv(R + "missing_stress.csv", index=False)
agg = miss.groupby(["scenario", "level", "model"]).mean(numeric_only=True).drop(columns="rep").reset_index()
out["missing"] = agg.to_dict(orient="records")
print(agg.round(3).to_string())

# ---------------------------------------------------------------- explanation fidelity
errs, widths, ncontrib = [], [], []
traces = []
ZL, ZU = full.bounds(test)
for i in range(len(test)):
    row = test.iloc[i][BASE]
    t = full.explain(row)
    if all(r["firing_lower"] == r["firing_upper"] for r in t["rules"]):
        lo, hi = reconstruct_from_trace(json.loads(json.dumps(t)))  # serialise -> re-read
        errs.append(max(abs(lo - ZL[i]), abs(hi - ZU[i])))
    fired = [r for r in t["rules"] if r["firing_upper"] > 0]
    ncontrib.append(len(fired))
    if i < 5 or (t["missing_inputs"] and len(traces) < 8):
        traces.append(dict(index=int(i), y=int(test.y.iloc[i]), **t))
out["fidelity"] = dict(n=len(errs), n_test=len(test), max_abs_error=float(np.max(errs)), mean_abs_error=float(np.mean(errs)),
                       median_rules_firing=float(np.median(ncontrib)), mean_rules_firing=float(np.mean(ncontrib)),
                       p90_rules_firing=float(np.quantile(ncontrib, .9)))
json.dump(traces, open(R + "example_traces.json", "w"), indent=1)
json.dump(full.rulebase(), open(R + "rulebase_full.json", "w"), indent=1)
print(out["fidelity"])

# ---------------------------------------------------------------- sensitivity analyses
K = main[FULL]["K"]
sens = {}
for label, feats in [("Without sleep predictors", [f for f in BASE if f not in DOMAINS["Sleep"]]),
                     ("Without self-rated health", [f for f in BASE if f != "srh"]),
                     ("Extended sleep items (2017-2020 only)", BASE + ["told_sleep_trouble", "daytime_sleepy"])]:
    r = {}
    for nm, make in [(FULL, lambda f=feats: FOFS(f, n_rules=K, priors=PRIORS, structure="main")),
                     ("Logistic regression", lambda f=feats: LRModel(f))]:
        m = make().fit(dev, dev.y.values)
        r[nm] = {"internal_test": roc_auc_score(test.y, m.score(test))}
        if "Extended" not in label:
            r[nm]["temporal"] = roc_auc_score(L.y, m.score(L))
    sens[label] = r
out["sensitivity"] = sens
print(json.dumps(sens, indent=1))

# ---------------------------------------------------------------- subgroups
RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian", 7: "Other/multiracial"}
sub = []
for sname, df in [("internal_test", test), ("temporal", L), ("external_pooled", POOL)]:
    groups = {"Sex: male": df.female == 0, "Sex: female": df.female == 1,
              "Age 18-39": df.age < 40, "Age 40-59": (df.age >= 40) & (df.age < 60), "Age >=60": df.age >= 60}
    groups |= {f"Race/ethnicity: {v}": df.RIDRETH3 == k for k, v in RACE.items()}
    for g, msk in groups.items():
        msk = msk.values
        if df.y[msk].sum() < 10:
            continue
        rr = dict(set=sname, group=g, n=int(msk.sum()), positives=int(df.y[msk].sum()))
        for nm in [FULL, "Logistic regression", "Gradient-boosted trees"]:
            s = preds[nm][sname]["s"][msk]; y = df.y.values[msk]
            rr[nm + " AUROC"] = roc_auc_score(y, s)
            rr[nm + " Sens"] = ((s >= thr[nm]) & (y == 1)).sum() / y.sum()
        sub.append(rr)
subdf = pd.DataFrame(sub); subdf.to_csv(R + "subgroups.csv", index=False)
out["subgroups"] = subdf.to_dict(orient="records")

# ---------------------------------------------------------------- descriptive table
def desc(df):
    r = {"N": len(df), "PHQ-9 >= 10, n (%)": f"{int(df.y.sum())} ({100 * df.y.mean():.1f})",
         "Weighted prevalence (%)": f"{100 * np.average(df.y, weights=df.wt):.1f}"}
    for v, lab in [("age", "Age, years"), ("pir", "Income-to-poverty ratio"), ("bmi", "BMI, kg/m2"),
                   ("sleep_wd", "Sleep, workdays (h)"), ("sleep_we", "Sleep, free days (h)"),
                   ("sedentary", "Sedentary time (h/day)"), ("n_chronic", "Chronic conditions (count)")]:
        r[lab + ", median (IQR)"] = f"{df[v].median():.1f} ({df[v].quantile(.25):.1f}-{df[v].quantile(.75):.1f})"
    for v, val, lab in [("female", 1, "Female"), ("smoker", 1, "Current smoker"), ("diabetes", 1, "Diabetes"),
                        ("hypertension", 1, "Hypertension"), ("educ", 5, "College graduate"),
                        ("marital", 1, "Married/partnered")]:
        r[lab + ", %"] = f"{100 * (df[v] == val).sum() / df[v].notna().sum():.1f}"
    r["Self-rated health fair/poor, %"] = f"{100 * (df.srh >= 4).sum() / df.srh.notna().sum():.1f}"
    r["Any predictor missing, %"] = f"{100 * df[BASE].isna().any(axis=1).mean():.1f}"
    r["Income-to-poverty ratio missing, %"] = f"{100 * df.pir.isna().mean():.1f}"
    return r


exP = prepare(pd.read_pickle("/home/claude/data/excluded_P.pkl"))
t1 = pd.DataFrame({"Development": desc(dev), "Internal test": desc(test), "Temporal (2021-2023)": desc(L), "External (2005-2014)": desc(POOL)})
t1["Excluded (2017-2020)"] = pd.Series({"N": len(exP),
                                        "Age, years, median (IQR)": f"{exP.age.median():.1f} ({exP.age.quantile(.25):.1f}-{exP.age.quantile(.75):.1f})",
                                        "Female, %": f"{100 * exP.female.mean():.1f}"})
t1.to_csv(R + "table1.csv")
out["table1"] = t1.fillna("").to_dict()

json.dump(out, open(R + "secondary_results.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
col = {"Box missing bounds": "#b5838d", FULL: "#1b6ca8", "Logistic regression": "#d1495b", "Gradient-boosted trees": "#2e8b57",
       "Crisp ontology-guided": "#8d6a9f", "Version 1 (conjunctive rules)": "#e09f3e", "Fuzzy without ontology": "#e09f3e", "No interval missing handling": "#6c757d"}
PANELS = [("internal_test", test, "Internal test (2017-2020)"), ("temporal", L, "Temporal (2021-2023)"),
          ("external_pooled", POOL, "External (2005-2014)")]
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for k, (sname, df, ttl) in enumerate(PANELS):
    for nm in [FULL, "Logistic regression", "Gradient-boosted trees", "Version 1 (conjunctive rules)"]:
        fpr, tpr, _ = roc_curve(df.y, preds[nm][sname]["s"])
        ax[k].plot(fpr, tpr, color=col[nm], lw=1.4, label=f"{nm} ({roc_auc_score(df.y, preds[nm][sname]['s']):.3f})")
    ax[k].plot([0, 1], [0, 1], ls=":", c="grey", lw=0.8); ax[k].set_title(ttl)
    ax[k].set_xlabel("1 - specificity"); ax[k].set_ylabel("Sensitivity"); ax[k].legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(R + "fig_roc.png", dpi=300); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for k, (sname, df, ttl) in enumerate(PANELS):
    for nm in [FULL, "Logistic regression", "Gradient-boosted trees"]:
        p = preds[nm][sname]["p"]; y = df.y.values
        q = pd.qcut(p, 10, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q", observed=True).mean()
        ax[k].plot(g.p, g.y, "o-", ms=3, color=col[nm], lw=1.2, label=nm)
    ax[k].plot([0, .6], [0, .6], ls=":", c="grey", lw=0.8); ax[k].set_xlim(0, .6); ax[k].set_ylim(0, .6)
    ax[k].set_title(ttl); ax[k].set_xlabel("Predicted probability (decile mean)"); ax[k].set_ylabel("Observed proportion")
    ax[k].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(R + "fig_calibration.png", dpi=300); plt.close(fig)

mc = agg[agg.scenario == "MCAR"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for nm in cmp:
    d = mc[mc.model == nm].sort_values("level")
    ax[0].plot(d.level * 100, d.AUROC, "o-", ms=3, lw=1.3, color=col[nm], label=nm)
ax[0].set_xlabel("Predictor values masked at random (%)"); ax[0].set_ylabel("AUROC (point score)"); ax[0].legend(fontsize=7)
d = mc[mc.model == FULL].sort_values("level")
ax[1].plot(d.level * 100, d.Coverage, "o-", ms=3, color=col[FULL], label="Coverage (decided cases)")
ax[1].plot(d.level * 100, d.Sens_refer, "s--", ms=3, color="#d1495b", label="Sensitivity, refer-on-abstain")
ax[1].plot(d.level * 100, d.Spec_refer, "^--", ms=3, color="#2e8b57", label="Specificity, refer-on-abstain")
ax[1].set_xlabel("Predictor values masked at random (%)"); ax[1].set_ylabel("Proportion"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(R + "fig_missing.png", dpi=300); plt.close(fig)
print("figures saved")
