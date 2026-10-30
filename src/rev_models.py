"""Secondary analyses, part 1: additional comparators, record-typical predictor set, PHQ-2 reference point,
interpretability comparison, decision curves, recalibrated operating points, subgroup O/E, selection table."""
import json, pickle, os, time
import numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from rev_common import *  # noqa
from experiment_v2 import run, CYC
from experiment import platt, metrics
from fofs import prepare

os.makedirs(RV, exist_ok=True)
S = pickle.load(open(R + "models.pkl", "rb"))
models, preds, dev, test, L, POOL, EXT = S["models"], S["preds"], S["dev"], S["test"], S["L"], S["POOL"], S["EXT"]
main = json.load(open(R + "main_results.json"))
SETS = {"internal_test": test, "temporal": L, "external_pooled": POOL} | {f"external_{CYC[c]}": EXT[c] for c in "DEFGH"}
FULLCI = ("internal_test", "temporal", "external_pooled")
FULL = "Proposed (full)"
out = {}

# ------------------------------------------------------------------ 1. additional comparators
new = {"Sign-constrained logistic regression": lambda: SignLR(BASE),
       "Unconstrained linear LR (no indicators)": lambda: SignLR(BASE, constrained=False),
       "LR, median imputation without indicators": lambda: LRNoInd(BASE),
       "LR, multiple imputation": lambda: LRMI(BASE),
       "Monotone gradient-boosted trees": lambda: MonoGBT(BASE),
       "Explainable boosting machine (GA2M)": lambda: EBM(BASE)}
if os.path.exists(RV + "cache1.pkl"):
    NEWM, NEWP, NEWO, paired = pickle.load(open(RV + "cache1.pkl", "rb"))
else:
    NEWM, NEWP, NEWO = {}, {}, {}
    for nm, mk in new.items():
        t0 = time.time()
        m, o, p = run(nm, mk, SETS, FULLCI)
        NEWM[nm], NEWO[nm], NEWP[nm] = m, o, p
        print(nm, round(time.time() - t0), {k: round(v["AUROC"], 3) for k, v in o["sets"].items()}, flush=True)
    NEWO["Monotone gradient-boosted trees"]["params"] = NEWM["Monotone gradient-boosted trees"].params
    NEWO["Sign-constrained logistic regression"]["lambda"] = NEWM["Sign-constrained logistic regression"].lam
    paired = {}
    for sname, df in SETS.items():
        y = df.y.values; sp = preds[FULL][sname]["s"]
        for nm in NEWP:
            so = NEWP[nm][sname]["s"]; r = np.random.default_rng(SEED); d = []
            for _ in range(2000 if sname in FULLCI else 1000):
                i = r.integers(0, len(y), len(y)); d.append(roc_auc_score(y[i], sp[i]) - roc_auc_score(y[i], so[i]))
            d = np.array(d)
            paired[f"{sname}|{nm}"] = dict(diff=roc_auc_score(y, sp) - roc_auc_score(y, so), lo=float(np.quantile(d, .025)),
                                           hi=float(np.quantile(d, .975)),
                                           p_two_sided=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))

    pickle.dump((NEWM, NEWP, NEWO, paired), open(RV + "cache1.pkl", "wb"))
out["new_models"] = NEWO; out["new_paired"] = paired
sl = NEWM["Sign-constrained logistic regression"]; ul = NEWM["Unconstrained linear LR (no indicators)"]
out["signlr_coef"] = {c: [float(a), float(b)] for c, a, b in zip(sl.cols, sl.coef_raw, ul.coef_raw)}
out["signlr_n_zero"] = int(np.sum(np.abs(sl.w) < 1e-8))
out["unconstrained_lr_violations"] = [c for c, b in zip(ul.cols, ul.coef_raw) if RAW_SIGN.get(c, 0) * b < 0]
out["ebm_terms"] = NEWM["Explainable boosting machine (GA2M)"].n_terms()
out["ebm_pairs"] = [list(map(int, t)) for t in NEWM["Explainable boosting machine (GA2M)"].m.term_features_ if len(t) == 2]
out["ebm_feature_names"] = list(NEWM["Explainable boosting machine (GA2M)"].m.feature_names_in_)

# ------------------------------------------------------------------ 2. pinned coefficients
full = models[FULL]
out["full_n_at_zero"] = int(np.sum((full.signs != 0) & (np.abs(full.beta) < 1e-8)))
out["full_n_signed"] = int(np.sum(full.signs != 0))
for nm in ["Version 1 (conjunctive rules)", "Main effects + interactions"]:
    m = models[nm]
    out[f"at_zero|{nm}"] = [int(np.sum((m.signs != 0) & (np.abs(m.beta) < 1e-8))), int(np.sum(m.signs != 0)), len(m.rules)]
# unconstrained fit at the same K and structure
from fofs import FOFS
for nm, kw, K in [("Version 1 (conjunctive rules)", dict(), 32), ("Main effects + interactions", dict(structure="main"), 16)]:
    mu = FOFS(BASE, n_rules=K, priors=PRIORS, direction=False, **kw).fit(dev, dev.y.values)
    out[f"unconstrained_same_K|{nm}"] = {s: float(roc_auc_score(d.y, mu.score(d))) for s, d in
                                         [("internal_test", test), ("temporal", L), ("external_pooled", POOL)]}
    out[f"constrained_same_K|{nm}"] = {s: float(roc_auc_score(SETS[s].y, preds[nm][s]["s"])) for s in ("internal_test", "temporal", "external_pooled")}

# ------------------------------------------------------------------ 3. record-typical predictor sets
rec_sets = {}
for lab, feats in [("EHR9", EHR), ("EHR7", EHR_CORE)]:
    r = {}
    for nm, mk in [("Proposed", lambda f=feats: FOFS(f, n_rules=0, priors=PRIORS, structure="main")),
                   ("Logistic regression", lambda f=feats: LRModel(f)),
                   ("Gradient-boosted trees", lambda f=feats: GBTModel(f))]:
        m, o, p = run(nm, mk, {k: SETS[k] for k in FULLCI}, FULLCI)
        r[nm] = {s: dict(AUROC=v["AUROC"], CI=v["CI"]["AUROC"], Sensitivity=v["Sensitivity"], Specificity=v["Specificity"],
                         Flagged=v["Flagged"], OE=v["OE"], Coverage=v.get("Coverage")) for s, v in o["sets"].items()}
    rec_sets[lab] = r
    print(lab, {k: round(v["internal_test"]["AUROC"], 3) for k, v in r.items()}, flush=True)
out["record_typical"] = rec_sets

# ------------------------------------------------------------------ 4. PHQ-2 >= 3 reference point
phq2 = {}
for sname, df in SETS.items():
    q = df.phq2.values; y = df.y.values; ok = ~np.isnan(q); f = q[ok] >= 3; yy = y[ok]
    phq2[sname] = dict(n=int(ok.sum()), Sensitivity=float((f & (yy == 1)).sum() / (yy == 1).sum()),
                       Specificity=float((~f & (yy == 0)).sum() / (yy == 0).sum()), Flagged=float(f.mean()),
                       PPV=float((f & (yy == 1)).sum() / max(f.sum(), 1)), AUROC_phq2=float(roc_auc_score(yy, q[ok])))
out["phq2"] = phq2

# ------------------------------------------------------------------ 5. interpretability comparison
lrm = models["Logistic regression"].m
lr_coef = lrm[-1].coef_[0]; lr_names = list(lrm[0].get_feature_names_out())
Xt = test
fire = full._rule_mu(full._impute_mu(full._mu(prepare(Xt))))
act_full = (np.abs(fire * full.beta) > 1e-12).sum(1)
out["interp"] = dict(full_terms=len(full.rules) + 1, full_active_median=float(np.median(act_full)),
                     full_active_p90=float(np.quantile(act_full, .9)),
                     lr_terms=int(np.sum(np.abs(lr_coef) > 1e-12)) + 1, lr_names=lr_names,
                     lr_coef=[float(c) for c in lr_coef],
                     lr_sign_violations=[n for n, c in zip(lr_names, lr_coef) if RAW_SIGN.get(n, 0) * c < 0],
                     ebm_terms=out["ebm_terms"] + 1)

# ------------------------------------------------------------------ 6. decision curves, calibration, recalibrated operating points
TH = np.round(np.arange(0.02, 0.301, 0.01), 3)
allp = {nm: preds[nm] for nm in [FULL, "Logistic regression", "Gradient-boosted trees"]}
allp["Explainable boosting machine (GA2M)"] = NEWP["Explainable boosting machine (GA2M)"]
dca = {}
for sname in FULLCI:
    y = SETS[sname].y.values; n = len(y); prev = y.mean()
    d = {"treat_all": [float(prev - (1 - prev) * t / (1 - t)) for t in TH], "thresholds": TH.tolist()}
    for nm, pp in allp.items():
        p = pp[sname]["p"]
        d[nm] = [float(((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t)) for t in TH]
        # recalibrated in the large (intercept estimated on the same cohort; illustrative upper reference)
        lp = logit(np.clip(p, 1e-6, 1 - 1e-6)); dl = brentq(lambda a: expit(lp + a).mean() - y.mean(), -5, 5)
        q = expit(lp + dl)
        d[nm + " (recalibrated)"] = [float(((q >= t) & (y == 1)).sum() / n - ((q >= t) & (y == 0)).sum() / n * t / (1 - t)) for t in TH]
    dca[sname] = d
out["dca"] = dca

opr = {}
cal_full = None
for sname in ("temporal", "external_pooled"):
    for nm in [FULL, "Logistic regression", "Gradient-boosted trees"]:
        y = SETS[sname].y.values; p = np.clip(preds[nm][sname]["p"], 1e-6, 1 - 1e-6); s = preds[nm][sname]["s"]
        thr_s = main[nm]["threshold"]
        # probability-scale threshold equivalent to the score threshold (Platt map is monotone)
        tp = float(np.min(p[s >= thr_s])) if (s >= thr_s).any() else 1.0
        lp = logit(p); rng = np.random.default_rng(SEED); res = []
        for _ in range(200):
            i = rng.permutation(len(y)); a, b = i[: len(y) // 2], i[len(y) // 2:]
            dl = brentq(lambda d: expit(lp[a] + d).mean() - y[a].mean(), -5, 5)
            q = expit(lp[b] + dl); f = q >= tp; yy = y[b]
            res.append([(f & (yy == 1)).sum() / yy.sum(), (~f & (yy == 0)).sum() / (yy == 0).sum(), f.mean(),
                        (~f).mean() * 1000, ((~f) & (yy == 1)).sum() / len(yy) * 1000])
        res = np.array(res)
        f0 = s >= thr_s
        before = [(f0 & (y == 1)).sum() / y.sum(), (~f0 & (y == 0)).sum() / (y == 0).sum(), f0.mean(), (~f0).mean() * 1000,
                  ((~f0) & (y == 1)).sum() / len(y) * 1000]
        opr[f"{sname}|{nm}"] = dict(prob_threshold=tp, before=[float(x) for x in before], after=res.mean(0).tolist(),
                                    after_lo=np.quantile(res, .025, 0).tolist(), after_hi=np.quantile(res, .975, 0).tolist())
for nm in [FULL, "Logistic regression", "Gradient-boosted trees"]:
    y = test.y.values; s = preds[nm]["internal_test"]["s"]; f0 = s >= main[nm]["threshold"]
    opr[f"internal_test|{nm}"] = dict(before=[float((f0 & (y == 1)).sum() / y.sum()), float((~f0 & (y == 0)).sum() / (y == 0).sum()),
                                              float(f0.mean()), float((~f0).mean() * 1000), float(((~f0) & (y == 1)).sum() / len(y) * 1000)])
out["operating"] = opr

# ------------------------------------------------------------------ 7. subgroup O/E
RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian", 7: "Other/multiracial"}
sub = []
for sname, df in [("internal_test", test), ("temporal", L), ("external_pooled", POOL)]:
    groups = {"Sex: male": df.female == 0, "Sex: female": df.female == 1, "Age 18-39": df.age < 40,
              "Age 40-59": (df.age >= 40) & (df.age < 60), "Age >=60": df.age >= 60}
    groups |= {f"Race/ethnicity: {v}": df.RIDRETH3 == k for k, v in RACE.items()}
    for g, msk in groups.items():
        msk = msk.values; y = df.y.values[msk]
        if y.sum() < 10: continue
        rr = dict(set=sname, group=g)
        for nm in [FULL, "Logistic regression", "Gradient-boosted trees"]:
            rr[nm + " OE"] = float(y.mean() / preds[nm][sname]["p"][msk].mean())
        sub.append(rr)
pd.DataFrame(sub).to_csv(RV + "subgroup_oe.csv", index=False)

# ------------------------------------------------------------------ 8. selection into the analytic cohort (minor 9)
inc = prepare(pd.read_pickle("/home/claude/data/cohort_P.pkl")); exc = prepare(pd.read_pickle("/home/claude/data/excluded_P.pkl"))
incL = prepare(pd.read_pickle("/home/claude/data/cohort_L.pkl")); excL = prepare(pd.read_pickle("/home/claude/data/excluded_L.pkl"))
def d_(df):
    r = {"N": f"{len(df):,}"}
    for v, lab in [("age", "Age, years"), ("pir", "Income-to-poverty ratio"), ("bmi", "BMI, kg/m²"), ("sleep_wd", "Sleep, workdays (h)"),
                   ("sleep_we", "Sleep, free days (h)"), ("sedentary", "Sedentary time (h/day)")]:
        x = df[v].dropna()
        r[lab + ", median (IQR)"] = f"{x.median():.1f} ({x.quantile(.25):.1f}–{x.quantile(.75):.1f})" if len(x) else "–"
    for v, val, lab in [("female", 1, "Female, %"), ("smoker", 1, "Current smoker, %"), ("diabetes", 1, "Diabetes, %"),
                        ("hypertension", 1, "Hypertension, %"), ("educ", 5, "College graduate, %"), ("marital", 1, "Married/partnered, %")]:
        x = df[v].dropna(); r[lab] = f"{100 * (x == val).mean():.1f}" if len(x) else "–"
    x = df.srh.dropna(); r["Self-rated health fair/poor, %"] = f"{100 * (x >= 4).mean():.1f}" if len(x) else "–"
    x = df.n_chronic.dropna(); r["≥ 2 chronic conditions, %"] = f"{100 * (x >= 2).mean():.1f}" if len(x) else "–"
    r["Any predictor missing, %"] = f"{100 * df[BASE].isna().any(axis=1).mean():.1f}"
    return r
sel = pd.DataFrame({"Included 2017–2020": d_(inc), "Excluded 2017–2020": d_(exc), "Included 2021–2023": d_(incL), "Excluded 2021–2023": d_(excL)})
sel.to_csv(RV + "selection.csv")

# ------------------------------------------------------------------ 9. weighted vs unweighted AUROC gaps (minor 2)
gaps = {}
for nm in main:
    if isinstance(main[nm], dict) and "sets" in main[nm]:
        gaps[nm] = {s: main[nm]["sets"][s]["AUROC"] - main[nm]["sets"][s]["AUROC_weighted"] for s in FULLCI}
for nm, o in NEWO.items():
    gaps[nm] = {s: o["sets"][s]["AUROC"] - o["sets"][s]["AUROC_weighted"] for s in FULLCI}
out["wgap"] = gaps

json.dump(out, open(RV + "rev_models.json", "w"), indent=1, default=float)
pickle.dump(dict(models=NEWM, preds=NEWP), open(RV + "new_models.pkl", "wb"))
print("done rev_models")
