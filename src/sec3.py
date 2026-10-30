"""Third set of secondary analyses: abstention based on the spread of multiple imputations and risk-coverage curves;
sensitivity by sex with sex-specific thresholds and recalibration; spline LR on all 10 masks at 20% and 50% MCAR."""
import json, pickle, time
import numpy as np, pandas as pd
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from rev_common import *  # noqa
from fofs import FOFS, prepare

S = pickle.load(open(R + "models.pkl", "rb")); NEW = pickle.load(open(RV + "new_models.pkl", "rb")); RC = pickle.load(open(RV + "rcs_model.pkl", "rb"))
models, preds, dev, test, L, POOL = S["models"], S["preds"], S["dev"], S["test"], S["L"], S["POOL"]
main = json.load(open(R + "main_results.json")); rmj = json.load(open(RV + "rev_models.json")); s2 = json.load(open(RV + "sec2_models.json"))
FULL = "Proposed (full)"; full = models[FULL]; slr = NEW["models"]["Sign-constrained logistic regression"]; mi = NEW["models"]["LR, multiple imputation"]
t_full = main[FULL]["threshold"]; t_slr = rmj["new_models"]["Sign-constrained logistic regression"]["threshold"]; t_mi = rmj["new_models"]["LR, multiple imputation"]["threshold"]
MASKABLE = [f for f in BASE if f not in ("age", "female")]
out = {}; t0 = time.time()


def mi_draws(d, n_per=4):
    X = ml_matrix(d, BASE); P = []
    for k, (imp, sc, lr) in enumerate(mi.ms):
        for j in range(n_per):
            imp.random_state_ = np.random.RandomState(1000 * k + j)
            P.append(lr.predict_proba(sc.transform(imp.transform(X)))[:, 1])
    return np.column_stack(P)


def margin_bounds(zl, zu, t):
    a, b, tt = logit(np.clip(zl, 1e-9, 1 - 1e-9)), logit(np.clip(zu, 1e-9, 1 - 1e-9)), logit(t)
    return np.where(a >= tt, a - tt, np.where(b < tt, tt - b, -np.minimum(tt - a, b - tt)))


def rc_curve(y, conf, dec, grid=np.linspace(0.05, 1.0, 20)):
    o = np.argsort(-conf, kind="stable"); res = []
    for c in grid:
        k = max(1, int(round(c * len(y)))); i = o[:k]; yy, dd = y[i], dec[i]
        sens = (dd & (yy == 1)).sum() / max((yy == 1).sum(), 1); spec = (~dd & (yy == 0)).sum() / max((yy == 0).sum(), 1)
        res.append(dict(coverage=float(c), sens=float(sens), spec=float(spec), bacc=float((sens + spec) / 2), err=float((dd != (yy == 1)).mean())))
    return res


def at_cov(y, conf, dec, c):
    return rc_curve(y, conf, dec, grid=np.array([c]))[0]


def evaluate(d):
    y = d.y.values; r = {}
    zl, zu = full.bounds(d); s = full.score(d); cov_b = (zl >= t_full) | (zu < t_full)
    conf_b = margin_bounds(zl, zu, t_full); dec_b = s >= t_full
    a, b = slr.bounds(d); ss = slr.score(d); cov_l = (a >= t_slr) | (b < t_slr)
    conf_l = margin_bounds(a, b, t_slr); dec_l = ss >= t_slr
    P = mi_draws(d); pm = P.mean(1); share = (P >= t_mi).mean(1); dec_m = pm >= t_mi
    cov_m = (share >= 0.9) | (share <= 0.1)
    conf_m = np.abs(share - 0.5) + 1e-3 * np.abs(logit(np.clip(pm, 1e-9, 1 - 1e-9)) - logit(t_mi))
    for nm, cov, conf, dec in (("bounds", cov_b, conf_b, dec_b), ("linear", cov_l, conf_l, dec_l), ("mi", cov_m, conf_m, dec_m)):
        own = dict(coverage=float(cov.mean()),
                   sens=float((dec & cov & (y == 1)).sum() / max((cov & (y == 1)).sum(), 1)),
                   spec=float((~dec & cov & (y == 0)).sum() / max((cov & (y == 0)).sum(), 1)),
                   sens_refer=float(((dec | ~cov) & (y == 1)).sum() / (y == 1).sum()),
                   spec_refer=float((~dec & cov & (y == 0)).sum() / (y == 0).sum()), flag_refer=float((dec | ~cov).mean()))
        r[nm] = dict(own=own, curve=rc_curve(y, conf, dec), at_bounds_cov=at_cov(y, conf, dec, float(cov_b.mean())))
    r["mi_auroc"] = float(roc_auc_score(y, pm)); r["n_draws"] = int(P.shape[1])
    return r


res = {}
for nm, d in [("internal_test", test), ("temporal", L), ("external_pooled", POOL)]:
    res["natural|" + nm] = evaluate(d); print(nm, round(time.time() - t0), flush=True)
rng = np.random.default_rng(SEED)
mc = []
for rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
    for rep in range(10 if rate > 0 else 1):
        d = test.copy(); mask = rng.random((len(d), len(MASKABLE))) < rate
        for j, f in enumerate(MASKABLE): d.loc[mask[:, j], f] = np.nan
        if rate == 0.3: mc.append(evaluate(d))
res["mcar30_reps"] = mc
r2 = np.random.default_rng(SEED + 101); mn = {"income_low": [], "srh_poor": [], "both": []}
for rep in range(10):
    for scen in ("income_low", "srh_poor", "both"):
        d = test.copy(); u = r2.random((len(d), 2))
        if scen in ("income_low", "both"):
            p_ = np.where(d.pir.values < 1.3, 0.6, 0.1); d.loc[u[:, 0] < p_, "pir"] = np.nan
        if scen in ("srh_poor", "both"):
            p_ = np.where(d.srh.values >= 4, 0.6, 0.1); d.loc[u[:, 1] < p_, "srh"] = np.nan
        mn[scen].append(evaluate(d))
    print("mnar", rep, round(time.time() - t0), flush=True)
res["mnar_reps"] = mn


def avg(list_):
    o = {}
    for meth in ("bounds", "linear", "mi"):
        o[meth] = {"own": {k: float(np.mean([x[meth]["own"][k] for x in list_])) for k in list_[0][meth]["own"]},
                   "at_bounds_cov": {k: float(np.mean([x[meth]["at_bounds_cov"][k] for x in list_])) for k in list_[0][meth]["at_bounds_cov"]},
                   "curve": [{k: float(np.mean([x[meth]["curve"][i][k] for x in list_])) for k in list_[0][meth]["curve"][i]} for i in range(len(list_[0][meth]["curve"]))]}
    return o


summ = {k: v for k, v in res.items() if k.startswith("natural")}
summ["mcar30"] = avg(mc)
for scen, l in mn.items(): summ["mnar|" + scen] = avg(l)
out["mi_abstention"] = summ

# ---------------------------------------------------------------- sensitivity by sex; sex-specific thresholds; recalibration
from experiment import cv_scores, pick_threshold, platt
oof = cv_scores(lambda: FOFS(BASE, n_rules=0, priors=PRIORS, structure="main"), dev)
thr_sex = {sx: pick_threshold(dev.y.values[dev.female.values == sx], oof[dev.female.values == sx]) for sx in (0, 1)}
oof_lr = cv_scores(lambda: LRModel(BASE), dev)
thr_sex_lr = {sx: pick_threshold(dev.y.values[dev.female.values == sx], oof_lr[dev.female.values == sx]) for sx in (0, 1)}
sx_out = {"thr_common": t_full, "thr_sex": thr_sex}
for nm, d in [("internal_test", test), ("temporal", L), ("external_pooled", POOL)]:
    y = d.y.values; fem = d.female.values; s = preds[FULL][nm]["s"]; sl = preds["Logistic regression"][nm]["s"]
    rr = {}
    for sx, lab in ((0, "male"), (1, "female")):
        m_ = fem == sx; yy = y[m_]
        f_c = s[m_] >= t_full; f_s = s[m_] >= thr_sex[sx]; f_lr = sl[m_] >= main["Logistic regression"]["threshold"]; f_lrs = sl[m_] >= thr_sex_lr[sx]
        # sex-specific recalibration-in-the-large on half of the cohort (same probability threshold), averaged over 200 splits
        p = np.clip(preds[FULL][nm]["p"][m_], 1e-6, 1 - 1e-6); tp = float(np.min(preds[FULL][nm]["p"][s >= t_full]))
        rg = np.random.default_rng(SEED); sens_rec = []
        if nm != "internal_test":
            from scipy.optimize import brentq
            for _ in range(200):
                i = rg.permutation(m_.sum()); a_, b_ = i[: len(i) // 2], i[len(i) // 2:]
                dl = brentq(lambda q: expit(logit(p[a_]) + q).mean() - yy[a_].mean(), -5, 5)
                q_ = expit(logit(p[b_]) + dl) >= tp; sens_rec.append(((q_) & (yy[b_] == 1)).sum() / (yy[b_] == 1).sum())
        rr[lab] = dict(n=int(m_.sum()), pos=int(yy.sum()), prev=float(yy.mean()),
                       sens_common=float((f_c & (yy == 1)).sum() / yy.sum()), spec_common=float((~f_c & (yy == 0)).sum() / (yy == 0).sum()),
                       flag_common=float(f_c.mean()),
                       sens_sex=float((f_s & (yy == 1)).sum() / yy.sum()), spec_sex=float((~f_s & (yy == 0)).sum() / (yy == 0).sum()), flag_sex=float(f_s.mean()),
                       sens_lr=float((f_lr & (yy == 1)).sum() / yy.sum()), sens_lr_sex=float((f_lrs & (yy == 1)).sum() / yy.sum()),
                       auroc=float(roc_auc_score(yy, s[m_])), oe=float(yy.mean() / preds[FULL][nm]["p"][m_].mean()),
                       sens_recal=float(np.mean(sens_rec)) if sens_rec else None)
    sx_out[nm] = rr
out["sex"] = sx_out
print("sex", json.dumps(sx_out)[:600], flush=True)

# ---------------------------------------------------------------- spline LR on all 10 masks at 20% and 50% MCAR
rcs = RC["model"]; rng = np.random.default_rng(SEED); aucs = {"20": [], "50": []}
for rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
    for rep in range(10 if rate > 0 else 1):
        d = test.copy(); mask = rng.random((len(d), len(MASKABLE))) < rate
        for j, f in enumerate(MASKABLE): d.loc[mask[:, j], f] = np.nan
        if rate in (0.2, 0.5):
            t_r = s2["rcs"]["threshold"]; a, b = rcs.bounds(d)
            aucs[str(int(rate * 100))].append((roc_auc_score(d.y, rcs.score(d)), float(((a >= t_r) | (b < t_r)).mean())))
out["rcs_full10"] = {k: dict(AUROC=float(np.mean([x[0] for x in v])), Coverage=float(np.mean([x[1] for x in v]))) for k, v in aucs.items()}
json.dump(out, open(RV + "sec3.json", "w"), indent=1, default=float)
print("done sec3", round(time.time() - t0))
