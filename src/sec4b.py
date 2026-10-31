"""Smaller analyses for the fourth set: design-based intervals for the primary comparison, minimum sample size (Riley
et al.), calibration-map variants, the log-odds contribution of the uncollected free-day sleep, missingness excluding
the uncollected input, and acquisition compared on expected item cost."""
import json, pickle, time
import numpy as np, pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from rev_common import *  # noqa
from experiment import cv_scores, platt, cal_slope_int
from exact import ref_distribution

S = pickle.load(open(R + "models.pkl", "rb")); NEW = pickle.load(open(RV + "new_models.pkl", "rb"))
RC = pickle.load(open(RV + "rcs_model.pkl", "rb")); main = json.load(open(R + "main_results.json"))
dev, test, L, POOL, preds = S["dev"], S["test"], S["L"], S["POOL"], S["preds"]
full = S["models"]["Proposed (full)"]; FULL = "Proposed (full)"
out = {}; t0 = time.time()

# ---------------------------------------------------------------- A. design-based (stratified cluster) bootstrap, 2005-2014
y = POOL.y.values
SC = {"full": preds[FULL]["external_pooled"]["s"], "lr": preds["Logistic regression"]["external_pooled"]["s"],
      "lrni": NEW["preds"]["LR, median imputation without indicators"]["external_pooled"]["s"],
      "rcs": RC["preds"]["external_pooled"]["s"]}
strat = (POOL.cycle.astype(str) + "_" + POOL.SDMVSTRA.astype(int).astype(str)).values
psu = (pd.Series(strat) + "_" + POOL.SDMVPSU.astype(int).astype(str)).values
groups = {}
for i, (h, c) in enumerate(zip(strat, psu)): groups.setdefault(h, {}).setdefault(c, []).append(i)
groups = {h: [np.array(v) for v in d.values()] for h, d in groups.items()}
rg = np.random.default_rng(SEED); D = {k: [] for k in ("lr", "lrni", "rcs")}
for b in range(2000):
    idx = []
    for h, cl in groups.items():
        pick = rg.integers(0, len(cl), len(cl)); idx += [cl[j] for j in pick]
    i = np.concatenate(idx); a = roc_auc_score(y[i], SC["full"][i])
    for k in D: D[k].append(a - roc_auc_score(y[i], SC[k][i]))
out["design_boot"] = {k: dict(d=float(roc_auc_score(y, SC["full"]) - roc_auc_score(y, SC[k])),
                              lo=float(np.quantile(v, .025)), hi=float(np.quantile(v, .975)),
                              p=float(min(1, 2 * min(np.mean(np.array(v) <= 0), np.mean(np.array(v) >= 0)))))
                      for k, v in D.items()}
out["design_boot"]["n_strata"] = len(groups); out["design_boot"]["n_psu"] = int(sum(len(v) for v in groups.values()))
print("design", out["design_boot"], round(time.time() - t0), flush=True)

# ---------------------------------------------------------------- B. minimum sample size (Riley et al. 2020)
phi = float(dev.y.mean()); P = len(full.beta); C = 0.78
def sim_r2(sig, n=400000, seed=1):
    g = np.random.default_rng(seed); z = g.normal(0, sig, n)
    mu = 0.0
    for _ in range(60):  # intercept that gives prevalence phi
        mu -= (expit(mu + z).mean() - phi) * 4
    lp = mu + z; yy = g.random(n) < expit(lp)
    ll1 = np.sum(np.where(yy, np.log(expit(lp)), np.log(1 - expit(lp)))); p0 = yy.mean()
    ll0 = np.sum(np.where(yy, np.log(p0), np.log(1 - p0)))
    return 1 - np.exp(-2 * (ll1 - ll0) / n), roc_auc_score(yy, lp)
lo_, hi_ = 0.2, 3.0
for _ in range(30):
    mid = (lo_ + hi_) / 2; r2, c = sim_r2(mid)
    lo_, hi_ = (mid, hi_) if c < C else (lo_, mid)
R2 = sim_r2((lo_ + hi_) / 2)[0]
R2max = 1 - (phi ** phi * (1 - phi) ** (1 - phi)) ** 2
n1 = P / ((0.9 - 1) * np.log(1 - R2 / 0.9))
S2 = R2 / (R2 + 0.05 * R2max); n2 = P / ((S2 - 1) * np.log(1 - R2 / S2))
n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi)
nreq = int(np.ceil(max(n1, n2, n3)))
out["riley"] = dict(P=P, phi=phi, C=C, R2cs=float(R2), R2max=float(R2max), n1=float(n1), n2=float(n2), n3=float(n3),
                    n_required=nreq, events_required=int(np.ceil(nreq * phi)), n_dev=int(len(dev)), events_dev=int(dev.y.sum()),
                    epp_required=float(nreq * phi / P), epp_dev=float(dev.y.sum() / P))
print("riley", out["riley"], flush=True)

# ---------------------------------------------------------------- C. calibration-map variants on the internal test set
yt = test.y.values; cal = {}
for nm, mk in [("lr", lambda: LRModel(BASE)), ("full", lambda: FOFS(BASE, n_rules=0, priors=PRIORS, structure="main"))]:
    key = "Logistic regression" if nm == "lr" else FULL
    oof = cv_scores(mk, dev); m = S["models"][key]; s_dev = m.score(dev); s_t = preds[key]["internal_test"]["s"]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(oof, dev.y.values)
    cal[nm] = dict(raw=float(cal_slope_int(yt, s_t)), platt_oof=float(cal_slope_int(yt, platt(dev.y.values, oof)(s_t))),
                   platt_insample=float(cal_slope_int(yt, platt(dev.y.values, s_dev)(s_t))),
                   isotonic_oof=float(cal_slope_int(yt, np.clip(iso.predict(s_t), 1e-4, 1 - 1e-4))))
out["calibration_maps"] = cal; print("cal", cal, flush=True)

# ---------------------------------------------------------------- D. log-odds added to every 2005-2014 point score for free-day sleep
cvar = np.array([full.concepts[c]["variable"] for c in full.cnames])
we = [k for k, r in enumerate(full.rules) if all(cvar[j] == "sleep_we" for j in r)]
contrib = float(sum(full.beta[k] * full.mu_fill[full.rules[k][0]] for k in we))
out["sleep_we_fill"] = dict(contrib=contrib, terms={full.cnames[full.rules[k][0]]: [float(full.beta[k]), float(full.mu_fill[full.rules[k][0]])] for k in we})
print("fill", out["sleep_we_fill"], flush=True)

# ---------------------------------------------------------------- E. any predictor missing, excluding free-day sleep
oth = [v for v in BASE if v != "sleep_we"]
out["anymiss_excl_we"] = {nm: float(d[oth].isna().any(axis=1).mean()) for nm, d in [("dev", dev), ("internal_test", test), ("temporal", L), ("external_pooled", POOL)]}
out["anymiss_excl_we_sed"] = float(POOL[[v for v in oth if v != "sedentary"]].isna().any(axis=1).mean())
print("anymiss", out["anymiss_excl_we"], flush=True)

# ---------------------------------------------------------------- F. acquisition compared on expected item cost
MASKABLE = [f for f in BASE if f not in ("age", "female")]
ref = {v: ref_distribution(dev, v) for v in MASKABLE}
acqO = json.load(open(RV + "rev_missing.json"))["acq_fixed_order"]
COST = ["diabetes", "hypertension", "n_chronic", "bmi", "smoker", "marital", "educ", "pir", "srh", "sleep_wd", "sleep_we", "sedentary"]
REC = {"diabetes", "hypertension", "n_chronic", "bmi", "smoker", "marital", "educ"}
def item_cost(v, interview=3.0): return 1.0 if v in REC else (2.0 if v == "pir" else interview)
thr = main[FULL]["threshold"]


def acquire(row_m, row_t, policy, r, interview=3.0):
    cur = row_m.copy(); asked = []
    while True:
        zl, zu = full.bounds(cur)
        if zl[0] >= thr or zu[0] < thr: return asked
        miss = [v for v in MASKABLE if pd.isna(cur[v].iloc[0]) and pd.notna(row_t[v].iloc[0])]
        if not miss: return asked
        if policy == "random": v = miss[r.integers(len(miss))]
        elif policy == "fixed": v = [x for x in acqO if x in miss][0]
        elif policy == "cost": v = [x for x in COST if x in miss][0]
        else:
            best, v = -1.0, miss[0]
            for c in miss:
                vals, w = ref[c]; Rr = pd.concat([cur] * len(vals), ignore_index=True); Rr[c] = vals
                a, b = full.bounds(Rr); sc = float(np.sum(w * ((a >= thr) | (b < thr))))
                if policy == "bound_cost": sc /= item_cost(c, interview)
                if sc > best + 1e-12: best, v = sc, c
        cur[v] = row_t[v].iloc[0]; asked.append(v)


acq = {}
d = test.copy(); mask = np.random.default_rng(SEED + 11).random((len(d), len(MASKABLE))) < 0.3
for j, f in enumerate(MASKABLE): d.loc[mask[:, j], f] = np.nan
zl, zu = full.bounds(d); ab = np.where((zl < thr) & (zu >= thr))[0]
seqs = {}
for pol in ("bound", "bound_cost", "fixed", "cost", "random"):
    rr = np.random.default_rng(SEED); seqs[pol] = [acquire(d.iloc[[i]].copy(), test.iloc[[i]].copy(), pol, rr) for i in ab]
res = {"n": int(len(ab))}
for ic in (3.0, 5.0):
    for pol, sq in seqs.items():
        res[f"{pol}_cost{int(ic)}"] = float(np.mean([sum(item_cost(v, ic) for v in s) for s in sq]))
for pol, sq in seqs.items(): res[f"{pol}_n"] = float(np.mean([len(s) for s in sq]))
bs = np.random.default_rng(SEED)
for ic in (3.0, 5.0):
    cb = np.array([sum(item_cost(v, ic) for v in s) for s in seqs["bound_cost"]])
    for other in ("bound", "fixed", "cost", "random"):
        co = np.array([sum(item_cost(v, ic) for v in s) for s in seqs[other]]); diff = co - cb
        boots = [diff[bs.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        res[f"d_{other}_cost{int(ic)}"] = [float(diff.mean()), float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]
acq["masked30_internal"] = res; out["acq_cost"] = acq
print("acq", res, round(time.time() - t0), flush=True)
json.dump(out, open(RV + "sec4b.json", "w"), indent=1, default=float)
print("done sec4b")
