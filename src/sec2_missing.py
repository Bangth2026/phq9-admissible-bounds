"""Second set of secondary analyses, part 2: bounds under natural missingness (internal, temporal, external),
missing-not-at-random masking, decision value at matched workload, added comparators in the MCAR stress test,
narrowed ranges on the same masks, and Algorithm 2 with per-record paired comparison and a cost-based order."""
import json, pickle, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from rev_common import *  # noqa
from fofs import FOFS, prepare, membership
from exact import ref_distribution

S = pickle.load(open(R + "models.pkl", "rb")); NEW = pickle.load(open(RV + "new_models.pkl", "rb"))
RC = pickle.load(open(RV + "rcs_model.pkl", "rb"))
models, dev, test, L, POOL, EXT = S["models"], S["dev"], S["test"], S["L"], S["POOL"], S["EXT"]
main = json.load(open(R + "main_results.json")); rmj = json.load(open(RV + "rev_models.json")); s2 = json.load(open(RV + "sec2_models.json"))
M = models | NEW["models"] | {"LR with restricted cubic splines": RC["model"]}
thr = {nm: main[nm]["threshold"] for nm in models} | {nm: rmj["new_models"][nm]["threshold"] for nm in NEW["models"]}
thr["LR with restricted cubic splines"] = s2["rcs"]["threshold"]
FULL = "Proposed (full)"; full = M[FULL]
MASKABLE = [f for f in BASE if f not in ("age", "female")]
BOUNDED = [FULL, "Box missing bounds", "Sign-constrained logistic regression", "LR with restricted cubic splines"]
POINT = ["Logistic regression", "LR, median imputation without indicators", "LR, multiple imputation", "Sign-constrained logistic regression",
         "LR with restricted cubic splines", "Gradient-boosted trees", "Monotone gradient-boosted trees", "Explainable boosting machine (GA2M)",
         "No interval missing handling"]
out = {}
cont = ["age", "pir", "bmi", "sleep_wd", "sleep_we", "sedentary"]
P99 = {v: (float(np.nanquantile(dev[v], .01)), float(np.nanquantile(dev[v], .99))) for v in cont}


def dec(y, pos, neg):
    ab = ~(pos | neg); cov = ~ab; ref = pos | ab
    return dict(Coverage=cov.mean(), Abstain=ab.mean(), Sens_refer=(ref & (y == 1)).sum() / y.sum(),
                Spec_refer=(neg & (y == 0)).sum() / (y == 0).sum(), Flag_refer=ref.mean(),
                Sens_covered=(pos & (y == 1)).sum() / max(((y == 1) & cov).sum(), 1),
                Spec_covered=(neg & (y == 0)).sum() / max(((y == 0) & cov).sum(), 1))


def nb(y, flag, pt):
    n = len(y); return float((flag & (y == 1)).sum() / n - (flag & (y == 0)).sum() / n * pt / (1 - pt))


def evaluate(d, narrowed=False):
    """All metrics on one (possibly masked) data set."""
    y = d.y.values; res = []
    for nm in BOUNDED:
        m = M[nm]; t = thr[nm]; zl, zu = m.bounds(d); r = dict(model=nm, kind="bounds", **dec(y, zl >= t, zu < t))
        r["MeanWidth"] = float(np.mean(zu - zl))
        for pt in (0.05, 0.10): r[f"NB_refer_{pt}"] = nb(y, (zl >= t) | ((zl < t) & (zu >= t)), pt)
        if narrowed and nm in (FULL, "Sign-constrained logistic regression", "LR with restricted cubic splines"):
            if isinstance(m, FOFS):
                from exact import ExactBounds
                a, b = ExactBounds(m, ranges=P99).bounds(d)
            else:
                a, b = m.bounds(d, ranges=P99)
            r["Coverage_p99"] = float(1 - ((a < t) & (b >= t)).mean())
        if nm == FULL:
            s = m.score(d); uv = BASE; comp = ~d[uv].isna().any(axis=1).values
            res.append(dict(model="Abstain if any input missing", kind="bounds", **dec(y, comp & (s >= t), comp & (s < t)),
                            **{f"NB_refer_{pt}": nb(y, ~(comp & (s < t)), pt) for pt in (0.05, 0.10)}))
            if "sleep_we" in d and d.sleep_we.isna().any():
                dec_mask = (zl >= t) | (zu < t)
                r["Share_decided_with_we_missing"] = float((dec_mask & d.sleep_we.isna().values).sum() / max(dec_mask.sum(), 1))
        res.append(r)
    for nm in POINT:
        m = M[nm]; t = thr[nm]; s = m.score(d)
        r = dict(model=nm, kind="point", AUROC=roc_auc_score(y, s), Sens=((s >= t) & (y == 1)).sum() / y.sum(),
                 Spec=((s < t) & (y == 0)).sum() / (y == 0).sum(), Flag=(s >= t).mean())
        for pt in (0.05, 0.10): r[f"NB_{pt}"] = nb(y, s >= t, pt)
        res.append(r)
    sp = M[FULL].score(d); res.append(dict(model=FULL, kind="point", AUROC=roc_auc_score(y, sp),
                                           Sens=((sp >= thr[FULL]) & (y == 1)).sum() / y.sum(), Spec=((sp < thr[FULL]) & (y == 0)).sum() / (y == 0).sum(),
                                           Flag=(sp >= thr[FULL]).mean(), **{f"NB_{pt}": nb(y, sp >= thr[FULL], pt) for pt in (0.05, 0.10)}))
    # matched workload: every point-score model flagged at the proportion referred by the proposed refer-on-abstain policy
    zl, zu = M[FULL].bounds(d); t = thr[FULL]; q = float(((zl >= t) | ((zl < t) & (zu >= t))).mean())
    for nm in POINT + [FULL]:
        s = M[nm].score(d); k = int(round(q * len(s)))
        if k <= 0: continue
        cut = np.sort(s)[::-1][k - 1]; f = s >= cut
        res.append(dict(model=nm, kind="matched", Flag=float(f.mean()), Sens=float((f & (y == 1)).sum() / y.sum()),
                        Spec=float((~f & (y == 0)).sum() / (y == 0).sum())))
    return res


t0 = time.time()
# ---------------------------------------------------------------- A. natural missingness
nat = {}
for nmset, d in [("internal_test", test), ("temporal", L), ("external_pooled", POOL)] + [(f"external_{k}", EXT[k]) for k in "DEFGH"]:
    nat[nmset] = evaluate(d, narrowed=nmset in ("internal_test", "external_pooled"))
    nat[nmset + "_missing_any"] = float(d[BASE].isna().any(axis=1).mean())
    nat[nmset + "_pir_missing"] = float(d.pir.isna().mean())
    print("natural", nmset, round(time.time() - t0), flush=True)
out["natural"] = nat

# ---------------------------------------------------------------- B. MCAR (same masks as the first stress test) + MNAR
rows = []
rng = np.random.default_rng(SEED)
for rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
    for rep in range(10 if rate > 0 else 1):
        d = test.copy(); mask = rng.random((len(d), len(MASKABLE))) < rate
        for j, f in enumerate(MASKABLE): d.loc[mask[:, j], f] = np.nan
        if rate in (0.1, 0.3, 0.0) or rep < 3:
            for r in evaluate(d, narrowed=rate in (0.0, 0.1, 0.3)): rows.append(dict(scenario="MCAR", level=str(int(rate * 100)), rep=rep, **r))
    print("mcar", rate, round(time.time() - t0), flush=True)
r2 = np.random.default_rng(SEED + 101)
for rep in range(10):
    for scen in ("income_low", "srh_poor", "both"):
        d = test.copy(); u = r2.random((len(d), 2))
        if scen in ("income_low", "both"):
            p_ = np.where(d.pir.values < 1.3, 0.6, 0.1); d.loc[u[:, 0] < p_, "pir"] = np.nan
        if scen in ("srh_poor", "both"):
            p_ = np.where(d.srh.values >= 4, 0.6, 0.1); d.loc[u[:, 1] < p_, "srh"] = np.nan
        for r in evaluate(d): rows.append(dict(scenario="MNAR", level=scen, rep=rep, **r))
    print("mnar", rep, round(time.time() - t0), flush=True)
st = pd.DataFrame(rows); st.to_csv(RV + "sec2_stress.csv", index=False)
agg = st.groupby(["scenario", "level", "model", "kind"]).mean(numeric_only=True).drop(columns="rep").reset_index()
out["stress"] = agg.to_dict(orient="records")
json.dump(out, open(RV + "sec2_missing.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- C. Algorithm 2: per-record comparison, cost order
ref = {v: ref_distribution(dev, v) for v in MASKABLE}
acqO = json.load(open(RV + "rev_missing.json"))["acq_fixed_order"]
COST = ["diabetes", "hypertension", "n_chronic", "bmi", "smoker", "marital", "educ", "pir", "srh", "sleep_wd", "sleep_we", "sedentary"]


def acquire(row_m, row_t, policy, r):
    cur = row_m.copy(); asked = 0; t = thr[FULL]
    while True:
        zl, zu = full.bounds(cur)
        if zl[0] >= t: return asked, 1
        if zu[0] < t: return asked, 0
        miss = [v for v in MASKABLE if pd.isna(cur[v].iloc[0]) and pd.notna(row_t[v].iloc[0])]
        if not miss: return asked, -1
        if policy == "random": v = miss[r.integers(len(miss))]
        elif policy == "fixed": v = [x for x in acqO if x in miss][0]
        elif policy == "cost": v = [x for x in COST if x in miss][0]
        else:
            best, v = -1.0, miss[0]
            for c in miss:
                vals, w = ref[c]; Rr = pd.concat([cur] * len(vals), ignore_index=True); Rr[c] = vals
                a, b = full.bounds(Rr); sc = float(np.sum(w * ((a >= t) | (b < t))))
                if sc > best + 1e-12: best, v = sc, c
        cur[v] = row_t[v].iloc[0]; asked += 1


acq = {}
for scen, rate, base in [("masked10_internal", 0.1, test), ("masked30_internal", 0.3, test), ("masked30_temporal", 0.3, L)]:
    d = base.copy()
    mask = np.random.default_rng(SEED + 11).random((len(d), len(MASKABLE))) < rate
    for j, f in enumerate(MASKABLE): d.loc[mask[:, j], f] = np.nan
    zl, zu = full.bounds(d); ab = np.where((zl < thr[FULL]) & (zu >= thr[FULL]))[0]
    rec = {}
    for pol in ("bound", "fixed", "cost", "random"):
        rr = np.random.default_rng(SEED); rec[pol] = np.array([acquire(d.iloc[[i]].copy(), base.iloc[[i]].copy(), pol, rr)[0] for i in ab])
    dec_ = np.array([acquire(d.iloc[[i]].copy(), base.iloc[[i]].copy(), "fixed", None)[1] for i in ab]) if False else None
    res = {"n": int(len(ab))}
    for pol, v in rec.items(): res[pol + "_mean_all"] = float(v.mean())
    bs = np.random.default_rng(SEED)
    for other in ("fixed", "cost", "random"):
        diff = rec[other] - rec["bound"]; boots = [diff[bs.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        res[f"diff_{other}"] = [float(diff.mean()), float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]
        res[f"frac_fewer_{other}"] = float((diff > 0).mean()); res[f"frac_more_{other}"] = float((diff < 0).mean())
    acq[scen] = res
    print("acq", scen, res, round(time.time() - t0), flush=True)
out["acquisition2"] = acq; out["cost_order"] = COST
json.dump(out, open(RV + "sec2_missing.json", "w"), indent=1, default=float)
print("done sec2_missing")
