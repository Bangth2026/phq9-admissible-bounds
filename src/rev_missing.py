"""Secondary analyses, part 2: exactness of bounds (Algorithm 1), admissible ranges, fair missing-data stress tests,
point score with and without clipping, complete-case abstention baseline, bound-guided acquisition (Algorithm 2)."""
import json, pickle, time
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from rev_common import *  # noqa
from experiment import DOMAINS
from exact import ExactBounds, brute_force, ref_distribution, _is_cont
from fofs import VARIABLES, prepare

S = pickle.load(open(R + "models.pkl", "rb")); NEW = pickle.load(open(RV + "new_models.pkl", "rb"))
models, dev, test, L = S["models"], S["dev"], S["test"], S["L"]
main = json.load(open(R + "main_results.json")); rmj = json.load(open(RV + "rev_models.json"))
thr = {nm: main[nm]["threshold"] for nm in models} | {nm: rmj["new_models"][nm]["threshold"] for nm in NEW["models"]}
M = models | NEW["models"]
FULL = "Proposed (full)"; full = M[FULL]
MASKABLE = [f for f in BASE if f not in ("age", "female")]
out = {}

# ------------------------------------------------------------------ A. exactness of the bounds
rng = np.random.default_rng(SEED)
ex = {}
for nm in [FULL, "Main effects + interactions", "Version 1 (conjunctive rules)"]:
    m = M[nm]; eb = ExactBounds(m)
    d = test.copy(); mask = rng.random((len(d), len(MASKABLE))) < 0.3
    for j, f in enumerate(MASKABLE): d.loc[mask[:, j], f] = np.nan
    t0 = time.time(); zl, zu = eb.bounds(d); te = time.time() - t0
    t0 = time.time(); gl, gu = m.bounds(d); tg = time.time() - t0
    lg = lambda p: np.log(p / (1 - p))
    inner = np.maximum(lg(gl) - lg(zl), lg(zu) - lg(gu))       # how much the 41-point grid under-covers (log-odds)
    outer = np.maximum(lg(zl) - lg(gl), lg(gu) - lg(zu))       # grid wider than exact (should be <= 0)
    dec_e = np.where(zl >= thr[nm], 1, np.where(zu < thr[nm], 0, -1)); dec_g = np.where(gl >= thr[nm], 1, np.where(gu < thr[nm], 0, -1))
    ex[nm] = dict(max_grid_shortfall_logodds=float(inner.max()), n_shortfall=int((inner > 1e-9).sum()),
                  max_grid_excess_logodds=float(outer.max()), decisions_changed=int((dec_e != dec_g).sum()),
                  n=len(d), seconds_exact=te, seconds_grid=tg, fallbacks=eb.fallbacks, components=eb.components_seen)
    # dense brute force on 1-2 masked variables
    errs, outside = [], []
    cc = test[test[MASKABLE].notna().all(1)].reset_index(drop=True)
    r2 = np.random.default_rng(SEED + 1)
    for k in (1, 2):
        for i in r2.choice(len(cc), 40, replace=False):
            row = cc.iloc[[i]].copy(); vs = list(r2.choice(MASKABLE, k, replace=False)); row[vs] = np.nan
            el, eu = eb.bounds(row, return_eta=True)
            bl, bu = brute_force(m, row, vs, n=4001)
            errs.append(max(abs(bl - el[0]), abs(eu[0] - bu)))
            outside.append(max(el[0] - bl, bu - eu[0]))     # > 0 would mean brute force found a value outside the exact bounds
    ex[nm]["brute_max_abs_logodds"] = float(np.max(errs)); ex[nm]["brute_n"] = len(errs)
    ex[nm]["brute_max_outside"] = float(np.max(outside))
    print(nm, ex[nm], flush=True)
out["exactness"] = ex

# ------------------------------------------------------------------ B. admissible ranges and narrowed ranges
cont = [v for v in BASE if _is_cont(full, v)]
pr = {v: (float(np.nanquantile(dev[v], .01)), float(np.nanquantile(dev[v], .99))) for v in cont}
out["ranges"] = {v: dict(admissible=list(VARIABLES[v][2]), p01_p99=list(pr[v]), unit=VARIABLES[v][3]) for v in cont}
cov = {}
eb_w, eb_n = ExactBounds(full), ExactBounds(full, ranges=pr)
for lvl in (0.0, 0.1, 0.3):
    d = test.copy()
    if lvl:
        mask = np.random.default_rng(SEED + 7).random((len(d), len(MASKABLE))) < lvl
        for j, f in enumerate(MASKABLE): d.loc[mask[:, j], f] = np.nan
    r = {}
    for lab, eb in [("admissible", eb_w), ("p01_p99", eb_n)]:
        zl, zu = eb.bounds(d); ab = (zl < thr[FULL]) & (zu >= thr[FULL])
        r[lab] = dict(coverage=float(1 - ab.mean()), width=float(np.mean(zu - zl)))
    cov[str(lvl)] = r
out["narrowed"] = cov
print(cov, flush=True)

# ------------------------------------------------------------------ C. stress tests
def used_vars(nm):
    m = M[nm]
    if hasattr(m, "rules"):
        cv = [m.concepts[c]["variable"] for c in m.cnames]
        return sorted({cv[j] for r in m.rules for j in r})
    return list(BASE)

CMP = [FULL, "Box missing bounds", "No interval missing handling", "Version 1 (conjunctive rules)",
       "Logistic regression", "LR, median imputation without indicators", "LR, multiple imputation",
       "Sign-constrained logistic regression", "Gradient-boosted trees", "Monotone gradient-boosted trees",
       "Explainable boosting machine (GA2M)"]


def dec_metrics(y, pos, neg):
    ab = ~(pos | neg); cov = ~ab
    return dict(Coverage=cov.mean(), Sens_refer=((pos | ab) & (y == 1)).sum() / y.sum(),
                Spec_refer=(neg & (y == 0)).sum() / (y == 0).sum(),
                Sens_covered=(pos & (y == 1)).sum() / max(((y == 1) & cov).sum(), 1),
                Spec_covered=(neg & (y == 0)).sum() / max(((y == 0) & cov).sum(), 1))


def evaluate(d, masked_cells):
    rows = []; y = d.y.values
    for nm in CMP:
        m = M[nm]; s = m.score(d); t = thr[nm]
        r = dict(model=nm, AUROC=roc_auc_score(y, s))
        uv = used_vars(nm)
        r["eff_mask_rate"] = float(d[uv].isna().values.mean()) if masked_cells is not None else np.nan
        if isinstance(m, FOFS) and m.interval_missing:
            zl, zu = m.bounds(d); r.update(dec_metrics(y, zl >= t, zu < t)); r["MeanWidth"] = float(np.mean(zu - zl))
            if nm == FULL:
                raw = expit(m._eta(m._impute_mu(m._mu(prepare(d)))))
                r["AUROC_unclipped"] = roc_auc_score(y, raw)
                r["clip_changed"] = float(np.mean(np.abs(raw - s) > 1e-12))
                comp = ~d[uv].isna().any(axis=1).values          # complete-case abstention baseline
                rows.append(dict(model="Abstain if any input missing", AUROC=np.nan,
                                 **dec_metrics(y, comp & (s >= t), comp & (s < t))))
        elif isinstance(m, SignLR):
            zl, zu = m.bounds(d); r.update(dec_metrics(y, zl >= t, zu < t)); r["MeanWidth"] = float(np.mean(zu - zl))
            r["Sens_point"] = ((s >= t) & (y == 1)).sum() / y.sum(); r["Spec_point"] = ((s < t) & (y == 0)).sum() / (y == 0).sum()
        else:
            r.update(dec_metrics(y, s >= t, s < t))
        rows.append(r)
    return rows


rows = []
rng = np.random.default_rng(SEED)
for rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
    for rep in range(10 if rate > 0 else 1):
        d = test.copy(); mask = rng.random((len(d), len(MASKABLE))) < rate
        for j, f in enumerate(MASKABLE): d.loc[mask[:, j], f] = np.nan
        for r in evaluate(d, mask): rows.append(dict(scenario="MCAR", level=str(int(rate * 100)), rep=rep, **r))
    print("mask", rate, flush=True)
for dom, vars_ in list(DOMAINS.items()) + [("Self-rated health", ["srh"])]:
    d = test.copy(); d[vars_] = np.nan
    for r in evaluate(d, None): rows.append(dict(scenario="Domain removed", level=dom, rep=0, **r))
st = pd.DataFrame(rows); st.to_csv(RV + "stress.csv", index=False)
agg = st.groupby(["scenario", "level", "model"]).mean(numeric_only=True).drop(columns="rep").reset_index()
out["stress"] = agg.to_dict(orient="records")
for sname, d in [("temporal", L)]:
    out["natural_" + sname] = evaluate(d, None)

# ------------------------------------------------------------------ D. Algorithm 2: bound-guided acquisition
ref = {v: ref_distribution(dev, v) for v in MASKABLE}
# fixed order baseline: variables ranked by the range of their total contribution to eta over admissible values
contrib = {}
cvar = np.array([full.concepts[c]["variable"] for c in full.cnames])
for v in MASKABLE:
    idx = [k for k, r in enumerate(full.rules) if cvar[r[0]] == v]
    vals = ref[v][0]; mu = np.column_stack([np.nan_to_num(membership_v) for membership_v in
                                            [__import__("fofs").membership(vals, full.concepts[full.cnames[full.rules[k][0]]]["mf"]) for k in idx]]) if idx else np.zeros((len(vals), 0))
    g = mu @ full.beta[idx] if idx else np.zeros(len(vals))
    contrib[v] = float(g.max() - g.min())
ORDER = sorted(MASKABLE, key=lambda v: -contrib[v])
out["acq_fixed_order"] = ORDER; out["acq_contrib_range"] = contrib


def acquire(row_m, row_t, policy, r):
    cur = row_m.copy(); asked = 0; t = thr[FULL]
    while True:
        zl, zu = full.bounds(cur)
        if zl[0] >= t: return asked, 1
        if zu[0] < t: return asked, 0
        miss = [v for v in MASKABLE if pd.isna(cur[v].iloc[0]) and pd.notna(row_t[v].iloc[0])]  # only revealable inputs
        if not miss: return asked, -1
        if policy == "random": v = miss[r.integers(len(miss))]
        elif policy == "fixed": v = [x for x in ORDER if x in miss][0]
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
    res = {}
    for pol in ("bound", "fixed", "random"):
        rr = np.random.default_rng(SEED); rec = []
        for i in ab:
            row_m = d.iloc[[i]].copy(); row_t = base.iloc[[i]].copy()
            n_, dec = acquire(row_m, row_t, pol, rr); rec.append((n_, dec, int(d.y.iloc[i])))
        rec = np.array(rec)
        resolved = rec[:, 1] >= 0
        res[pol] = dict(n=int(len(rec)), resolved=float(resolved.mean()), mean_asked=float(rec[resolved, 0].mean()) if resolved.any() else np.nan,
                        one_question=float((resolved & (rec[:, 0] == 1)).mean()),
                        max_asked=int(rec[resolved, 0].max()) if resolved.any() else 0,
                        sens_after=float(((rec[:, 1] == 1) & (rec[:, 2] == 1)).sum() / max((rec[:, 2] == 1).sum(), 1)),
                        spec_after=float(((rec[:, 1] == 0) & (rec[:, 2] == 0)).sum() / max((rec[:, 2] == 0).sum(), 1)))
    acq[scen] = res
    print(scen, res, flush=True)
out["acquisition"] = acq
json.dump(out, open(RV + "rev_missing.json", "w"), indent=1, default=float)
print("done rev_missing")
