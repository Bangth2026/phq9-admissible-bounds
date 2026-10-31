"""Fourth set of secondary analyses: abstention rules compared with the base model held fixed.

Rules: worst-case bounds over admissible values (wc), imputation spread (mi: 20 chained-equation completions passed
through the same model and threshold), complete-case abstention (cc) and pattern submodels (ps: one refit of the same
model class per missingness pattern, threshold chosen on development out-of-fold scores). Base models: the fuzzy
additive model, sign-constrained linear logistic regression and logistic regression with restricted cubic splines.
Scenarios: the three cohorts as observed; MCAR 30%; MNAR 0.6/0.1 (as before) and 0.9/0.05; outcome-dependent masking;
structural absence of one input in a target sample whose relation between that input and the observed inputs is
shifted away from the development data."""
import json, pickle, sys, time
import numpy as np, pandas as pd
from scipy.special import logit, expit
from sklearn.linear_model import LinearRegression
from rev_common import *  # noqa
from fofs import FOFS, prepare
from experiment import cv_scores, pick_threshold

S = pickle.load(open(R + "models.pkl", "rb")); NEW = pickle.load(open(RV + "new_models.pkl", "rb"))
RC = pickle.load(open(RV + "rcs_model.pkl", "rb"))
dev, test, L, POOL = S["dev"], S["test"], S["L"], S["POOL"]
main = json.load(open(R + "main_results.json")); rmj = json.load(open(RV + "rev_models.json"))
s2 = json.load(open(RV + "sec2_models.json"))
mi = NEW["models"]["LR, multiple imputation"]
MODELS = {"fuzzy": (S["models"]["Proposed (full)"], main["Proposed (full)"]["threshold"]),
          "slr": (NEW["models"]["Sign-constrained logistic regression"],
                  rmj["new_models"]["Sign-constrained logistic regression"]["threshold"]),
          "rcs": (RC["model"], s2["rcs"]["threshold"])}
T_MI = rmj["new_models"]["LR, multiple imputation"]["threshold"]
MASKABLE = [f for f in BASE if f not in ("age", "female")]
DISCRETE = {"female", "educ", "srh", "smoker", "diabetes", "hypertension", "n_chronic"}
MAKE = {"fuzzy": lambda f: FOFS(f, n_rules=0, priors=PRIORS, structure="main"),
        "slr": lambda f: SignLR(f), "rcs": lambda f: RCSLR(f)}
t0 = time.time()


# ------------------------------------------------------------------------------------------------ helpers
def completions(d, n_per=4):
    """20 completed copies of d: four posterior draws from each of the five chained-equation imputation models.
    Discrete inputs are rounded and every input is clipped to its admissible range."""
    X = ml_matrix(d, BASE); out = []
    for k, (imp, sc, lr) in enumerate(mi.ms):
        for j in range(n_per):
            imp.random_state_ = np.random.RandomState(1000 * k + j)
            Z = pd.DataFrame(imp.transform(X), columns=X.columns, index=d.index)
            dc = d.copy()
            for v in BASE:
                if v == "marital": continue
                lo, hi = VARIABLES[v][2]; z = Z[v].clip(lo, hi)
                if v in DISCRETE: z = z.round()
                dc[v] = d[v].where(d[v].notna(), z)
            mm = Z[["m_married", "m_prev", "m_never"]].values.argmax(1) + 1.0
            dc["marital"] = d["marital"].where(d["marital"].notna(), pd.Series(mm, index=d.index))
            out.append((dc, lr.predict_proba(sc.transform(Z.values))[:, 1]))
    return out


PS_CACHE = {}
def submodel(mname, missing):
    key = (mname, missing)
    if key not in PS_CACHE:
        f = [v for v in BASE if v not in missing]
        oof = cv_scores(lambda: MAKE[mname](f), dev)
        m = MAKE[mname](f).fit(dev, dev.y.values)
        PS_CACHE[key] = (m, pick_threshold(dev.y.values, oof))
    return PS_CACHE[key]


def pattern_scores(d, mname):
    miss = d[BASE].isna().values; s = np.zeros(len(d)); t = np.zeros(len(d))
    keys = [tuple(v for v, b in zip(BASE, row) if b) for row in miss]
    for k in set(keys):
        idx = np.array([i for i, kk in enumerate(keys) if kk == k])
        m, th = submodel(mname, k); s[idx] = m.score(d.iloc[idx]); t[idx] = th
    return s, t


def lmargin(a, t):
    return logit(np.clip(a, 1e-9, 1 - 1e-9)) - logit(np.clip(t, 1e-9, 1 - 1e-9))


GRID = np.linspace(0.05, 1.0, 20)
def rc_curve(y, conf, dec, grid=GRID):
    o = np.argsort(-conf, kind="stable"); n = len(y); res = []
    for c in grid:
        k = max(1, int(round(c * n))); i = o[:k]; yy, dd = y[i], dec[i]
        npos = max((yy == 1).sum(), 1); nneg = max((yy == 0).sum(), 1)
        sens = (dd & (yy == 1)).sum() / npos; spec = (~dd & (yy == 0)).sum() / nneg
        res.append((sens + spec) / 2)
    return np.array(res)


def aurc(y, conf, dec):  # area under the balanced-error / coverage curve, normalised to the coverage range
    return float(np.trapezoid(1 - rc_curve(y, conf, dec), GRID) / (GRID[-1] - GRID[0]))


def summarise(y, cov, dec, conf, cref):
    n = len(y); P = (y == 1); refer = dec | ~cov
    nb = lambda pt: float(((refer & P).sum() - (refer & ~P).sum() * pt / (1 - pt)) / n)
    k = max(1, int(round(cref * n))); i = np.argsort(-conf, kind="stable")[:k]
    yy, dd = y[i], dec[i]
    at = dict(sens=float((dd & (yy == 1)).sum() / max((yy == 1).sum(), 1)),
              spec=float((~dd & (yy == 0)).sum() / max((yy == 0).sum(), 1)))
    at["bacc"] = (at["sens"] + at["spec"]) / 2
    sens_c = float((dec & cov & P).sum() / max((cov & P).sum(), 1))
    spec_c = float((~dec & cov & ~P).sum() / max((cov & ~P).sum(), 1))
    return dict(coverage=float(cov.mean()), cov_refer=float((cov & dec).mean()), cov_norefer=float((cov & ~dec).mean()),
                sens=sens_c, spec=spec_c, bacc=(sens_c + spec_c) / 2,
                sens_refer=float((refer & P).sum() / P.sum()), spec_refer=float((~refer & ~P).sum() / (~P).sum()),
                flag_refer=float(refer.mean()), per1000_referred=1000 * float(refer.mean()),
                per1000_missed=1000 * float((~refer & P).sum() / n), per1000_avoided=1000 * float((~refer).mean()),
                nb05=nb(0.05), nb10=nb(0.10), at=at, aurc=aurc(y, conf, dec))


def evaluate(d, with_ps=True, keep=False):
    """All rules on all models for one data set. Returns summaries (and, if keep, the arrays for bootstrapping)."""
    y = d.y.values; comp = completions(d); res = {}; arrays = {}
    complete = ~d[BASE].isna().any(axis=1).values
    for mname, (m, t) in MODELS.items():
        zl, zu = m.bounds(d); s = m.score(d)
        rules = {}
        cov = (zl >= t) | (zu < t); dec = np.where(cov, zl >= t, s >= t)
        conf = np.where(zl >= t, lmargin(zl, t), np.where(zu < t, -lmargin(zu, t), -np.minimum(-lmargin(zl, t), lmargin(zu, t))))
        rules["wc"] = (cov, dec, conf)
        P = np.column_stack([m.score(dc) for dc, _ in comp]); share = (P >= t).mean(1); pm = P.mean(1)
        cov = (share >= 0.9) | (share <= 0.1); dec = pm >= t
        rules["mi"] = (cov, dec, np.abs(share - 0.5) + 1e-3 * np.abs(lmargin(pm, t)))
        rules["cc"] = (complete.copy(), s >= t, np.where(complete, np.abs(lmargin(s, t)), -1e6 + np.abs(lmargin(s, t))))
        if with_ps:
            sp, tp = pattern_scores(d, mname)
            rules["ps"] = (np.ones(len(d), bool), sp >= tp, np.abs(lmargin(sp, tp)))
        cref = rules["wc"][0].mean()
        res[mname] = {r: summarise(y, *v, cref) for r, v in rules.items()}
        arrays[mname] = rules
    # chained-equation logistic regression with its own threshold (the rule reported in the previous analysis)
    P = np.column_stack([p for _, p in comp]); share = (P >= T_MI).mean(1); pm = P.mean(1)
    cov = (share >= 0.9) | (share <= 0.1)
    rules = {"mi": (cov, pm >= T_MI, np.abs(share - 0.5) + 1e-3 * np.abs(lmargin(pm, T_MI)))}
    res["lrmi"] = {"mi": summarise(y, *rules["mi"], arrays["fuzzy"]["wc"][0].mean())}; arrays["lrmi"] = rules
    res["n"] = int(len(d)); res["prev"] = float(y.mean())
    return (res, arrays) if keep else res


def boot_aurc(y, arrays, B=500, seed=SEED, groups=None):
    """Bootstrap intervals of AURC per model x rule and of the paired difference to worst-case bounds."""
    rg = np.random.default_rng(seed); n = len(y); vals = {}
    for b in range(B):
        i = rg.integers(0, n, n)
        for mname, rules in arrays.items():
            for r, (cov, dec, conf) in rules.items():
                vals.setdefault((mname, r), []).append(aurc(y[i], conf[i], dec[i]))
    out = {}
    for (mname, r), v in vals.items():
        v = np.array(v); o = dict(lo=float(np.quantile(v, .025)), hi=float(np.quantile(v, .975)))
        if r != "wc" and (mname, "wc") in vals:
            dd = v - np.array(vals[(mname, "wc")])
            o.update(d_lo=float(np.quantile(dd, .025)), d_hi=float(np.quantile(dd, .975)))
        out[f"{mname}|{r}"] = o
    return out


def avg(lst):
    def rec(xs):
        if isinstance(xs[0], dict): return {k: rec([x[k] for x in xs]) for k in xs[0]}
        return float(np.mean(xs))
    out = rec(lst)
    # spread over repetitions for AURC
    for mname in MODELS:
        for r in lst[0][mname]:
            out[mname][r]["aurc_sd"] = float(np.std([x[mname][r]["aurc"] for x in lst], ddof=1)) if len(lst) > 1 else 0.0
    return out


OUT = {}
# ------------------------------------------------------------------------------------------------ natural missingness
for nm, d in [("internal_test", test), ("temporal", L), ("external_pooled", POOL)]:
    r, a = evaluate(d, keep=True)
    r["boot"] = boot_aurc(d.y.values, a, B=500 if nm != "external_pooled" else 300)
    if nm == "external_pooled":  # weighted coverage (interview weights) as a population-level check
        w = d.wt.values; r["weighted_coverage"] = {f"{m}|{rr}": float(np.sum(w * v[0]) / w.sum()) for m, rs in a.items() for rr, v in rs.items()}
    OUT["natural|" + nm] = r
    print(nm, round(time.time() - t0), {m: round(r[m]["wc"]["coverage"], 3) for m in MODELS}, flush=True)
    json.dump(OUT, open(RV + "sec4.json", "w"), indent=1, default=float)


def run_reps(name, maker, reps=10, with_ps=True, boot=True):
    lst = []; first = None
    for rep in range(reps):
        d = maker(rep)
        if rep == 0 and boot:
            r, a = evaluate(d, with_ps, keep=True); first = boot_aurc(d.y.values, a, B=300, seed=SEED + 7)
        else:
            r = evaluate(d, with_ps)
        lst.append(r)
    o = avg(lst)
    if first: o["boot_rep0"] = first
    OUT[name] = o; json.dump(OUT, open(RV + "sec4.json", "w"), indent=1, default=float)
    print(name, round(time.time() - t0), {m: (round(o[m]["wc"]["coverage"], 3), round(o[m]["mi"]["coverage"], 3)) for m in MODELS}, flush=True)


# ------------------------------------------------------------------------------------------------ MCAR 30% (same masks as before)
rng = np.random.default_rng(SEED); MC = []
for rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
    for rep in range(10 if rate > 0 else 1):
        d = test.copy(); mask = rng.random((len(d), len(MASKABLE))) < rate
        for j, f in enumerate(MASKABLE): d.loc[mask[:, j], f] = np.nan
        if rate == 0.3: MC.append(d)
run_reps("mcar30", lambda rep: MC[rep], with_ps=False)

# ------------------------------------------------------------------------------------------------ MNAR 0.6/0.1 (same masks as before)
r2 = np.random.default_rng(SEED + 101); MN = {"income_low": [], "srh_poor": [], "both": []}
for rep in range(10):
    for scen in ("income_low", "srh_poor", "both"):
        d = test.copy(); u = r2.random((len(d), 2))
        if scen in ("income_low", "both"): d.loc[u[:, 0] < np.where(d.pir.values < 1.3, 0.6, 0.1), "pir"] = np.nan
        if scen in ("srh_poor", "both"): d.loc[u[:, 1] < np.where(d.srh.values >= 4, 0.6, 0.1), "srh"] = np.nan
        MN[scen].append(d)
for scen in MN: run_reps("mnar|" + scen, lambda rep, s=scen: MN[s][rep])


# ------------------------------------------------------------------------------------------------ stronger MNAR 0.9/0.05
def strong(rep, var):
    g = np.random.default_rng(SEED + 202 + rep); d = test.copy(); u = g.random(len(d))
    if var == "srh": d.loc[u < np.where(d.srh.values >= 4, 0.9, 0.05), "srh"] = np.nan
    else: d.loc[u < np.where(d.pir.values < 1.3, 0.9, 0.05), "pir"] = np.nan
    return d
for var in ("srh", "pir"): run_reps("mnar_strong|" + var, lambda rep, v=var: strong(rep, v))


# ------------------------------------------------------------------------------------------------ outcome-dependent masking
def outcome_dep(rep):
    g = np.random.default_rng(SEED + 303 + rep); d = test.copy(); u = g.random((len(d), 2)); p_ = np.where(d.y.values == 1, 0.6, 0.1)
    d.loc[u[:, 0] < p_, "srh"] = np.nan; d.loc[u[:, 1] < p_, "pir"] = np.nan
    return d
run_reps("outcome_dep", outcome_dep)


# ------------------------------------------------------------------------------------------------ structural absence with shift
def residual_z(var):
    """Standardised residual of var given the other inputs, from a linear model fitted on development data."""
    oth = [v for v in BASE if v != var]
    Xd = ml_matrix(dev, oth); med = Xd.median(); ok = dev[var].notna().values
    lm = LinearRegression().fit(Xd.fillna(med).values[ok], dev[var].values[ok])
    sd = np.std(dev[var].values[ok] - lm.predict(Xd.fillna(med).values[ok]))
    def z(d):
        r = (d[var].values - lm.predict(ml_matrix(d, oth).fillna(med).values)) / sd
        return np.nan_to_num(r)
    return z


SHIFT = {"srh": +1, "sleep_we": -1}  # direction in which the target departs from development: worse health, shorter free-day sleep
for var, sgn in SHIFT.items():
    zf = residual_z(var); zt = zf(test)
    for gam in (0.0, 0.5, 1.0):
        def mk(rep, var=var, gam=gam, sgn=sgn, zt=zt):
            g = np.random.default_rng(SEED + 404 + rep); w = np.exp(sgn * gam * zt); w /= w.sum()
            i = g.choice(len(test), len(test), replace=True, p=w); d = test.iloc[i].reset_index(drop=True).copy()
            d[var + "_true"] = d[var]; d[var] = np.nan
            return d
        dd = mk(0); dr = dd.copy(); dr[var] = dr[var + "_true"]
        OUT.setdefault("shift_meta", {})[f"{var}|{gam}"] = dict(
            mean_resid=float(np.mean(zf(dr)[dr[var].notna().values])), prev=float(dd.y.mean()),
            mean_true=float(np.nanmean(dd[var + "_true"])), mean_dev=float(np.nanmean(dev[var])))
        run_reps(f"shift|{var}|{gam}", mk)

OUT["pattern_submodels"] = {f"{m}|{'+'.join(k) or 'none'}": float(t) for (m, k), (_, t) in PS_CACHE.items()}
OUT["seconds"] = time.time() - t0
json.dump(OUT, open(RV + "sec4.json", "w"), indent=1, default=float)
print("done sec4", round(time.time() - t0))
