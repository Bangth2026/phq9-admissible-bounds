"""Analyses, part 5: point-score selective prediction as a fifth abstention rule; bootstrap intervals for
net-benefit differences against worst-case bounds; prevalence and sensitivity of the shifted target samples; a
repeat of the self-rated-health shift with residuals from an ordinal (proportional-odds) model; SD of AURC over
repetitions for every simulated scenario."""
import json, time
import numpy as np, pandas as pd
from scipy.special import logit, expit
exec(open("/home/claude/src/sec4.py").read().split("OUT = {}")[0])
from statsmodels.miscmodels.ordinal_model import OrderedModel

t0 = time.time(); OUT5 = {}
# ---------------------------------------------------------------- point-score selective prediction: band chosen on development OOF scores
TARGET = 0.90
DELTA = {}
for mname, (m, t) in MODELS.items():
    oof = cv_scores(lambda mn=mname: MAKE[mn](BASE), dev)
    DELTA[mname] = float(np.quantile(np.abs(lmargin(oof, t)), 1 - TARGET))
OUT5["delta"] = DELTA; print("delta", DELTA, flush=True)


def ps_sel(d, mname):
    m, t = MODELS[mname]; s = m.score(d); mg = np.abs(lmargin(s, t))
    return mg >= DELTA[mname], s >= t, mg


def eval_sel(d):
    y = d.y.values; r = {}
    for mname, (m, t) in MODELS.items():
        zl, zu = m.bounds(d); cref = float(((zl >= t) | (zu < t)).mean())
        r[mname] = summarise(y, *ps_sel(d, mname), cref)
    r["prev"] = float(y.mean())
    return r


def nb(y, refer, pt): P = y == 1; return ((refer & P).sum() - (refer & ~P).sum() * pt / (1 - pt)) / len(y)


def boot_nb(y, arrays, B=1000, seed=SEED):
    rg = np.random.default_rng(seed); n = len(y); out = {}
    for mname, rules in arrays.items():
        if mname not in MODELS: continue
        ref = {r: (dec | ~cov) for r, (cov, dec, conf) in rules.items()}
        for pt in (0.05, 0.10):
            D = {r: [] for r in ref if r != "wc"}
            for b in range(B):
                i = rg.integers(0, n, n); w = nb(y[i], ref["wc"][i], pt)
                for r in D: D[r].append(nb(y[i], ref[r][i], pt) - w)
            for r, v in D.items():
                out[f"{mname}|{r}|{pt}"] = dict(d=float(nb(y, ref[r], pt) - nb(y, ref["wc"], pt)),
                                               lo=float(np.quantile(v, .025)), hi=float(np.quantile(v, .975)))
    return out


def with_sel(d, arrays):
    for mname in MODELS: arrays[mname]["sel"] = ps_sel(d, mname)
    return arrays


# ---------------------------------------------------------------- scenario data (identical to sec4)
rng = np.random.default_rng(SEED); MC = []
for rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
    for rep in range(10 if rate > 0 else 1):
        d = test.copy(); mask = rng.random((len(d), len(MASKABLE))) < rate
        for j, f in enumerate(MASKABLE): d.loc[mask[:, j], f] = np.nan
        if rate == 0.3: MC.append(d)
r2 = np.random.default_rng(SEED + 101); MN = {"income_low": [], "srh_poor": [], "both": []}
for rep in range(10):
    for scen in ("income_low", "srh_poor", "both"):
        d = test.copy(); u = r2.random((len(d), 2))
        if scen in ("income_low", "both"): d.loc[u[:, 0] < np.where(d.pir.values < 1.3, 0.6, 0.1), "pir"] = np.nan
        if scen in ("srh_poor", "both"): d.loc[u[:, 1] < np.where(d.srh.values >= 4, 0.6, 0.1), "srh"] = np.nan
        MN[scen].append(d)


def strong(rep, var):
    g = np.random.default_rng(SEED + 202 + rep); d = test.copy(); u = g.random(len(d))
    if var == "srh": d.loc[u < np.where(d.srh.values >= 4, 0.9, 0.05), "srh"] = np.nan
    else: d.loc[u < np.where(d.pir.values < 1.3, 0.9, 0.05), "pir"] = np.nan
    return d


def outcome_dep(rep):
    g = np.random.default_rng(SEED + 303 + rep); d = test.copy(); u = g.random((len(d), 2)); p_ = np.where(d.y.values == 1, 0.6, 0.1)
    d.loc[u[:, 0] < p_, "srh"] = np.nan; d.loc[u[:, 1] < p_, "pir"] = np.nan
    return d


def residual_lin(var):
    from sklearn.linear_model import LinearRegression
    oth = [v for v in BASE if v != var]
    Xd = ml_matrix(dev, oth); med = Xd.median(); ok = dev[var].notna().values
    lm = LinearRegression().fit(Xd.fillna(med).values[ok], dev[var].values[ok])
    sd = np.std(dev[var].values[ok] - lm.predict(Xd.fillna(med).values[ok]))
    return lambda d: np.nan_to_num((d[var].values - lm.predict(ml_matrix(d, oth).fillna(med).values)) / sd)


def residual_ord():
    """Probability-scale residual of an ordinal outcome (Li and Shepherd): P(Y < y) − P(Y > y) under a
    proportional-odds model of self-rated health on the other inputs, standardised in development data."""
    oth = [v for v in BASE if v != "srh"]
    Xd = ml_matrix(dev, oth); med = Xd.median(); ok = dev.srh.notna().values
    Z = (Xd.fillna(med) - Xd.fillna(med).mean()) / Xd.fillna(med).std().replace(0, 1)
    mod = OrderedModel(dev.srh.values[ok].astype(int), Z.values[ok], distr="logit").fit(method="bfgs", disp=False, maxiter=500)
    mu, sdv = Xd.fillna(med).mean(), Xd.fillna(med).std().replace(0, 1)
    def pres(d):
        Zt = ((ml_matrix(d, oth).fillna(med) - mu) / sdv).values
        pr = mod.model.predict(mod.params, exog=Zt)  # n x 5, categories 1..5
        y = d.srh.values.astype(float); k = np.nan_to_num(y - 1).astype(int)
        cum = np.cumsum(pr, 1); below = np.where(k > 0, cum[np.arange(len(k)), np.maximum(k - 1, 0)], 0.0)
        above = 1 - cum[np.arange(len(k)), k]; r = below - above
        return np.where(np.isnan(y), 0.0, r)
    r_dev = pres(dev[ok]); s = r_dev.std()
    return lambda d: pres(d) / s


def shift_maker(var, gam, sgn, zt):
    def mk(rep):
        g = np.random.default_rng(SEED + 404 + rep); w = np.exp(sgn * gam * zt); w /= w.sum()
        i = g.choice(len(test), len(test), replace=True, p=w); d = test.iloc[i].reset_index(drop=True).copy()
        d[var + "_true"] = d[var]; d[var] = np.nan
        return d
    return mk


SCEN = {"mcar30": lambda rep: MC[rep], "mnar|income_low": lambda rep: MN["income_low"][rep], "mnar|srh_poor": lambda rep: MN["srh_poor"][rep],
        "mnar|both": lambda rep: MN["both"][rep], "mnar_strong|srh": lambda rep: strong(rep, "srh"), "mnar_strong|pir": lambda rep: strong(rep, "pir"),
        "outcome_dep": outcome_dep}
zsrh = residual_lin("srh")(test); zwe = residual_lin("sleep_we")(test)
for g in (0.0, 0.5, 1.0):
    SCEN[f"shift|srh|{g}"] = shift_maker("srh", g, +1, zsrh); SCEN[f"shift|sleep_we|{g}"] = shift_maker("sleep_we", g, -1, zwe)
zord = residual_ord()(test)
OUT5["ord_corr_lin"] = float(np.corrcoef(zord, zsrh)[0, 1])
SCEN["shift_ord|srh|1.0"] = shift_maker("srh", 1.0, +1, zord)

# ---------------------------------------------------------------- natural cohorts: point-score rule and NB bootstraps (all rules)
for nm, d in [("internal_test", test), ("temporal", L), ("external_pooled", POOL)]:
    r, a = evaluate(d, keep=True); a = with_sel(d, a)
    o = {m: r[m] | {"sel": summarise(d.y.values, *a[m]["sel"], a[m]["wc"][0].mean())} for m in MODELS}
    o["boot_nb"] = boot_nb(d.y.values, a, B=1000 if nm != "external_pooled" else 500)
    sub = {m: {k: a[m][k] for k in ("wc", "sel", "ps") if k in a[m]} for m in MODELS}
    o["boot_aurc_sel"] = boot_aurc(d.y.values, sub, B=300 if nm == "external_pooled" else 500, seed=SEED + 9)
    OUT5["natural|" + nm] = o; print(nm, round(time.time() - t0), flush=True)
    json.dump(OUT5, open(RV + "sec5.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- simulated scenarios: point-score rule (10 reps), prevalence, AURC SD from sec4
S4 = json.load(open(RV + "sec4.json"))
for key, mk in SCEN.items():
    lst = [eval_sel(mk(rep)) for rep in range(10)]
    o = {m: {k: float(np.mean([x[m][k] for x in lst])) for k in ("coverage", "cov_refer", "cov_norefer", "sens", "spec", "bacc", "sens_refer",
                                                                   "flag_refer", "per1000_referred", "per1000_missed", "per1000_avoided", "nb05", "nb10", "aurc")}
         | {"atb": float(np.mean([x[m]["at"]["bacc"] for x in lst])), "aurc_sd": float(np.std([x[m]["aurc"] for x in lst], ddof=1))} for m in MODELS}
    o["prev"] = float(np.mean([x["prev"] for x in lst])); o["prev_sd"] = float(np.std([x["prev"] for x in lst], ddof=1))
    OUT5[key] = o; print(key, round(time.time() - t0), flush=True)
    json.dump(OUT5, open(RV + "sec5.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- full rule comparison for the ordinal-residual shift (10 reps) and NB bootstraps for rep 0 of key scenarios
lst = []
for rep in range(10):
    d = SCEN["shift_ord|srh|1.0"](rep); lst.append(evaluate(d))
OUT5["shift_ord_full"] = avg(lst); OUT5["shift_ord_meta"] = dict(prev=float(np.mean([x["prev"] for x in lst])))
for key in ("mnar_strong|srh", "outcome_dep", "shift|srh|1.0", "shift|sleep_we|1.0"):
    d = SCEN[key](0); r, a = evaluate(d, keep=True); a = with_sel(d, a)
    OUT5.setdefault("boot_nb_rep0", {})[key] = boot_nb(d.y.values, a, B=1000, seed=SEED + 5)
    print("bootnb", key, round(time.time() - t0), flush=True)
    json.dump(OUT5, open(RV + "sec5.json", "w"), indent=1, default=float)
OUT5["seconds"] = time.time() - t0
json.dump(OUT5, open(RV + "sec5.json", "w"), indent=1, default=float)
print("done sec5", round(time.time() - t0))
