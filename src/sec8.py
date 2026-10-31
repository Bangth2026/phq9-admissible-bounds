"""Analyses, part 8: (a) sensitivity of imputation spread to its configuration (agreement cut-offs 80/20, 90/10 and
95/5; 10 or 20 completed data sets); (b) the rules in 2021-2023 after recalibration-in-the-large of the fuzzy model,
i.e. with the referral threshold moved by the estimated intercept shift, to see whether the advantage of worst-case
bounds at 0.05 there comes from referring more people to offset under-prediction."""
import json, time
import numpy as np
from scipy.optimize import brentq
from scipy.special import logit, expit
exec(open("/home/claude/src/sec5.py").read().split("# ---------------------------------------------------------------- natural cohorts")[0])
t0 = time.time(); OUT8 = {}


def mi_variants(d):
    m, t = MODELS["fuzzy"]; comp = completions(d); y = d.y.values
    P = np.column_stack([m.score(dc) for dc, _ in comp]); zl, zu = m.bounds(d); cref = float(((zl >= t) | (zu < t)).mean())
    res = {}
    for nimp, cols in (("20", slice(None)), ("10", slice(0, 20, 2))):
        Q = P[:, cols]; share = (Q >= t).mean(1); pm = Q.mean(1)
        for hi, lo in ((0.8, 0.2), (0.9, 0.1), (0.95, 0.05)):
            cov = (share >= hi) | (share <= lo)
            res[f"{nimp}|{int(hi * 100)}"] = summarise(y, cov, pm >= t, np.abs(share - 0.5) + 1e-3 * np.abs(lmargin(pm, t)), cref)
    return res


def short(v): return {k: v[k] for k in ("coverage", "nb05", "nb10", "per1000_missed", "per1000_avoided")} | {"atb": v["at"]["bacc"], "aurc": v["aurc"]}


for nm, d in [("natural|internal_test", test), ("natural|temporal", L), ("natural|external_pooled", POOL)]:
    OUT8[nm] = {k: short(v) for k, v in mi_variants(d).items()}; print(nm, round(time.time() - t0), flush=True)
for key in ("mcar30", "shift|srh|1.0"):
    lst = [mi_variants(SCEN[key](rep)) for rep in range(10)]
    OUT8[key] = {k: {mt: float(np.mean([short(x[k])[mt] for x in lst])) for mt in short(lst[0][k])} for k in lst[0]}
    print(key, round(time.time() - t0), flush=True)
json.dump(OUT8, open(RV + "sec8.json", "w"), indent=1)

# ---------------------------------------------------------------- 2021-2023 after recalibration-in-the-large
y = L.y.values; out = {}
for mname, (m, t) in MODELS.items():
    p = np.clip(m.score(L), 1e-9, 1 - 1e-9)
    shift = brentq(lambda q: expit(logit(p) + q).mean() - y.mean(), -5, 5)
    out[mname] = {"shift": float(shift), "t_raw": float(t), "t_recal": float(expit(logit(t) - shift))}
OUT8["recal_tmp"] = out
orig = dict(MODELS)
for mname, (m, t) in orig.items(): MODELS[mname] = (m, out[mname]["t_recal"])
for mname in orig: pattern_scores(L, mname)  # fit the submodels for the 2021-2023 patterns
PS_BAK = dict(PS_CACHE)
for (mname, k), (mm, th) in list(PS_CACHE.items()): PS_CACHE[(mname, k)] = (mm, float(expit(logit(th) - out[mname]["shift"])))
DELTA_BAK = dict(DELTA)
r, a = evaluate(L, keep=True); a = with_sel(L, a)
OUT8["recal_tmp_rules"] = {mname: {rr: short(v) for rr, v in r[mname].items()} | {"sel": short(summarise(y, *a[mname]["sel"], a[mname]["wc"][0].mean()))}
                           for mname in MODELS}
OUT8["recal_tmp_bootnb"] = boot_nb(y, a, B=1000, seed=SEED + 8)
MODELS.update(orig); PS_CACHE.clear(); PS_CACHE.update(PS_BAK)
json.dump(OUT8, open(RV + "sec8.json", "w"), indent=1)
print("done sec8", round(time.time() - t0))
