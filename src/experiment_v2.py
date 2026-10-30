"""Version-2 experiments, run once according to results/protocol_v2.md.
Primary model: fuzzy additive model (structure='main'). Evaluation: internal test (2017-2020), temporal (2021-2023)
and external NHANES 2005-2014 (pooled and per cycle)."""
import json, pickle, time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from experiment import (BASE, PRIORS, SEED, dev, test, L, LRModel, GBTModel, FOFS, cv_scores, tune_K,
                        pick_threshold, platt, metrics, bootstrap, cal_slope_int)
from fofs import prepare

warnings.filterwarnings("ignore")
R = "/home/claude/results/"
EXT = {c: prepare(pd.read_pickle(f"/home/claude/data/cohort_{c}.pkl")) for c in "DEFGH"}
CYC = {"D": "2005-2006", "E": "2007-2008", "F": "2009-2010", "G": "2011-2012", "H": "2013-2014"}
POOL = pd.concat([EXT[c].assign(cycle=CYC[c]) for c in "DEFGH"], ignore_index=True)

MAIN = dict(structure="main")
VARIANTS = {   # name: (kwargs, fixed K or None -> tuned by the parsimony rule)
    "Proposed (full)": (MAIN, 0),                       # locked in protocol_v2.md
    "Version 1 (conjunctive rules)": (dict(), 32),      # as published in v1 analysis
    "Main effects + interactions": (MAIN, 16),          # best CV K, reported as a variant
    "No direction constraints": (dict(structure="main", direction=False), None),
    "No interval missing handling": (dict(structure="main", interval_missing=False), None),
    "Crisp ontology-guided": (dict(structure="main", crisp=True), None),
    "Box missing bounds": (dict(structure="main", bound_mode="box"), 0),
    "Fuzzy without ontology": (dict(structure="main", direction=False, interval_missing=False, concepts="quantile"), None),
    "Normalized TSK aggregation": (dict(aggregation="normalized"), 32),
}
GRID = (0, 8, 16, 32, 64)


def run(name, make, sets, full_ci):
    t0 = time.time()
    oof = cv_scores(make, dev); cal = platt(dev.y.values, oof); thr = pick_threshold(dev.y.values, oof)
    m = make().fit(dev, dev.y.values)
    out = {"threshold": thr, "oof_auroc": roc_auc_score(dev.y, oof), "sets": {}}
    preds = {}
    for sname, df in sets.items():
        zl, zu = m.bounds(df); s = m.score(df); p = cal(s); y = df.y.values
        fz = isinstance(m, FOFS)
        met = metrics(y, s, p, thr, zl if fz else None, zu if fz else None)
        met["AUROC_weighted"] = roc_auc_score(y, s, sample_weight=df.wt.values)
        met["cal_slope"] = cal_slope_int(y, p)
        if sname in full_ci:
            met["CI"] = bootstrap(y, {"s": s, "p": p}, lambda y, s, p: metrics(y, s, p, thr), B=1000)
        else:
            met["CI"] = bootstrap(y, {"s": s}, lambda y, s: {"AUROC": roc_auc_score(y, s)}, B=500)
        out["sets"][sname] = met; preds[sname] = dict(s=s, p=p, zl=zl, zu=zu)
    out["seconds"] = time.time() - t0
    return m, out, preds


if __name__ == "__main__":
    sets = {"internal_test": test, "temporal": L, "external_pooled": POOL}
    sets |= {f"external_{CYC[c]}": EXT[c] for c in "DEFGH"}
    full_ci = ("internal_test", "temporal", "external_pooled")
    res = {"n": {k: int(len(v)) for k, v in sets.items()} | {"dev": len(dev)},
           "pos": {k: int(v.y.sum()) for k, v in sets.items()} | {"dev": int(dev.y.sum())}}
    models, preds, tuning = {}, {}, {}
    for nm, cls in [("Logistic regression", LRModel), ("Gradient-boosted trees", GBTModel)]:
        m, o, p = run(nm, lambda c=cls: c(BASE), sets, full_ci)
        models[nm], res[nm], preds[nm] = m, o, p
        print(nm, {k: round(v["AUROC"], 3) for k, v in o["sets"].items()}, flush=True)
    res["Gradient-boosted trees"]["params"] = models["Gradient-boosted trees"].params
    for nm, (kw, K) in VARIANTS.items():
        if K is None:
            cvres = {k: roc_auc_score(dev.y, cv_scores(lambda k=k: FOFS(BASE, n_rules=k, priors=PRIORS, **kw), dev)) for k in GRID}
            best = max(cvres.values()); K = min(k for k, v in cvres.items() if v >= best - 0.005)
            tuning[nm] = {"K": K, "cv": cvres}
        else:
            tuning[nm] = {"K": K, "cv": None}
        m, o, p = run(nm, lambda K=K, kw=kw: FOFS(BASE, n_rules=K, priors=PRIORS, **kw), sets, full_ci)
        o["K"] = K; o["n_rules"] = len(m.rules)
        models[nm], res[nm], preds[nm] = m, o, p
        print(nm, K, len(m.rules), {k: round(v["AUROC"], 3) for k, v in o["sets"].items()}, flush=True)
    res["tuning"] = tuning
    paired = {}
    for sname, df in sets.items():
        y = df.y.values; sp = preds["Proposed (full)"][sname]["s"]
        for nm in preds:
            if nm == "Proposed (full)": continue
            so = preds[nm][sname]["s"]; r = np.random.default_rng(SEED); d = []
            for _ in range(2000 if sname in full_ci else 1000):
                i = r.integers(0, len(y), len(y)); d.append(roc_auc_score(y[i], sp[i]) - roc_auc_score(y[i], so[i]))
            d = np.array(d)
            paired[f"{sname}|{nm}"] = dict(diff=roc_auc_score(y, sp) - roc_auc_score(y, so), lo=float(np.quantile(d, .025)),
                                           hi=float(np.quantile(d, .975)),
                                           p_two_sided=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    res["paired_auroc"] = paired
    pickle.dump(dict(models=models, preds=preds, dev=dev, test=test, L=L, POOL=POOL, EXT=EXT),
                open(R + "models.pkl", "wb"))
    json.dump(res, open(R + "main_results.json", "w"), indent=1, default=float)
    print("done main v2")
