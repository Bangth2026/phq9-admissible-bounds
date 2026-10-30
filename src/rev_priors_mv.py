"""Multivariable (mutually adjusted) direction priors from NHANES 2015-2016: one logistic model with every concept
indicator (membership >= 0.5) of the 14 base variables, one reference concept per variable (most prevalent in
2015-2016). Directions are read from the 95% CI of each coefficient; reference concepts get direction 0."""
import json, pickle
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from rev_common import *  # noqa
from fofs import FOFS, knowledge_concepts, membership, prepare

S = pickle.load(open(R + "models.pkl", "rb")); dev, test, L, POOL = S["dev"], S["test"], S["L"], S["POOL"]
d = prepare(pd.read_pickle("/home/claude/data/cohort_I.pkl"))
C = {k: v for k, v in knowledge_concepts().items() if v["variable"] in BASE and v["variable"] != "sleep_we"}
X = pd.DataFrame({k: (membership(d[v["variable"]].values, v["mf"]) >= 0.5).astype(float) for k, v in C.items()})
miss = np.column_stack([d[v["variable"]].isna().values for v in C.values()]).any(1)
X, y = X[~miss], d.y.values[~miss]
ref = {}
for var in dict.fromkeys(v["variable"] for v in C.values()):
    ks = [k for k, v in C.items() if v["variable"] == var]
    ref[var] = max(ks, key=lambda k: X[k].mean())
cols = [k for k in C if k not in ref.values()]
fit = sm.Logit(y, sm.add_constant(X[cols])).fit(disp=0)
mv = {}
rows = []
for k in C:
    if k in cols:
        b, se = fit.params[k], fit.bse[k]; lo, hi = np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
        mv[k] = 1 if lo > 1 else (-1 if hi < 1 else 0)
        rows.append(dict(concept=k, OR=np.exp(b), lo=lo, hi=hi, direction_mv=mv[k], direction_crude=PRIORS[k]))
    else:
        mv[k] = 0; rows.append(dict(concept=k, direction_mv=0, direction_crude=PRIORS[k], note="reference"))
for k in ["ShortSleepFreeDay", "AdequateSleepFreeDay", "LongSleepFreeDay"]:
    mv[k] = mv[k.replace("FreeDay", "Workday")]
tab = pd.DataFrame(rows); tab.to_csv(RV + "priors_multivariable.csv", index=False)
out = {"n": int(len(y)), "references": ref,
       "changes": tab.loc[tab.direction_mv != tab.direction_crude, ["concept", "direction_crude", "direction_mv"]].to_dict(orient="records")}
for lab, kw, K in [("Proposed", dict(structure="main"), 0), ("Version 1", dict(), 32)]:
    m = FOFS(BASE, n_rules=K, priors=mv, **kw).fit(dev, dev.y.values)
    out[lab] = {s: float(roc_auc_score(df.y, m.score(df))) for s, df in [("internal_test", test), ("temporal", L), ("external_pooled", POOL)]}
    out[lab]["n_at_zero"] = int(np.sum((m.signs != 0) & (np.abs(m.beta) < 1e-8)))
    out[lab]["n_signed"] = int(np.sum(m.signs != 0))
    out[lab]["rules"] = [m.rule_text(k) + f" ({b:+.3f})" for k, b in enumerate(m.beta)]
    print(lab, {k: v for k, v in out[lab].items() if k != "rules"}, flush=True)
json.dump(out, open(RV + "rev_priors_mv.json", "w"), indent=1, default=float)
print(out["changes"])
