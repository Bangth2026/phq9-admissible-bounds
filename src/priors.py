"""Direction-of-association priors for ontology concepts, estimated on NHANES 2015-2016
(an earlier, non-overlapping cycle that is never used for model fitting or testing)."""
import json, numpy as np, pandas as pd
from fofs import knowledge_concepts, membership, prepare

d = prepare(pd.read_pickle("/home/claude/data/cohort_I.pkl"))
y = d.y.values
rows, priors = [], {}
for name, spec in knowledge_concepts(extended=True).items():
    v = spec["variable"]
    src = v if v != "sleep_we" else None
    if src is None or d[src].isna().all():
        continue
    m = membership(d[src].values, spec["mf"])
    ok = ~np.isnan(m); inn = m[ok] >= 0.5; yy = y[ok]
    a, n1 = yy[inn].sum(), inn.sum(); b, n0 = yy[~inn].sum(), (~inn).sum()
    if n1 < 30 or n0 < 30 or a == 0 or b == 0:
        rr, lo, hi = np.nan, np.nan, np.nan
    else:
        rr = (a / n1) / (b / n0); se = np.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n0)
        lo, hi = np.exp(np.log(rr) - 1.96 * se), np.exp(np.log(rr) + 1.96 * se)
    dirn = 1 if lo > 1 else (-1 if hi < 1 else 0)
    priors[name] = dirn
    rows.append(dict(concept=name, variable=v, n_in=int(n1), rate_in=a / n1 if n1 else np.nan,
                     rate_out=b / n0 if n0 else np.nan, RR=rr, lo=lo, hi=hi, direction=dirn))
for name in ["ShortSleepFreeDay", "AdequateSleepFreeDay", "LongSleepFreeDay"]:  # not fielded in 2015-16
    src = name.replace("FreeDay", "Workday"); priors[name] = priors[src]
    rows.append(dict(concept=name, variable="sleep_we", direction=priors[src], note=f"inherited from {src}"))
tab = pd.DataFrame(rows)
tab.to_csv("/home/claude/results/priors_2015_2016.csv", index=False)
json.dump(priors, open("/home/claude/results/priors.json", "w"), indent=1)
print(tab.round(3).to_string())
