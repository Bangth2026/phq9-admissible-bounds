"""Coverage of exact linear bounds (sign-constrained logistic regression) with admissible versus 1st-99th percentile
ranges (single mask per level; superseded in the paper by the 10-mask values in sec2_missing.py)."""
import json, pickle, numpy as np
from rev_common import *  # noqa
S = pickle.load(open(R + "models.pkl", "rb")); N = pickle.load(open(RV + "new_models.pkl", "rb"))
dev, test = S["dev"], S["test"]; m = N["models"]["Sign-constrained logistic regression"]
thr = json.load(open(RV + "rev_models.json"))["new_models"]["Sign-constrained logistic regression"]["threshold"]
MASK = [f for f in BASE if f not in ("age", "female")]
pr = {v: (float(np.nanquantile(dev[v], .01)), float(np.nanquantile(dev[v], .99))) for v in ["age", "pir", "bmi", "sleep_wd", "sleep_we", "sedentary"]}
out = {}
for lvl in (0.0, 0.1, 0.3):
    d = test.copy()
    if lvl:
        mk = np.random.default_rng(SEED + 7).random((len(d), len(MASK))) < lvl
        for j, f in enumerate(MASK): d.loc[mk[:, j], f] = np.nan
    r = {}
    for lab, rg in (("admissible", None), ("p01_p99", pr)):
        zl, zu = m.bounds(d, ranges=rg); ab = (zl < thr) & (zu >= thr); r[lab] = dict(coverage=float(1 - ab.mean()), width=float(np.mean(zu - zl)))
    out[str(lvl)] = r
json.dump(out, open(RV + "narrowed_slr.json", "w"))
