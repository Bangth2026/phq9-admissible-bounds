"""Check admissible-value bounds against brute-force evaluation (301-point grids for continuous
variables, all categories for discrete ones) for test participants with 1 or 2 masked variables."""
import json, pickle, numpy as np, pandas as pd
from experiment import LRModel, GBTModel, BASE, SEED  # noqa
from fofs import VARIABLES
S = pickle.load(open("/home/claude/results/models.pkl", "rb")); m = S["models"]["Proposed (full)"]; test = S["test"]
CONT = ("pir", "bmi", "sleep_wd", "sleep_we", "sedentary", "age")
MASK = [f for f in BASE if f not in ("age", "female")]
def grid(v):
    lo, hi = VARIABLES[v][2]
    return np.linspace(lo, hi, 301) if v in CONT else np.arange(lo, hi + 1)
rng = np.random.default_rng(SEED); errs = {1: [], 2: []}
full = test[test[MASK].notna().all(1)].reset_index(drop=True)
for k in (1, 2):
    for i in rng.choice(len(full), 40, replace=False):
        row = full.iloc[[i]].copy(); vs = list(rng.choice(MASK, k, replace=False)); row[vs] = np.nan
        zl, zu = m.bounds(row)
        g = np.array(np.meshgrid(*[grid(v) for v in vs])).reshape(k, -1).T
        R = pd.concat([row] * len(g), ignore_index=True)
        for j, v in enumerate(vs): R[v] = g[:, j]
        z = m.bounds(R)[0]
        errs[k].append(max(abs(z.min() - zl[0]), abs(z.max() - zu[0])))
out = {f"{k}_missing": dict(n=len(e), max_abs_diff=float(max(e))) for k, e in errs.items()}
json.dump(out, open("/home/claude/results/bounds_check.json", "w"), indent=1); print(out)
