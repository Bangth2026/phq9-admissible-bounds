"""Exactness of the earlier grid versus Algorithm 1, on records without box fallback (30% MCAR, internal test)."""
import json, pickle, numpy as np
from rev_common import *  # noqa
from exact import ExactBounds
S = pickle.load(open(R + "models.pkl", "rb")); test = S["test"]; thr = json.load(open(R + "main_results.json"))
MASK = [f for f in BASE if f not in ("age", "female")]
out = {}
lg = lambda p: np.log(p / (1 - p))
for nm in ["Proposed (full)", "Main effects + interactions", "Version 1 (conjunctive rules)"]:
    m = S["models"][nm]; eb = ExactBounds(m); t = thr[nm]["threshold"]
    rng = np.random.default_rng(SEED)
    d = test.copy(); mask = rng.random((len(d), len(MASK))) < 0.3
    for j, f in enumerate(MASK): d.loc[mask[:, j], f] = np.nan
    el, eu = eb.bounds(d, return_eta=True); gl, gu = m.bounds(d)
    ok = np.ones(len(d), bool); ok[list(eb.fb_rows)] = False
    short = np.maximum(lg(gl) - el, eu - lg(gu))
    zl, zu = 1 / (1 + np.exp(-el)), 1 / (1 + np.exp(-eu))
    de = np.where(zl >= t, 1, np.where(zu < t, 0, -1)); dg = np.where(gl >= t, 1, np.where(gu < t, 0, -1))
    out[nm] = dict(n=len(d), n_fallback_rows=int((~ok).sum()), n_short=int((short[ok] > 1e-9).sum()),
                   max_short=float(short[ok].max()), dec_changed=int((de[ok] != dg[ok]).sum()),
                   components=eb.components_seen, fallback_components=eb.fallbacks)
    print(nm, out[nm], flush=True)
json.dump(out, open(RV + "rev_exact.json", "w"), indent=1)
