"""Temporal recalibration-in-the-large: intercept update estimated on a random half of 2021-2023,
evaluated on the other half (repeated 200 times)."""
import json, pickle, numpy as np
from experiment import LRModel, GBTModel  # noqa: needed for unpickling
from scipy.optimize import brentq
from scipy.special import logit, expit
from sklearn.metrics import brier_score_loss
S = pickle.load(open("/home/claude/results/models.pkl", "rb"))
out = {}
for sname, D in [("temporal", S["L"]), ("external_pooled", S["POOL"])]:
  y = D.y.values
  for nm in ["Proposed (full)", "Logistic regression", "Gradient-boosted trees"]:
    p = np.clip(S["preds"][nm][sname]["p"], 1e-6, 1 - 1e-6); lp = logit(p)
    rng = np.random.default_rng(2026); oe_b, oe_a, br_b, br_a, sh = [], [], [], [], []
    for _ in range(200):
        i = rng.permutation(len(y)); a, b = i[: len(y) // 2], i[len(y) // 2:]
        d = brentq(lambda d: expit(lp[a] + d).mean() - y[a].mean(), -5, 5); sh.append(d)
        q = expit(lp[b] + d)
        oe_b.append(y[b].mean() / p[b].mean()); oe_a.append(y[b].mean() / q.mean())
        br_b.append(brier_score_loss(y[b], p[b])); br_a.append(brier_score_loss(y[b], q))
    out[f"{sname}|{nm}"] = {k: [float(np.mean(v)), float(np.quantile(v, .025)), float(np.quantile(v, .975))]
               for k, v in dict(intercept_shift=sh, OE_before=oe_b, OE_after=oe_a, Brier_before=br_b, Brier_after=br_a).items()}
json.dump(out, open("/home/claude/results/recalibration.json", "w"), indent=1)
print(json.dumps(out, indent=1))
