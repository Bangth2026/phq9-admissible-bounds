"""Operational meaning of small AUROC differences: proportion flagged per 1,000 participants when each model's
threshold is set to reach 80% sensitivity in the cohort itself (2005-2014 and 2021-2023)."""
import json, pickle
import numpy as np
R = "/home/claude/results/"; RV = R + "secondary/"
from experiment import LRModel, GBTModel  # noqa
S = pickle.load(open(R + "models.pkl", "rb")); NEW = pickle.load(open(RV + "new_models.pkl", "rb")); RC = pickle.load(open(RV + "rcs_model.pkl", "rb"))
out = {}
for st in ("external_pooled", "temporal"):
    d = S["POOL"] if st == "external_pooled" else S["L"]; y = d.y.values
    P = {"full": S["preds"]["Proposed (full)"][st]["s"], "lrni": NEW["preds"]["LR, median imputation without indicators"][st]["s"],
         "rcs": RC["preds"][st]["s"]}
    for k, s in P.items():
        t = np.sort(s[y == 1])[int(np.floor(0.2 * (y == 1).sum()))]
        fl = (s >= t); out[f"{st}|{k}"] = dict(flag_per1000=1000 * fl.mean(), sens=float((fl & (y == 1)).sum() / (y == 1).sum()),
                                              fp_per1000=1000 * float((fl & (y == 0)).mean()))
json.dump(out, open(RV + "sec6.json", "w"), indent=1)
print(json.dumps(out, indent=1))
