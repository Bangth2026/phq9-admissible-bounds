"""Range of each input's possible contribution to the linear predictor: sign-constrained linear regression versus the
proposed model (used to explain coverage differences under domain removal)."""
import json, pickle
from rev_common import *  # noqa
N = pickle.load(open(RV + "new_models.pkl", "rb")); m = N["models"]["Sign-constrained logistic regression"]
cw = dict(zip(m.cols, m.coef_raw)); lin = {}
for v in m.cols:
    if v.startswith("m_"): continue
    lo, hi = VARIABLES[v][2]; lin[v] = abs(cw[v]) * (hi - lo)
o = [cw[c] for c in ("m_married", "m_prev", "m_never")]; lin["marital"] = max(o) - min(o)
fz = json.load(open(RV + "rev_missing.json"))["acq_contrib_range"]
json.dump({"fuzzy": fz, "linear": lin}, open(RV + "contrib_ranges.json", "w"))
