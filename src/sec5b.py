"""Point-score selective prediction against pattern submodels in 2005-2014 (paired bootstrap of AURC and net benefit)."""
import json
import numpy as np
exec(open("/home/claude/src/sec5.py").read().split("# ---------------------------------------------------------------- scenario data")[0])
d = POOL; y = d.y.values; out = {}
for mname in MODELS:
    sp, tp = pattern_scores(d, mname); A = {"ps": (np.ones(len(d), bool), sp >= tp, np.abs(lmargin(sp, tp))), "sel": ps_sel(d, mname)}
    rg = np.random.default_rng(SEED + 21); da, d05, d10 = [], [], []
    ref = {r: dec | ~cov for r, (cov, dec, conf) in A.items()}
    for b in range(500):
        i = rg.integers(0, len(y), len(y))
        da.append(aurc(y[i], A["sel"][2][i], A["sel"][1][i]) - aurc(y[i], A["ps"][2][i], A["ps"][1][i]))
        d05.append(nb(y[i], ref["sel"][i], .05) - nb(y[i], ref["ps"][i], .05)); d10.append(nb(y[i], ref["sel"][i], .10) - nb(y[i], ref["ps"][i], .10))
    q = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    out[mname] = dict(d_aurc=aurc(y, A["sel"][2], A["sel"][1]) - aurc(y, A["ps"][2], A["ps"][1]), d_aurc_ci=q(da),
                      d_nb05=float(nb(y, ref["sel"], .05) - nb(y, ref["ps"], .05)), d_nb05_ci=q(d05),
                      d_nb10=float(nb(y, ref["sel"], .10) - nb(y, ref["ps"], .10)), d_nb10_ci=q(d10))
    print(mname, out[mname], flush=True)
json.dump(out, open(RV + "sec5b.json", "w"), indent=1)
