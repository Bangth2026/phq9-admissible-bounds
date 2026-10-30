"""Second set of secondary analyses, part 1: spline logistic regression comparator, refits without free-day sleep,
priors estimated for the reference contrast that is actually fitted."""
import json, pickle, os, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from rev_common import *  # noqa
from experiment_v2 import run, CYC
from fofs import FOFS, knowledge_concepts, membership, prepare

S = pickle.load(open(R + "models.pkl", "rb"))
models, preds, dev, test, L, POOL, EXT = S["models"], S["preds"], S["dev"], S["test"], S["L"], S["POOL"], S["EXT"]
SETS = {"internal_test": test, "temporal": L, "external_pooled": POOL} | {f"external_{CYC[c]}": EXT[c] for c in "DEFGH"}
FULLCI = ("internal_test", "temporal", "external_pooled"); FULL = "Proposed (full)"
out = {}


def paired(sp, so, y, B=2000):
    r = np.random.default_rng(SEED); d = []
    for _ in range(B):
        i = r.integers(0, len(y), len(y)); d.append(roc_auc_score(y[i], sp[i]) - roc_auc_score(y[i], so[i]))
    d = np.array(d)
    return dict(diff=roc_auc_score(y, sp) - roc_auc_score(y, so), lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)),
                p_two_sided=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


# ---------------------------------------------------------------- 1. spline logistic regression
t0 = time.time()
m, o, p = run("LR with restricted cubic splines", lambda: RCSLR(BASE), SETS, FULLCI)
out["rcs"] = o; print("RCS", round(time.time() - t0), {k: round(v["AUROC"], 3) for k, v in o["sets"].items()}, flush=True)
out["rcs_paired"] = {s: paired(preds[FULL][s]["s"], p[s]["s"], SETS[s].y.values, 2000 if s in FULLCI else 1000) for s in SETS}
pickle.dump(dict(model=m, preds=p), open(RV + "rcs_model.pkl", "wb"))

# ---------------------------------------------------------------- 2. all models refitted without free-day sleep
NOWE = [f for f in BASE if f != "sleep_we"]
MK = {"Proposed (full)": lambda: FOFS(NOWE, n_rules=0, priors=PRIORS, structure="main"),
      "Logistic regression": lambda: LRModel(NOWE),
      "LR, median imputation without indicators": lambda: LRNoInd(NOWE),
      "Sign-constrained logistic regression": lambda: SignLR(NOWE),
      "LR with restricted cubic splines": lambda: RCSLR(NOWE),
      "Gradient-boosted trees": lambda: GBTModel(NOWE),
      "Explainable boosting machine (GA2M)": lambda: EBM(NOWE)}
nowe, nowep = {}, {}
for nm, mk in MK.items():
    m_, o_, p_ = run(nm, mk, {k: SETS[k] for k in FULLCI}, ())
    nowe[nm] = {s: dict(AUROC=v["AUROC"], OE=v["OE"], slope=v["cal_slope"], Sens=v["Sensitivity"], Spec=v["Specificity"],
                        Flagged=v["Flagged"], Coverage=v.get("Coverage")) for s, v in o_["sets"].items()}
    nowep[nm] = p_
    print("noWE", nm, {s: round(v["AUROC"], 3) for s, v in nowe[nm].items()}, flush=True)
out["no_sleep_we"] = nowe
out["no_sleep_we_paired"] = {f"{s}|{nm}": paired(nowep[FULL][s]["s"], nowep[nm][s]["s"], SETS[s].y.values)
                             for s in ("internal_test", "external_pooled") for nm in nowep if nm != FULL}

# ---------------------------------------------------------------- 3. priors for the fitted reference contrast
full = models[FULL]
cv = [full.concepts[c]["variable"] for c in full.cnames]
used = {full.cnames[r[0]] for r in full.rules}
ref = {}
for c, v in zip(full.cnames, cv):
    if c not in used: ref[v] = c
d = prepare(pd.read_pickle("/home/claude/data/cohort_I.pkl")); y = d.y.values
C = knowledge_concepts(); rows = []; pri_ref = dict(PRIORS)
for c in sorted(used):
    v = C[c]["variable"]
    if v == "sleep_we" or v not in ref: continue
    mk = membership(d[v].values, C[c]["mf"]) >= 0.5; mr = membership(d[v].values, C[ref[v]]["mf"]) >= 0.5
    ok = ~np.isnan(d[v].values)
    a, n1 = y[ok & mk].sum(), (ok & mk).sum(); b, n0 = y[ok & mr].sum(), (ok & mr).sum()
    if min(a, b) == 0 or min(n1, n0) < 30:
        rr = lo = hi = np.nan; dirn = 0
    else:
        rr = (a / n1) / (b / n0); se = np.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n0)
        lo, hi = np.exp(np.log(rr) - 1.96 * se), np.exp(np.log(rr) + 1.96 * se); dirn = 1 if lo > 1 else (-1 if hi < 1 else 0)
    pri_ref[c] = dirn
    rows.append(dict(concept=c, reference=ref[v], RR=rr, lo=lo, hi=hi, direction_reference_contrast=dirn, direction_crude=PRIORS[c]))
for c in ["ShortSleepFreeDay", "AdequateSleepFreeDay", "LongSleepFreeDay"]:
    pri_ref[c] = pri_ref.get(c.replace("FreeDay", "Workday"), PRIORS[c])
    if c in used: rows.append(dict(concept=c, reference=ref.get("sleep_we"), direction_reference_contrast=pri_ref[c], direction_crude=PRIORS[c], note="inherited"))
tab = pd.DataFrame(rows); tab.to_csv(RV + "priors_reference_contrast.csv", index=False)
mref = FOFS(BASE, n_rules=0, priors=pri_ref, structure="main").fit(dev, dev.y.values)
out["prior_refcontrast"] = dict(
    changes=tab.loc[tab.direction_reference_contrast != tab.direction_crude, ["concept", "reference", "direction_crude", "direction_reference_contrast"]].to_dict(orient="records"),
    auc={s: float(roc_auc_score(df.y, mref.score(df))) for s, df in [("internal_test", test), ("temporal", L), ("external_pooled", POOL)]},
    n_at_zero=int(np.sum((mref.signs != 0) & (np.abs(mref.beta) < 1e-8))), n_signed=int(np.sum(mref.signs != 0)),
    max_abs_beta_change=float(np.max(np.abs(mref.beta - full.beta))) if len(mref.beta) == len(full.beta) else None)
print(out["prior_refcontrast"], flush=True)
json.dump(out, open(RV + "sec2_models.json", "w"), indent=1, default=float)
print("done sec2_models")
