"""Secondary analyses, part 3: age- and sex-adjusted direction priors, repeated random splits of the
2017-2020 release for the model and ablation comparisons."""
import json, pickle, time, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from rev_common import *  # noqa
from fofs import FOFS, knowledge_concepts, membership, prepare

S = pickle.load(open(R + "models.pkl", "rb"))
dev, test, L, POOL = S["dev"], S["test"], S["L"], S["POOL"]
out = {}
part = sys.argv[1] if len(sys.argv) > 1 else "all"

if part in ("all", "priors"):
    # ---------------------------------------------------------------- adjusted priors in 2015-2016
    d = prepare(pd.read_pickle("/home/claude/data/cohort_I.pkl")); y = d.y.values
    rows, adj = [], {}
    for name, spec in knowledge_concepts(extended=True).items():
        v = spec["variable"]
        if v == "sleep_we": continue
        m = membership(d[v].values, spec["mf"]); ok = ~np.isnan(m) & d.age.notna().values
        X = pd.DataFrame({"c": (m[ok] >= 0.5).astype(float)})
        if v != "age": X["age"] = d.age.values[ok]; X["age2"] = (d.age.values[ok] - 50) ** 2 / 100
        if v != "female": X["female"] = d.female.values[ok]
        fit = sm.Logit(y[ok], sm.add_constant(X)).fit(disp=0)
        b, se = fit.params["c"], fit.bse["c"]; lo, hi = np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
        adj[name] = 1 if lo > 1 else (-1 if hi < 1 else 0)
        rows.append(dict(concept=name, OR=np.exp(b), lo=lo, hi=hi, direction_adj=adj[name], direction_crude=PRIORS[name]))
    for nm in ["ShortSleepFreeDay", "AdequateSleepFreeDay", "LongSleepFreeDay"]:
        adj[nm] = adj[nm.replace("FreeDay", "Workday")]
        rows.append(dict(concept=nm, direction_adj=adj[nm], direction_crude=PRIORS[nm], note="inherited"))
    tab = pd.DataFrame(rows); tab.to_csv(RV + "priors_adjusted.csv", index=False)
    out["prior_changes"] = tab.loc[tab.direction_adj != tab.direction_crude, ["concept", "direction_crude", "direction_adj"]].to_dict(orient="records")
    res = {}
    for lab, kw, K in [("Proposed", dict(structure="main"), 0), ("Version 1", dict(), 32)]:
        for plab, pri in [("crude", PRIORS), ("adjusted", adj)]:
            m = FOFS(BASE, n_rules=K, priors=pri, **kw).fit(dev, dev.y.values)
            res[f"{lab}|{plab}"] = {s: float(roc_auc_score(df.y, m.score(df))) for s, df in
                                    [("internal_test", test), ("temporal", L), ("external_pooled", POOL)]}
            res[f"{lab}|{plab}"]["rules"] = [m.rule_text(k) + f" ({b:+.3f})" for k, b in enumerate(m.beta)]
            res[f"{lab}|{plab}"]["n_at_zero"] = int(np.sum((m.signs != 0) & (np.abs(m.beta) < 1e-8)))
            print(lab, plab, {k: v for k, v in res[f'{lab}|{plab}'].items() if k != 'rules'}, flush=True)
    out["adjusted_priors"] = res
    json.dump(out, open(RV + "rev_priors.json", "w"), indent=1, default=float)

if part in ("all", "splits"):
    # ---------------------------------------------------------------- repeated random splits
    P = prepare(pd.read_pickle("/home/claude/data/cohort_P.pkl"))
    gbt_par = json.load(open(R + "main_results.json"))["Gradient-boosted trees"]["params"]
    MK = {
        "Proposed (full)": lambda: FOFS(BASE, n_rules=0, priors=PRIORS, structure="main"),
        "Version 1 (conjunctive rules)": lambda: FOFS(BASE, n_rules=32, priors=PRIORS),
        "Main effects + interactions": lambda: FOFS(BASE, n_rules=16, priors=PRIORS, structure="main"),
        "No direction constraints": lambda: FOFS(BASE, n_rules=0, priors=PRIORS, structure="main", direction=False),
        "Crisp ontology-guided": lambda: FOFS(BASE, n_rules=0, priors=PRIORS, structure="main", crisp=True),
        "Fuzzy without ontology": lambda: FOFS(BASE, n_rules=0, priors=PRIORS, structure="main", direction=False,
                                               interval_missing=False, concepts="quantile"),
        "Logistic regression": lambda: LRModel(BASE),
        "Sign-constrained logistic regression": lambda: SignLR(BASE),
        "Gradient-boosted trees": lambda: _GBTfixed(),
        "Explainable boosting machine (GA2M)": lambda: EBM(BASE),
    }

    class _GBTfixed(GBTModel):
        def __init__(self): self.feats = BASE
        def fit(self, df, y):
            self.m = HistGradientBoostingClassifier(min_samples_leaf=50, random_state=SEED, **gbt_par).fit(ml_matrix(df, BASE), y)
            return self

    B = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    rec = []
    for b in range(B):
        t0 = time.time()
        tr, te = train_test_split(P, test_size=0.30, stratify=P.y, random_state=1000 + b)
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        r = {"split": b}
        for nm, mk in MK.items():
            m = mk().fit(tr, tr.y.values); r[nm] = roc_auc_score(te.y, m.score(te))
        rec.append(r); print(b, round(time.time() - t0), flush=True)
        pd.DataFrame(rec).to_csv(RV + "repeated_splits100.csv", index=False)
    df = pd.DataFrame(rec); summ = {}
    for nm in MK:
        if nm == "Proposed (full)": continue
        dd = df["Proposed (full)"] - df[nm]
        summ[nm] = dict(mean_auc=float(df[nm].mean()), mean_diff=float(dd.mean()), sd_diff=float(dd.std()),
                        lo=float(dd.quantile(.025)), hi=float(dd.quantile(.975)), frac_pos=float((dd > 0).mean()))
    summ["Proposed (full)"] = dict(mean_auc=float(df["Proposed (full)"].mean()), sd=float(df["Proposed (full)"].std()))
    json.dump(summ, open(RV + "rev_splits100.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
