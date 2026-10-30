"""Main experiments: development (NHANES 2017-Mar 2020), internal test, temporal validation
(NHANES Aug 2021-Aug 2023), ablations, missing-data stress tests, explanation fidelity."""
import json, pickle, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from fofs import FOFS, prepare, reconstruct_from_trace

warnings.filterwarnings("ignore")
SEED = 2026
R = "/home/claude/results/"
rng = np.random.default_rng(SEED)
PRIORS = json.load(open(R + "priors.json"))

BASE = ["age", "female", "pir", "educ", "marital", "bmi", "sleep_wd", "sleep_we", "srh", "smoker",
        "diabetes", "hypertension", "sedentary", "n_chronic"]
DOMAINS = {"Sleep": ["sleep_wd", "sleep_we"], "Socioeconomic": ["pir", "educ", "marital"],
           "Somatic": ["bmi", "srh", "diabetes", "hypertension", "n_chronic"], "Lifestyle": ["smoker", "sedentary"]}

P = prepare(pd.read_pickle("/home/claude/data/cohort_P.pkl"))
L = prepare(pd.read_pickle("/home/claude/data/cohort_L.pkl"))
dev, test = train_test_split(P, test_size=0.30, stratify=P.y, random_state=SEED)
dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)


def ml_matrix(df, feats):
    X = df[[f for f in feats if f != "marital"]].copy()
    if "marital" in feats:
        for k, nm in [(1, "m_married"), (2, "m_prev"), (3, "m_never")]:
            X[nm] = (df.marital == k).astype(float).where(df.marital.notna())
    return X


class LRModel:
    name = "Logistic regression"
    def __init__(self, feats): self.feats = feats
    def fit(self, df, y):
        self.m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                               LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", max_iter=2000))
        self.m.fit(ml_matrix(df, self.feats), y); return self
    def score(self, df): return self.m.predict_proba(ml_matrix(df, self.feats))[:, 1]
    def bounds(self, df): s = self.score(df); return s, s


class GBTModel:
    name = "Gradient-boosted trees"
    def __init__(self, feats): self.feats = feats
    def fit(self, df, y):
        g = GridSearchCV(HistGradientBoostingClassifier(min_samples_leaf=50, random_state=SEED),
                         {"learning_rate": [0.03, 0.1], "max_depth": [2, 3], "max_iter": [100, 300]},
                         cv=5, scoring="roc_auc")
        g.fit(ml_matrix(df, self.feats), y); self.m = g.best_estimator_; self.params = g.best_params_; return self
    def score(self, df): return self.m.predict_proba(ml_matrix(df, self.feats))[:, 1]
    def bounds(self, df): s = self.score(df); return s, s


VARIANTS = {
    "Proposed (full)": dict(),
    "No direction constraints": dict(direction=False),
    "No interval missing handling": dict(interval_missing=False),
    "Crisp ontology-guided": dict(crisp=True),
    "Fuzzy without ontology": dict(direction=False, interval_missing=False, concepts="quantile"),
    "Box missing bounds": dict(bound_mode="box"),
    "Normalized TSK aggregation": dict(aggregation="normalized"),
}


def fofs_factory(feats, kw, K):
    return lambda: FOFS(feats, n_rules=K, priors=PRIORS, **kw)


def cv_scores(make, df, folds=5):
    oof = np.zeros(len(df))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(df, df.y):
        m = make().fit(df.iloc[tr], df.y.iloc[tr].values)
        oof[va] = m.score(df.iloc[va])
    return oof


def tune_K(feats, kw, df, grid=(8, 16, 32, 64)):
    res = {K: roc_auc_score(df.y, cv_scores(fofs_factory(feats, kw, K), df)) for K in grid}
    best = max(res.values())
    K = min(k for k, v in res.items() if v >= best - 0.005)  # prespecified parsimony rule
    return K, res


def pick_threshold(y, s, target=0.80):
    ts = np.unique(s)[::-1]
    for t in ts:
        if ((s >= t) & (y == 1)).sum() / (y == 1).sum() >= target:
            return float(t)
    return float(ts[-1])


def platt(y, s):
    lr = LogisticRegression(C=1e6).fit(np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))[:, None], y)
    return lambda x: lr.predict_proba(np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))[:, None])[:, 1]


def cal_slope_int(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = LogisticRegression(C=1e6).fit(lp[:, None], y).coef_[0, 0]
    return slope


def metrics(y, s, p, t, zl=None, zu=None, w=None):
    pred = s >= t
    out = dict(AUROC=roc_auc_score(y, s), AUPRC=average_precision_score(y, s),
               Sensitivity=(pred & (y == 1)).sum() / (y == 1).sum(),
               Specificity=(~pred & (y == 0)).sum() / (y == 0).sum(),
               PPV=(pred & (y == 1)).sum() / max(pred.sum(), 1), NPV=(~pred & (y == 0)).sum() / max((~pred).sum(), 1),
               Brier=brier_score_loss(y, p), OE=y.mean() / p.mean(), Flagged=pred.mean())
    if zl is not None:
        pos, neg = zl >= t, zu < t; ab = ~(pos | neg); cov = ~ab
        out.update(Coverage=cov.mean(),
                   Sens_covered=(pos & (y == 1)).sum() / max(((y == 1) & cov).sum(), 1),
                   Spec_covered=(neg & (y == 0)).sum() / max(((y == 0) & cov).sum(), 1),
                   AbstainRate_pos=ab[y == 1].mean(), AbstainRate_neg=ab[y == 0].mean(),
                   Sens_refer=((pos | ab) & (y == 1)).sum() / (y == 1).sum(),
                   Spec_refer=(neg & (y == 0)).sum() / (y == 0).sum())
    return out


def bootstrap(y, cols, fn, B=1000, seed=SEED):
    """cols: dict name->array; fn(y, **arrays)->dict. Returns CI per metric."""
    r = np.random.default_rng(seed); n = len(y); acc = []
    for _ in range(B):
        i = r.integers(0, n, n)
        if y[i].sum() == 0: continue
        acc.append(fn(y[i], **{k: v[i] for k, v in cols.items()}))
    df = pd.DataFrame(acc)
    return {k: (float(df[k].quantile(.025)), float(df[k].quantile(.975))) for k in df}


def run_model(name, make, feats_dev, sets):
    t0 = time.time()
    oof = cv_scores(make, feats_dev)
    cal = platt(feats_dev.y.values, oof)
    thr = pick_threshold(feats_dev.y.values, oof)
    m = make().fit(feats_dev, feats_dev.y.values)
    out = {"threshold": thr, "oof_auroc": roc_auc_score(feats_dev.y, oof), "fit_seconds": time.time() - t0, "sets": {}}
    preds = {}
    for sname, df in sets.items():
        zl, zu = m.bounds(df); s = m.score(df); p = cal(s); y = df.y.values
        is_fuzzy = isinstance(m, FOFS)
        met = metrics(y, s, p, thr, zl if is_fuzzy else None, zu if is_fuzzy else None)
        met["AUROC_weighted"] = roc_auc_score(y, s, sample_weight=df.wt.values)
        cis = bootstrap(y, {"s": s, "p": p}, lambda y, s, p: metrics(y, s, p, thr), B=1000)
        met["CI"] = cis
        met["cal_slope"] = cal_slope_int(y, p)
        out["sets"][sname] = met
        preds[sname] = dict(s=s, p=p, zl=zl, zu=zu)
    return m, out, preds


if __name__ == "__main__":
    results = {"n": {"dev": len(dev), "test": len(test), "temporal": len(L),
                     "dev_pos": int(dev.y.sum()), "test_pos": int(test.y.sum()), "temporal_pos": int(L.y.sum())}}
    sets = {"internal_test": test, "temporal": L}
    models, preds, tuning = {}, {}, {}
    for nm, cls in [("Logistic regression", LRModel), ("Gradient-boosted trees", GBTModel)]:
        m, o, p = run_model(nm, lambda c=cls: c(BASE), dev, sets)
        models[nm], results[nm], preds[nm] = m, o, p
        print(nm, {k: round(v["AUROC"], 3) for k, v in o["sets"].items()}, flush=True)
    results["Gradient-boosted trees"]["params"] = models["Gradient-boosted trees"].params
    for nm, kw in VARIANTS.items():
        K, res = tune_K(BASE, kw, dev); tuning[nm] = {"K": K, "cv": res}
        m, o, p = run_model(nm, fofs_factory(BASE, kw, K), dev, sets)
        o["K"] = K; o["n_rules_incl_default"] = len(m.rules)
        models[nm], results[nm], preds[nm] = m, o, p
        print(nm, K, {k: round(v["AUROC"], 3) for k, v in o["sets"].items()}, flush=True)
    results["tuning"] = tuning

    # paired bootstrap: proposed vs each comparator (AUROC difference)
    paired = {}
    for sname, df in sets.items():
        y = df.y.values; sp = preds["Proposed (full)"][sname]["s"]
        for nm in preds:
            if nm == "Proposed (full)": continue
            so = preds[nm][sname]["s"]; r = np.random.default_rng(SEED); d = []
            for _ in range(2000):
                i = r.integers(0, len(y), len(y)); d.append(roc_auc_score(y[i], sp[i]) - roc_auc_score(y[i], so[i]))
            d = np.array(d)
            paired[f"{sname}|{nm}"] = dict(diff=roc_auc_score(y, sp) - roc_auc_score(y, so),
                                           lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)),
                                           p_two_sided=float(2 * min((d <= 0).mean(), (d >= 0).mean())))
    results["paired_auroc"] = paired
    pickle.dump(dict(models=models, preds=preds, dev=dev, test=test, L=L), open(R + "models.pkl", "wb"))
    json.dump(results, open(R + "main_results.json", "w"), indent=1, default=float)
    print("done main")
