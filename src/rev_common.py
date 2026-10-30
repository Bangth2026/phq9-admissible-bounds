"""Shared pieces for the secondary analyses."""
import json, pickle, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from experiment import BASE, PRIORS, SEED, LRModel, GBTModel, ml_matrix  # noqa
from fofs import FOFS, VARIABLES

warnings.filterwarnings("ignore")
R = "/home/claude/results/"
RV = "/home/claude/results/secondary/"

# Direction of each raw input implied by the concept priors (used for sign-constrained LR and monotone GBT).
# Columns as produced by ml_matrix(): BASE without 'marital' plus three marital dummies.
RAW_SIGN = {"age": 0, "female": 1, "pir": -1, "educ": -1, "bmi": 0, "sleep_wd": 0, "sleep_we": 0, "srh": 1,
            "smoker": 1, "diabetes": 1, "hypertension": 1, "sedentary": 1, "n_chronic": 1,
            "m_married": -1, "m_prev": 1, "m_never": 1}
# Variables typically held as structured EHR data
EHR = ["age", "female", "bmi", "diabetes", "hypertension", "n_chronic", "smoker", "educ", "marital"]
EHR_CORE = ["age", "female", "bmi", "diabetes", "hypertension", "n_chronic", "smoker"]


class SignLR:
    """Logistic regression on raw inputs, median imputation (no indicators), standardised, ridge penalty chosen by
    5-fold CV, coefficient signs constrained by the direction priors. Exposes exact
    admissible-value bounds (linear predictor -> extremes at range ends / categories)."""
    name = "Sign-constrained logistic regression"

    def __init__(self, feats, constrained=True, lams=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1)):
        self.feats = feats; self.constrained = constrained; self.lams = lams

    def _X(self, df):
        X = ml_matrix(df, self.feats)
        return X[self.cols].values if hasattr(self, "cols") else X

    def _fit(self, Z, y, lam):
        n, p = Z.shape
        sg = np.array([RAW_SIGN.get(c, 0) if self.constrained else 0 for c in self.cols])
        def obj(t):
            b, w = t[0], t[1:]; q = np.clip(expit(b + Z @ w), 1e-9, 1 - 1e-9)
            L = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)) + lam * w @ w
            g = (q - y) / n
            return L, np.r_[g.sum(), Z.T @ g + 2 * lam * w]
        bnds = [(None, None)] + [((0, None) if s > 0 else (None, 0) if s < 0 else (None, None)) for s in sg]
        r = minimize(obj, np.zeros(p + 1), jac=True, method="L-BFGS-B", bounds=bnds)
        return r.x[0], r.x[1:]

    def fit(self, df, y):
        X = ml_matrix(df, self.feats); self.cols = list(X.columns)
        self.med = X.median(); self.mu = X.fillna(self.med).mean(); self.sd = X.fillna(self.med).std().replace(0, 1)
        Z = ((X.fillna(self.med) - self.mu) / self.sd).values; y = np.asarray(y, float)
        best = (-1, None)
        for lam in self.lams:
            oof = np.zeros(len(y))
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Z, y):
                b, w = self._fit(Z[tr], y[tr], lam); oof[va] = b + Z[va] @ w
            a = roc_auc_score(y, oof)
            if a > best[0]: best = (a, lam)
        self.lam = best[1]; self.b0, self.w = self._fit(Z, y, self.lam)
        self.coef_raw = self.w / self.sd.values
        return self

    def eta(self, df):
        X = ml_matrix(df, self.feats)[self.cols].fillna(self.med)
        return self.b0 + ((X - self.mu) / self.sd).values @ self.w

    def score(self, df): return expit(self.eta(df))

    def bounds(self, df, ranges=None):
        """Exact bounds over admissible values: each missing raw input moves eta linearly, so its extremes are at
        the ends of its admissible range; marital status is enumerated over its three categories."""
        X = ml_matrix(df, self.feats)[self.cols]
        base = self.eta(df); lo = base.copy(); hi = base.copy()
        cw = dict(zip(self.cols, self.coef_raw))
        rngs = {v: VARIABLES[v][2] for v in self.feats} | (ranges or {})
        for v in [c for c in self.cols if not c.startswith("m_")]:
            miss = X[v].isna().values
            if not miss.any(): continue
            a, b = rngs[v]
            d = np.array([cw[v] * (a - self.med[v]), cw[v] * (b - self.med[v])])
            lo[miss] += d.min(); hi[miss] += d.max()
        if "marital" in self.feats:
            mc = ["m_married", "m_prev", "m_never"]; miss = X["m_married"].isna().values
            if miss.any():
                fill = sum(cw[c] * self.med[c] for c in mc)
                opts = np.array([cw[c] - fill for c in mc])
                lo[miss] += opts.min(); hi[miss] += opts.max()
        return expit(lo), expit(hi)


class LRNoInd(LRModel):
    name = "LR, median imputation without indicators"
    def fit(self, df, y):
        self.m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", max_iter=2000))
        self.m.fit(ml_matrix(df, self.feats), y); return self


class LRMI(LRModel):
    """Multiple imputation by chained equations (5 imputations with posterior draws), logistic regression fitted on
    each completed data set, predictions averaged."""
    name = "LR, multiple imputation"
    def fit(self, df, y):
        X = ml_matrix(df, self.feats); self.ms = []
        for k in range(5):
            imp = IterativeImputer(sample_posterior=True, random_state=SEED + k, max_iter=10)
            Z = imp.fit_transform(X)
            sc = StandardScaler().fit(Z)
            lr = LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", max_iter=2000).fit(sc.transform(Z), y)
            self.ms.append((imp, sc, lr))
        return self
    def score(self, df):
        X = ml_matrix(df, self.feats)
        return np.mean([lr.predict_proba(sc.transform(imp.transform(X)))[:, 1] for imp, sc, lr in self.ms], 0)


class MonoGBT(GBTModel):
    name = "Monotone gradient-boosted trees"
    def fit(self, df, y):
        from sklearn.model_selection import GridSearchCV
        X = ml_matrix(df, self.feats)
        cst = [RAW_SIGN.get(c, 0) for c in X.columns]
        g = GridSearchCV(HistGradientBoostingClassifier(min_samples_leaf=50, random_state=SEED, monotonic_cst=cst),
                         {"learning_rate": [0.03, 0.1], "max_depth": [2, 3], "max_iter": [100, 300]}, cv=5, scoring="roc_auc")
        g.fit(X, y); self.m = g.best_estimator_; self.params = g.best_params_; return self


class EBM(GBTModel):
    """Explainable Boosting Machine (GA2M): cyclic boosted shape functions plus automatically detected pairwise
    interactions; missing values handled in a separate bin (interpret 0.7.8 defaults, 10 interactions)."""
    name = "Explainable boosting machine (GA2M)"
    def fit(self, df, y):
        from interpret.glassbox import ExplainableBoostingClassifier
        X = ml_matrix(df, self.feats)
        self.m = ExplainableBoostingClassifier(interactions=10, random_state=SEED, n_jobs=1)
        self.m.fit(X, y); self.params = {"interactions": 10}; return self
    def n_terms(self): return len(self.m.term_features_)


def pick_threshold(y, s, target=0.80):
    ts = np.unique(s)[::-1]
    for t in ts:
        if ((s >= t) & (y == 1)).sum() / (y == 1).sum() >= target: return float(t)
    return float(ts[-1])


def rcs_basis(x, knots):
    """Restricted (natural) cubic spline basis (Harrell): linear term plus k-2 non-linear terms; linear beyond the
    boundary knots."""
    k = np.asarray(knots, float); K = len(k); x = np.asarray(x, float)
    cols = [x]
    p = lambda u: np.maximum(u, 0) ** 3
    sc = (k[-1] - k[0]) ** 2
    for j in range(K - 2):
        cols.append((p(x - k[j]) - p(x - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
                     + p(x - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2])) / sc)
    return np.column_stack(cols)


class RCSLR:
    """Logistic regression with restricted cubic splines (4 knots at the 5th, 35th, 65th and 95th development
    percentiles) for age, BMI and both sleep durations; other inputs linear; median imputation without indicators;
    L2 penalty chosen by 5-fold CV. Bounds over admissible values: the model is additive in the inputs, so each
    missing input is optimised separately (dense 2,001-point grid for spline inputs, range ends for linear inputs,
    categories for marital status)."""
    name = "LR with restricted cubic splines"
    SPL = ("age", "bmi", "sleep_wd", "sleep_we")

    def __init__(self, feats): self.feats = feats

    def _design(self, X):
        parts, names = [], []
        for c in X.columns:
            v = X[c].values
            if c in self.knots:
                b = rcs_basis(v, self.knots[c]); parts.append(b); names += [f"{c}_{j}" for j in range(b.shape[1])]
            else:
                parts.append(v[:, None]); names.append(c)
        return np.hstack(parts), names

    def fit(self, df, y):
        X = ml_matrix(df, self.feats); self.cols = list(X.columns); self.med = X.median()
        self.knots = {c: np.nanquantile(X[c], [0.05, 0.35, 0.65, 0.95]) for c in self.SPL if c in self.cols}
        Z, self.names = self._design(X.fillna(self.med))
        self.sc = StandardScaler().fit(Z)
        self.m = LogisticRegressionCV(Cs=10, cv=5, scoring="roc_auc", max_iter=4000).fit(self.sc.transform(Z), y)
        return self

    def _eta_from_X(self, X):
        Z, _ = self._design(X.fillna(self.med)); return self.m.decision_function(self.sc.transform(Z))

    def score(self, df): return expit(self._eta_from_X(ml_matrix(df, self.feats)[self.cols]))

    def bounds(self, df, ranges=None):
        X = ml_matrix(df, self.feats)[self.cols]
        base = self._eta_from_X(X); lo = base.copy(); hi = base.copy()
        rng = {v: VARIABLES[v][2] for v in self.feats} | (ranges or {})
        ref = pd.DataFrame([self.med[self.cols].values], columns=self.cols)
        for v in [c for c in self.cols if not c.startswith("m_")]:
            miss = X[v].isna().values
            if not miss.any(): continue
            grid = np.linspace(*rng[v], 2001) if v in self.knots else np.array(rng[v], float)
            G = pd.concat([ref] * len(grid), ignore_index=True); G[v] = grid
            e = self._eta_from_X(G); e0 = self._eta_from_X(ref)[0]
            lo[miss] += (e - e0).min(); hi[miss] += (e - e0).max()
        if "marital" in self.feats:
            miss = X["m_married"].isna().values
            if miss.any():
                opts = []
                for c in ("m_married", "m_prev", "m_never"):
                    G = ref.copy(); G[["m_married", "m_prev", "m_never"]] = 0.0; G[c] = 1.0; opts.append(self._eta_from_X(G)[0])
                e0 = self._eta_from_X(ref)[0]; opts = np.array(opts) - e0
                lo[miss] += opts.min(); hi[miss] += opts.max()
        return expit(lo), expit(hi)
