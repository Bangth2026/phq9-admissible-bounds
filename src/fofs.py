"""Ontology-guided fuzzy prescreening system (FOFS).

Zero-order TSK rule base whose concepts, admissible ranges, mutual exclusivity and
direction-of-association priors are declared in an explicit ontology. Missing inputs
are propagated as membership intervals [0,1] and the score bounds are computed exactly
(Karnik-Mendel style switch-point enumeration).
"""
from __future__ import annotations
import itertools, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------
VARIABLES = {
    # name: domain, type, admissible range, unit, NHANES source
    "age": ("Demographic", "continuous", (18, 80), "years", "DEMO.RIDAGEYR"),
    "female": ("Demographic", "binary", (0, 1), "-", "DEMO.RIAGENDR"),
    "pir": ("Socioeconomic", "continuous", (0, 5), "ratio", "DEMO.INDFMPIR"),
    "educ": ("Socioeconomic", "ordinal", (1, 5), "level", "DEMO.DMDEDUC2"),
    "marital": ("Socioeconomic", "nominal", (1, 3), "category", "DEMO.DMDMARTZ"),
    "bmi": ("Somatic", "continuous", (12, 90), "kg/m2", "BMX.BMXBMI"),
    "sleep_wd": ("Sleep", "continuous", (2, 14), "h", "SLQ.SLD012"),
    "sleep_we": ("Sleep", "continuous", (2, 14), "h", "SLQ.SLD013"),
    "srh": ("Somatic", "ordinal", (1, 5), "level", "HUQ.HUQ010"),
    "smoker": ("Lifestyle", "binary", (0, 1), "-", "SMQ.SMQ020/SMQ040"),
    "diabetes": ("Somatic", "binary", (0, 1), "-", "DIQ.DIQ010"),
    "hypertension": ("Somatic", "binary", (0, 1), "-", "BPQ.BPQ020"),
    "sedentary": ("Lifestyle", "continuous", (0, 24), "h/day", "PAQ.PAD680"),
    "n_chronic": ("Somatic", "count", (0, 10), "conditions", "MCQ (10 items)"),
    "told_sleep_trouble": ("Sleep", "binary", (0, 1), "-", "SLQ.SLQ050"),
    "daytime_sleepy": ("Sleep", "ordinal", (0, 4), "level", "SLQ.SLQ120"),
}


def ruspini(var, names, breaks, h, lo, hi, prov):
    """Trapezoids forming a Ruspini partition around knowledge-based break points."""
    out, edges = {}, [lo] + list(breaks) + [hi]
    for k, nm in enumerate(names):
        a = edges[k] - h if k > 0 else lo - 1e-9
        b = edges[k] + h if k > 0 else lo - 1e-9
        c = edges[k + 1] - h if k < len(names) - 1 else hi + 1e-9
        d = edges[k + 1] + h if k < len(names) - 1 else hi + 1e-9
        out[nm] = {"variable": var, "mf": ("trap", a, b, c, d), "provenance": prov}
    return out


def sets(var, mapping, prov):
    return {nm: {"variable": var, "mf": ("set", tuple(v)), "provenance": prov} for nm, v in mapping.items()}


def knowledge_concepts(extended=False):
    C = {}
    C |= ruspini("age", ["YoungAdult", "MiddleAged", "OlderAdult"], [35, 60], 5, 18, 80,
                 "Conventional adult age bands; transition width 5 y")
    C |= sets("female", {"Female": [1], "Male": [0]}, "Self-reported sex")
    C |= ruspini("pir", ["LowIncome", "MiddleIncome", "HighIncome"], [1.3, 3.5], 0.3, 0, 5,
                 "130% FPL (SNAP gross-income limit) and 350% FPL; transition width 0.3")
    C |= sets("educ", {"LessThanHighSchool": [1, 2], "HighSchoolOrSomeCollege": [3, 4],
                       "CollegeGraduate": [5]}, "NHANES education categories")
    C |= sets("marital", {"MarriedOrPartnered": [1], "PreviouslyMarried": [2], "NeverMarried": [3]},
              "NHANES marital status categories")
    C |= ruspini("bmi", ["Underweight", "NormalWeight", "Overweight", "Obese"], [18.5, 25, 30], 1.0,
                 12, 90, "WHO BMI cut-points; transition width 1 kg/m2")
    C |= ruspini("sleep_wd", ["ShortSleepWorkday", "AdequateSleepWorkday", "LongSleepWorkday"],
                 [6.5, 9.5], 0.5, 2, 14, "AASM/SRS consensus (>=7 h recommended); transition width 0.5 h")
    C |= ruspini("sleep_we", ["ShortSleepFreeDay", "AdequateSleepFreeDay", "LongSleepFreeDay"],
                 [6.5, 9.5], 0.5, 2, 14, "AASM/SRS consensus (>=7 h recommended); transition width 0.5 h")
    C |= sets("srh", {"GoodOrBetterHealth": [1, 2, 3], "FairHealth": [4], "PoorHealth": [5]},
              "Self-rated general health categories")
    C |= sets("smoker", {"CurrentSmoker": [1], "NonSmoker": [0]}, "SMQ020/SMQ040")
    C |= sets("diabetes", {"Diabetes": [1], "NoDiabetes": [0]}, "Doctor-told diabetes")
    C |= sets("hypertension", {"Hypertension": [1], "NoHypertension": [0]}, "Doctor-told hypertension")
    C |= ruspini("sedentary", ["LowSedentary", "ModerateSedentary", "HighSedentary"], [4, 8], 1.0, 0, 24,
                 "Sedentary time bands (h/day); transition width 1 h")
    C |= sets("n_chronic", {"NoChronicCondition": [0], "OneChronicCondition": [1],
                            "Multimorbidity": [2, 3, 4, 5, 6, 7, 8, 9, 10]}, "Count of 10 doctor-told conditions")
    if extended:
        C |= sets("told_sleep_trouble", {"ReportedSleepTrouble": [1], "NoReportedSleepTrouble": [0]}, "SLQ050")
        C |= sets("daytime_sleepy", {"FrequentDaytimeSleepiness": [3, 4], "OccasionalDaytimeSleepiness": [2],
                                     "RareDaytimeSleepiness": [0, 1]}, "SLQ120 frequency")
    return C


def quantile_concepts(df, base):
    """Ablation: data-driven tertile memberships replacing knowledge-based breaks."""
    C = {}
    for nm, spec in base.items():
        C[nm] = dict(spec)
    by_var = {}
    for nm, spec in base.items():
        if spec["mf"][0] == "trap":
            by_var.setdefault(spec["variable"], []).append(nm)
    for var, names in by_var.items():
        lo, hi = VARIABLES[var][2]
        qs = np.nanquantile(df[var], np.linspace(0, 1, len(names) + 1)[1:-1])
        h = 0.1 * (np.nanquantile(df[var], .9) - np.nanquantile(df[var], .1))
        C.update(ruspini(var, names, list(qs), h, lo, hi, "Training-data quantiles (ablation)"))
    return C


def prepare(df):
    df = df.copy()
    if "marital" not in df:
        m = np.where(df.married == 1, 1, np.where(df.prev_married == 1, 2, 3)).astype(float)
        m[df.married.isna().values] = np.nan
        df["marital"] = m
    return df


# ---------------------------------------------------------------------------
# Membership evaluation
# ---------------------------------------------------------------------------
def membership(x, mf, crisp=False):
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    ok = ~np.isnan(x)
    if mf[0] == "set":
        out[ok] = np.isin(x[ok], mf[1]).astype(float)
        return out
    _, a, b, c, d = mf
    v = x[ok]
    if crisp:
        lo, hi = (a + b) / 2, (c + d) / 2
        out[ok] = ((v >= lo) & (v < hi)).astype(float)
        return out
    m = np.zeros_like(v)
    m[(v >= b) & (v <= c)] = 1.0
    up = (v > a) & (v < b); m[up] = (v[up] - a) / (b - a)
    dn = (v > c) & (v < d); m[dn] = (d - v[dn]) / (d - c)
    out[ok] = m
    return out


# ---------------------------------------------------------------------------
# Exact bounds of a weighted average with interval weights (switch-point enumeration)
# ---------------------------------------------------------------------------
def wavg_bounds(aL, aU, c):
    """aL,aU: (n,R) activation bounds; c: (R,) consequents. Returns (zmin, zmax)."""
    o = np.argsort(c); c = c[o]; aL = aL[:, o]; aU = aU[:, o]
    R = len(c)
    # prefix (first k rules) / suffix sums
    def bound(first, rest):
        cf_n = np.concatenate([np.zeros((first.shape[0], 1)), np.cumsum(first * c, 1)], 1)
        cf_d = np.concatenate([np.zeros((first.shape[0], 1)), np.cumsum(first, 1)], 1)
        rs_n = np.concatenate([np.cumsum((rest * c)[:, ::-1], 1)[:, ::-1], np.zeros((rest.shape[0], 1))], 1)
        rs_d = np.concatenate([np.cumsum(rest[:, ::-1], 1)[:, ::-1], np.zeros((rest.shape[0], 1))], 1)
        num, den = cf_n + rs_n, cf_d + rs_d
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den
    zmin = np.nanmin(bound(aU, aL), 1)  # upper weights on small consequents
    zmax = np.nanmax(bound(aL, aU), 1)  # upper weights on large consequents
    return zmin, zmax


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FOFS:
    """aggregation='additive': p = sigmoid(b0 + sum_r beta_r * mu_r)  (default; exact additive trace)
       aggregation='normalized': zero-order TSK weighted average sum(a c)/sum(a)  (ablation)."""

    def __init__(self, features, n_rules=16, max_ante=2, crisp=False, direction=True,
                 interval_missing=True, concepts="knowledge", priors=None, aggregation="additive",
                 smooth=10.0, lam=1e-3, min_support=0.01, redundancy=0.95, bound_mode="admissible",
                 max_enum=20000, structure="conjunctive", seed=0):
        self.features = list(features); self.K = n_rules; self.max_ante = max_ante
        self.crisp = crisp; self.direction = direction; self.interval_missing = interval_missing
        self.concept_mode = concepts; self.priors = priors or {}; self.aggregation = aggregation
        self.smooth = smooth; self.lam = lam; self.min_support = min_support; self.redundancy = redundancy
        self.bound_mode = bound_mode; self.max_enum = max_enum
        # structure='main': every single-concept rule (one reference concept per variable omitted) enters first,
        # then K sign-constrained two-concept rules are added greedily (fuzzy additive model with interactions).
        # structure='conjunctive': K rules of any arity chosen greedily (version 1 of the system).
        self.structure = structure

    # -- concepts ------------------------------------------------------------
    def _concepts(self, df):
        ext = any(f in self.features for f in ("told_sleep_trouble", "daytime_sleepy"))
        base = knowledge_concepts(extended=ext)
        if self.concept_mode == "quantile":
            base = quantile_concepts(df, base)
        return {k: v for k, v in base.items() if v["variable"] in self.features}

    def _mu(self, df):
        cols = [membership(df[s["variable"]].values, s["mf"], self.crisp) for s in self.concepts.values()]
        return np.column_stack(cols)  # NaN where variable missing

    def _impute_mu(self, M):
        M = M.copy()
        for j in range(M.shape[1]):
            M[np.isnan(M[:, j]), j] = self.mu_fill[j]
        return M

    def _allowed_sign(self, r):
        """Ontology coherence: +1 (beta>=0), -1 (beta<=0), 0 (free) or None (inadmissible)."""
        if not self.direction:
            return 0
        d = {self.priors.get(self.cnames[k], 0) for k in r} - {0}
        if len(d) > 1:
            return None
        return d.pop() if d else 0

    # -- fitting ---------------------------------------------------------------
    def fit(self, df, y):
        df = prepare(df); y = np.asarray(y, float)
        if not self.interval_missing:  # median imputation in input space (ablation)
            self.medians = df[self.features].median()
            df = df.copy(); df[self.features] = df[self.features].fillna(self.medians)
        self.concepts = self._concepts(df)
        self.cnames = list(self.concepts); cvar = [self.concepts[c]["variable"] for c in self.cnames]
        M = self._mu(df); self.mu_fill = np.nanmean(M, 0); Mi = self._impute_mu(M)
        p0 = y.mean(); self.p0 = p0; n = len(y)
        cand = [(j,) for j in range(len(self.cnames))]
        if self.max_ante >= 2:   # ontology: one concept per variable in an antecedent
            cand += [(i, j) for i, j in itertools.combinations(range(len(self.cnames)), 2) if cvar[i] != cvar[j]]
        rules, signs, cols, rates = [], [], [], []
        for r in cand:
            sg = self._allowed_sign(r)
            if sg is None:
                continue
            a = Mi[:, list(r)].min(1)
            if a.mean() < self.min_support:
                continue
            c = (a @ y + self.smooth * p0) / (a.sum() + self.smooth)
            if self.aggregation == "normalized" and sg != 0 and np.sign(c - p0) != sg:
                continue
            rules.append(r); signs.append(sg); cols.append(a); rates.append(c)
        Ac = np.column_stack(cols); signs = np.array(signs); rates = np.array(rates)
        sel = []
        if self.aggregation == "additive":
            b0 = logit(p0); beta = np.zeros(0)
            if self.structure == "main":
                for v in dict.fromkeys(cvar):
                    idx = [k for k, r in enumerate(rules) if len(r) == 1 and cvar[r[0]] == v]
                    if not idx: continue
                    ref = max(idx, key=lambda k: Ac[:, k].mean())  # most prevalent concept = reference
                    sel += [k for k in idx if k != ref]
                b0, beta = self._fit_additive(Ac[:, sel], y, signs[sel], b0, np.zeros(len(sel)))
                pair_only = np.array([len(r) == 2 for r in rules])
            for _ in range(self.K):
                eta = b0 + (Ac[:, sel] @ beta if sel else np.zeros(n)); p = expit(eta); r_ = y - p
                g = Ac.T @ r_; h = (Ac ** 2).T @ (p * (1 - p)) + 1e-9
                gain = g ** 2 / h
                gain[(signs > 0) & (g < 0)] = 0; gain[(signs < 0) & (g > 0)] = 0; gain[sel] = 0
                if self.structure == "main": gain[~pair_only] = 0
                j = self._pick(gain, Ac, sel)
                if j is None: break
                sel.append(j); b0, beta = self._fit_additive(Ac[:, sel], y, signs[sel], b0, np.append(beta, 0.0))
            self.rules = [rules[j] for j in sel]; self.signs = signs[sel]
            self.b0, self.beta = b0, beta
        else:
            num = np.full(n, p0); den = np.ones(n)
            for _ in range(self.K):
                z = num / den; gain = (Ac.T @ (y - z)) ** 2 / (Ac.sum(0) + self.smooth); gain[sel] = 0
                j = self._pick(gain, Ac, sel)
                if j is None: break
                sel.append(j); num += Ac[:, j] * rates[j]; den += Ac[:, j]
            self.rules = [()] + [rules[j] for j in sel]
            self._fit_normalized(self._rule_mu(Mi), y, np.array([p0] + [rates[j] for j in sel]))
        return self

    def _pick(self, gain, Ac, sel):
        for j in np.argsort(-gain):
            if gain[j] <= 1e-12:
                return None
            if all(abs(np.corrcoef(Ac[:, j], Ac[:, k])[0, 1]) <= self.redundancy for k in sel):
                return j
        return None

    def _fit_additive(self, A, y, sg, b0, beta):
        lam, n = self.lam, len(y)
        def obj(t):
            b, be = t[0], t[1:]; p = expit(b + A @ be); p = np.clip(p, 1e-9, 1 - 1e-9)
            L = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + lam * np.sum(be ** 2)
            g = (p - y) / n
            return L, np.concatenate([[g.sum()], A.T @ g + 2 * lam * be])
        bnds = [(None, None)] + [((0, None) if s > 0 else (None, 0) if s < 0 else (None, None)) for s in sg]
        res = minimize(obj, np.concatenate([[b0], beta]), jac=True, method="L-BFGS-B", bounds=bnds)
        return res.x[0], res.x[1:]

    def _fit_normalized(self, A, y, c0):
        u0 = logit(np.clip(c0, 1e-4, 1 - 1e-4)); v0 = np.zeros(len(c0)); n = len(y); lam = self.lam
        def obj(theta):
            u, v = theta[:len(c0)], theta[len(c0):]
            c, w = expit(u), np.exp(v)
            a = A * w; S = a.sum(1); z = np.clip((a @ c) / S, 1e-6, 1 - 1e-6)
            L = -np.mean(y * np.log(z) + (1 - y) * np.log(1 - z)) + lam * (np.sum((u - u0) ** 2) + np.sum(v ** 2))
            g = (z - y) / (z * (1 - z)) / n
            gc = (a / S[:, None]).T @ g
            gw = ((A * (c[None, :] - z[:, None])) / S[:, None]).T @ g
            return L, np.concatenate([gc * c * (1 - c) + 2 * lam * (u - u0), gw * w + 2 * lam * v])
        res = minimize(obj, np.concatenate([u0, v0]), jac=True, method="L-BFGS-B")
        self.c = expit(res.x[:len(c0)]); self.w = np.exp(res.x[len(c0):])

    def _rule_mu(self, M, bound=None):
        cols = []
        for r in self.rules:
            if len(r) == 0:
                cols.append(np.ones(M.shape[0])); continue
            sub = M[:, list(r)]
            if bound == "L":
                sub = np.where(np.isnan(sub), 0.0, sub)
            elif bound == "U":
                sub = np.where(np.isnan(sub), 1.0, sub)
            cols.append(sub.min(1))
        return np.column_stack(cols) if cols else np.zeros((M.shape[0], 0))

    # -- inference -------------------------------------------------------------
    def _M(self, df):
        df = prepare(df)
        if not self.interval_missing:
            df = df.copy(); df[self.features] = df[self.features].fillna(self.medians)
        return df, self._mu(df)

    def _grid(self, var):
        """Admissible values of a variable: category values or a grid incl. all trapezoid corners."""
        specs = [self.concepts[c] for c in self.cnames if self.concepts[c]["variable"] == var]
        lo, hi = VARIABLES[var][2]
        if all(sp["mf"][0] == "set" for sp in specs):
            vals = sorted({v for sp in specs for v in sp["mf"][1]})
        else:
            pts = [p for sp in specs for p in sp["mf"][1:] if lo <= p <= hi]
            vals = sorted(set(np.round(np.r_[np.linspace(lo, hi, 41), pts], 6)))
        vals = np.array(vals, float)
        mv = np.column_stack([membership(vals, sp["mf"], self.crisp) for sp in specs])
        _, keep = np.unique(np.round(mv, 9), axis=0, return_index=True)  # distinct membership vectors
        return vals[np.sort(keep)]

    def _eta(self, M):
        return self.b0 + self._rule_mu(M) @ self.beta

    def bounds(self, df):
        df, M = self._M(df)
        if self.aggregation != "additive":
            mL, mU = self._rule_mu(M, "L"), self._rule_mu(M, "U")
            return wavg_bounds(mL * self.w, mU * self.w, self.c)
        mL, mU = self._rule_mu(M, "L"), self._rule_mu(M, "U")
        pos = self.beta > 0
        eL = self.b0 + mL[:, pos] @ self.beta[pos] + mU[:, ~pos] @ self.beta[~pos]
        eU = self.b0 + mU[:, pos] @ self.beta[pos] + mL[:, ~pos] @ self.beta[~pos]
        if self.bound_mode == "box":
            return expit(eL), expit(eU)
        # ontology-constrained bounds: optimise over admissible values of the missing variables
        cvar = np.array([self.concepts[c]["variable"] for c in self.cnames])
        rule_vars = [sorted({cvar[j] for j in r}) for r in self.rules]
        used = sorted({v for rv in rule_vars for v in rv})
        miss = df[used].isna().values
        pats = {}
        for i, row in enumerate(miss):
            if row.any():
                pats.setdefault(tuple(row), []).append(i)
        for pat, idx in pats.items():
            idx = np.array(idx)
            mv = [v for v, b in zip(used, pat) if b]
            # connected components of missing variables (coupled through rules)
            parent = {v: v for v in mv}
            def find(v):
                while parent[v] != v: v = parent[v]
                return v
            for rv in rule_vars:
                mm = [v for v in rv if v in parent]
                for a, b in zip(mm, mm[1:]): parent[find(a)] = find(b)
            comps = {}
            for v in mv: comps.setdefault(find(v), []).append(v)
            sub = df.iloc[idx]
            Msub = M[idx]
            lo_tot = self.b0 + np.zeros(len(idx)); hi_tot = lo_tot.copy()
            done_rules = np.zeros(len(self.rules), bool)
            for comp in comps.values():
                rk = np.array([any(v in comp for v in rv) for rv in rule_vars])
                done_rules |= rk
                grids = [self._grid(v) for v in comp]
                if np.prod([len(g) for g in grids]) > self.max_enum:  # fallback: box bound
                    lo_tot += mL[idx][:, rk & pos] @ self.beta[rk & pos] + mU[idx][:, rk & ~pos] @ self.beta[rk & ~pos]
                    hi_tot += mU[idx][:, rk & pos] @ self.beta[rk & pos] + mL[idx][:, rk & ~pos] @ self.beta[rk & ~pos]
                    continue
                combos = np.array(list(itertools.product(*grids)))  # (A, |comp|)
                total = np.zeros((len(idx), len(combos)))
                for k in np.where(rk)[0]:
                    act = np.ones((len(idx), len(combos)))
                    for j in self.rules[k]:
                        v = cvar[j]
                        if v in comp:
                            g = membership(combos[:, comp.index(v)], self.concepts[self.cnames[j]]["mf"], self.crisp)
                            act = np.minimum(act, g[None, :])
                        else:
                            act = np.minimum(act, Msub[:, [j]])
                    total += self.beta[k] * act
                best_lo, best_hi = total.min(1), total.max(1)
                lo_tot += best_lo; hi_tot += best_hi
            rest = ~done_rules  # rules with no missing variable
            lo_tot += mL[idx][:, rest] @ self.beta[rest]; hi_tot += mL[idx][:, rest] @ self.beta[rest]
            eL[idx], eU[idx] = lo_tot, hi_tot
        return expit(eL), expit(eU)

    def score(self, df):
        """Point score: missing concept memberships set to their development-sample means,
        clipped into the admissible interval."""
        zl, zu = self.bounds(df)
        if self.aggregation != "additive":
            return (zl + zu) / 2
        _, M = self._M(df)
        z = expit(self._eta(self._impute_mu(M)))
        return np.clip(z, zl, zu)

    # -- explanation -----------------------------------------------------------
    def rule_text(self, k):
        r = self.rules[k]
        return "TRUE (default)" if not r else " AND ".join(self.cnames[j] for j in r)

    def rulebase(self):
        if self.aggregation == "additive":
            return [{"id": 0, "rule": "intercept", "beta": float(self.b0)}] + [
                {"id": k + 1, "rule": self.rule_text(k), "beta": float(self.beta[k]),
                 "ontology_sign": int(self.signs[k]),
                 "concept_provenance": [self.concepts[self.cnames[j]]["provenance"] for j in self.rules[k]]}
                for k in range(len(self.rules))]
        return [{"id": k, "rule": self.rule_text(k), "weight": float(self.w[k]), "consequent": float(self.c[k])}
                for k in range(len(self.rules))]

    def explain(self, row: pd.Series):
        """Executable trace (additive aggregation)."""
        df, M = self._M(row.to_frame().T.astype(float))
        mL, mU = self._rule_mu(M, "L")[0], self._rule_mu(M, "U")[0]
        zl, zu = self.bounds(row.to_frame().T.astype(float))
        miss = [f for f in self.features if pd.isna(df[f].iloc[0])]
        t = {"inputs": {f: (None if pd.isna(df[f].iloc[0]) else float(df[f].iloc[0])) for f in self.features},
             "missing_inputs": miss, "intercept": float(self.b0), "concepts": {}, "rules": [],
             "prob_lower": float(zl[0]), "prob_upper": float(zu[0])}
        for j in sorted({j for r in self.rules for j in r}):
            m = M[0, j]
            t["concepts"][self.cnames[j]] = {"variable": self.concepts[self.cnames[j]]["variable"],
                                             "membership": None if np.isnan(m) else float(m),
                                             "provenance": self.concepts[self.cnames[j]]["provenance"]}
        for k in range(len(self.rules)):
            e = {"id": k + 1, "rule": self.rule_text(k), "beta": float(self.beta[k]),
                 "firing_lower": float(mL[k]), "firing_upper": float(mU[k])}
            if mL[k] == mU[k]:
                e["logit_contribution"] = float(self.beta[k] * mL[k])
            t["rules"].append(e)
        return t


def reconstruct_from_trace(t):
    """Independent recomputation of the probability bounds from an exported trace."""
    lo = t["intercept"] + sum(min(r["beta"] * r["firing_lower"], r["beta"] * r["firing_upper"]) for r in t["rules"])
    hi = t["intercept"] + sum(max(r["beta"] * r["firing_lower"], r["beta"] * r["firing_upper"]) for r in t["rules"])
    return 1 / (1 + np.exp(-lo)), 1 / (1 + np.exp(-hi))
