"""Exact admissible-value bounds for additive fuzzy rule models (Algorithm 1) and
bound-guided input acquisition (Algorithm 2).

Model:  eta(x) = b0 + sum_r beta_r * min_{k in A_r} mu_k(x_{v(k)}),  |A_r| <= 2, trapezoidal / set memberships.

Algorithm 1 (exact bounds). Missing variables are grouped into connected components (two missing variables are
connected if a rule contains both). eta is a constant plus a sum of independent functions of the components, so
its minimum and maximum are sums of per-component extrema. For a component:
  * categorical variables are enumerated;
  * a single continuous variable x: g(x) = sum_r beta_r min(mu_k(x), c_r) is continuous and piecewise linear; its
    breakpoints are the trapezoid corners of the concepts of x and the solutions of mu_k(x) = c_r for every observed
    partner membership c_r. Extrema over [lo, hi] lie in this finite set together with lo and hi (Proposition 1).
  * two continuous variables (u, w): on every rectangle between consecutive breakpoints of u and of w all
    memberships are affine, so F(u, w) is piecewise linear with cells cut by the switching lines mu_k(u) = mu_l(w)
    of the pair rules. Each cell is a convex polygon on which F is linear, so extrema lie at cell vertices:
    rectangle corners, intersections of switching lines with rectangle edges, and pairwise intersections of
    switching lines inside the rectangle (Proposition 2).
  * components with more than two continuous variables fall back to the box bound, which is a valid outer bound;
    the number of fallbacks is reported.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.special import expit
from fofs import VARIABLES, membership, prepare

EPS = 1e-12


def _cvar(m):
    return np.array([m.concepts[c]["variable"] for c in m.cnames])


def _ranges(m, override=None):
    out = {v: VARIABLES[v][2] for v in m.features}
    if override:
        out.update(override)
    return out


def _is_cont(m, v):
    return any(m.concepts[c]["mf"][0] == "trap" for c in m.cnames if m.concepts[c]["variable"] == v)


def _cats(m, v, rng):
    vals = sorted({x for c in m.cnames if m.concepts[c]["variable"] == v for x in m.concepts[c]["mf"][1]})
    return [x for x in vals if rng[0] <= x <= rng[1]]


def _corners(m, v, rng):
    lo, hi = rng
    pts = [lo, hi]
    for c in m.cnames:
        sp = m.concepts[c]
        if sp["variable"] == v and sp["mf"][0] == "trap":
            pts += [p for p in sp["mf"][1:] if lo < p < hi]
    return pts


def _solve(mf, level, lo, hi):
    """All x in [lo, hi] with mu(x) = level for a trapezoid."""
    _, a, b, c, d = mf
    out = []
    if b > a: out.append(a + level * (b - a))
    if d > c: out.append(d - level * (d - c))
    return [x for x in out if lo <= x <= hi]


class ExactBounds:
    def __init__(self, model, ranges=None):
        assert model.aggregation == "additive" and model.interval_missing and not model.crisp
        self.m = model
        self.cvar = _cvar(model)
        self.rng = _ranges(model, ranges)
        self.rule_vars = [sorted({self.cvar[j] for j in r}) for r in model.rules]
        self.used = sorted({v for rv in self.rule_vars for v in rv})
        self.fallbacks = 0
        self.fb_rows = set()
        self.components_seen = 0

    # ---- evaluation of the component part of eta at candidate points ------------------------------------
    def _eval(self, rk, vals, obs_mu):
        """rk: rule indices; vals: dict var -> array (N,); obs_mu: membership row (concepts,) of the person."""
        N = len(next(iter(vals.values())))
        tot = np.zeros(N)
        for k in rk:
            act = np.ones(N)
            for j in self.m.rules[k]:
                v = self.cvar[j]
                if v in vals:
                    act = np.minimum(act, membership(vals[v], self.m.concepts[self.m.cnames[j]]["mf"]))
                else:
                    act = np.minimum(act, obs_mu[j])
            tot += self.m.beta[k] * act
        return tot

    def _breaks(self, v, rk, obs_mu):
        """Breakpoints of the single-variable part for v: corners, endpoints, and mu_k(x) = c (observed partner)."""
        lo, hi = self.rng[v]
        pts = _corners(self.m, v, (lo, hi))
        for k in rk:
            r = self.m.rules[k]
            on_v = [j for j in r if self.cvar[j] == v]
            others = [j for j in r if self.cvar[j] != v]
            if on_v and others and all(not np.isnan(obs_mu[j]) for j in others):
                lvl = min(obs_mu[j] for j in others)
                for j in on_v:
                    mf = self.m.concepts[self.m.cnames[j]]["mf"]
                    if mf[0] == "trap":
                        pts += _solve(mf, lvl, lo, hi)
        return np.unique(np.round(np.array(pts, float), 12))

    def _affine(self, mf, x0, x1):
        """mu on [x0, x1] as alpha + gamma * x (trapezoid is affine between consecutive breakpoints)."""
        m0, m1 = membership(np.array([x0, x1]), mf)
        g = (m1 - m0) / (x1 - x0) if x1 > x0 else 0.0
        return m0 - g * x0, g

    def _cand_2d(self, u, w, rk, obs_mu):
        bu, bw = self._breaks(u, rk, obs_mu), self._breaks(w, rk, obs_mu)
        pairs = [k for k in rk if {u, w} <= {self.cvar[j] for j in self.m.rules[k]}]
        U, W = [], []
        for (u0, u1), (w0, w1) in itertools.product(zip(bu[:-1], bu[1:]), zip(bw[:-1], bw[1:])):
            pts = [(u0, w0), (u0, w1), (u1, w0), (u1, w1)]
            lines = []
            for k in pairs:
                ju = [j for j in self.m.rules[k] if self.cvar[j] == u][0]
                jw = [j for j in self.m.rules[k] if self.cvar[j] == w][0]
                a1, g1 = self._affine(self.m.concepts[self.m.cnames[ju]]["mf"], u0, u1)
                a2, g2 = self._affine(self.m.concepts[self.m.cnames[jw]]["mf"], w0, w1)
                lines.append((g1, -g2, a2 - a1))          # g1*u - g2*w = a2 - a1
            for (p, q, r) in lines:
                if abs(q) > EPS:                          # intersections with u = u0, u1
                    for uu in (u0, u1):
                        ww = (p * uu - r) / q
                        if w0 - 1e-12 <= ww <= w1 + 1e-12: pts.append((uu, ww))
                if abs(p) > EPS:                          # intersections with w = w0, w1
                    for ww in (w0, w1):
                        uu = (r + q * ww) / p
                        if u0 - 1e-12 <= uu <= u1 + 1e-12: pts.append((uu, ww))
            for (p1, q1, r1), (p2, q2, r2) in itertools.combinations(lines, 2):
                det = p1 * q2 - p2 * q1
                if abs(det) > EPS:
                    uu = (r1 * q2 - r2 * q1) / det; ww = (p1 * r2 - p2 * r1) / det
                    if u0 - 1e-12 <= uu <= u1 + 1e-12 and w0 - 1e-12 <= ww <= w1 + 1e-12: pts.append((uu, ww))
            for a, b in pts:
                U.append(min(max(a, u0), u1)); W.append(min(max(b, w0), w1))
        return np.array(U), np.array(W)

    def _component(self, comp, rk, obs_mu):
        cat = [v for v in comp if not _is_cont(self.m, v)]
        cont = [v for v in comp if _is_cont(self.m, v)]
        if len(cont) > 2:
            return None
        lo_best, hi_best = np.inf, -np.inf
        for combo in itertools.product(*[_cats(self.m, v, self.rng[v]) for v in cat]) if cat else [()]:
            if len(cont) == 0:
                vals = {v: np.array([x], float) for v, x in zip(cat, combo)}
            elif len(cont) == 1:
                xs = self._breaks(cont[0], rk, obs_mu)
                vals = {cont[0]: xs} | {v: np.full(len(xs), x, float) for v, x in zip(cat, combo)}
            else:
                U, W = self._cand_2d(cont[0], cont[1], rk, obs_mu)
                vals = {cont[0]: U, cont[1]: W} | {v: np.full(len(U), x, float) for v, x in zip(cat, combo)}
            # categorical values fixed: observed-partner breakpoints must use the categorical memberships too
            if cat and cont:
                mu2 = obs_mu.copy()
                for v, x in zip(cat, combo):
                    for j in np.where(self.cvar == v)[0]:
                        mu2[j] = membership(np.array([x]), self.m.concepts[self.m.cnames[j]]["mf"])[0]
                if len(cont) == 1:
                    xs = self._breaks(cont[0], rk, mu2)
                    vals = {cont[0]: xs} | {v: np.full(len(xs), x, float) for v, x in zip(cat, combo)}
                else:
                    U, W = self._cand_2d(cont[0], cont[1], rk, mu2)
                    vals = {cont[0]: U, cont[1]: W} | {v: np.full(len(U), x, float) for v, x in zip(cat, combo)}
            f = self._eval(rk, vals, obs_mu)
            lo_best = min(lo_best, f.min()); hi_best = max(hi_best, f.max())
        return lo_best, hi_best

    def bounds(self, df, return_eta=False):
        m = self.m
        df = prepare(df)
        M = m._mu(df)
        mL, mU = m._rule_mu(M, "L"), m._rule_mu(M, "U")
        pos = m.beta > 0
        miss = df[self.used].isna().values
        eL = m.b0 + mL @ m.beta; eU = eL.copy()          # complete rows: exact
        for i in np.where(miss.any(1))[0]:
            mv = [v for v, b in zip(self.used, miss[i]) if b]
            parent = {v: v for v in mv}
            def find(v):
                while parent[v] != v: v = parent[v]
                return v
            for rv in self.rule_vars:
                mm = [v for v in rv if v in parent]
                for a, b in zip(mm, mm[1:]): parent[find(a)] = find(b)
            comps = {}
            for v in mv: comps.setdefault(find(v), []).append(v)
            done = np.zeros(len(m.rules), bool); lo = hi = m.b0
            for comp in comps.values():
                rk = np.array([any(v in comp for v in rv) for rv in self.rule_vars])
                done |= rk
                self.components_seen += 1
                res = self._component(comp, np.where(rk)[0], M[i])
                if res is None:
                    self.fallbacks += 1; self.fb_rows.add(int(i))
                    lo += mL[i, rk & pos] @ m.beta[rk & pos] + mU[i, rk & ~pos] @ m.beta[rk & ~pos]
                    hi += mU[i, rk & pos] @ m.beta[rk & pos] + mL[i, rk & ~pos] @ m.beta[rk & ~pos]
                else:
                    lo += res[0]; hi += res[1]
            rest = ~done
            lo += mL[i, rest] @ m.beta[rest]; hi += mL[i, rest] @ m.beta[rest]
            eL[i], eU[i] = lo, hi
        if return_eta:
            return eL, eU
        return expit(eL), expit(eU)


def brute_force(model, row, vars_, n=2001, ranges=None):
    """Dense brute force for one person with 1-2 continuous/categorical missing variables (verification)."""
    import pandas as pd
    rng = _ranges(model, ranges)
    grids = []
    for v in vars_:
        if _is_cont(model, v):
            grids.append(np.linspace(*rng[v], n if len(vars_) == 1 else int(np.sqrt(n * 200))))
        else:
            grids.append(np.array(_cats(model, v, rng[v]), float))
    g = np.array(np.meshgrid(*grids)).reshape(len(vars_), -1).T
    R = pd.concat([row] * len(g), ignore_index=True)
    for j, v in enumerate(vars_): R[v] = g[:, j]
    _, Mm = model._M(R)
    eta = model.b0 + model._rule_mu(Mm) @ model.beta
    return eta.min(), eta.max()


# ------------------------------------------------------------------------------------------------------------
# Algorithm 2: bound-guided input acquisition
# ------------------------------------------------------------------------------------------------------------
def acquisition(model, eb, row_masked, row_true, thr, ref_values, policy="bound", rng=None, order=None):
    """Ask for missing inputs one at a time until the decision is resolved (refer / do not refer).
    policy='bound': ask the variable with the highest probability of resolving the decision on its own, estimated
    by evaluating the bounds at reference values of that variable (development-set empirical distribution),
    other inputs kept missing. Returns (n_asked, resolved, decision)."""
    import pandas as pd
    cur = row_masked.copy()
    asked = 0
    while True:
        zl, zu = eb.bounds(cur)
        if zl[0] >= thr: return asked, True, 1
        if zu[0] < thr: return asked, True, 0
        miss = [v for v in eb.used if pd.isna(cur[v].iloc[0])]
        if not miss: return asked, False, -1
        if policy == "random":
            v = miss[rng.integers(len(miss))]
        elif policy == "fixed":
            v = [x for x in order if x in miss][0]
        else:
            best, v = -1.0, miss[0]
            for cand in miss:
                vals, w = ref_values[cand]
                R = pd.concat([cur] * len(vals), ignore_index=True); R[cand] = vals
                a, b = eb.bounds(R)
                res = float(np.sum(w * ((a >= thr) | (b < thr))))
                if res > best + 1e-12: best, v = res, cand
        cur[v] = row_true[v].iloc[0]; asked += 1


def ref_distribution(dev, v, max_points=25):
    """Weighted support of a variable in development data (quantile bins for continuous variables)."""
    x = dev[v].dropna().values
    u = np.unique(x)
    if len(u) <= max_points:
        w = np.array([(x == a).mean() for a in u]); return u, w
    q = np.quantile(x, (np.arange(max_points) + 0.5) / max_points)
    return q, np.full(max_points, 1.0 / max_points)
