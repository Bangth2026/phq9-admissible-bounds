"""Compare admissible-value bounds with dense brute-force evaluation (4,001-point grids for one missing input,
about 900 x 900 points for two) for the proposed model and the conjunctive rule bases. Requires results/models.pkl
(created by src/experiment_v2.py)."""
import pickle, sys, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from experiment import BASE, LRModel, GBTModel  # noqa
from fofs import VARIABLES

S = pickle.load(open(ROOT / "results" / "models.pkl", "rb")); TEST = S["test"]
MASK = [f for f in BASE if f not in ("age", "female")]
CONT = {"age", "pir", "bmi", "sleep_wd", "sleep_we", "sedentary"}


def brute(model, row, vs):
    grids = []
    for v in vs:
        lo, hi = VARIABLES[v][2]
        if v in CONT: grids.append(np.linspace(lo, hi, 4001 if len(vs) == 1 else 900))
        else: grids.append(np.arange(lo, hi + 1, dtype=float))
    g = np.array(np.meshgrid(*grids)).reshape(len(vs), -1).T
    Rr = pd.concat([row] * len(g), ignore_index=True)
    for j, v in enumerate(vs): Rr[v] = g[:, j]
    _, M = model._M(Rr); eta = model.b0 + model._rule_mu(M) @ model.beta
    return eta.min(), eta.max()


def test_bounds(n=20, seed=7):
    from exact import ExactBounds
    cc = TEST[TEST[MASK].notna().all(1)].reset_index(drop=True); rng = np.random.default_rng(seed)
    for nm in ("Proposed (full)", "Version 1 (conjunctive rules)", "Main effects + interactions"):
        m = S["models"][nm]; eb = ExactBounds(m)
        for k in (1, 2):
            for i in rng.choice(len(cc), n, replace=False):
                row = cc.iloc[[i]].copy(); vs = list(rng.choice(MASK, k, replace=False)); row[vs] = np.nan
                el, eu = eb.bounds(row, return_eta=True); bl, bu = brute(m, row, vs)
                assert el[0] <= bl + 1e-9 and eu[0] >= bu - 1e-9, (nm, vs)          # bounds contain every brute-force value
                assert bl - el[0] < 1e-3 and eu[0] - bu < 1e-3, (nm, vs)          # and are tight up to the grid
                if nm == "Proposed (full)":                                       # separable model: grid-free bounds agree
                    gl, gu = m.bounds(row); lg = lambda p: np.log(p / (1 - p))
                    assert abs(lg(gl[0]) - el[0]) < 1e-9 and abs(lg(gu[0]) - eu[0]) < 1e-9


if __name__ == "__main__":
    test_bounds(); print("bounds contain and match brute force")
