"""Re-evaluate the probability from each serialised JSON trace with code that does not import the model, for all
internal test participants with complete inputs. Requires results/models.pkl."""
import json, math, pickle, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from experiment import BASE, LRModel, GBTModel  # noqa


def prob_from_trace(t):
    """Independent re-computation: intercept plus beta x firing of every rule, then the logistic function."""
    eta = t["intercept"] + sum(r["beta"] * r["firing_lower"] for r in t["rules"])
    return 1.0 / (1.0 + math.exp(-eta))


def test_traces():
    S = pickle.load(open(ROOT / "results" / "models.pkl", "rb")); m = S["models"]["Proposed (full)"]; test = S["test"]
    zl, _ = m.bounds(test); worst = 0.0; n = 0
    for i in range(len(test)):
        row = test.iloc[i][BASE]
        if row.isna().any(): continue
        t = json.loads(json.dumps(m.explain(row)))
        worst = max(worst, abs(prob_from_trace(t) - zl[i])); n += 1
    assert worst < 1e-12, worst
    return n, worst


if __name__ == "__main__":
    print("traces reproduced: n = %d, max abs error = %.1e" % test_traces())
