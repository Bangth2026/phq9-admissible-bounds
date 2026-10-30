"""Reproduce the cohort flow and the direction priors from the public NHANES files and compare them with the stored
results (results/flow.json, results/priors.json). Run after src/cohort.py and src/priors.py."""
import json, sys, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"


def test_flow_and_priors():
    stored_flow = json.load(open(R / "flow.json")); stored_pri = json.load(open(R / "priors.json"))
    subprocess.run([sys.executable, "cohort.py"], cwd=ROOT / "src", check=True, capture_output=True)
    subprocess.run([sys.executable, "priors.py"], cwd=ROOT / "src", check=True, capture_output=True)
    assert json.load(open(R / "flow.json")) == stored_flow
    assert json.load(open(R / "priors.json")) == stored_pri


if __name__ == "__main__":
    test_flow_and_priors(); print("cohort flow and priors reproduced")
