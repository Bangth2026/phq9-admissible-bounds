# Abstaining when inputs are missing: worst-case bounds, imputation spread and pattern submodels (NHANES)

Code, knowledge schema and results for the manuscript *"Abstaining when inputs are missing: worst-case bounds,
imputation spread and pattern submodels compared for prescreening elevated depressive symptoms in NHANES"*.

The model flags adults likely to score ≥ 10 on the PHQ-9 from 14 non-symptom survey variables. It is **not** a
diagnostic tool and is not a substitute for the PHQ-2 where symptom questions can be asked.

## Contents

| Path | What it is |
| --- | --- |
| `src/fofs.py` | Model: variables, admissible ranges, concepts, provenance; sign-constrained fuzzy additive rules; JSON trace |
| `src/fofs.py` `bounds()` | **Algorithm 1** (Box 1): worst-case bounds for a main-effects rule base |
| `src/exact.py` | **Algorithm 2** (Box 2: exact worst-case bounds for rule bases with conjunctions) and **Algorithm 3** (Box 3: bound-guided input acquisition) |
| `src/sec4.py` | Abstention rules with the base model held fixed: worst-case bounds, imputation spread, pattern submodels, complete cases; natural, MCAR, MNAR, outcome-dependent and shifted missingness; risk–coverage, AURC, decision consequences |
| `src/sec5.py`, `src/sec5b.py` | Point-score selective prediction, net-benefit intervals, prevalence of shifted samples, ordinal-residual shift |
| `src/sec4b.py` | Design-based bootstrap, minimum sample size, calibration maps, acquisition with item costs |
| `src/sec3.py` | Earlier imputation-spread analysis, sex-specific thresholds |
| `tests/` | Independent checks: cohort flow and priors, bounds against brute force, trace re-evaluation |
| `src/rev_common.py` | Comparators: sign-constrained LR (exact linear bounds), spline LR (restricted cubic splines, grid bounds), LR without indicators, LR with multiple imputation, monotone GBT, EBM (GA²M) |
| `schema/knowledge_schema.jsonld` | Knowledge schema as JSON-LD 1.1 with a SKOS concept scheme |
| `results/protocol_v2.md` | Analysis protocol of version 2 with its addendum (not registered; time stamps are file modification times recorded in the addendum) |
| `results/paper_stats_r5.json` | Every number in the manuscript, written by `src/stats_r5.py` (earlier statistics files are kept) |
| `results/v1/` | Archived outputs of version 1 (conjunctive rule base), used as inputs |
| `results/secondary/` | Outputs of the secondary analyses |
| `run_all.sh` | Full pipeline in order |

## Running

Python ≥ 3.11 (`pip install -r requirements.txt`); Node.js with `docx` for the manuscript. Scripts use absolute paths
under `/home/claude/` (`data/`, `results/`, `src/`); create that layout or edit the `R`/`D` constants. Then
`bash run_all.sh` (about 5 hours on two CPU cores). The tests run at the end (`python tests/test_*.py`). Random seed 2026 throughout. Building the manuscript fails if
any statistic is missing, so no number is typed by hand. Model pickles are not stored (they are regenerated).

## Use of generative AI

The analysis code was drafted with the assistance of Claude (Anthropic) under the authors' specification and
reviewed by the authors; the tests were drafted with the same tool; their construction and checks are described in Section 2.13 of the manuscript.

## Licence

MIT (see `LICENSE`). NHANES data are public and are not redistributed here.
