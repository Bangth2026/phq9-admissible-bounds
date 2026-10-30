# Exact admissible-value bounds for missing inputs in an interpretable fuzzy additive prescreening model (NHANES)

Code, knowledge schema and results for the manuscript *"Exact admissible-value bounds for missing inputs in an
interpretable fuzzy additive model to prescreen elevated depressive symptoms: development and validation in NHANES"*.

The model flags adults likely to score ≥ 10 on the PHQ-9 from 14 non-symptom survey variables. It is **not** a
diagnostic tool and is not a substitute for the PHQ-2 where symptom questions can be asked.

## Contents

| Path | What it is |
| --- | --- |
| `src/fofs.py` | Model: variables, admissible ranges, concepts, provenance; sign-constrained fuzzy additive rules; JSON trace |
| `src/exact.py` | **Algorithm 1** (exact admissible-value bounds, Propositions 1–2) and **Algorithm 2** (bound-guided input acquisition) |
| `src/rev_common.py` | Comparators: sign-constrained LR (exact linear bounds), spline LR (restricted cubic splines, grid bounds), LR without indicators, LR with multiple imputation, monotone GBT, EBM (GA²M) |
| `schema/knowledge_schema.jsonld` | Knowledge schema as JSON-LD 1.1 with a SKOS concept scheme |
| `results/protocol_v2.md` | Analysis protocol of version 2 with its addendum (not registered; time stamps are file modification times recorded in the addendum) |
| `results/paper_stats_r2.json` | Every number in the manuscript, written by `src/stats_r2.py` |
| `results/v1/` | Archived outputs of version 1 (conjunctive rule base), used as inputs |
| `results/secondary/` | Outputs of the secondary analyses |
| `run_all.sh` | Full pipeline in order |

## Running

Python ≥ 3.11 (`pip install -r requirements.txt`); Node.js with `docx` for the manuscript. Scripts use absolute paths
under `/home/claude/` (`data/`, `results/`, `src/`); create that layout or edit the `R`/`D` constants. Then
`bash run_all.sh` (about 3 hours on two CPU cores). Random seed 2026 throughout. Building the manuscript fails if
any statistic is missing, so no number is typed by hand. Model pickles are not stored (they are regenerated).

## Use of generative AI

The analysis code was drafted with the assistance of Claude (Anthropic) under the authors' specification and
reviewed by the authors; independent checks are described in Section 2.14 of the manuscript.

## Licence

MIT (see `LICENSE`). NHANES data are public and are not redistributed here.
