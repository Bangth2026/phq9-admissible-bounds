#!/usr/bin/env bash
# Full pipeline from the public NHANES files to the statistics file and manuscript.
# Scripts use absolute paths under /home/claude/ (data/, results/, src/); edit them or create that layout.
# results/v1/ (version-1 outputs) and results/protocol_v2.md are inputs and are not regenerated.
set -e
cd "$(dirname "$0")/src"
bash download_nhanes.sh
python cohort.py && python priors.py
python experiment_v2.py                      # version 2: internal, temporal, external validation, ablations
python experiment2.py && python recal.py && python verify_bounds.py && python figs_extra.py && python cv_structure.py
python rev_models.py                         # added comparators, PHQ-2, record-typical model, DCA, operating points
python rev_missing.py && python rev_exact.py # Algorithm 1 exactness, MCAR stress tests, Algorithm 2
python rev_priors_splits.py priors && python rev_priors_splits.py splits 20 && python rev_priors_mv.py
python narrowed_slr.py && python contrib_ranges.py
python sec2_models.py && python sec2_missing.py   # spline LR, refits without free-day sleep, natural/MNAR missingness
python sec3.py                               # imputation-spread abstention, risk-coverage, sex-specific thresholds
python export_schema.py && python figs_r1.py && python figs_r3.py
python stats.py && python stats_r1.py && python stats_r2.py && python stats_r3.py && python fig_flow.py
python sec4.py && python sec4b.py && python rev_splits100.py splits 100 && python figs_r4.py   # abstention rules with the model held fixed
python sec5.py && python sec5b.py && python stats_r4.py && python stats_r5.py   # point-score selective prediction, net-benefit intervals
python sec6.py && python stats_r6.py && python fig_design.py   # operational AUROC differences, scenario summary, design figure
python sec7.py && python stats_r7.py && python fig_export7.py   # pooled intervals, figure export
python sec8.py && python stats_r8.py && python fig_tradeoff.py   # configuration sensitivity, recalibration, trade-off figure
python manuscript_full7.py && python manuscript_r8.py && DOUBLE=1 node render.js ../manuscript.docx ../results/manuscript_r8.json lines && DOUBLE=1 node render.js ../supplementary.docx ../results/supplement_r8.json nolines
cd .. && python tests/test_cohort_priors.py && python tests/test_trace_fidelity.py && python tests/test_bounds_bruteforce.py
