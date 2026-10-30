# Analysis protocol, version 2 (locked before any evaluation of the revised model on test data)

Locked at: see timestamp in file name log below. Written before NHANES 2005–2014 files were downloaded and before
the revised model was applied to the 2017–2020 internal test set or the 2021–2023 temporal set.

## Reason for revision
Version 1 (conjunctive rule base, K=32) was 0.012–0.023 AUROC below logistic regression on the internal and
temporal test sets. Diagnosis on development data only: 30/32 rules were two-concept conjunctions, so the rule base
lacked main effects. Results of version 1 were seen before this revision; version 1 is retained and reported.

## Primary model (version 2)
- Structure 'main': all single-concept rules, one reference concept per variable omitted (the concept with the
  highest mean membership in development data), fitted jointly; then K two-concept rules added greedily with the
  same score statistic, sign constraints and redundancy filter as version 1.
- K chosen from {0, 8, 16, 32, 64} by 5-fold CV AUROC in development data with the same parsimony rule as version 1
  (smallest K within 0.005 of the best). CV AUROC: K0 0.7589, K8 0.7562, K16 0.7596, K32 0.7544, K64 0.7586.
  Selected K = 0 (main effects only, 25 rules plus intercept).
- Concepts, cut-points, direction priors, ridge penalty, admissible-value bounds, point score, Platt calibration and
  threshold rule (largest threshold with out-of-fold sensitivity >= 0.80) are unchanged from version 1.

## Comparators and ablations (each re-tuned for K by the same rule where applicable)
Logistic regression; gradient-boosted trees; version 1 (conjunctive); main + interactions (K=16, best CV);
no direction constraints; crisp memberships; box bounds; median imputation; quantile memberships without ontology;
normalised TSK aggregation.

## Evaluation sets
1. Internal test, NHANES 2017–March 2020 (already used for version 1).
2. Temporal, NHANES Aug 2021–Aug 2023 (already used for version 1).
3. NEW external (backward-temporal) validation: NHANES 2005–2006, 2007–2008, 2009–2010, 2011–2012, 2013–2014,
   pooled and per cycle. Never used for any decision. Harmonisation: weekday sleep from SLD010H; free-day sleep not
   fielded (treated as missing by every model); COPD item absent before 2013 (chronic-condition count over the
   available items); variables absent in a cycle are set to missing, never imputed from other cycles.
Primary metric: AUROC with bootstrap CI and paired difference versus logistic regression. Secondary: AUPRC,
sensitivity, specificity, Brier, O/E, coverage under abstention.

## Decision rule for the paper
The version-2 model is reported as the proposed system whatever its external-set result. No further changes to
the model after the external evaluation.

## Addendum (written after the analysis; not part of the locked protocol)
- The protocol file was last saved at 13:45:10 UTC on 28 October 2026; the first 2005–2014 file was downloaded at
  13:45:18 UTC. The "timestamp log" mentioned at the top of this file was not created; the file modification time
  is the only record.
- After the download, code changes were limited to reading the new files: recoding marital status (DMDMARTL to the
  three DMDMARTZ categories, as already done for 2015–2016), deriving race and ethnicity for subgroup analysis from
  RIDRETH1 before 2011 (no separate Asian category), and not constructing the extended sleep items for 2005–2014.
  The model, K, thresholds and comparators were not changed.
- The structure-selection CV is reproduced by src/cv_structure.py (results/cv_structure_v2.json).
