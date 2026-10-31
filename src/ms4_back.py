# Discussion, conclusions, declarations and appendices of version 4.
from ms4_helpers import *  # noqa

h1("4. Discussion")
h2("4.1. Principal findings")
p("For RQ1, with the base model held fixed, the abstention rules differed little where few inputs were missing and "
  "clearly where many were. Imputation spread and pattern submodels decided more participants than worst-case bounds, "
  "classified the decided participants more accurately at equal coverage, and gave higher net benefit at a threshold "
  "probability of 0.10 under structural, random and non-random missingness. Pattern submodels were the best response "
  "to an input that was never collected. Worst-case bounds missed the fewest cases in every scenario, because they "
  "referred more people; this bought the highest net benefit at a threshold probability of 0.05 only when missingness "
  "depended on self-rated health or on the outcome itself. For RQ2, the fuzzy additive model ranked people about as "
  "well as linear logistic regression in all three periods and slightly less well than spline regression; the "
  "pre-specified primary contrast favoured it only because the comparator coded an input that was not collected "
  "with an indicator. For RQ3, the bound-guided request rule saved a small number of inputs, and a cost-weighted "
  "version had the lowest expected cost of the orders compared.")

h2("4.2. When are worst-case bounds worth their cost?")
p("We had proposed that worst-case bounds are worth having where an input is absent by design and its distribution in "
  "the target population is unknown. The 2005–2014 cohort is that setting, and the bounds lost there: the imputation "
  "model drew free-day sleep from its 2017–2020 conditional distribution, an assumption that cannot be checked, and "
  "still decided more participants more accurately. The added analyses explain why and show where the ordering "
  "changes. When the imputation model was made wrong on purpose, by sampling a target population whose self-rated "
  "health or free-day sleep departed systematically from what the other inputs predict, imputation spread and pattern "
  "submodels became less sensitive, missing about twice as many cases per 1,000 as without shift, but the "
  "records they chose to decide were still classified better than those decided by the bounds. The bounds did not "
  "become more accurate; they stayed conservative and referred almost everyone. Under non-random and outcome-dependent "
  "masking the picture was similar, with one difference: because a missing value then signals risk, the bounds' habit "
  "of referring incomplete records happened to be right, and for the fuzzy model their risk–coverage curve was better than "
  "those of imputation spread and pattern submodels. The ordering therefore reverses only on measures that weigh missed cases heavily, and only when "
  "missingness carries information about the outcome.")
p("Three features of the bounds explain the rest. First, the guarantee is relative to the model and its frozen "
  "threshold, not to the outcome: in 2021–2023, where every model under-predicted by a factor of about 1.6, a decision "
  "that holds for every admissible completion is wrong in the same way as the model's decision on a complete record. "
  "Second, the width of the bounds is set by single coefficients and admissible ranges rather than by how plausible the "
  "extreme completions are for the person. In 2005–2014 the free-day sleep coefficients made the bounds asymmetric, so "
  "they decided many participants to refer and few not to refer; with self-rated health missing, its range spans most "
  "of the linear predictor and the regression models' bounds decided almost nobody. Third, the distance of the bounds "
  "from the threshold is a poor confidence measure: ranking participants by the distance of the point score from the "
  "threshold gave a better risk–coverage curve in 2005–2014 than ranking by the bounds. Bounded terms, which the "
  "schema's saturating memberships provide, keep the bounds narrower than those of linear or spline regression, but "
  "they do not change these three features.")
p("For practice this suggests a simple order of preference. If an input is missing by design in a whole population, "
  "fit a model without it (a pattern submodel); the cost in discrimination here was nil. If inputs are missing "
  "sporadically and the mechanism is plausibly ignorable given the observed inputs, abstain on imputation spread. "
  "Worst-case bounds are a defensible choice only when missingness is likely to signal risk and a missed case weighs "
  "much more than an unnecessary referral, and even then their advantage over referring every incomplete record is "
  "small.")

h2("4.3. Relation to other approaches and what the schema contributes")
p("Worst-case bounds are the credal-classifier logic [@zaffalon2002,@corani2008] applied to an additive model with "
  "declared admissible sets, and imputation spread is the abstention counterpart of impute-then-predict, which can be "
  "consistent even when values are not missing at random if the predictor is flexible enough [@lemorvan2021]. Pattern "
  "submodels [@fletchermercaldo2020] did well here because the patterns were few; with many sporadic patterns they "
  "need sharing of information across patterns. Conformal prediction with missing values [@zaffran2023] guarantees the "
  "coverage of label sets within patterns under exchangeability; its abstentions depend on the score band rather than "
  "on which inputs are missing, and the guarantee is lost under the period shift studied here, which is why we did not "
  "use it as a baseline. Risk–coverage evaluation comes from selective classification [@geifman2017].")
p("The knowledge schema did not improve ranking: the fitted model is a sign-constrained piecewise-linear additive model "
  "whose constraints were not binding, and removing the schema left discrimination within noise. What the schema "
  "contributes is the admissible set of each input, which worst-case bounds need and a fitted model does not carry, "
  "and auditability: each of the {{n_rules}} terms is a named concept with a declared cut-point source and prior in a "
  "machine-readable file that the code loads, in line with interpretability criteria for fuzzy systems [@alonso2015]. The audit also exposes the declarations' limits, such as the prior for "
  "free-day sleep that was inherited rather than estimated and that sets the width of the 2005–2014 bounds. The "
  "sign priors mattered only for the conjunctive version 1, where they pruned incoherent conjunctions; there, "
  "univariable and mutually adjusted priors gave similar discrimination ({{v1_con_ext}} and {{mvpri_v1_ext}} in "
  "2005–2014), so we draw no conclusion about which kind of prior is better.")

h2("4.4. Transport across survey periods")
p("All models under-predicted in 2021–2023 by a similar factor, and a single intercept shift repaired it. NHANES "
  "documentation records several changes after the pandemic suspension that plausibly contributed: the depression "
  "screener moved from an interviewer-administered computer-assisted interview to audio computer-assisted "
  "self-interview [@nhanes_dpq_l], a mode change known to increase reporting of sensitive symptoms [@tourangeau2007]; "
  "oversampling by race, Hispanic origin and income was discontinued [@nhanes_l_overview]; and unweighted interview "
  "and examination response rates fell to 34.5% and 25.6% from 51.0% and 46.9% in 2017–March 2020 "
  "[@nhanes_rr_2021,@nhanes_rr_2017]. We cannot separate these from a real rise in prevalence [@brody2025,@daly2021], "
  "but the self-administration change alone is reason to expect a model trained on interviewer-administered data to "
  "under-predict. Any deployment would require local recalibration and monitoring regardless of model type or "
  "abstention rule [@vancalster2019].")

h2("4.5. Practical relevance")
p("With the full predictor set, which includes self-rated health, sleep and sedentary time asked in an interview, the "
  "model flagged about half of participants to find about five in six with elevated symptoms; the PHQ-2 flagged about "
  "one in ten. Even allowing for its incorporation bias, two symptom questions are far more efficient, and where they "
  "can be asked they should be. A setting without them is a population survey whose core omits symptom scales. The "
  "core questionnaire of the Behavioral Risk Factor Surveillance System [@brfss2018] is an example, but the mapping is "
  "partial: it asks general health on the same five-level scale, smoking, diabetes and other chronic conditions and "
  "demographics, but only one sleep-duration item and no sedentary time, and it also asks about days of poor mental "
  "health and a previous diagnosis of a depressive disorder, items too close to the outcome for a model of this kind. "
  "A model for that survey would have to be refitted on its items. Whether a prioritisation step saves effort in a "
  "real service, and whether its explanations help staff, has not been evaluated.")

h2("4.6. Limitations")
p("The fuzzy model was designed after version-1 results on the internal and later-period sets had been seen, and the "
  "protocol that fixed it was not registered; that it preceded the download of the 2005–2014 files rests on our "
  "records only. The abstention-rule comparisons, including every scenario that could have reversed the ordering, "
  "were specified after the original results were known. Both validation cohorts are periods of the same survey. The "
  "simulated scenarios are stylised: the shift was induced by resampling one test set, and the masking probabilities "
  "were chosen to be strong rather than estimated. Pattern submodels were not evaluated under many sporadic patterns. "
  "Imputation spread used one chained-equation model with 20 draws and a 90/10 rule; other imputation models or cut-offs "
  "could change coverage. All estimates are unweighted, and only the primary comparison has a design-based interval. "
  "The outcome is a questionnaire threshold, not a diagnostic interview [@levis2019], and self-rated health, the "
  "strongest predictor, overlaps in content with it. The acquisition rule was evaluated only under random masking with "
  "true values revealed, its support points come from the marginal development distribution of each input, and its "
  "item costs are illustrative. Conformal abstention, credal classifiers for non-additive models and learned "
  "acquisition policies [@shim2018] were discussed but not fitted. Subgroup analyses were descriptive and underpowered "
  "for several groups.")

h1("5. Conclusions")
p("With the base model held fixed, abstaining on the spread of multiple imputations or using pattern submodels "
  "decided more participants than abstaining on worst-case bounds over admissible input values, decided them more "
  "accurately, and gave higher net benefit at a threshold probability of 0.10, including for an input never collected "
  "and when the imputation model was wrong for the target population. Worst-case bounds missed fewer cases because "
  "they referred more people; this paid off only when missingness signalled risk and missed cases were weighted "
  "heavily. Their guarantee concerns the model's output, not the person's outcome. For an input missing by design, a "
  "model fitted without it was the simplest and best option. The fuzzy additive model built on the knowledge schema "
  "ranked people as well as linear logistic regression, and two symptom questions remain far more efficient than any "
  "non-symptom model where they can be asked.")

h1("Declarations")
p("*Ethics approval and consent:* NHANES protocols were approved by the NCHS Research Ethics Review Board and all "
  "participants provided informed consent. This secondary analysis of public de-identified data did not require "
  "further approval under the policy of Saigon International University.")
p("*Data availability:* NHANES data are publicly available from the National Center for Health Statistics. Components "
  "used: DEMO, DPQ, SLQ, BMX, HUQ, SMQ, PAQ, MCQ, DIQ and BPQ for 2005–2006 to 2015–2016, 2017–March 2020 and "
  "August 2021–August 2023.")
p("*Code availability:* The full pipeline (cohort construction, knowledge schema in JSON-LD, models, Algorithms 1–3, "
  "abstention rules, comparators, experiments, tests, figures, the analysis protocol with its addendum, and the "
  "statistics file that populates this manuscript) is publicly available at {{repo_url}} (release v4.0.0) and archived "
  "at Zenodo ({{zenodo_doi_v4}}). Re-run from the public NHANES files, the archived release reproduces every number "
  "reported here except computing times.")
p("*Funding:* This research did not receive any specific grant from funding agencies in the public, commercial, or "
  "not-for-profit sectors.")
p("*Declaration of competing interest:* The authors declare that they have no known competing financial interests or "
  "personal relationships that could have appeared to influence the work reported in this paper.")
p("*CRediT authorship contribution statement:* Truong Hai Bang: Conceptualization, Methodology, Software, Formal "
  "analysis, Visualization, Writing – original draft, Writing – review & editing, Supervision. Van An Phan Thi: "
  "Conceptualization, Data curation, Investigation, Validation, Writing – review & editing.")
p("*Declaration of generative AI and AI-assisted technologies in the manuscript preparation process:* During the "
  "preparation of this work the authors used Claude (Anthropic) to draft and edit manuscript text and to check "
  "bibliographic records; its use in writing the analysis code and tests is described in Section 2.13. After using this "
  "tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the "
  "published article.")

h1("References")
B.append({"t": "refs"})
pb()
h1("Appendix A. Additional tables")
p("Tables A1–A5 give the direction priors, subgroup results, the version-1 rule base, included and excluded adults, and "
  "the admissible ranges. Table A6 gives decision consequences for the regression base models, Table A7 the 2005–2014 "
  "results by cycle, Table A8 the paired AUROC differences for all comparators, variants and ablations, Table A9 the "
  "point-score AUROC and worst-case coverage under further MCAR rates and domain removal, and Table A10 worst-case "
  "coverage by cycle. The variants and ablations of the fuzzy model were: version 1 (K = 32 rules of any arity); main "
  "effects plus 16 pair rules; normalised zero-order TSK aggregation p = Σ_r w_r μ_r c_r / Σ_r w_r μ_r (K = 32); no "
  "direction constraints; crisp memberships at the same cut-points; quantile-based memberships with no constraints and "
  "median imputation ('fuzzy without schema'); median imputation instead of bounds; and box bounds, in which each "
  "missing membership ranges over [0, 1].")
for k in ("tableS1", "tableS2", "tableS3", "tableS4", "tableS5", "tableA6", "tableCycle", "tableAbl", "tableMiss", "tableMiss2", "tableNat"):
    tab(k)
h1("Appendix B. TRIPOD+AI checklist")
tab("tableTRIPOD")
