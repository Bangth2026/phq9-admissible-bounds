# Main text of version 7, formatted for Computer Methods and Programs in Biomedicine (original research,
# main text within about 3,500 words; extended methods and results in the Supplementary Material).
from ms4_helpers import *  # noqa

title("Abstaining when inputs are missing: worst-case bounds, imputation spread and pattern submodels compared for "
      "prescreening elevated depressive symptoms in NHANES")
meta("Truong Hai Bang^{a,*}, Van An Phan Thi^{b}")
meta("^{a}Faculty of Engineering and Computer Science, Saigon International University, 8C Tong Huu Dinh Street, An Khanh "
     "Ward, Ho Chi Minh City, Vietnam")
meta("^{b}VNU University of Education, Vietnam National University, Hanoi, G7 Building, No. 144 Xuan Thuy Street, Cau Giay "
     "District, Hanoi, Vietnam")
meta("*Corresponding author: Truong Hai Bang, Faculty of Engineering and Computer Science, Saigon International University, "
     "8C Tong Huu Dinh Street, An Khanh Ward, Ho Chi Minh City, Vietnam. Email: truonghaibang@siu.edu.vn. Telephone: {{phone}}")
meta("ORCID: Truong Hai Bang, https://orcid.org/0000-0001-5094-3128; Van An Phan Thi, https://orcid.org/0009-0005-4135-5689 "
     "(phanthivanantlls@gmail.com)")
meta("Word count (main text, excluding abstract, tables, boxes, figure captions, declarations and references): {{wordcount}}")
pb()

h1("Abstract")
p("*Background and objective:* Models applied to incomplete records must impute or abstain. We compared abstention "
  "rules for predicting a Patient Health Questionnaire-9 (PHQ-9) score ≥ 10 from non-symptom survey data.")
p("*Methods:* NHANES 2017–March 2020 provided development (n = {{n_dev}}) and test (n = {{n_int}}) data; 2021–2023 "
  "(n = {{n_tmp}}) and 2005–2014 (n = {{n_ext}}; free-day sleep not collected) served as validation. Five rules were "
  "applied to three fixed base models (a fuzzy additive model built from a schema of admissible input values, and two "
  "logistic regressions): worst-case bounds, which abstain when the probability range over all admissible values of "
  "the missing inputs contains the threshold; imputation spread; pattern submodels; point-score selective "
  "prediction; and complete cases. Missingness was natural in three cohorts and simulated in 13 scenarios. The rule "
  "comparison was specified after initial results were known and is exploratory.")
p("*Results:* The pre-specified primary contrast (2005–2014 AUROC, fuzzy model minus logistic regression with missing "
  "indicators) was {{d_lr_ext}} ({{d_lr_ext_ci}}), an artefact of indicating an uncollected input. With the fuzzy model "
  "fixed, worst-case bounds decided {{a4_ext_fuzzy_wc_cov}}% of 2005–2014 participants, imputation spread "
  "{{a4_ext_fuzzy_mi_cov}}% and pattern submodels all; balanced accuracy at equal coverage was {{a4_ext_fuzzy_wc_atb}}, "
  "{{a4_ext_fuzzy_mi_atb}} and {{a4_ext_fuzzy_ps_atb}}, and net benefit at threshold 0.10 {{a4_ext_fuzzy_wc_nb10}}, "
  "{{a4_ext_fuzzy_mi_nb10}} and {{a4_ext_fuzzy_ps_nb10}}. Worst-case bounds guarantee only that a decision holds for "
  "every admissible value under the model, not that it is correct; they missed fewer cases by referring more people, "
  "and their net benefit at 0.10 was never higher than that of the best other rule. As a benchmark, PHQ-2 ≥ 3 flagged "
  "{{phq2_int_flag}}% of participants with sensitivity {{phq2_int_sens}} (inflated by incorporation); the fuzzy model "
  "flagged {{full_int_flag}}% with sensitivity {{full_int_Sensitivity}}.")
p("*Conclusions:* Where two symptom questions can be asked, the PHQ-2 should be used. Where they cannot, worst-case "
  "abstention was better only in narrow situations; imputation spread, pattern submodels or point-score selective "
  "prediction were usually better, and for an input missing by design a model fitted without it was a simple choice. "
  "These exploratory comparisons need confirmation in other data.")
p("*Keywords:* Missing data; Abstention; Selective prediction; Multiple imputation; Depression screening; NHANES")

h1("1. Introduction")
p("A prediction model deployed on routine data meets incomplete records. The usual response is to impute the missing "
  "inputs and predict, which hides how much a decision depends on what was not observed [@vansmeden2020,@kompa2021]. "
  "The alternative is to abstain: to return no decision for some records, as in the reject option [@chow1970] and "
  "selective classification [@geifman2017]. Several abstention rules exist, and they assume different things about "
  "the missing values. Credal classifiers compute the prediction over all completions of a record and abstain when the "
  "completions disagree, assuming nothing about why values are missing [@zaffalon2002,@corani2008]. Rules based on "
  "multiple imputation abstain when completed data sets disagree, assuming that the conditional distribution of the "
  "missing values is known. Pattern submodels fit one model per missingness pattern and never need the missing values "
  "[@fletchermercaldo2020]. Conformal prediction with missing values calibrates label sets within patterns "
  "[@zaffran2023], and work on imputation for prediction shows when impute-then-predict is consistent [@lemorvan2021]. "
  "To our knowledge these rules have not been compared on one task with the base model held fixed, on real structural "
  "missingness and on simulated missingness built to violate their assumptions. That comparison is the contribution "
  "of this study; we do not propose a new model for depression screening.")
p("The task is to flag adults likely to score ≥ 10 on the PHQ-9, the most widely used threshold for elevated "
  "depressive symptoms [@kroenke2001,@levis2019], from non-symptom survey data, as studied before in the National "
  "Health and Nutrition Examination Survey (NHANES) [@zhang2022,@liu2024,@vu2025]. Such a model cannot replace the "
  "PHQ-9 or its two-item form, the PHQ-2 [@kroenke2003,@levis2020]. It can only help where no symptom questions were "
  "asked, for example in surveys, cohorts or registries collected without a symptom scale, to describe risk or decide "
  "whom to follow up. NHANES suits the comparison because sleep duration on free days was not collected in 2005–2014, "
  "so every record in those cycles has a structural missingness pattern.")
p("Worst-case abstention needs each input's set of admissible values, which a fitted model does not carry. One of our "
  "three base models is therefore a fuzzy additive model built from a schema that declares admissible sets and named "
  "concepts with sourced cut-points. We chose it because its bounded terms keep worst-case bounds finite and narrow, "
  "giving that rule its best chance, not because we expected it to predict better than regression "
  "[@hastie1986,@rudin2019,@lou2013]. We asked how five abstention rules compare, with the base model fixed, in "
  "coverage, accuracy among decided participants and decision consequences, across natural and simulated "
  "missingness.")

h1("2. Methods")
h2("2.1. Study design, data and outcome")
p("This retrospective study is reported following TRIPOD+AI [@collins2024] (checklist in Supplementary Section S5). "
  "NHANES 2017–March 2020 [@akinbami2022] was split 70/30, stratified by outcome, into development "
  "(n = {{n_dev}}, {{rl_events_dev}} events) and internal test (n = {{n_int}}) sets. August 2021–August 2023 "
  "(n = {{n_tmp}}) and the five cycles 2005–2006 to 2013–2014 (n = {{n_ext}}) served as later- and earlier-period "
  "validation of frozen models (Fig. 1). Participants were adults with valid responses to all nine PHQ-9 items "
  "[@nhanes_dpq]; the outcome was a total score ≥ 10. Fourteen predictors were plausibly available without a symptom "
  "questionnaire: age, sex, income-to-poverty ratio, education, marital status, body mass index, workday and free-day "
  "sleep, self-rated health, smoking, diabetes, hypertension, sedentary time and a count of chronic conditions. "
  "Free-day sleep was not collected in 2005–2014. Cohort construction, harmonisation and a sample-size check "
  "[@riley2020] are in Supplementary Section S1. All estimates are unweighted and refer to study participants.")
fig("fig_flow.png", "Fig. 1. Participant flow. The 2015–2016 cycle was used only to estimate direction priors. Models "
    "and thresholds were frozen before evaluation on the later- and earlier-period cohorts.", 6.3)

h2("2.2. Base models")
p("Three base models were fitted once on development data. (i) A fuzzy additive model: a knowledge schema, released as "
  "a JSON-LD file [@jsonld2020], declares each input's admissible set and {{n_concepts}} concepts with trapezoidal "
  "memberships at externally sourced cut-points [@who2000,@watson2015]; the model sums sign-constrained log-odds "
  "contributions of {{n_rules}} single-concept rules, so each term is bounded and constant beyond its outermost "
  "cut-points. (ii) Sign-constrained linear logistic regression. (iii) Logistic regression with restricted cubic "
  "splines [@harrell2015]. Each model's referral threshold was the largest raw score reaching an out-of-fold "
  "sensitivity of at least 0.80 in development data ({{thr_full}} for the fuzzy model). Discrimination was also "
  "compared with logistic regression with and without missing indicators, gradient-boosted trees and GA²M "
  "[@nori2019]. Details are in Supplementary Section S1.")

h2("2.3. Abstention rules")
p("A rule *decides* a person when it returns *refer* (recommend completing the PHQ-9) or *do not refer*; otherwise it "
  "*abstains*. *Coverage* is the proportion decided. Under *refer-on-abstain*, every abstention is referred, which "
  "turns any rule into a referral decision for everyone. Fig. 2 summarises the design: each base model is held fixed "
  "while five rules are applied to the same records.")
fig("fig_design.png", "Fig. 2. Design of the comparison. Each base model is fitted once and held fixed; each rule turns "
    "its output into refer, do not refer or abstain for the same records; rules are compared on coverage, accuracy at "
    "matched coverage, risk–coverage curves and decision consequences.", 6.8)
p("*Worst-case bounds.* With O and M the observed and missing inputs and 𝒳_M the product of their admissible sets, the "
  "rule computes p_L = min σ(η(x_O, x_M)) and p_U = max σ(η(x_O, x_M)) over x_M ∈ 𝒳_M, refers if p_L ≥ t, does not "
  "refer if p_U < t, and abstains otherwise. A decision then holds for every admissible value of the missing inputs, "
  "but only under the model: if the model is miscalibrated, such a decision can still be wrong.")
p("*Imputation spread.* Five chained-equation models (Bayesian ridge, posterior sampling, 10 iterations), fitted once on "
  "development predictors without the outcome, produced 20 completed copies of each record; the base model scored "
  "each copy, and the rule refers if at least 90% of the 20 probabilities exceed t, does not refer if at most 10% do, "
  "and abstains otherwise.")
p("*Pattern submodels.* For each missingness pattern, the base model was refitted without the missing inputs, with its "
  "own threshold, and applied to records with that pattern; every record is decided. They were not fitted under "
  "random masking, where patterns are numerous.")
p("*Point-score selective prediction.* The rule abstains when the single-imputation point score lies within a band "
  "around the threshold, |logit p − logit t| < δ, with δ set so that 90% of development participants are decided. It "
  "ignores which inputs are missing. Split-conformal prediction for a binary outcome reduces to such a band "
  "[@angelopoulos2023], so we did not fit it separately.")
p("*Complete cases.* Only records with every input observed are decided.")

h2("2.4. Computing worst-case bounds")
p("Because σ is monotone it suffices to bound η. For the fuzzy main-effects model, η is a sum of univariate "
  "piecewise-linear terms with breakpoints at the trapezoid corners, so its extremes are sums of each missing input's "
  "extremes, attained at a range end or a corner (Algorithm 1, Box 1). Rule bases with two-concept conjunctions, such "
  "as the conjunctive first version of the system kept as a comparator, couple inputs; Algorithm 2 (Box 2) enumerates "
  "a finite candidate set per connected component of missing inputs, which is exact by two statements. *Statement 1:* "
  "with one continuous missing input x, each term β_r min(μ_k(x), c_r) is piecewise linear with breakpoints at the "
  "trapezoid corners and where μ_k(x) = c_r, so the extremes lie on that finite set. *Statement 2:* with two "
  "continuous missing inputs, the switching lines μ_k(x) = μ_m(z) cut each rectangle of the breakpoint grid into "
  "convex polygons on which η is affine, so the extremes lie at polygon vertices. Components with more than two "
  "continuous missing inputs use an outer box bound. Proofs, costs and checks against brute force are in "
  "Supplementary Section S1.7. For linear regression the bounds lie at range ends; for spline regression they were "
  "found on a 2,001-point grid per input.")
tab("tableAlg")
tab("tableAlgGen")

h2("2.5. Missingness scenarios")
p("The three cohorts were analysed as observed. Thirteen scenarios were simulated on the internal test set "
  "(10 repetitions each), chosen to violate one rule's assumption at a time rather than to reproduce a particular "
  "survey: 30% missing completely at random (MCAR); income or self-rated health missing not at random (MNAR) with "
  "probabilities 0.6/0.1 or 0.9/0.05 in low-income or fair-or-poor-health participants versus others, or both inputs; "
  "outcome-dependent masking of both (0.6 in cases, 0.1 otherwise); and self-rated health or free-day sleep absent "
  "for everyone in target samples resampled with probability ∝ exp(γ s z), where z is the standardised residual of the "
  "input given the other inputs (γ = 0, 0.5, 1; s chosen so that the imputation model imputes healthier values and "
  "longer sleep than the truth). Real missingness is likely milder and mixed, so the scenarios bracket rather than "
  "represent practice. Supplementary Section S1.8 gives details, including a repeat with ordinal residuals [@li2012].")

h2("2.6. Outcomes and statistical analysis")
p("Rules were compared with the base model fixed on (1) coverage, split into decisions to refer and not to refer; (2) "
  "sensitivity, specificity and balanced accuracy among the participants each rule is most confident about, up to the "
  "coverage of worst-case bounds; (3) the area under the risk–coverage curve (AURC; balanced error against coverage "
  "0.05–1; lower is better); and (4) under refer-on-abstain, participants referred, cases missed and questionnaires "
  "avoided per 1,000, and net benefit [@vickers2006] at threshold probabilities 0.05 and 0.10. Intervals are 95% "
  "percentile bootstrap intervals: paired resamples of participants for the cohorts, and resamples within each of "
  "the 10 repetitions, pooled, for simulated scenarios. A difference whose interval includes zero is called equivalent. "
  "The pre-specified primary analysis was the 2005–2014 AUROC of the fuzzy model against logistic regression with "
  "missing indicators, with a design-based bootstrap over strata and primary sampling units as a check. Other "
  "pre-specified analyses are secondary. The comparison of abstention rules was specified after the original results "
  "were known and is exploratory; tables mark exploratory analyses with ‡.")

h2("2.7. Software, code and use of generative AI")
p("Analyses used Python 3.13, scikit-learn 1.9.1, SciPy 1.18.1, statsmodels and interpret 0.7.8. The analysis code, "
  "including three verification tests, was drafted with a generative AI tool (Claude, Anthropic) under the authors' "
  "specification; the tests check the bounds against dense brute force, recompute probabilities from serialised "
  "traces, and reproduce the cohort flow from public files. The authors reviewed the code, outputs and text. The "
  "analyses were re-run from the archived release in a clean directory ({{rerun_statement}}).")

h1("3. Results")
h2("3.1. Cohorts and base models")
p("Prevalence of PHQ-9 ≥ 10 was {{prev_ext}}% in 2005–2014, {{flow_dev_prevalence_unweighted}}% in 2017–2020 and "
  "{{prev_tmp}}% in 2021–2023 (Table 1). Excluding the uncollected free-day sleep, {{amx_ext}}% of 2005–2014 "
  "participants had at least one missing predictor. The primary contrast was {{d_lr_ext}} ({{d_lr_ext_ci}}; "
  "design-based {{db_lr_ci}}), but it reflects the indicator for an input never collected: logistic regression without "
  "indicators reached {{lrni_ext_AUROC}}, the same as the fuzzy model ({{full_ext_AUROC}}). Spline regression ranked "
  "best ({{rcs_ext_AUROC}}). The base models therefore discriminated about equally well: at 80% sensitivity in "
  "2005–2014 the fuzzy model flagged {{op80_ext_full_flag}} participants per 1,000 and spline regression "
  "{{op80_ext_rcs_flag}}. All models under-predicted in 2021–2023 (O/E {{full_tmp_OE}} for the fuzzy model). Full "
  "results for nine models are in Supplementary Section S2.")
tab("table1")

h2("3.2. Abstention rules in the cohorts as observed")
p("Where few inputs were missing, the rules differed little (Table 2). In 2005–2014, with the fuzzy model, worst-case "
  "bounds decided {{a4_ext_fuzzy_wc_cov}}% of participants ({{a4_ext_fuzzy_wc_covr}}% referred, "
  "{{a4_ext_fuzzy_wc_covn}}% not referred), imputation spread {{a4_ext_fuzzy_mi_cov}}%, point-score selective "
  "prediction {{a4_ext_fuzzy_sel_cov}}% and pattern submodels everyone. The asymmetry of worst-case decisions follows "
  "from one coefficient: with free-day sleep unknown, the ShortSleepFreeDay term (+{{we_beta_short}} log-odds, from a "
  "prior that was never estimated) can raise the upper bound but not the lower one. At the coverage of worst-case "
  "bounds, balanced accuracy was {{a4_ext_fuzzy_wc_atb}} for worst-case bounds, {{a4_ext_fuzzy_mi_atb}} for imputation "
  "spread, {{a4_ext_fuzzy_ps_atb}} for pattern submodels and {{a4_ext_fuzzy_sel_atb}} for point-score selective "
  "prediction (Fig. 3). Ranking by the distance of the point score from the threshold gave the lowest AURC "
  "({{a4_ext_fuzzy_sel_aurc}} against {{a4_ext_fuzzy_wc_aurc}} for worst-case bounds). The same ordering held with "
  "the regression base models.")
tab("tableAbst5")
fig("fig_riskcov4.png", "Fig. 3. Risk–coverage curves with the fuzzy additive model as the fixed base model: balanced "
    "accuracy among decided participants against the percentage decided, participants ranked by each rule's "
    "confidence. Open circles mark each rule's own operating point. Simulated panels show the first of 10 "
    "repetitions; pattern submodels were not fitted under MCAR.", 6.8)

h2("3.3. Simulated missingness")
p("Table 3 summarises all 16 scenarios. Three results held wherever much was missing. Worst-case bounds decided the "
  "fewest participants; their net benefit at 0.10 was never above that of the best other rule (lower in "
  "{{sum_n_wc_nb_lower}} scenarios, equivalent in {{sum_n_wc_nb_eq}}); and when an input was absent for everyone, "
  "pattern submodels or point-score selective prediction did best. Under 30% MCAR, worst-case bounds decided "
  "{{a4_mc30_fuzzy_wc_cov}}% and imputation spread {{a4_mc30_fuzzy_mi_cov}}%, with balanced accuracy at matched "
  "coverage {{a4_mc30_fuzzy_wc_atb}} against {{a4_mc30_fuzzy_mi_atb}}. Other results were confined to particular "
  "mechanisms. Worst-case bounds had the lowest AURC in {{sum_n_wc_aurc_best}} scenarios, all with non-random or "
  "outcome-dependent masking or in 2021–2023, by at most {{sum_wc_lead_max}}. When self-rated health was absent and "
  "the imputation model was made wrong (γ = 1), prevalence rose from {{prv_sh0srh}}% to {{prv_sh1srh}}%, and the "
  "sensitivity of imputation spread fell from {{a4_sh0srh_fuzzy_mi_sensr}}% to {{a4_sh1srh_fuzzy_mi_sensr}}%; worst-case "
  "bounds kept {{a4_sh1srh_fuzzy_wc_sensr}}% by deciding only {{a4_sh1srh_fuzzy_wc_cov}}% of participants, and the "
  "records they decided were classified less accurately ({{a4_sh1srh_fuzzy_wc_atb}} against {{a4_sh1srh_fuzzy_mi_atb}}). "
  "{{dsel_sentence}} Results for every base model are in Supplementary Table S16.")
tab("tableSum")

h2("3.4. Decision consequences")
p("Under refer-on-abstain in 2005–2014, per 1,000 participants, worst-case bounds referred {{a4_ext_fuzzy_wc_ref}}, "
  "missed {{a4_ext_fuzzy_wc_mis}} cases and avoided {{a4_ext_fuzzy_wc_avd}} questionnaires; pattern submodels "
  "referred {{a4_ext_fuzzy_ps_ref}}, missed {{a4_ext_fuzzy_ps_mis}} and avoided {{a4_ext_fuzzy_ps_avd}} (Table 4). "
  "Worst-case bounds missed fewer cases than imputation spread and pattern submodels in every scenario, because they "
  "referred more people. At a threshold probability of 0.05, which weighs a missed case as 19 unnecessary referrals, "
  "worst-case bounds exceeded imputation spread and pattern submodels in 2021–2023 ({{dnb_tmp_fuzzy_mi_05}} for "
  "imputation spread minus worst-case bounds), where every model under-predicted, exceeded pattern submodels and "
  "point-score selective prediction under outcome-dependent masking, and exceeded pattern submodels with self-rated "
  "health absent and shifted (γ = 1). Complete cases, which refer everyone when an input is absent, are not counted in "
  "these comparisons.")
tab("tableCons")

h2("3.5. PHQ-2 benchmark")
p("PHQ-2 ≥ 3 had sensitivity {{phq2_int_sens}} and specificity {{phq2_int_spec}} internally and flagged "
  "{{phq2_int_flag}}% of participants, against {{full_int_Sensitivity}}, {{full_int_Specificity}} and "
  "{{full_int_flag}}% for the fuzzy model. Both PHQ-2 items are part of the outcome, so its figures are inflated; "
  "against diagnostic interviews its pooled sensitivity was 0.72 at specificity 0.85 [@levis2020], still far more "
  "efficient than any non-symptom model here (Supplementary Section S2.9).")

h1("4. Discussion")
h2("4.1. Principal findings")
p("With the base model held fixed, the choice of abstention rule mattered more than the choice among base models, "
  "which discriminated about equally well. Imputation spread, pattern submodels and point-score selective prediction "
  "decided more participants than worst-case bounds and classified them more accurately. For an input never collected, "
  "a model fitted without it decided everyone and gave the highest net benefit at 0.10, while ranking by the point "
  "score gave the best risk–coverage curve. Worst-case bounds missed fewer cases only by referring more people.")

h2("4.2. When are worst-case bounds worth their cost?")
p("We had proposed that worst-case bounds suit an input absent by design whose distribution in the target population "
  "is unknown. In 2005–2014 that is the situation, and the bounds lost: imputation drew free-day sleep from its "
  "2017–2020 distribution, an assumption that cannot be checked, and still decided more participants more "
  "accurately. When the imputation model was made wrong on purpose, imputation lost some sensitivity but the bounds "
  "did not become more accurate. The bounds came closest to the other rules where missingness signalled risk, "
  "because their habit of referring incomplete records was then right, and in 2021–2023, where referring more people "
  "partly offset under-prediction; a recalibrated model with a lower threshold might do the same. Three features "
  "explain this. The guarantee concerns the model's output and frozen threshold, not the outcome. The width of the "
  "bounds is set by single coefficients and admissible ranges, not by how plausible an extreme completion is for "
  "the person. And distance of the bounds from the threshold is a poor confidence measure. Refer-on-abstain also has "
  "a cost: in 2005–2014 worst-case bounds referred {{a4_ext_fuzzy_wc_ref}} of every 1,000 participants to miss "
  "{{a4_ext_fuzzy_wc_mis}} cases, against {{a4_ext_fuzzy_ps_ref}} referrals and {{a4_ext_fuzzy_ps_mis}} missed cases "
  "for pattern submodels. In practice, an input missing by design calls for a model fitted without it; sporadic "
  "missingness for imputation spread or a band around the point score; and worst-case bounds only where missingness "
  "is likely to signal risk and a missed case weighs much more than an unnecessary referral.")

h2("4.3. Limitations and generalisability")
p("The rule comparison and every simulated scenario except MCAR were specified after the original results were known, "
  "so the findings are exploratory. The fuzzy model was designed after a first version had been evaluated, and its "
  "protocol was not registered. Both validation cohorts come from one survey; NHANES is a United States household "
  "survey with standardised interviews, and the predictors, their missingness, the prevalence of symptoms and model "
  "calibration will differ in health records, other surveys and other countries. The simulated mechanisms are "
  "stylised. Imputation used one model and a 90/10 rule. The outcome is a questionnaire threshold, not a diagnosis "
  "[@levis2019], and self-rated health overlaps in content with it. Estimates are unweighted. Further limitations are "
  "discussed in Supplementary Section S3.")

h1("5. Conclusions")
p("Where two symptom questions can be asked, the PHQ-2 is far more efficient than any non-symptom model studied here. "
  "Where they cannot and inputs are missing, the abstention rule matters more than the base model. In these exploratory "
  "comparisons, worst-case bounds over admissible values decided fewer participants, less accurately, than imputation "
  "spread, pattern submodels or point-score selective prediction; their net benefit at 0.10 was never higher, and at "
  "0.05 it was higher than some rules only where all models under-predicted or where missingness was related to the "
  "outcome or to a shifted input. Their guarantee "
  "concerns the model's output, not the person's outcome. For an input missing by design, a model fitted without it "
  "was a simple and good choice.")

h1("Acknowledgements")
p("None.")
h1("Declarations")
p("*Ethics approval and consent:* NHANES protocols were approved by the NCHS Research Ethics Review Board and all "
  "participants provided informed consent. This secondary analysis of public de-identified data did not require "
  "further approval under the policy of Saigon International University.")
p("*Data availability:* NHANES data are publicly available from the National Center for Health Statistics "
  "(components DEMO, DPQ, SLQ, BMX, HUQ, SMQ, PAQ, MCQ, DIQ and BPQ, 2005–2006 to August 2021–August 2023).")
p("*Code availability:* Code, the knowledge schema, the analysis protocol and the statistics file that populates this "
  "manuscript are available at {{repo_url}} (release v7.0.0) and archived at Zenodo ({{zenodo_doi_v7}}).")
p("*Funding:* This research did not receive any specific grant from funding agencies in the public, commercial, or "
  "not-for-profit sectors.")
p("*Declaration of competing interest:* The authors declare that they have no known competing financial interests or "
  "personal relationships that could have appeared to influence the work reported in this paper.")
p("*CRediT authorship contribution statement:* Truong Hai Bang: Conceptualization, Methodology, Software, Formal "
  "analysis, Visualization, Writing – original draft, Writing – review & editing, Supervision. Van An Phan Thi: "
  "Conceptualization, Data curation, Investigation, Validation, Writing – review & editing.")
h1("Declaration of generative AI and AI-assisted technologies in the manuscript preparation process")
p("During the preparation of this work the authors used Claude (Anthropic) to draft the analysis code and the "
  "verification tests under the authors' specification (Section 2.7), to help structure the manuscript and to edit its "
  "language. After using this tool, the authors reviewed and edited the content as needed and take full "
  "responsibility for the content of the published article.")
h1("Appendix A. Supplementary material")
p("Supplementary material to this article (extended methods and results, Supplementary Tables S1–S17, Supplementary "
  "Figs. S1–S6 and the TRIPOD+AI checklist) is provided as a separate file.")
h1("References")
B.append({"t": "refs"})
