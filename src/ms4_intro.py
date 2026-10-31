# Title page, highlights, abstract and introduction of version 4.
from ms4_helpers import *  # noqa

title("Abstaining when inputs are missing: worst-case bounds, imputation spread and pattern submodels compared for "
      "prescreening elevated depressive symptoms in NHANES")
meta("Truong Hai Bang^{1,*}, Van An Phan Thi^{2}")
meta("^{1}Faculty of Engineering and Computer Science, Saigon International University, 8C Tong Huu Dinh Street, An Khanh "
     "Ward, Ho Chi Minh City, Vietnam. ORCID: https://orcid.org/0000-0001-5094-3128")
meta("^{2}VNU University of Education, Vietnam National University, Hanoi, G7 Building, No. 144 Xuan Thuy Street, Cau Giay "
     "District, Hanoi, Vietnam. Email: phanthivanantlls@gmail.com. ORCID: https://orcid.org/0009-0005-4135-5689")
meta("*Corresponding author: Truong Hai Bang, truonghaibang@siu.edu.vn")

h1("Highlights")
bullets([
    "Four abstention rules for missing inputs were compared with the base model held fixed.",
    "Pattern submodels decided every 2005–2014 participant with the highest net benefit.",
    "Worst-case bounds decided fewest participants, least accurately at equal coverage.",
    "Bounds missed fewer cases only by referring more people, even when imputation was wrong.",
    "PHQ-2 flagged {{phq2_int_flag}}% of participants (sensitivity {{phq2_int_sens}}); the model flagged {{full_int_flag}}% ({{full_int_Sensitivity}}).",
])

h1("Abstract")
p("*Background and objective:* Prediction models applied to incomplete records either impute the missing inputs or "
  "abstain. We compared rules for abstaining in a prescreening task: predicting a Patient Health Questionnaire-9 "
  "(PHQ-9) score ≥ 10 from non-symptom survey data.")
p("*Methods:* NHANES 2017–March 2020 provided development (n = {{n_dev}}) and internal test (n = {{n_int}}) data; "
  "August 2021–August 2023 (n = {{n_tmp}}) and 2005–2014 (n = {{n_ext}}, free-day sleep not collected) served as "
  "validation. Three base models were fitted: a fuzzy additive model built from a schema that declares each input's "
  "admissible values, sign-constrained linear and spline logistic regression. Four rules were applied to each: "
  "worst-case bounds (abstain when the range of predicted probabilities over all admissible values of the missing "
  "inputs contains the threshold), imputation spread over 20 multiple imputations, pattern submodels and complete "
  "cases. Missingness was natural, random, not at random, outcome-dependent, or structural with a shifted "
  "conditional distribution.")
p("*Results:* The pre-specified primary contrast, 2005–2014 AUROC of the fuzzy model minus logistic regression with "
  "missing indicators, was {{d_lr_ext}} ({{d_lr_ext_ci}}); it reflects the indicator for an input never collected, and "
  "regression without indicators matched the fuzzy model ({{lrni_ext_AUROC}}). With the fuzzy model fixed, worst-case "
  "bounds decided {{a4_ext_fuzzy_wc_cov}}% of 2005–2014 participants, imputation spread {{a4_ext_fuzzy_mi_cov}}% and pattern "
  "submodels all; at equal coverage, balanced accuracy was {{a4_ext_fuzzy_wc_atb}}, {{a4_ext_fuzzy_mi_atb}} and "
  "{{a4_ext_fuzzy_ps_atb}}. With self-rated health absent for everyone in a "
  "sample shifted away from the imputation model, imputation spread missed {{a4_sh1srh_fuzzy_mi_mis}} cases per 1,000 and "
  "worst-case bounds {{a4_sh1srh_fuzzy_wc_mis}}, but the bounds referred {{a4_sh1srh_fuzzy_wc_ref}} per 1,000 and still "
  "decided less accurately. Worst-case bounds had the highest net benefit only when missingness depended on health or "
  "on the outcome and a missed case was weighed as 19 unnecessary referrals. PHQ-2 ≥ 3 flagged {{phq2_int_flag}}% of participants with sensitivity "
  "{{phq2_int_sens}}, inflated because both items are part of the outcome.")
p("*Conclusions:* With the base model fixed, worst-case abstention decided fewer participants, and "
  "less accurately, than imputation spread or pattern submodels; it missed fewer cases only by referring more people. "
  "For an input missing by design, a model fitted without it was the better choice. No non-symptom model approached "
  "the PHQ-2.")
p("*Keywords:* Missing data; Abstention; Selective prediction; Pattern submodels; Multiple imputation; Depression "
  "screening; PHQ-9; Fuzzy rule-based systems; NHANES")

h1("1. Introduction")
p("A prediction model deployed on routine data meets incomplete records. The usual response is to impute the missing "
  "inputs and predict, which hides how much a decision depends on what was not observed [@vansmeden2020,@kompa2021]. "
  "The alternative is to abstain: to return no decision for some records and route them elsewhere, as in the reject "
  "option [@chow1970] and selective classification [@geifman2017]. Abstention needs a rule that says when a record is "
  "too incomplete to decide. Several such rules exist, and they rest on different assumptions about the missing "
  "values. Credal classifiers compute the prediction over all completions of a record and abstain when the completions "
  "disagree, assuming nothing about how values went missing [@zaffalon2002,@corani2008]. Rules based on multiple "
  "imputation abstain when predictions disagree across completed data sets, assuming that the conditional distribution "
  "of the missing values given the observed ones is known. Pattern submodels fit one model per missingness pattern and "
  "so never need the missing values [@fletchermercaldo2020]. Conformal prediction with missing values calibrates "
  "label sets within patterns [@zaffran2023], and work on imputation for prediction shows when impute-then-predict is "
  "consistent [@lemorvan2021]. To our knowledge, these rules have not been compared on the same prediction task with "
  "the base model held fixed, on real structured missingness as well as simulated missingness that violates their "
  "assumptions.")
p("We make that comparison on a well-understood task: flagging adults likely to score ≥ 10 on the PHQ-9, the most "
  "widely used threshold for elevated depressive symptoms [@kroenke2001,@levis2019], from non-symptom survey data. The "
  "task has been studied in the National Health and Nutrition Examination Survey (NHANES) with logistic nomograms "
  "[@zhang2022,@liu2024] and machine-learning classifiers explained with SHAP values [@vu2025,@salimi2026]; Zhang et "
  "al. developed a nomogram on NHANES 2007–2012 and assessed it in a separate validation cohort [@zhang2022]. Fuzzy "
  "expert systems for depression [@fazel2019,@ghorbani2023,@mendez2021,@saha2024] use symptom ratings or sensor data. "
  "A model of this kind is useful only where neither the PHQ-9 nor its two-item form, the PHQ-2, is available "
  "[@kroenke2003,@levis2020]; we use the task mainly as a test bed. NHANES suits a comparison of abstention rules "
  "because one input, sleep duration on free days, was not collected in 2005–2014, so a validation in those cycles "
  "has a structural missingness pattern in every record.")
p("Worst-case abstention needs the set of admissible values of each input, which a fitted model does not carry. We "
  "therefore build one base model, a fuzzy additive model, from a schema that declares each input's admissible set, "
  "concepts with externally sourced cut-points and direction priors. Its terms are bounded, which keeps worst-case "
  "bounds narrow, and it is a sign-constrained piecewise-linear generalised additive model [@hastie1986]; it has to be "
  "judged against the regression family and inherently interpretable alternatives such as GA²M "
  "[@rudin2019,@lou2013,@caruana2015,@nori2019]. Two logistic regression models with unbounded terms serve as further "
  "base models.")
p("The study asks three questions. (RQ1) With the base model held fixed, how do worst-case bounds, imputation spread, "
  "pattern submodels and complete-case abstention compare in coverage, accuracy of decided records and decision "
  "consequences, under natural, random, non-random, outcome-dependent and structural missingness, including a target "
  "population in which the imputation model is wrong? (RQ2) How do the base models compare in discrimination and "
  "calibration across three survey periods? (RQ3) Does a request rule guided by the bounds resolve abstentions with "
  "fewer inputs or at lower cost?")
p("The contributions, in order of evidential weight, are: (i) a comparison of four abstention rules across three base "
  "models and sixteen missingness scenarios on real survey data, with coverage split by decision, accuracy at matched "
  "coverage, risk–coverage curves and decision consequences per 1,000 participants; (ii) a validation of the base "
  "models in three survey periods against eight comparators, with the PHQ-2 as a reference point; (iii) algorithms for "
  "worst-case bounds of fuzzy additive rule bases, elementary for main effects (Algorithm 1) and exact for components "
  "with up to two continuous missing variables when rules contain conjunctions (Algorithm 2), together with a "
  "bound-guided acquisition rule (Algorithm 3); and (iv) the schema, released as a JSON-LD file that the code loads. "
  "We report the development history, including an abandoned first version, and make no claim of clinical utility.")
