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
    "Five abstention rules for missing inputs were compared with the base model fixed.",
    "Pattern submodels and point-score selective prediction did best in 2005–2014.",
    "Worst-case bounds decided fewest participants, least accurately at equal coverage.",
    "Worst-case bounds missed fewer cases only by referring more people.",
    "PHQ-2 flagged {{phq2_int_flag}}% (sensitivity {{phq2_int_sens}}); the model flagged {{full_int_flag}}% ({{full_int_Sensitivity}}).",
])

h1("Abstract")
p("*Background and objective:* Models applied to incomplete records must impute or abstain. We compared abstention "
  "rules for predicting a PHQ-9 score ≥ 10 from non-symptom survey data.")
p("*Methods:* NHANES 2017–March 2020 provided development (n = {{n_dev}}) and test (n = {{n_int}}) data; 2021–2023 "
  "(n = {{n_tmp}}) and 2005–2014 (n = {{n_ext}}; free-day sleep not collected) served as validation. Five rules were "
  "applied to three fixed base models (a fuzzy additive model built from a schema of admissible input values, and two "
  "logistic regressions): worst-case bounds, which abstain when the probability range over all admissible values of "
  "the missing inputs contains the threshold; imputation spread; pattern submodels; point-score selective "
  "prediction; and complete cases. Missingness was natural or simulated in 13 scenarios.")
p("*Results:* The pre-specified primary contrast (2005–2014 AUROC, fuzzy model minus logistic regression with missing "
  "indicators) was {{d_lr_ext}} ({{d_lr_ext_ci}}), an artefact of indicating an uncollected input. With the fuzzy model "
  "fixed, worst-case bounds decided {{a4_ext_fuzzy_wc_cov}}% of 2005–2014 participants, imputation spread "
  "{{a4_ext_fuzzy_mi_cov}}% and pattern submodels all; balanced accuracy at equal coverage was {{a4_ext_fuzzy_wc_atb}}, "
  "{{a4_ext_fuzzy_mi_atb}} and {{a4_ext_fuzzy_ps_atb}}, and net benefit at threshold 0.10 {{a4_ext_fuzzy_wc_nb10}}, "
  "{{a4_ext_fuzzy_mi_nb10}} and {{a4_ext_fuzzy_ps_nb10}}. Worst-case bounds guarantee only that a decision holds for "
  "every admissible value under the model, not that it is correct; they missed fewer cases by referring more people, "
  "and their net benefit was equivalent or higher only where little was missing, where missingness signalled risk or where models under-predicted. As "
  "a benchmark, PHQ-2 ≥ 3 flagged {{phq2_int_flag}}% of participants with sensitivity {{phq2_int_sens}} (inflated by "
  "incorporation); the fuzzy model flagged {{full_int_flag}}% with sensitivity {{full_int_Sensitivity}}.")
p("*Conclusions:* Where two symptom questions can be asked, the PHQ-2 should be used. Where they cannot, worst-case "
  "abstention was better only in narrow situations; imputation spread, pattern submodels or point-score selective "
  "prediction were usually better, and for an input missing by design a model fitted without it was a simple choice.")
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
  "assumptions. That comparison is the contribution of this study. We do not propose a new model for depression "
  "screening.")
p("We make that comparison on a well-understood task: flagging adults likely to score ≥ 10 on the PHQ-9, the most "
  "widely used threshold for elevated depressive symptoms [@kroenke2001,@levis2019], from non-symptom survey data. The "
  "task has been studied in the National Health and Nutrition Examination Survey (NHANES) with logistic nomograms "
  "[@zhang2022,@liu2024] and machine-learning classifiers explained with SHAP values [@vu2025,@salimi2026]; Zhang et "
  "al. developed a nomogram on NHANES 2007–2012 and assessed it in a separate validation cohort [@zhang2022]. Fuzzy "
  "expert systems for depression [@fazel2019,@ghorbani2023,@mendez2021,@saha2024] use symptom ratings or sensor data. "
  "A model of this kind cannot replace the PHQ-9 or its two-item form, the PHQ-2 "
  "[@kroenke2003,@levis2020]. It can only help where no symptom questions were asked, for example in secondary analyses "
  "of population surveys, cohort studies or registries collected without a symptom scale, to describe risk or to "
  "decide whom to follow up. We use the task mainly as a test bed. NHANES suits a comparison of abstention rules "
  "because one input, sleep duration on free days, was not collected in 2005–2014, so a validation in those cycles "
  "has a structural missingness pattern in every record.")
p("Worst-case abstention needs the set of admissible values of each input, which a fitted model does not carry. We "
  "therefore include, as one of three base models, a fuzzy additive model built from a schema that declares each "
  "input's admissible set, concepts with externally sourced cut-points and direction priors. We chose it for the "
  "comparison, not for its accuracy. Its terms are bounded, so worst-case bounds stay finite and narrow; this gives "
  "the worst-case rule its best chance, whereas a model with linear terms gives bounds that widen with every admissible "
  "range. Its terms are also named concepts with declared sources, so each bound can be traced to a concept and a "
  "range. Mathematically it is a sign-constrained piecewise-linear generalised additive model [@hastie1986], and we "
  "expected it to rank people about as well as regression, not better [@rudin2019,@lou2013,@caruana2015,@nori2019]. "
  "Two logistic regression models with unbounded terms serve as the other base models.")
p("The study asks three questions. (RQ1) With the base model held fixed, how do worst-case bounds, imputation spread, "
  "pattern submodels, point-score selective prediction and complete-case abstention compare in coverage, accuracy of decided records and decision "
  "consequences, under natural, random, non-random, outcome-dependent and structural missingness, including a target "
  "population in which the imputation model is wrong? (RQ2) How do the base models compare in discrimination and "
  "calibration across three survey periods? (RQ3) Does a request rule guided by the bounds resolve abstentions with "
  "fewer inputs or at lower cost?")
p("The main contribution is (i) a comparison of five abstention rules across three base "
  "models and sixteen missingness scenarios (three cohorts as observed and 13 simulated; Table @SUM@) on real survey data, with coverage split by decision, accuracy at matched "
  "coverage, risk–coverage curves and decision consequences per 1,000 participants. Supporting contributions are (ii) a validation of the base "
  "models in three survey periods against eight comparators, with the PHQ-2 as a reference point; (iii) algorithms for "
  "worst-case bounds of fuzzy additive rule bases, elementary for main effects (Algorithm 1) and exact for components "
  "with up to two continuous missing variables when rules contain conjunctions (Algorithm 2), together with a "
  "bound-guided acquisition rule (Algorithm 3); and (iv) the schema, released as a JSON-LD file that the code loads. "
  "We report the development history, including an abandoned first version, and make no claim of clinical utility.")
