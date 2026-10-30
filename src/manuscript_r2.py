"""Manuscript. Every number is injected from results/paper_stats_r2.json via {{key}};
a missing key raises an error."""
import json, re, sys
sys.path.insert(0, "/home/claude/src")
import pandas as pd
from refs import REFS

R = "/home/claude/results/"; RV = R + "revision/"
S = json.load(open(R + "paper_stats_r2.json"))
FIG = R


def V(k):
    if k not in S:
        raise KeyError(f"missing statistic: {k}")
    return S[k]


B = []
def title(t): B.append({"t": "title", "x": t})
def h1(t): B.append({"t": "h1", "x": t})
def h2(t): B.append({"t": "h2", "x": t})
def p(t): B.append({"t": "p", "x": t})
def eq(t, n): B.append({"t": "eq", "x": t, "n": n})
def bullets(xs): B.append({"t": "bullets", "x": xs})
def fig(path, cap, w=6.3): B.append({"t": "fig", "path": FIG + path, "cap": cap, "w": w})
def pb(): B.append({"t": "pb"})
def meta(t): B.append({"t": "meta", "x": t})
def tab(key): B.append({"t": key})

# ============================================================================================================
title("Exact admissible-value bounds for missing inputs in an interpretable fuzzy additive model to prescreen elevated "
      "depressive symptoms: development and validation in NHANES")
meta("Truong Hai Bang^{1,*}, Van An Phan Thi^{2}")
meta("^{1}Faculty of Engineering and Computer Science, Saigon International University, Ho Chi Minh City, Vietnam")
meta("^{2}Nguyen Tat Thanh University, Ho Chi Minh City, Vietnam")
meta("*Corresponding author: truonghaibang@siu.edu.vn · ORCID 0000-0001-5094-3128 (T.H. Bang); ORCID 0009-0005-4135-5689 (V.A. Phan Thi)")

h1("Highlights")
bullets([
    "Exact bounds over admissible values replace imputation for missing inputs.",
    "Bounds were tested on real missingness in 26,057 external records.",
    "Ranking matched linear regression; spline regression ranked slightly better.",
    "At equal workload, bounds found no more cases than imputation did.",
    "PHQ-2 flagged {{phq2_int_flag}}% of adults; the prescreening model flagged {{full_int_flag}}%.",
])

h1("Abstract")
p("*Background and objective:* Models that flag adults likely to score ≥ 10 on the Patient Health Questionnaire-9 "
  "(PHQ-9) from non-symptom information are judged mostly on discrimination, while incomplete records are left to "
  "imputation. We developed an interpretable fuzzy additive model whose admissible input values are declared in a "
  "machine-readable knowledge schema, and evaluated what exact bounds over those values add when inputs are missing.")
p("*Methods:* NHANES 2017–March 2020 provided development (n = {{n_dev}}) and internal test (n = {{n_int}}) data; "
  "August 2021–August 2023 (n = {{n_tmp}}) and 2005–2014 (n = {{n_ext}}) served for temporal and external validation "
  "of frozen models. Fourteen non-symptom variables were mapped to {{n_concepts}} concepts. "
  "Algorithm 1 computes exact probability bounds over the admissible values of missing inputs; the system abstains "
  "when the bounds straddle the threshold. Algorithm 2 requests the missing input most likely to resolve an "
  "abstention. Comparators were logistic regression (linear, spline, sign-constrained, with three missing-data "
  "strategies), GA²M and gradient-boosted trees. Bounds were evaluated on natural missingness, including free-day "
  "sleep absent externally, and on random and non-random masking.")
p("*Results:* External AUROC was {{full_ext_AUROC}} ({{full_ext_AUROC_ci}}). It did not differ from logistic "
  "regression without missing indicators ({{lrni_ext_AUROC}}) and was {{d_rcs_ext_abs}} below spline regression "
  "({{rcs_ext_AUROC}}); the protocol's primary comparison with indicator-based regression ({{lr_ext_AUROC}}) reflects "
  "how that comparator handled the absent sleep variable. Externally, admissible-value bounds decided "
  "{{nat_ext_full_Coverage}}% of participants, against {{nat_ext_slr_Coverage}}% for exact bounds of linear "
  "regression and none under complete-case abstention. At the same proportion referred, sensitivity was the same as "
  "for imputation-based models ({{natm_ext_full_Sens}}% versus {{natm_ext_lrni_Sens}}%). Algorithm 2 needed "
  "{{acq2_m30_d_fixed}} fewer inputs per abstention than a fixed order ({{acq2_m30_d_fixed_ci}}). PHQ-2 ≥ 3 flagged "
  "{{phq2_int_flag}}% of adults with sensitivity {{phq2_int_sens}}; the model flagged {{full_int_flag}}% with "
  "sensitivity {{full_int_Sensitivity}}.")
p("*Conclusions:* The knowledge schema did not improve ranking. Exact bounds identified which decisions hold for "
  "every admissible value of the missing inputs, decided more records than linear bounds when a variable was "
  "structurally absent, and found no more cases at equal workload. The model may support record-based "
  "prioritisation where no symptom question can be asked; it is not a substitute for the PHQ-2.")
p("*Keywords:* Depression screening; PHQ-9; Fuzzy rule-based systems; Missing data; Abstention; Interpretable machine "
  "learning; External validation; NHANES")
pb()

# ============================================================================================================
h1("1. Introduction")
p("The PHQ-9 is a nine-item self-report instrument, and a total score of 10 or more is the most widely used "
  "threshold for elevated depressive symptoms [@kroenke2001,@levis2019]. Its two-item short form, the PHQ-2, is the "
  "usual first step of two-stage screening [@kroenke2003,@levis2020]. Anyone proposing an indirect model therefore has "
  "to answer a blunt question: why predict the score instead of asking two questions? The defensible use case is "
  "narrow. When no contact with the person has yet taken place, for example when a service reviews records or survey "
  "data to decide whom to invite for assessment, symptom questions cannot be asked, while age, sex, income, "
  "education, body mass index, chronic conditions, sleep duration and self-rated health may already be on file. A "
  "model on such data can prioritise who is asked to complete the questionnaire; it cannot replace it and it does "
  "not diagnose depression. Prevalence of elevated symptoms in United States adults has also risen over the past "
  "decade [@brody2025,@daly2021], which makes behaviour across survey periods part of the question.")
p("Several studies have used the National Health and Nutrition Examination Survey (NHANES) to predict PHQ-9-defined "
  "depression from non-questionnaire variables, with logistic nomograms [@zhang2022,@liu2024] or machine-learning "
  "classifiers explained post hoc with SHAP values [@vu2025,@salimi2026]. A separate line of work has built fuzzy "
  "expert systems and fuzzy ontologies for depression [@fazel2019,@ghorbani2023,@mendez2021,@saha2024]; these are "
  "interpretable by construction, but their inputs are symptom ratings, PHQ-9 answers or sensor streams. Between the "
  "two lines sits a well-developed family of inherently interpretable models for tabular data [@rudin2019,@holzinger2019,@arrieta2020]: "
  "generalised additive models and their pairwise-interaction extension GA²M, implemented as the explainable boosting "
  "machine [@hastie1986,@lou2013,@caruana2015,@nori2019]; rule ensembles such as RuleFit [@friedman2008]; sparse "
  "integer scoring systems [@ustun2017]; and gradient-boosted trees with monotone constraints. A fuzzy rule system on "
  "non-symptom data has to be judged against this family, not only against black boxes.")
p("Our earlier view was that a knowledge structure could improve inference itself. The data did not support that "
  "view (Section 3.4): with knowledge-based cut-points and sign priors, the fitted model is close to a "
  "sign-constrained piecewise-linear additive model, and it ranks people about as well as linear logistic "
  "regression. The more useful question concerns incomplete records. Routine pipelines impute silently, which hides "
  "how much a decision depends on what was not observed [@vansmeden2020,@kompa2021]. Three lines of work address "
  "this. Credal classifiers treat missing values conservatively, compute predictions over all completions of a "
  "record and return an indeterminate output when the completions disagree [@zaffalon2002,@corani2008]. Pattern "
  "submodels fit a separate model for each missingness pattern and so avoid imputation at prediction time "
  "[@fletchermercaldo2020]. Active feature acquisition chooses at test time which unobserved feature to acquire next "
  "[@shim2018]. Interval-valued fuzzy aggregation [@zywica2015] and exact bounds of weighted averages under interval "
  "weights [@karnik2001,@wu2007] are related ideas from fuzzy systems. Our approach is closest to the credal view: "
  "it bounds the prediction of a fitted additive model over declared admissible values and abstains when the bound "
  "contains the threshold, and it adds a simple acquisition rule driven by those bounds. Its specific features are "
  "that admissible values are declared per variable in a released schema, that the bounds are exact for an additive "
  "rule class with piecewise-linear memberships, and that both are tested on real, structured missingness.")
p("This study asks three questions. (RQ1) How does a knowledge-schema-constrained fuzzy additive model compare with "
  "linear, spline and sign-constrained logistic regression, GA²M and gradient-boosted trees in discrimination, "
  "calibration and net benefit, internally, in a later survey period and in earlier survey cycles? (RQ2) Which "
  "design choices carry measurable value: rule structure, aggregation, fuzzy memberships, knowledge-based "
  "cut-points or sign constraints? (RQ3) When inputs are missing, under natural and simulated missingness, what do "
  "exact admissible-value bounds and bound-guided acquisition add over imputation, missing indicators, multiple "
  "imputation, linear bounds and complete-case abstention, including at an equal proportion referred?")
p("The contributions are as follows. (i) A knowledge schema in four layers (observation, concept, rule, provenance), "
  "released as a JSON-LD file with a SKOS concept scheme, which declares each variable's admissible range, each "
  "concept's membership function and cut-point source, and sign priors estimated on an independent survey cycle. "
  "(ii) Algorithm 1, exact bounds of the predicted probability over all admissible completions of a record, for "
  "additive rule models with one- and two-concept antecedents; we prove exactness for components with up to two "
  "continuous missing variables. For the recommended main-effects model the computation reduces to a separable "
  "search over each variable's breakpoints; the propositions matter for the general rule class. (iii) Algorithm 2, "
  "a bound-guided rule for which missing input to request. (iv) An evaluation in three survey periods against seven "
  "comparators, with the PHQ-2 as a reference point, and of the bounds under natural missingness, non-random "
  "masking and matched workload. We report the development history, including an abandoned first version, and "
  "make no claim of clinical utility.")

# ============================================================================================================
h1("2. Materials and methods")
h2("2.1. Study design, intended use and reporting")
p("This is a retrospective development, temporal and external validation study of a prediction model, reported "
  "following TRIPOD+AI [@collins2024] (checklist in Appendix B). The intended user is a service that "
  "reviews existing survey or record information before any symptom questionnaire has been administered; the "
  "intended action is to recommend completing the PHQ-9 or an equivalent assessment. Where a person can be asked two "
  "questions, the PHQ-2 is the appropriate first step, and we report it as a reference point (Section 2.13). The model "
  "does not diagnose major depressive disorder, does not estimate future incidence and is not a tool for assessing "
  "self-harm risk. Predictors and outcome are measured in the same survey, so the target is concurrent elevated "
  "symptoms.")

h2("2.2. Data sources")
p("NHANES is a continuous cross-sectional survey of the non-institutionalised United States population. Development "
  "data came from the 2017–March 2020 pre-pandemic release, which combines the 2017–2018 cycle with data collected "
  "until fieldwork was suspended in March 2020 [@akinbami2022]; 2017–2018 therefore cannot serve as independent "
  "validation. Temporal validation used August 2021–August 2023, and external validation the five two-year cycles "
  "2005–2006 to 2013–2014. The 2015–2016 cycle was used only to estimate direction priors (Section 2.6). Components "
  "were merged by the respondent identifier SEQN: demographics, depression screener, sleep, body measures, hospital "
  "utilisation, smoking, physical activity, medical conditions, diabetes and blood pressure questionnaires.")

h2("2.3. Participants and outcome")
p("We included respondents aged ≥ 18 years with valid responses (0–3) to all nine PHQ-9 items (DPQ010–DPQ090) "
  "[@nhanes_dpq]. Refused, don't-know and missing responses were not treated as symptom values; DPQ100 (functional "
  "difficulty) is not part of the total. With q_{ij} the response of participant i to item j,")
eq("S_i = Σ_{j=1}^{9} q_{ij},     Y_i = 1[S_i ≥ 10].", 1)
p("All PHQ-9 items, the total score and any derived quantity were excluded from the predictors. Of {{flow_dev_adults}} "
  "adults in 2017–March 2020, {{flow_dev_noDPQ}} had no depression screener record and {{flow_dev_invalid}} had at "
  "least one invalid item, leaving {{flow_dev_analytic}} participants ({{flow_dev_positives}} with PHQ-9 ≥ 10, "
  "{{flow_dev_prevalence_unweighted}}%). Of {{flow_tmp_adults}} adults in 2021–2023, {{flow_tmp_noDPQ}} had no "
  "screener record and {{flow_tmp_invalid}} an invalid item, leaving {{n_tmp}} ({{pos_tmp}} positive, {{prev_tmp}}%). "
  "In 2005–2014, {{ext_noDPQ}} of {{ext_adults}} adults had no screener record and {{ext_invalid}} an invalid item, "
  "leaving {{n_ext}} ({{pos_ext}} positive, {{prev_ext}}%) (Fig. 1). Included and excluded participants are compared "
  "in Table A4. The development release was split 70/30 at participant level, stratified by outcome "
  "(seed 2026), into a development set (n = {{n_dev}}) and an internal test set (n = {{n_int}}).")
fig("fig_flow.png", "Fig. 1. Participant flow. The 2015–2016 cycle was used only to estimate the direction priors stored "
    "in the knowledge schema. Models and thresholds were frozen before evaluation on the temporal and external cohorts.", 6.3)

h2("2.4. Predictors and harmonisation")
p("Candidate predictors had to be plausibly available before a symptom questionnaire, clearly defined, not derived "
  "from the outcome, and fielded with the same coding in 2017–2020 and 2021–2023. Fourteen variables qualified: age, "
  "sex, family income-to-poverty ratio, education (five levels), marital status (married/partnered, previously "
  "married, never married), body mass index, usual sleep duration on workdays and on free days, self-rated general "
  "health (five levels), current smoking, doctor-told diabetes, doctor-told hypertension, daily sedentary time, and a "
  "count of ten doctor-told chronic conditions (asthma, arthritis, congestive heart failure, coronary heart disease, "
  "angina, myocardial infarction, stroke, liver condition, chronic obstructive pulmonary disease, cancer). All are "
  "survey items. Several would not be found in a typical electronic health record: self-rated health, sleep duration "
  "and sedentary time are interview questions, and the chronic conditions are self-reported 'doctor-told' answers "
  "rather than coded diagnoses. Table 2 marks which variables are usually available as structured record data, and "
  "Section 2.13 describes a model restricted to them.")
p("Contact with a mental health professional was not used, because it is close to the outcome. Race and ethnicity "
  "served only for subgroup evaluation. Two sleep items with strong associations in the earlier cycle (SLQ050, "
  "SLQ120) were not fielded in 2021–2023 and were examined only in a development-period sensitivity analysis. Sleep "
  "duration overlaps conceptually with the PHQ-9 sleep item but is collected by a separate questionnaire; we report "
  "an analysis without sleep variables. In the development set {{any_missing_dev}}% of participants had at least one "
  "missing predictor, mostly income-to-poverty ratio ({{pir_missing_dev}}%).")
p("The 2005–2014 cycles required three harmonisation decisions, written into the analysis protocol (Section 2.8) "
  "before the files were downloaded: workday sleep came from SLD010H, recorded in whole hours; free-day sleep was not "
  "collected and was missing for every external participant; and the chronic-condition count used the nine items "
  "available before the COPD item was introduced in 2013–2014. Sedentary time was not collected in 2005–2006. Two "
  "routine recodings were made when the files were read (marital status mapped to the three categories above; race "
  "and ethnicity before 2011 from RIDRETH1, which has no separate Asian category). Variables absent in a cycle were "
  "left missing for every model.")

h2("2.5. Knowledge schema")
p("We use the term knowledge schema rather than ontology, because no description-logic reasoner [@bobillo2016] is "
  "involved and the structure is a typed, versioned declaration rather than an axiomatised theory. It has four layers "
  "(Fig. 2). The observation layer declares each variable with its domain, type, unit, NHANES source field and "
  "admissible range (Table 2; Table A5). The concept layer holds {{n_concepts}} linguistic concepts "
  "(fuzzy sets [@zadeh1965]), each bound to exactly one variable. The rule layer stores antecedent–coefficient pairs "
  "with their sign constraints. The provenance layer records the source of each cut-point and each prior. The schema "
  "is distributed as a JSON-LD 1.1 file [@jsonld2020] in which concepts are SKOS concepts [@skos2009] of one concept "
  "scheme, with membership functions, priors and rules as typed nodes; the file is loaded by the model code, so the "
  "released artefact is the one that produced the results.")
fig("fig_framework.png", "Fig. 2. Architecture of the prescreening system. The knowledge schema (orange) declares "
    "variables, admissible ranges, concepts, cut-point sources and sign priors; Algorithms 1 and 2 use these declarations "
    "when inputs are missing.", 6.6)
p("Continuous variables were partitioned into trapezoidal concepts forming Ruspini partitions [@ruspini1969], so that "
  "memberships of a variable's concepts sum to one. For break points b_1 < … < b_m and half-width h, concept k has "
  "support [b_{k−1} − h, b_k + h] and core [b_{k−1} + h, b_k − h], with membership")
eq("μ(x; a,b,c,d) = max(0, min((x − a)/(b − a), 1, (d − x)/(d − c))).", 2)
p("Cut-points came from external conventions: WHO body-mass-index categories (18.5, 25, 30 kg/m²; h = 1) [@who2000], "
  "the consensus recommendation of at least 7 h of sleep (short < 6.5 h, long > 9.5 h; h = 0.5) [@watson2015], 130% "
  "and 350% of the federal poverty level (h = 0.3), age bands at 35 and 60 years (h = 5) and sedentary-time bands at 4 "
  "and 8 h/day (h = 1). Ordinal and categorical variables have crisp set memberships (e.g. FairHealth, PoorHealth, "
  "Multimorbidity for ≥ 2 conditions).")

h2("2.6. Direction priors from an independent cycle")
p("For each concept we estimated, in NHANES 2015–2016 (n = {{flow_pri_analytic}}, {{flow_pri_positives}} positive), the "
  "risk ratio of PHQ-9 ≥ 10 for membership ≥ 0.5 versus < 0.5 with a log-scale 95% confidence interval; the direction "
  "D(k) was +1 if the lower limit exceeded 1, −1 if the upper limit was below 1, and 0 otherwise. The procedure "
  "estimated {{n_pri_est_pos}} positive, {{n_pri_est_neg}} negative and {{n_pri_est_zero}} neutral directions for the "
  "{{n_concepts}} concepts other than free-day sleep and the five extended-sleep concepts (Table A1). Free-day sleep "
  "was not asked in 2015–2016, so its {{n_pri_inh}} concepts inherited the direction of the workday concepts; these "
  "priors were never estimated and are marked in Tables 2 and 3. Each prior compares a concept with all other values "
  "of its variable, whereas each coefficient of the main-effects model (Section 2.7) is a contrast with one reference "
  "concept. A sign constraint therefore restricts a different estimand from the one the prior measured. We report "
  "the consequences in three sensitivity analyses: priors from age- and sex-adjusted models, priors from one mutually "
  "adjusted model, and priors estimated for the reference contrast actually fitted (concept versus reference concept).")

h2("2.7. Model: sign-constrained fuzzy additive rules")
p("A rule r has an antecedent A_r of one or two concepts and fires with strength")
eq("μ_r(x) = min_{k ∈ A(r)} μ_k(x_k).", 3)
p("A two-concept antecedent must refer to two different variables, and the non-zero directions of its concepts must "
  "agree; the rule inherits the common direction s_r ∈ {−1, 0, +1}. Rules combine additively on the log-odds scale,")
eq("η(x) = logit p(x) = β_0 + Σ_{r=1}^{R} β_r μ_r(x),   subject to s_r β_r ≥ 0,", 4)
p("a zero-order Takagi–Sugeno system [@takagi1985] with log-odds consequents. In the selected structure every "
  "single-concept rule enters except one reference concept per variable (the concept with the highest mean "
  "membership in development data), and K two-concept rules may then be added greedily by the score statistic "
  "g_r²/h_r, with g_r = Σ_i μ_{ir}(y_i − p_i) and h_r = Σ_i μ_{ir}² p_i(1 − p_i), provided the sign of g_r is compatible "
  "with s_r and the rule's firing vector does not correlate above 0.95 with a selected rule. Coefficients were fitted "
  "by L-BFGS-B on the ridge-penalised log-likelihood (λ = 10⁻³) with the sign constraints as box bounds. During "
  "fitting, and for the point score, a missing membership was replaced by its development-sample mean. With Ruspini "
  "partitions the single-concept part is a piecewise-linear additive function of each continuous variable with knots "
  "at the schema's cut-points, i.e. a constrained generalised additive model [@hastie1986] whose terms carry names, "
  "and two-concept rules play the role of GA²M interactions [@lou2013]. K was chosen from {0, 8, 16, 32, 64} by "
  "5-fold cross-validated AUROC in the development set, taking the smallest K within 0.005 of the best. "
  "Cross-validated AUROC was {{cvmain_K0}}, {{cvmain_K8}}, {{cvmain_K16}}, {{cvmain_K32}} and {{cvmain_K64}} "
  "(logistic regression {{cv_lr_dev}}), so K = {{K_full}}: the final model has {{n_rules}} single-concept rules and an "
  "intercept.")

h2("2.8. Development history and analysis protocol")
p("The structure above is the second version of the system. Version 1 selected K = 32 rules of any arity by the "
  "same greedy procedure with no forced main effects; {{v1_n_pairs}} of its 32 rules were two-concept conjunctions. On "
  "the internal and temporal sets it trailed logistic regression by {{v1old_d_lr_int_abs}} ({{v1old_d_lr_int_ci}}) and "
  "{{v1old_d_lr_tmp_abs}} ({{v1old_d_lr_tmp_ci}}) AUROC. We attributed the gap to the absence of main effects, tested "
  "that diagnosis with development-set cross-validation only, and wrote a protocol fixing the revised structure, the "
  "K grid and selection rule, the comparators and ablations of the first submission, the external cohorts and their "
  "harmonisation, and the primary comparison (external AUROC versus logistic regression with missing indicators). "
  "According to the protocol's addendum, written at the time, the protocol file was last saved at {{t_protocol}} and "
  "the first 2005–2014 file was downloaded eight seconds later, at {{t_first_download}}; no external file was opened "
  "before that. These times are file-system modification times on the analysis machine. They can be altered by "
  "copying or editing, and no independently verifiable record (registry entry, commit or deposit) predates the "
  "external download. The earliest verifiable record of the protocol is its deposit with the code at Zenodo "
  "({{zenodo_doi}}), made after all analyses. We therefore describe the external comparison as free of earlier "
  "results according to our records, not as verified to be so. Version-1 results on the internal and temporal sets "
  "had been seen before the redesign, and all comparators and analyses added during revision (Sections 2.12–2.13) "
  "were specified after the original results were known; they are secondary. Version 1 is retained as a comparator.")

h2("2.9. Algorithm 1: exact admissible-value bounds")
p("Let O and M be the observed and missing variables of a person. Instead of imputing x_M, the system reports the "
  "range of probabilities the model can produce for any admissible completion,")
eq("p_L = min σ(η(x_O, x_M)),   p_U = max σ(η(x_O, x_M)),   over x_M ∈ 𝒳_M,", 5)
p("where 𝒳_M is the product of the admissible sets declared in the observation layer (category sets for discrete "
  "variables, intervals for continuous ones; Table A5). Because σ is monotone it suffices to bound η. "
  "Each rule involves at most two variables, so η is a constant plus a sum of functions of connected components of "
  "missing variables (two missing variables are connected if a rule contains both), and the extrema of η are sums of "
  "per-component extrema. Algorithm 1 (Box 1) computes them exactly.")
p("*Proposition 1 (one continuous missing variable).* Let x ∈ [l, u] be the only continuous missing variable of a "
  "component, with all discrete variables of the component fixed. Then g(x) = Σ_r β_r min(μ_{k(r)}(x), c_r), where "
  "c_r ∈ [0, 1] is the observed partner membership of rule r (c_r = 1 for single-concept rules). Each μ_k is "
  "continuous and piecewise linear with breakpoints at its trapezoid corners, and min(μ_k(x), c_r) adds breakpoints "
  "only where μ_k(x) = c_r, i.e. at x = a + c_r(b − a) and x = d − c_r(d − c). Hence g is continuous and piecewise "
  "linear on [l, u] with breakpoints in the finite set 𝒞 = {l, u} ∪ {corners} ∪ {solutions of μ_k(x) = c_r}. A "
  "continuous piecewise-linear function on a closed interval attains its extrema at an endpoint or a breakpoint, so "
  "min g and max g are attained on 𝒞 and enumerating 𝒞 is exact. □")
p("*Proposition 2 (two continuous missing variables).* Let (x, z) ∈ [l_x, u_x] × [l_z, u_z] be connected through "
  "pair rules min(μ_k(x), μ_m(z)). Partition each axis at its breakpoints from Proposition 1. On each resulting "
  "rectangle every membership is affine, so each pair term equals one of two affine functions, switching along the "
  "line μ_k(x) = μ_m(z). These lines cut the rectangle into convex polygons on which F(x, z) is affine; F is "
  "continuous, so its extrema over the rectangle lie at polygon vertices. Every vertex is a rectangle corner, an "
  "intersection of a switching line with a rectangle edge, or an intersection of two switching lines inside the "
  "rectangle. Enumerating these points on all rectangles is exact. □")
p("Discrete variables in a component are enumerated over their categories, and Propositions 1–2 are applied for "
  "each combination. Components with more than two continuous missing variables fall back to the box bound obtained "
  "by letting each missing membership range over [0, 1]; this is a valid outer bound, not an exact one, and we report "
  "how often it occurs. For a component with m_c categorical combinations, n_b breakpoints per axis and q pair rules, "
  "the cost is O(m_c n_b) evaluations for one continuous variable and O(m_c n_b² q²) for two. For the selected "
  "main-effects model every component is a single variable with no partner, 𝒞 reduces to the trapezoid corners and "
  "range ends, and the bounds are exact. The first submission used a 41-point grid augmented with trapezoid corners; "
  "we compare it with Algorithm 1 in Section 3.6. The box bound, which ignores that memberships of one variable are "
  "linked through the partition, is retained as an ablation. For sign-constrained linear regression the analogous "
  "exact bound is closed-form: each missing input moves η linearly, so its extremes are at the ends of the admissible "
  "range, and marital status is enumerated over its categories.")
p("Two remarks place Algorithm 1 in context. First, for the recommended main-effects model every rule involves "
  "one variable and has no partner, so η is a sum of univariate piecewise-linear functions and its bounds follow "
  "from evaluating each term at its breakpoints and range ends; this is elementary. Propositions 1 and 2 are needed "
  "for rule bases with conjunctions (version 1 and the interaction variant), and we present them as covering the "
  "general rule class, not as a property of the deployed model. Second, the same separable bounds hold for any "
  "logistic model whose terms are piecewise linear with constant tails at the same knots; a spline or linear model "
  "with unbounded tails gives bounds that grow with the admissible range. Whatever coverage advantage the proposed "
  "bounds have over linear bounds (Section 3.6) therefore comes from bounded basis functions, which the schema "
  "supplies through saturating memberships, not from the schema as such.")
tab("tableAlg")
p("The point score used for ranking is the development-mean imputation of Section 2.7, clipped into [p_L, p_U]. For a "
  "main-effects model the clipping is never active: η is linear in the memberships, and the vector of development-mean "
  "memberships is a convex combination of admissible membership vectors, so the imputed η lies between the bounds. The "
  "referral threshold t, the point score and the bounds are all on the scale of the model's raw probability σ(η), "
  "before Platt recalibration. Decisions follow a reject option [@chow1970]:")
eq("refer if p_L ≥ t;   do not refer if p_U < t;   abstain otherwise.", 6)
p("Abstentions were never counted as negative predictions. We report decisions on covered cases and, as the "
  "conservative policy for prescreening, refer-on-abstain.")

h2("2.10. Algorithm 2: bound-guided input acquisition")
p("An abstention means that at least one missing input could change the decision. Algorithm 2 asks for one missing "
  "input at a time. For each candidate input v, it estimates the probability that observing v alone resolves the "
  "decision, ρ_v = Σ_j w_{vj} 1[p_L(x_O, v = v_j) ≥ t or p_U(x_O, v = v_j) < t], where the other missing inputs remain "
  "unknown and v_j, w_{vj} are support points and weights of the development-set distribution of v (all categories "
  "for discrete variables, 25 quantile points for continuous ones). The input with the largest ρ_v is requested, its "
  "value is entered, the bounds are recomputed, and the loop stops when the decision is resolved or no missing input "
  "remains. Using the marginal distribution ignores what the observed inputs say about v. A conditional version "
  "would draw the support points from the distribution of v given x_O, for example from the chained-equation "
  "imputation model fitted for the comparators; ρ_v would then favour inputs whose likely values, for this person, "
  "fall on one side of the threshold. We compared the marginal rule with three orders: a fixed order by the range of "
  "each input's total log-odds contribution, a cost order that requests structured record items first (diabetes, "
  "hypertension, chronic conditions, BMI, smoking, marital status, education), then income, then interview items "
  "(self-rated health, sleep, sedentary time), and a random order. Evaluation used internal test records with 10% or "
  "30% of inputs masked at random and temporal records with 30% masked; a requested masked input was revealed with its "
  "true value, and inputs missing in the original data could not be revealed. Differences in the number of inputs "
  "requested are paired by record, with bootstrap intervals (2,000 resamples). By construction every order resolves "
  "the same records with the same final decisions.")

h2("2.11. Explanation trace")
p("For each person the system exports a JSON trace with the inputs, missing-input flags, the membership of every "
  "concept used by the rule base and its provenance, each rule's firing strength (or lower and upper firing when an "
  "input is missing), coefficient and log-odds contribution β_r μ_r, the intercept, the probability bounds and, for "
  "abstentions, the input requested by Algorithm 2. For complete inputs the contributions add exactly to η. This "
  "additive decomposition holds for any generalised linear model, so it is not by itself an advantage over logistic "
  "regression; Section 3.7 compares the two on size and sign plausibility instead.")

h2("2.12. Comparators and ablations")
p("Comparators were logistic regression with median imputation, missing-value indicators, standardisation and an "
  "L2 penalty chosen over ten values by 5-fold cross-validated AUROC; gradient-boosted trees [@friedman2001] "
  "(scikit-learn HistGradientBoostingClassifier [@pedregosa2011], native missing-value routing, minimum leaf size 50, "
  "tuned over learning rate, depth and iterations; {{gbt_params}} selected); logistic regression with median "
  "imputation and no indicators; logistic regression after multiple imputation by chained equations [@vanbuuren2011] "
  "(five imputations with posterior draws, predictions averaged); logistic regression with restricted cubic splines "
  "[@harrell2015] (four knots at the 5th, 35th, 65th and 95th development percentiles) for age, BMI and both sleep "
  "durations, other inputs linear, median imputation without indicators; sign-constrained logistic regression on the "
  "raw inputs, with the direction priors imposed on main effects (age, BMI and sleep unconstrained because their "
  "associations are U-shaped; λ = {{slr_lambda}}); an explainable boosting machine (GA²M; interpret 0.7.8, 10 pairwise "
  "interactions, missing values in a separate bin) [@lou2013,@nori2019]; and gradient-boosted trees with monotone "
  "constraints derived from the same priors. Rule ensembles [@friedman2008] and integer scoring systems [@ustun2017] "
  "were not fitted. Bounds for the spline model were computed by a dense 2,001-point search per missing input, which "
  "is exact up to the grid because the model is additive in its inputs. Version-level comparators were version 1 "
  "(K = 32), main effects plus 16 pair rules, and normalised zero-order TSK aggregation p = Σ_r w_r μ_r c_r / Σ_r w_r μ_r "
  "(K = 32), which also differs in rule filtering, gain and point score. Single-element ablations were: no direction "
  "constraints; crisp memberships; quantile-based memberships with no constraints and median imputation ('fuzzy "
  "without schema'); median imputation instead of bounds; and box bounds.")

h2("2.13. Performance measures and secondary analyses")
p("Every model produced out-of-fold predictions in the development set by 5-fold cross-validation. These were used "
  "to fit a Platt recalibration map and to fix the referral threshold as the largest score reaching an out-of-fold "
  "sensitivity of at least 0.80. Models were then refitted on the whole development set and applied unchanged to all "
  "cohorts. We report AUROC, area under the precision–recall curve (AUPRC), sensitivity, specificity, predictive "
  "values, proportion flagged, Brier score, calibration slope and observed-to-expected ratio (O/E) [@vancalster2019], "
  "and net benefit across threshold probabilities 0.02–0.30 [@vickers2006,@vickers2016]. Confidence intervals came "
  "from 1,000 bootstrap resamples (500 for AUROC in single cycles); AUROC differences used 2,000 paired resamples "
  "(1,000 in single cycles) with percentile intervals and two-sided bootstrap p-values. The primary comparison is "
  "external AUROC of the proposed model versus logistic regression; all other comparisons are secondary and "
  "unadjusted for multiplicity, so we interpret them by effect size and interval width rather than by p < 0.05. "
  "AUROC weighted by NHANES interview weights is a sensitivity analysis.")
p("The bootstrap resamples the test set only. To include training variability, the proposed model, version 1, the "
  "interaction variant, three ablations and five comparators were refitted on {{spl_B}} repeated stratified 70/30 "
  "splits of the full 2017–2020 release, with hyperparameters fixed at the values selected above. Recalibration-in-"
  "the-large was examined by estimating an intercept shift on a random half of the temporal or external cohort and "
  "evaluating O/E, Brier score, sensitivity, specificity and proportion flagged on the other half (200 repetitions). "
  "Operating points are also expressed per 1,000 adults as questionnaires avoided and cases missed.")
p("The PHQ-2 reference used items DPQ010 and DPQ020 with the usual cut-point ≥ 3 [@kroenke2003,@levis2020]; it is not "
  "a competitor in the same input space but shows what two symptom questions achieve. A record-typical model used "
  "only age, sex, BMI, diabetes, hypertension, chronic-condition count, smoking, education and marital status (and a "
  "seven-variable version without the last two). Direction priors were re-estimated with age- and sex-adjusted and "
  "with mutually adjusted logistic models, and the proposed model and version 1 refitted with them.")
p("Missing inputs were examined in three settings. Natural missingness: all models were applied to the internal, "
  "temporal and external cohorts as observed; externally free-day sleep is absent for everyone and sedentary time in "
  "2005–2006, and income-to-poverty ratio is missing for 7.5–13.3% of participants in every cohort. Simulated "
  "missingness completely at random (MCAR) on the internal test set: each of the 12 predictors other than age and sex "
  "masked with probability 10–50% (10 repetitions; the same masks for every model and analysis), or whole domains "
  "removed. Missingness not at random (MNAR): income-to-poverty ratio masked with probability 0.6 if below 1.3 and "
  "0.1 otherwise, self-rated health masked with probability 0.6 if fair or poor and 0.1 otherwise, or both (10 "
  "repetitions). For every model we report AUROC and, at its frozen threshold, sensitivity, specificity and proportion "
  "flagged. For models with bounds (the proposed model, its box-bound ablation, sign-constrained and spline "
  "regression) we report coverage, abstention, decisions on covered cases and refer-on-abstain operating "
  "characteristics, and add a complete-case baseline that abstains whenever any input is missing. Because "
  "refer-on-abstain and a point-score threshold refer different numbers of people, decision value was compared at "
  "matched workload, setting each point-score model's threshold so that it flagged the same proportion as the "
  "proposed refer-on-abstain policy, and by net benefit at threshold probabilities 0.05 and 0.10. Coverage was "
  "recomputed with continuous ranges narrowed to the 1st–99th development percentiles. A further sensitivity analysis "
  "refitted every model without free-day sleep and evaluated it externally, to show how the treatment of a "
  "structurally absent variable affects the primary comparison. Subgroup AUROC, sensitivity and O/E were computed "
  "by sex, age band and race and ethnicity where at least 10 positive cases were available.")

h2("2.14. Software and use of generative AI in the research process")
p("Analyses used Python 3.13 with pandas 3.0.5, scikit-learn 1.9.1, SciPy 1.18.1, statsmodels and interpret 0.7.8. "
  "The analysis code was written with the assistance of a generative AI tool (Claude, Anthropic), which drafted the "
  "cohort construction, model, Algorithms 1–2, experiments and figure scripts under the authors' specification. The "
  "authors reviewed the code, and correctness was checked by tests that do not depend on the generated code: the "
  "cohort flow and direction priors were reproduced exactly from the public files; Algorithm 1 was checked against "
  "dense brute-force evaluation (4,001-point grids) and against the earlier grid implementation; traces were "
  "re-evaluated by independent code; and the archived release was re-run from the raw files in a clean "
  "directory, reproducing every number in this manuscript except the reported computing times. All numbers in this manuscript are written by the analysis code to one statistics "
  "file and inserted programmatically. The code, schema and statistics file are publicly available (Code "
  "availability).")

# ============================================================================================================
h1("3. Results")
h2("3.1. Cohorts")
p("Table 1 summarises the cohorts. Prevalence of PHQ-9 ≥ 10 was {{prev_ext}}% in 2005–2014, "
  "{{flow_dev_prevalence_unweighted}}% in 2017–2020 and {{prev_tmp}}% in 2021–2023. The 2021–2023 cohort was older and "
  "more educated, with fewer participants rating their health as fair or poor but more sedentary time and more chronic "
  "conditions. The 2005–2014 cohort was younger, had more current smokers and reported about half an hour less workday "
  "sleep, partly because sleep was recorded in whole hours. Weighted prevalence differed between the development and "
  "internal test sets (Table 1) although the split was stratified on the unweighted outcome, because interview weights "
  "were not balanced across the split. Excluded adults of the development release were slightly older and more often "
  "women than included adults (Table A4).")
tab("table1")

h2("3.2. Knowledge schema and rule base")
p("Direction priors from 2015–2016 were consistent with established epidemiology: low income (RR {{rr_LowIncome}}), "
  "current smoking ({{rr_CurrentSmoker}}), poor self-rated health ({{rr_PoorHealth}}), multimorbidity "
  "({{rr_Multimorbidity}}), short ({{rr_ShortSleepWorkday}}) and long ({{rr_LongSleepWorkday}}) workday sleep were "
  "positive; college education ({{rr_CollegeGraduate}}) and being married or partnered ({{rr_MarriedOrPartnered}}) were "
  "negative. Age- and sex-adjusted priors had the same direction for every concept. Mutually adjusted priors changed "
  "{{mvpri_n_changed}} non-reference directions ({{mvpri_changed_list}}). Of the {{n_rules}} rules, {{n_rules_signed}} "
  "carry a sign constraint and {{n_rules_free}} are free; none of the constrained coefficients sits at its zero bound, "
  "and refitting without constraints changed no sign ({{nodir_violations}} violations among {{nodir_signed_rules}}). "
  "The constraints were therefore not binding for this model. Table 3 lists the rules by average effect. Because one "
  "reference concept per variable is omitted, each coefficient is a contrast with that reference; for example the "
  "negative coefficients of NormalWeight and Overweight are contrasts with Obese, not claims that overweight protects "
  "against depression.")
tab("table2")
tab("table3")

h2("3.3. Discrimination and calibration")
p("Table 4 and Fig. 3 report performance; Table 6 gives paired differences. The primary comparison was external "
  "AUROC of {{full_ext_AUROC}} ({{full_ext_AUROC_ci}}) for the proposed model against {{lr_ext_AUROC}} for logistic "
  "regression with missing indicators (difference {{d_lr_ext}}, {{d_lr_ext_ci}}; p = {{d_lr_ext_p}}). This difference "
  "does not reflect the knowledge schema. Free-day sleep is missing for every external participant; the indicator "
  "model applies a learnt indicator coefficient to everyone and over-predicts (O/E {{lr_ext_OE}}), whereas logistic "
  "regression without indicators reached {{lrni_ext_AUROC}}, the same as the proposed model ({{d_lrni_ext}}, "
  "{{d_lrni_ext_ci}}). When every model was refitted without free-day sleep, the proposed model reached "
  "{{nowe_full_ext}} externally against {{nowe_lrni_ext}} for regression without indicators ({{dnowe_lrni_ext}}, "
  "{{dnowe_lrni_ext_ci}}) and {{nowe_lr_ext}} with indicators, whose remaining indicators still covered sedentary "
  "time in 2005–2006. Logistic regression with restricted cubic splines was the best-ranking model in the later and "
  "earlier periods: {{rcs_tmp_AUROC}} temporally and {{rcs_ext_AUROC}} externally, {{d_rcs_ext_abs}} above the "
  "proposed model ({{d_rcs_ext_ci}}). Internally the proposed model was the lowest of the main models "
  "({{full_int_AUROC}} versus {{int_min_oth}}–{{int_max_oth}}). Over {{spl_B}} repeated splits of 2017–2020 it "
  "averaged {{spl_full_mean}} (SD {{spl_full_sd}}), and the mean difference to linear logistic regression was "
  "{{spl_lr_mean}} ({{spl_lr_lo}} to {{spl_lr_hi}}). The proposed model therefore ranks people about as well as "
  "linear regression and slightly worse than spline regression in every period and than GA²M on 2017–2023 data.")
tab("table4")
fig("fig_roc.png", "Fig. 3. ROC curves on internal testing (2017–March 2020), temporal validation (2021–2023) and "
    "external validation (2005–2014). AUROC in parentheses.", 6.6)
p("Calibration differed more than ranking (Fig. 4). Internally O/E was close to 1 for every model; the calibration "
  "slope of logistic regression with indicators was {{lr_int_slope}}, i.e. its predictions were too moderate "
  "(insufficiently spread), which can follow Platt recalibration fitted on out-of-fold scores and applied to the "
  "full-data refit. In 2021–2023 every model under-predicted (O/E {{full_tmp_OE}} for the proposed model; calibration "
  "slope {{full_tmp_slope}}). An intercept update estimated on half the temporal cohort brought O/E on the other half "
  "to {{rec_tmp_full_OE_after}} ({{rec_tmp_full_OE_after_ci}}) and the Brier score from {{rec_tmp_full_Brier_before}} "
  "to {{rec_tmp_full_Brier_after}} on the evaluation halves. In 2005–2014 the proposed model ({{full_ext_OE}}), "
  "sign-constrained regression ({{slr_ext_OE}}), regression without indicators ({{lrni_ext_OE}}), with multiple "
  "imputation ({{lrmi_ext_OE}}) and with splines ({{rcs_ext_OE}}) were calibrated in the large, whereas logistic "
  "regression with indicators ({{lr_ext_OE}}), gradient-boosted trees ({{gbt_ext_OE}}), monotone trees ({{mgbt_ext_OE}}) "
  "and GA²M ({{ebm_ext_OE}}) over-predicted; the over-prediction is tied to how a model treats the absent free-day "
  "sleep. Results by external cycle are in Table 5. Unweighted minus weighted AUROC ranged from {{wgap_min}} to "
  "{{wgap_max}} across models and sets (Table 4).")
fig("fig_calibration.png", "Fig. 4. Calibration by deciles of predicted probability after development-set Platt "
    "recalibration. All models under-predict in 2021–2023.", 6.6)
tab("tableCycle")

h2("3.4. Rule structure, aggregation and ablations")
p("Structure mattered more than any knowledge-related element (Table 6). Version 1 was {{d_v1_ext_abs}} lower "
  "externally ({{d_v1_ext_ci}}) and lower than the proposed model in all {{spl_B}} repeated splits (mean "
  "{{spl_v1_mean}}). Adding 16 pair rules did not help on any test set ({{d_inter_ext}} externally) and was lower in "
  "{{spl_inter_frac}}% of splits. Normalised TSK aggregation was {{d_tsk_tmp_abs}} lower in 2021–2023 ({{d_tsk_tmp_ci}}). "
  "Version 1 is the one model in which constraints mattered: refitted without them at the same K = 32, its AUROC fell "
  "from {{v1_con_int}} to {{v1_unc_int}} internally and from {{v1_con_ext}} to {{v1_unc_ext}} externally; with mutually "
  "adjusted priors it reached {{mvpri_v1_int}} and {{mvpri_v1_ext}}. For the main-effects model, constraints were not "
  "binding: the difference from the unconstrained model was below 10⁻⁴ in every set. Fuzzy memberships gave a small "
  "but consistent advantage over crisp memberships at the same cut-points: {{d_crisp_int}} internally "
  "({{d_crisp_int_ci}}), {{d_crisp_tmp}} temporally ({{d_crisp_tmp_ci}}), {{d_crisp_ext}} externally, and "
  "{{spl_crisp_mean}} on average over repeated splits, with the proposed model higher in {{spl_crisp_frac}}% of them. "
  "Quantile memberships without the schema gave inconsistent differences ({{d_noonto_tmp}} temporally, "
  "{{d_noonto_ext}} externally, {{spl_noonto_mean}} over splits). Removing the schema as a whole has no detectable "
  "effect on discrimination.")
tab("tableAbl")

h2("3.5. PHQ-2 reference, net benefit and record-typical predictors")
p("Two symptom questions outperformed every non-symptom model by a wide margin (Table 7). PHQ-2 ≥ 3 had sensitivity "
  "{{phq2_int_sens}} and specificity {{phq2_int_spec}} internally and flagged {{phq2_int_flag}}% of adults; the "
  "proposed model at its development threshold had sensitivity {{full_int_Sensitivity}} at specificity "
  "{{full_int_Specificity}} and flagged {{full_int_flag}}%. Per 1,000 adults on the internal test set, the model avoided "
  "{{op_int_full_avoid}} questionnaires and missed {{op_int_full_missed}} cases. In 2021–2023 the frozen model avoided "
  "{{op_tmp_full_avoid_b}} and missed {{op_tmp_full_missed_b}}; after the intercept update, at the same probability "
  "threshold, it flagged {{op_tmp_full_flag_a}}%, avoided {{op_tmp_full_avoid_a}} and missed {{op_tmp_full_missed_a}}. "
  "Decision curves are in Fig. 5. Internally, net benefit of the proposed model exceeded that of referring everyone by "
  "{{dca_int_05_dall}} at a threshold probability of 0.05, {{dca_int_10_dall}} at 0.10 and {{dca_int_20_dall}} at 0.20; "
  "externally by {{dca_ext_05_dall}}, {{dca_ext_10_dall}} and {{dca_ext_20_dall}}. Differences from logistic regression "
  "were at most a few thousandths at these thresholds (for example {{dca_int_10_dlr}} internally and "
  "{{dca_ext_10_dlr}} externally at 0.10). At thresholds near 0.02 all curves coincide with referring everyone.")
p("Restricting inputs to the nine record-typical variables reduced AUROC to {{ehr9_full_int}} internally, "
  "{{ehr9_full_tmp}} temporally and {{ehr9_full_ext}} externally for the proposed model (logistic regression "
  "{{ehr9_lr_int}}, {{ehr9_lr_tmp}} and {{ehr9_lr_ext}}); the seven-variable version reached {{ehr7_full_int}}, "
  "{{ehr7_full_tmp}} and {{ehr7_full_ext}}. Without self-rated health, sleep and sedentary time, the model is a weak "
  "ranker in the later survey period.")
tab("tablePHQ2")
fig("fig_dca.png", "Fig. 5. Decision curves. Net benefit of referring for questionnaire completion when the predicted "
    "probability exceeds the threshold, compared with referring everyone or no one. Dotted: proposed model after "
    "recalibration-in-the-large on the same cohort (illustrative upper reference).", 6.6)

h2("3.6. Missing inputs: exact bounds, abstention and acquisition")
p("*Exactness.* For the proposed model the earlier grid and Algorithm 1 agreed on all {{n_int}} internal test "
  "records with 30% of inputs masked (largest difference {{ex_full_short}} log-odds), as expected for a main-effects "
  "model. For the interaction variant, {{ex_inter_fbrows}} of {{n_int}} records ({{ex_inter_fbrows_pct}}%) had a "
  "component with more than two continuous missing variables and received the outer bound; on the others the grid "
  "was exact. For version 1, whose conjunctions create breakpoints at μ_k(x) = c_r, the grid under-covered the exact "
  "range in {{ex_v1_nshort}} records, by at most {{ex_v1_short}} log-odds (Fig. 6), and changed no decision; "
  "{{ex_v1_fbrows}} of {{n_int}} records ({{ex_v1_fbrows_pct}}%) needed the outer bound. Against dense brute force in 80 "
  "records per model with one or two masked inputs, Algorithm 1 never returned a range that excluded a brute-force "
  "value (largest excess {{ex_max_outside}}) and differed from it by less than 10⁻⁴ log-odds. Bounds for {{n_int}} "
  "masked records took {{ex_full_sec}} s for the proposed model and {{ex_v1_sec}} s for version 1.")
fig("fig_alg1.png", "Fig. 6. Algorithm 1 on one participant of the internal test set under the version-1 rule base "
    "with one continuous input missing. The linear predictor is piecewise linear in the missing input; its maximum lies "
    "at a breakpoint where a membership equals the observed partner membership, which the 41-point grid with trapezoid "
    "corners does not contain.", 6.0)
p("*Natural missingness.* Table 8 reports the bounds on the cohorts as observed. Internally and temporally, where "
  "{{nat_int_anymiss}}% and {{nat_tmp_anymiss}}% of participants had at least one missing input (mostly income), every "
  "bounding method decided about 95% of participants ({{nat_int_full_Coverage}}% for the proposed model, "
  "{{nat_int_slr_Coverage}}% for linear and {{nat_int_rcs_Coverage}}% for spline bounds internally) and the complete-case "
  "baseline {{nat_int_cc_Coverage}}%. The external cohort is different: free-day sleep was absent for everyone, so the "
  "complete-case baseline decided nobody. Admissible-value bounds decided {{nat_ext_full_Coverage}}% of external "
  "participants, every decision made with free-day sleep unknown, against {{nat_ext_box_Coverage}}% for box bounds, "
  "{{nat_ext_slr_Coverage}}% for linear and {{nat_ext_rcs_Coverage}}% for spline bounds. Narrowing continuous ranges to "
  "the 1st–99th development percentiles left the proposed model's coverage unchanged ({{nat_ext_full_Coverage_p99}}%) "
  "and raised linear and spline coverage to {{nat_ext_slr_Coverage_p99}}% and {{nat_ext_rcs_Coverage_p99}}%. On decided "
  "external records sensitivity was {{nat_ext_full_Sens_covered}}% and specificity {{nat_ext_full_Spec_covered}}%; under "
  "refer-on-abstain, {{nat_ext_full_Sens_refer}}% and {{nat_ext_full_Spec_refer}}%, with {{nat_ext_full_Flag_refer}}% "
  "referred. Flagging the same {{nat_ext_full_Flag_refer}}% by point score gave sensitivity {{natm_ext_full_Sens}}% for "
  "the proposed model, {{natm_ext_lrni_Sens}}% for regression without indicators and {{natm_ext_rcs_Sens}}% for spline "
  "regression. At equal workload, then, the bounds did not find more cases; what they add is the knowledge of which "
  "{{nat_ext_full_Coverage}}% of decisions hold for every admissible value of the absent inputs.")
tab("tableNat")
p("*Simulated missingness.* Table 9 and Fig. 7 summarise the masking experiments. Under MCAR, logistic regression "
  "with indicators fell from {{st_lr_0_AUROC}} to {{st_lr_30_AUROC}} at 30% masking [@vansmeden2020,@groenwold2012], and GA²M, whose separate "
  "missing-value bin was learnt from structured missingness, to {{st_ebm_30_AUROC}}; without indicators regression held "
  "at {{st_lrni_30_AUROC}}, with multiple imputation at {{st_lrmi_30_AUROC}} and with splines at "
  "{{spmc_30_rcs_AUROC}}, against {{st_full_30_AUROC}} for the proposed point score. Clipping changed no point score, as "
  "Section 2.9 predicts for a main-effects model. At 30% masking admissible-value bounds decided "
  "{{sbmc_30_full_Coverage}}% of records, linear bounds {{sbmc_30_slr_Coverage}}%, spline bounds {{sbmc_30_rcs_Coverage}}% "
  "and the complete-case baseline {{sbmc_30_cc_Coverage}}%; on decided records sensitivity was "
  "{{sbmc_30_full_Sens_covered}}% and specificity {{sbmc_30_full_Spec_covered}}%, and under refer-on-abstain the "
  "policy referred {{sbmc_30_full_Flag_refer}}% of participants. At that workload every point-score model had "
  "sensitivity between {{smmc_30_min_Sens}}% and {{smmc_30_max_Sens}}%. The coverage advantage is not general. When "
  "whole domains were removed, linear bounds decided more participants than admissible-value bounds in the sleep "
  "({{st_slr_Sleep_Coverage}}% versus {{st_full_Sleep_Coverage}}%) and socioeconomic ({{st_slr_Socioeconomic_Coverage}}% "
  "versus {{st_full_Socioeconomic_Coverage}}%) scenarios and fewer in the somatic, lifestyle and self-rated-health "
  "scenarios. The pattern follows the width of each variable's possible contribution to the linear predictor. Across "
  "their admissible ranges, self-rated health and the chronic-condition count can move it by {{crl_srh}} and "
  "{{crl_n_chronic}} log-odds in sign-constrained regression but by {{crf_srh}} and {{crf_n_chronic}} in the proposed "
  "model, whose concepts saturate; for the sleep and socioeconomic variables the two models' ranges are similar in "
  "total (for example income {{crl_pir}} versus {{crf_pir}}, education {{crl_educ}} versus {{crf_educ}}, workday sleep "
  "{{crl_sleep_wd}} versus {{crf_sleep_wd}}), so linear bounds are no wider there. ""Under MNAR masking of income in low-income records, bounds and imputation behaved alike (coverage "
  "{{sbmn_incomelow_full_Coverage}}% versus {{sbmn_incomelow_slr_Coverage}}% for linear bounds). When self-rated "
  "health was masked preferentially in fair or poor health, admissible-value bounds decided "
  "{{sbmn_srhpoor_full_Coverage}}% against {{sbmn_srhpoor_slr_Coverage}}% (linear) and {{sbmn_srhpoor_rcs_Coverage}}% "
  "(spline), and decisions on covered records kept sensitivity {{sbmn_srhpoor_full_Sens_covered}}%. The imputation "
  "models' AUROC fell to {{spmn_srhpoor_lrni_AUROC}}–{{spmn_srhpoor_lr_AUROC}}, but at matched workload their "
  "sensitivity ({{smmn_srhpoor_lrni_Sens}}% without indicators) was no lower than that of the proposed policy "
  "({{smmn_srhpoor_full_Sens}}%). Net benefit of refer-on-abstain at a threshold probability of 0.10 was "
  "{{sbmn_srhpoor_full_NB_refer_01}} against {{spmn_srhpoor_lrni_NB_01}} for regression without indicators at its "
  "frozen threshold.")
tab("tableMiss")
tab("tableMiss2")
fig("fig_missing.png", "Fig. 7. Missing-data stress tests on the internal test set (12 predictors other than age and sex "
    "masked completely at random, 10 repetitions). Left: AUROC of each model's point score. Right: percentage of "
    "participants who receive a decision when the model abstains on uncertain records.", 6.5)
p("*Acquisition.* Table 10 reports Algorithm 2. Among records on which the model abstained under 30% masking, it "
  "requested {{acq2_m30_bound_mean}} inputs per abstention on average, against {{acq2_m30_fixed_mean}} for the fixed "
  "order (paired difference {{acq2_m30_d_fixed}}, {{acq2_m30_d_fixed_ci}}), {{acq2_m30_cost_mean}} for the cost order "
  "({{acq2_m30_d_cost}}, {{acq2_m30_d_cost_ci}}) and {{acq2_m30_random_mean}} for a random order. It needed fewer inputs "
  "than the fixed order for {{acq2_m30_fewer_fixed}}% of records and more for {{acq2_m30_more_fixed}}%. Results were "
  "similar under 10% masking ({{acq2_m10_d_fixed}} versus the fixed order, {{acq2_m10_d_fixed_ci}}) and in the "
  "temporal cohort ({{acq2_t30_d_fixed}}, {{acq2_t30_d_fixed_ci}}). The gain over a model-derived fixed order is small; "
  "the gain over the order in which records are usually consulted, about two-thirds of an input per abstention, is "
  "larger. After resolution, sensitivity among the abstained positives was {{acq_m30_bound_sens}}% and specificity "
  "among the abstained negatives {{acq_m30_bound_spec}}%; these are the decisions the model would make with complete "
  "information on the same, difficult records.")
tab("tableAcq")

h2("3.7. Explanation fidelity and interpretability")
p("For the {{fid_n}} internal test participants with complete inputs, the probability recomputed from the serialised "
  "trace by independent code differed from the model output by at most {{fid_max}}, i.e. machine precision. As noted "
  "in Section 2.11, any generalised linear model has such a decomposition. The comparison with logistic regression "
  "is therefore one of content. The proposed model has {{full_terms}} terms (intercept and {{n_rules}} named concepts), "
  "of which a median of {{full_active_med}} are non-zero for a given person (90th percentile {{full_active_p90}}); "
  "logistic regression with indicators has {{lr_terms}} non-zero terms, all active for everyone, including "
  "{{lr_n_ind}} missing-indicator coefficients without a clinical reading, and linear terms that cannot represent the "
  "U-shaped associations of BMI and sleep; GA²M has {{ebm_terms}} shape functions, {{ebm_n_pairs}} of them pairwise. "
  "No coefficient of either regression model contradicted a prior. Fig. 8 shows the trace of a {{ex_age}}-year-old "
  "{{ex_sex}} with poor self-rated health, {{ex_nchronic}} chronic conditions, diabetes, hypertension, an "
  "income-to-poverty ratio of {{ex_pir}} and {{ex_sleep}} h of workday sleep: poor health and multimorbidity dominate, "
  "long sleep and previous marriage add, older age and male sex subtract, giving a raw probability of {{ex_p}} against "
  "a threshold of {{thr_full}} on the same scale. Diabetes contributes {{ex_diab}} log-odds. For a {{exm_age}}-year-old "
  "participant missing {{exm_missing}}, the bounds were [{{exm_lo}}, {{exm_hi}}] around a point score of {{exm_pt}}; "
  "the lower bound exceeds the threshold, so the system refers without the missing inputs and the trace states that "
  "they could not have changed the decision.")
fig("fig_explanation.png", "Fig. 8. Trace of one internal-test participant with PHQ-9 ≥ 10. Bars show each rule's "
    "contribution to the log-odds (β_r × firing strength), labelled with its signed value (+ raises, − lowers); colour "
    "repeats the sign. The intercept is not drawn as a bar; the title gives the intercept, the sum of the bars and the "
    "resulting log-odds and probability.", 5.8)

h2("3.8. Sensitivity and subgroup analyses")
p("Removing sleep variables changed AUROC little ({{sens_nosleep_full_int}} internal, {{sens_nosleep_full_tmp}} "
  "temporal). Removing self-rated health had the largest effect ({{sens_nosrh_full_int}} and {{sens_nosrh_full_tmp}}; "
  "logistic regression {{sens_nosrh_lr_int}} and {{sens_nosrh_lr_tmp}}). Adding the two sleep items dropped in "
  "2021–2023 raised internal AUROC to {{sens_ext_full_int}} (logistic regression {{sens_ext_lr_int}}), consistent with "
  "longitudinal evidence linking insomnia and depression [@baglioni2011], although both overlap in content with PHQ-9 "
  "items. Refitting with age- and sex-adjusted priors left the model unchanged; with "
  "mutually adjusted priors its AUROC was {{mvpri_full_int}}, {{mvpri_full_tmp}} and {{mvpri_full_ext}}. Among age "
  "bands, discrimination was lowest in adults aged 18–39 in every set (external AUROC {{sub_ext_Age1839_full}}, "
  "logistic regression {{sub_ext_Age1839_lr}}). In 2021–2023 discrimination among Mexican American participants was "
  "low for all models ({{sub_tmp_MexicanAmerican_full}}, {{sub_tmp_MexicanAmerican_lr}}, "
  "{{sub_tmp_MexicanAmerican_gbt}}; {{sub_tmp_MexicanAmerican_pos}} positive cases), and under-prediction in that "
  "group was {{sub_oe_desc}} (O/E {{oe_tmp_MexicanAmerican_full}} versus {{oe_tmp_all_full}} overall). Subgroup "
  "estimates and O/E are in Table A2.")

# ============================================================================================================
h1("4. Discussion")
h2("4.1. Principal findings")
p("For RQ1, the proposed model ranked people about as well as linear logistic regression in all three periods and "
  "slightly less well than spline regression in every period and than GA²M on 2017–2023 data. The primary comparison favoured it only because the comparator "
  "handled a structurally absent variable with an indicator; against regression without indicators there was no "
  "difference. Calibration in earlier cycles held for every regression model without indicators and failed for "
  "every model that learnt a missing-value route. For RQ2, rule structure had the largest effect: conjunctive rule "
  "bases and normalised TSK aggregation transported worse. Fuzzy memberships gave a small, consistent advantage over "
  "crisp ones; knowledge-based cut-points and sign constraints did not change ranking. For RQ3, exact bounds "
  "separated decisions that hold for every admissible value of the missing inputs from those that do not. They "
  "decided more records than linear bounds when a variable was absent for everyone or missing preferentially in "
  "high-risk records, fewer when the missing variables entered the linear model with small slopes, and they did not "
  "find more cases than imputation at equal workload.")

h2("4.2. What the knowledge schema contributes, and what it does not")
p("The selected model is, mathematically, a sign-constrained piecewise-linear generalised additive model with knots "
  "at the schema's cut-points, and its discrimination is that of the regression family. We state this plainly because "
  "it relocates the contribution. The first submission attributed gains to the ontology; the revised analyses do not "
  "support that, and we have withdrawn the claim. Removing the schema as a whole, or any of its parts, left ranking "
  "within noise. The sign priors agreed with every unconstrained estimate of the main-effects model. They mattered "
  "only for the conjunctive version 1, where they pruned incoherent conjunctions; in that setting univariable priors "
  "led to better external discrimination than mutually adjusted ones, which suggests that a prior used to filter a "
  "search need not be the adjusted effect that a clinician would quote.")
p("What the schema does contribute is the information that Algorithms 1 and 2 need and that a fitted model does not "
  "carry: the admissible set of each variable and, through saturating memberships, bounded terms. Bounded terms are "
  "what make the external bounds narrow; a spline model with constant tails at the same knots would share this "
  "property, so the advantage belongs to bounded basis functions rather than to the schema itself. A second, "
  "practical contribution is auditability. Each of the {{n_rules}} terms is a named concept with a declared cut-point "
  "source and prior, released as a machine-readable file that the code loads, so a reader can check a coefficient "
  "against its declaration. The declarations have limits that the audit exposes: priors were estimated as concept "
  "versus all other values while coefficients are contrasts with a reference concept. Re-estimating the priors for "
  "the fitted contrast changed {{refpri_n_changed}} directions ({{refpri_list}}) and moved no coefficient by more than "
  "{{refpri_maxbeta}}. This is in line with interpretability criteria for fuzzy systems [@alonso2015], but it is a "
  "property of documentation, not of accuracy.")

h2("4.3. Missing data, abstention and acquisition")
p("Imputation and missing indicators produce a number for everyone and hide how much it depends on what was not "
  "observed. Bounds do not depend on the missingness mechanism: a decision that holds for every admissible value "
  "holds whatever the reason a value is missing. This is their main advantage, and the external cohort and the "
  "non-random masking scenarios show it in practice. It is a guarantee about individual decisions, not a gain in "
  "classification: at equal workload the bounds-based policy found as many cases as imputation, not more, and "
  "refer-on-abstain refers many people when much is missing. Algorithm 2 reduces that cost by asking for the input "
  "most likely to settle a decision; its advantage over a model-derived fixed order is small and over a "
  "record-consultation order moderate. Credal classifiers [@zaffalon2002,@corani2008] apply the same conservative "
  "logic to probabilistic classifiers, pattern submodels [@fletchermercaldo2020] avoid imputation by fitting one model "
  "per pattern, and learned acquisition policies [@shim2018] optimise the order of questions with more data than "
  "ours. A comparison with these methods on the same data is outside the scope of this study. Algorithm 1 applies "
  "to any additive model with piecewise-linear terms and at most pairwise interactions, including GA²M shape "
  "functions, and Algorithm 2 to any model with exact bounds.")

h2("4.4. Transport across survey periods")
p("All models under-predicted in 2021–2023 by a similar factor, and a single intercept shift repaired it. NHANES "
  "documentation records several changes after the pandemic suspension that plausibly contributed: the depression "
  "screener moved from an interviewer-administered computer-assisted interview to audio computer-assisted "
  "self-interview [@nhanes_dpq_l], a mode change known to increase reporting of sensitive symptoms [@tourangeau2007]; "
  "oversampling by race, Hispanic origin and income was discontinued [@nhanes_l_overview]; and unweighted interview "
  "and examination response rates (all ages, including the screener response) fell to 34.5% and 25.6% from 51.0% and "
  "46.9% in 2017–March 2020 [@nhanes_rr_2021,@nhanes_rr_2017]. We cannot separate these from a real rise in prevalence "
  "[@brody2025], but the self-administration change alone is reason enough to expect a model trained on "
  "interviewer-administered data to under-predict. The loss of discrimination in 2021–2023 was shared by every model "
  "class and is consistent with the same changes. Any deployment would require local recalibration and monitoring "
  "regardless of model type [@vancalster2019].")

h2("4.5. Practical relevance")
p("The model's use case is narrower than the first submission implied. With the full predictor set, which includes "
  "self-rated health, sleep and sedentary time asked in an interview, it flagged about half of adults to find about "
  "five in six with elevated symptoms; the PHQ-2 flagged about one in ten and found about two in three. With "
  "record-typical predictors alone, discrimination in the later period was modest. The model may therefore be useful "
  "for prioritising whom to invite when a survey or registry already holds self-rated health and lifestyle items but "
  "no symptom questions; it is not a substitute for asking the PHQ-2. Whether a prioritisation step of this kind "
  "saves effort in a real service, and whether its explanations help staff decide whom to contact, has not been "
  "evaluated.")

h2("4.6. Limitations")
p("The revised model was designed after version-1 results on the internal and temporal sets had been seen, and the "
  "protocol that fixed it was not registered; the only record that it preceded the external download is a pair of "
  "file modification times, which cannot be verified independently. All comparators and analyses added during "
  "revision were specified after the original results were known. The external cohort is earlier cycles of the same "
  "survey, not a different health system. The outcome is a questionnaire threshold, not a diagnostic interview "
  "[@levis2019]. Predictors and outcome are cross-sectional; a trace explains the model, not the person's depression. "
  "Most predictors are survey self-reports. Algorithm 1 is exact for components with at most two continuous missing "
  "variables and uses an outer bound otherwise; the spline comparator's bounds use a dense grid. Algorithm 2 uses the "
  "marginal development distribution of each input and was evaluated by revealing masked values, which assumes that "
  "a requested input can be obtained. Credal classifiers, pattern submodels, learned acquisition policies, rule "
  "ensembles and integer scoring systems were discussed but not fitted. No clinician evaluated the explanations. "
  "Subgroup analyses were descriptive and underpowered for several groups.")

h1("5. Conclusions")
p("A fuzzy additive model constrained by a knowledge schema ranked adults by the likelihood of elevated depressive "
  "symptoms about as well as linear logistic regression across three survey periods and slightly less well than "
  "spline regression; its knowledge-related elements did not improve discrimination. Its declared admissible values "
  "allow exact probability bounds that show which decisions hold whatever the missing inputs are, including when a "
  "variable is absent for everyone, and a bound-guided choice of which input to ask for next. At equal workload these "
  "bounds did not find more cases than imputation. Two symptom questions remain far more efficient than any "
  "non-symptom model where they can be asked.")

# ============================================================================================================
h1("Declarations")
p("*Ethics approval and consent:* NHANES protocols were approved by the NCHS Research Ethics Review Board and all "
  "participants provided informed consent. This secondary analysis of public de-identified data did not require "
  "further approval under the policy of Saigon International University.")
p("*Data availability:* NHANES data are publicly available from the National Center for Health Statistics. Components "
  "used: DEMO, DPQ, SLQ, BMX, HUQ, SMQ, PAQ, MCQ, DIQ and BPQ for 2005–2006 to 2015–2016, 2017–March 2020 and "
  "August 2021–August 2023.")
p("*Code availability:* The full pipeline (cohort construction, knowledge schema in JSON-LD, model, Algorithms 1–2, "
  "comparators, experiments, figures, the analysis protocol with its addendum, and the statistics file that "
  "populates this manuscript) is publicly available at {{repo_url}} (release v2.0.0) and archived at Zenodo "
  "({{zenodo_doi}}). Re-run from the public NHANES files, the archived release reproduces every number reported here except computing times.")
p("*Funding:* This research did not receive any specific grant from funding agencies in the public, commercial, or "
  "not-for-profit sectors.")
p("*Declaration of competing interest:* The authors declare that they have no known competing financial interests or "
  "personal relationships that could have appeared to influence the work reported in this paper.")
p("*CRediT authorship contribution statement:* Truong Hai Bang: Conceptualization, Methodology, Software, Formal "
  "analysis, Visualization, Writing – original draft, Writing – review & editing, Supervision. Van An Phan Thi: "
  "Conceptualization, Data curation, Investigation, Validation, Writing – review & editing.")
p("*Declaration of generative AI and AI-assisted technologies in the manuscript preparation process:* During the "
  "preparation of this work the authors used Claude (Anthropic) to draft and edit manuscript text and to check "
  "bibliographic records; its use in writing the analysis code is described in Section 2.14. After using this tool, "
  "the authors reviewed and edited the content as needed and take full responsibility for the content of the "
  "published article.")

h1("References")
B.append({"t": "refs"})
pb()
h1("Appendix A. Additional tables")
for k in ("tableS1", "tableS2", "tableS3", "tableS4", "tableS5"):
    tab(k)
h1("Appendix B. TRIPOD+AI checklist")
tab("tableTRIPOD")


# ============================================================================================================
def fill(text):
    return re.sub(r"\{\{(\w+)\}\}", lambda m: V(m.group(1)), text)


order = []
def cites(text):
    def rep(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(",")]
        nums = []
        for k in keys:
            if k not in REFS: raise KeyError("unknown reference " + k)
            if k not in order: order.append(k)
            nums.append(order.index(k) + 1)
        nums = sorted(nums); parts, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
            parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
            i = j + 1
        return "[" + ", ".join(parts) + "]"
    return re.sub(r"\[(@[^\]]+)\]", rep, text)


if __name__ == "__main__":
    from tables_r2 import tables
    TT = tables(V)
    out = []
    for b in B:
        b = dict(b)
        if b["t"] in TT:
            T = TT[b["t"]]
            out.append({"t": "table", **T, "cap": cites(fill(T["cap"])), "note": cites(fill(T.get("note", ""))),
                        "rows": [[cites(fill(str(c))) for c in r] for r in T["rows"]]}); continue
        for k in ("x", "cap"):
            if k in b:
                b[k] = [cites(fill(s)) for s in b[k]] if isinstance(b[k], list) else cites(fill(b[k]))
        out.append(b)
    out = json.loads(json.dumps(out, ensure_ascii=False).replace("p = <0.001", "p < 0.001"))
    for i, b in enumerate(out):
        if b["t"] == "refs":
            out[i] = {"t": "refs", "x": [f"[{n + 1}] {REFS[k]}" for n, k in enumerate(order)]}
    unused = [k for k in REFS if k not in order]
    json.dump(out, open(R + "manuscript_r2.json", "w"), ensure_ascii=False, indent=0)
    print("blocks", len(out), "refs cited", len(order), "unused", unused)
