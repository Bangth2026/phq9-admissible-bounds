# Methods of version 4 (imported by manuscript_r4.py; uses its block helpers).
from ms4_helpers import *  # noqa

h1("2. Materials and methods")
h2("2.1. Study design, intended use and reporting")
p("This is a retrospective study that develops prediction models and compares rules for abstaining when inputs are "
  "missing, reported following TRIPOD+AI [@collins2024] (checklist in Appendix B). Both validation cohorts come from "
  "later or earlier periods of the same survey, so in TRIPOD terms both are temporal validations; we call them "
  "later-period (2021–2023) and earlier-period (2005–2014) validation. The intended user is a service or analyst "
  "holding survey or record information without symptom items; the intended action is to recommend completing the "
  "PHQ-9 or an equivalent assessment. Where a person can be asked two questions, the PHQ-2 is the appropriate first "
  "step, and we report it as a reference point (Section 2.11). The models do not diagnose major depressive disorder, "
  "do not estimate future incidence and are not tools for assessing self-harm risk. Predictors and outcome are "
  "measured in the same survey, so the target is concurrent elevated symptoms. All estimates are unweighted and refer "
  "to study participants unless stated otherwise.")

h2("2.2. Data sources")
p("NHANES is a continuous cross-sectional survey of the non-institutionalised United States population. Development "
  "data came from the 2017–March 2020 pre-pandemic release, which combines the 2017–2018 cycle with data collected "
  "until fieldwork was suspended in March 2020 [@akinbami2022]. Later-period validation used August 2021–August 2023, "
  "and earlier-period validation the five two-year cycles 2005–2006 to 2013–2014. The 2015–2016 cycle was used only to "
  "estimate direction priors (Section 2.5). Components were merged by the respondent identifier SEQN: demographics, "
  "depression screener, sleep, body measures, hospital utilisation, smoking, physical activity, medical conditions, "
  "diabetes and blood pressure questionnaires.")

h2("2.3. Participants, outcome and sample size")
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
  "(seed 2026), into a development set (n = {{n_dev}}, {{rl_events_dev}} events) and an internal test set "
  "(n = {{n_int}}).")
p("All eligible participants were used. We checked the development set against the minimum sample size for a binary "
  "prediction model [@riley2020], assuming {{rl_P}} predictor parameters, outcome prevalence {{rl_phi}} and an "
  "anticipated C statistic of 0.78, which corresponds to a Cox–Snell R² of {{rl_R2}}. Shrinkage of at most 10% required "
  "{{rl_n1}} participants, a difference of at most 0.05 between apparent and adjusted Nagelkerke R² required "
  "{{rl_n2}}, and estimating the overall risk within ±0.05 required {{rl_n3}}. The largest, {{rl_n}} participants with "
  "{{rl_events}} events ({{rl_epp}} events per parameter), is below the {{n_dev}} participants and {{rl_events_dev}} "
  "events available ({{rl_epp_dev}} per parameter).")
fig("fig_flow.png", "Fig. 1. Participant flow. The 2015–2016 cycle was used only to estimate the direction priors stored "
    "in the knowledge schema. Models and thresholds were frozen before evaluation on the later- and earlier-period cohorts.", 6.3)

h2("2.4. Predictors and harmonisation")
p("Candidate predictors had to be plausibly available before a symptom questionnaire, clearly defined, not derived "
  "from the outcome, and fielded with the same coding in 2017–2020 and 2021–2023. Fourteen variables qualified: age, "
  "sex, family income-to-poverty ratio, education (five levels), marital status (married/partnered, previously "
  "married, never married), body mass index, usual sleep duration on workdays and on free days, self-rated general "
  "health (five levels), current smoking, doctor-told diabetes, doctor-told hypertension, daily sedentary time, and a "
  "count of ten doctor-told chronic conditions (asthma, arthritis, congestive heart failure, coronary heart disease, "
  "angina, myocardial infarction, stroke, liver condition, chronic obstructive pulmonary disease, cancer). All are "
  "survey items; self-rated health, sleep duration and sedentary time are interview questions, and the chronic "
  "conditions are self-reported 'doctor-told' answers rather than coded diagnoses (Table 2 marks which variables are "
  "usually held as structured record data).")
p("Two predictors overlap in content with the PHQ-9. Sleep duration touches the sleep item but is asked in a separate "
  "questionnaire. Self-rated general health is the strongest predictor and plausibly reflects fatigue, sleep and "
  "appetite problems that the PHQ-9 asks about directly; a person who rates their health as poor may be reporting "
  "some of the same experience. Both are therefore closer to the outcome than demographic or record items, and we "
  "report models without them (Section 3.7). Contact with a mental health professional was not used, because it is "
  "close to the outcome. Race and ethnicity served only for subgroup evaluation. Two sleep items with strong "
  "associations in the earlier cycle (SLQ050, SLQ120) were not fielded in 2021–2023 and were examined only in a "
  "development-period sensitivity analysis. In the development set {{any_missing_dev}}% of participants had at least "
  "one missing predictor, mostly income-to-poverty ratio ({{pir_missing_dev}}%).")
p("The 2005–2014 cycles required three harmonisation decisions, written into the analysis protocol (Section 2.12) "
  "before the files were downloaded: workday sleep came from SLD010H, recorded in whole hours; free-day sleep was not "
  "collected and was missing for every participant; and the chronic-condition count used the nine items available "
  "before the COPD item was introduced in 2013–2014. Sedentary time was not collected in 2005–2006. Marital status "
  "was mapped to the three categories above, and race and ethnicity before 2011 came from RIDRETH1, which has no "
  "separate Asian category. Variables absent in a cycle were left missing for every model.")

h2("2.5. Base models")
p("Abstention rules need a base model that produces a probability. We used three, chosen to separate the effect of "
  "the rule from that of the model: a fuzzy additive model whose terms are bounded, sign-constrained linear logistic "
  "regression, whose terms are unbounded, and logistic regression with restricted cubic splines, the best-ranking "
  "model in the later and earlier periods.")
p("*Knowledge schema and fuzzy additive model.* A knowledge schema declares, for each of the 14 variables, its type, "
  "unit, NHANES source and admissible set (category sets for discrete variables, intervals for continuous ones; "
  "Table 2), together with {{n_concepts}} linguistic concepts (fuzzy sets [@zadeh1965]), the source of each cut-point and a direction prior. It "
  "is a typed, versioned declaration rather than an ontology with a reasoner [@bobillo2016], and it is released as a "
  "JSON-LD 1.1 file [@jsonld2020] in which concepts belong to one SKOS concept scheme [@skos2009]; the model code loads "
  "this file, so the released artefact is the one that produced the results (Fig. 2). Continuous variables are "
  "partitioned into trapezoidal concepts forming Ruspini partitions [@ruspini1969]. For break points b_1 < … < b_m, "
  "half-width h and admissible interval [l, u], the inner concepts have the trapezoidal membership")
eq("μ(x; a, b, c, d) = max(0, min((x − a)/(b − a), 1, (d − x)/(d − c))),  a = b_{k−1} − h, b = b_{k−1} + h, c = b_k − h, d = b_k + h,", 2)
p("and the first and last concepts are shoulders that stay at 1 up to the range ends: μ_1(x) = min(1, max(0, "
  "(b_1 + h − x)/(2h))) and μ_{m+1}(x) = min(1, max(0, (x − b_m + h)/(2h))). Memberships of a variable's concepts "
  "therefore sum to one everywhere in [l, u], and every membership is constant beyond the outermost corners. "
  "Cut-points came from external conventions: WHO body-mass-index categories (18.5, 25, 30 kg/m²; h = 1) [@who2000], "
  "the consensus recommendation of at least 7 h of sleep (short < 6.5 h, long > 9.5 h; h = 0.5) [@watson2015], 130% "
  "and 350% of the federal poverty level (h = 0.3), age bands at 35 and 60 years (h = 5) and sedentary-time bands at 4 "
  "and 8 h/day (h = 1). Ordinal and categorical variables have crisp set memberships (e.g. FairHealth, PoorHealth, "
  "Multimorbidity for ≥ 2 conditions).")
fig("fig_framework.png", "Fig. 2. Architecture of the prescreening system. The knowledge schema (orange) declares "
    "variables, admissible sets, concepts, cut-point sources and sign priors; the worst-case bounds (Algorithms 1 and 2) "
    "and the acquisition rule (Algorithm 3) use these declarations when inputs are missing.", 6.6)
p("A rule r has an antecedent A_r of one or two concepts and fires with strength μ_r(x) = min_{k ∈ A_r} μ_k(x_k). "
  "Rules combine additively on the log-odds scale,")
eq("η(x) = logit p(x) = β_0 + Σ_{r=1}^{R} β_r μ_r(x),   subject to s_r β_r ≥ 0,", 3)
p("a zero-order Takagi–Sugeno system [@takagi1985] with log-odds consequents, where s_r ∈ {−1, 0, +1} is the direction "
  "prior of the rule's concepts. Priors were estimated in NHANES 2015–2016 (n = {{flow_pri_analytic}}, "
  "{{flow_pri_positives}} positive) as the sign of the risk ratio of PHQ-9 ≥ 10 for membership ≥ 0.5 versus < 0.5 when "
  "its 95% confidence interval excluded 1, and 0 otherwise ({{n_pri_est_pos}} positive, {{n_pri_est_neg}} negative, "
  "{{n_pri_est_zero}} neutral; Table A1). Free-day sleep was not asked in 2015–2016, so its {{n_pri_inh}} concepts "
  "inherited the direction of the workday concepts; these priors were never estimated. The selected structure contains "
  "every single-concept rule except one reference concept per variable; two-concept rules could be added greedily, "
  "but 5-fold cross-validated AUROC in the development set ({{cvmain_K0}}, {{cvmain_K8}}, {{cvmain_K16}}, {{cvmain_K32}} "
  "and {{cvmain_K64}} for K = 0, 8, 16, 32, 64 added rules) selected none under the pre-specified parsimony rule "
  "(smallest K within 0.005 of the best). The model therefore has {{n_rules}} single-concept rules and an intercept: a "
  "sign-constrained piecewise-linear additive model with knots at the schema's cut-points and constant tails. "
  "Coefficients were fitted by L-BFGS-B on the mean log-likelihood plus λ Σ_r β_r² (λ = 10⁻³), with the sign "
  "constraints as box bounds. For the point score a missing membership was replaced by its development-sample mean.")
p("*Regression base models and other comparators.* Sign-constrained logistic regression used the raw inputs with "
  "median imputation, the direction priors imposed on main effects (age, BMI and sleep unconstrained because their "
  "associations are U-shaped) and a cross-validated L2 penalty (λ = {{slr_lambda}}). Logistic regression with "
  "restricted cubic splines [@harrell2015] used four knots at the 5th, 35th, 65th and 95th development percentiles "
  "for age, BMI and both sleep durations, other inputs linear, and median imputation. Discrimination was also "
  "compared with logistic regression with missing-value indicators (the comparator named in the protocol), without "
  "indicators, and after multiple imputation by chained equations [@vanbuuren2011]; gradient-boosted trees "
  "[@friedman2001,@pedregosa2011] with and without monotone constraints; and an explainable boosting machine (GA²M) "
  "[@lou2013,@nori2019]. Version-level variants and single-element ablations of the fuzzy model (conjunctive version 1, "
  "16 added pair rules, normalised TSK aggregation, no sign constraints, crisp memberships, quantile memberships "
  "without the schema, median imputation) are described in Appendix A.")

h2("2.6. Abstention rules")
p("Every rule returns one of three outputs for a person: refer for questionnaire completion, do not refer, or abstain. "
  "Abstentions were never counted as negative predictions; we report decisions among decided participants and, as "
  "the conservative policy for prescreening, refer-on-abstain, in which every abstention is referred [@chow1970]. Each "
  "rule uses its base model's threshold t, fixed on development data (Section 2.11) and applied on the raw probability "
  "scale σ(η), before recalibration; for the fuzzy model t = {{thr_full}}. The rules differ in what they assume about "
  "the missing inputs.")
p("*Worst-case bounds* assume nothing beyond the admissible set. Let O and M be the observed and missing variables of "
  "a person and 𝒳_M the product of the admissible sets of the missing variables. The rule computes")
eq("p_L = min σ(η(x_O, x_M)),   p_U = max σ(η(x_O, x_M)),   over x_M ∈ 𝒳_M,", 4)
p("and refers if p_L ≥ t, does not refer if p_U < t, and abstains otherwise. A decision that is made therefore holds "
  "for every admissible value of the missing inputs. The guarantee concerns the model's own output and the frozen "
  "threshold, not the outcome: if the model is miscalibrated in a population, as all models were in 2021–2023, a "
  "decision that holds for every completion can still be wrong, and it is wrong in the same way as the model's "
  "decision on a complete record. Section 2.7 describes how the bounds are computed for each base model.")
p("*Imputation spread* uses the conditional distribution of the missing inputs given the observed ones, learnt in the "
  "development data. The chained-equation model fitted for the comparator produced 20 completed copies of each record "
  "(four posterior draws from each of five imputation models; discrete inputs rounded and all inputs clipped to their "
  "admissible sets), the base model scored each copy, and the rule refers if at least 90% of the 20 probabilities "
  "exceed t, does not refer if at most 10% do, and abstains otherwise. Unlike the original analysis, in which the rule "
  "ran on the chained-equation logistic model, the imputations here pass through the same base model as the bounds, so "
  "the comparison holds the model fixed.")
p("*Pattern submodels* [@fletchermercaldo2020] avoid assumptions about the missing values by not using them: for each "
  "missingness pattern in the evaluation data, the base model was refitted on the development set without the "
  "missing variables, with its own threshold chosen on out-of-fold predictions, and applied to the records with that "
  "pattern. Every record therefore receives a decision. Because the number of patterns grows quickly when many inputs "
  "are masked at random, pattern submodels were evaluated under natural, non-random, outcome-dependent and shifted "
  "missingness, where patterns are few, but not under 30% MCAR. *Complete-case abstention* decides only records with "
  "every input observed.")
p("We did not use conformal prediction as a baseline. For a binary outcome, split-conformal prediction with a "
  "score-based nonconformity measure returns the set {0, 1}, the analogue of abstention, whenever the predicted "
  "probability falls in a band set by the target coverage level [@geifman2017]; the band does not depend on which "
  "inputs are missing. Conformal methods designed for missing inputs calibrate the band within missingness patterns "
  "and guarantee coverage of the label set marginally or conditionally on the pattern, under exchangeability between "
  "calibration and test data [@zaffran2023]. That guarantee concerns a different quantity from ours (how often the "
  "true label lies in the set, rather than whether the decision depends on the missing values), and it is lost under "
  "the period shift and non-random missingness studied here. Work on imputation for prediction shows that, with "
  "enough data, imputation followed by a flexible model can be consistent even when values are not missing at random "
  "[@lemorvan2021]; imputation spread is the abstention counterpart of that approach.")

h2("2.7. Computing worst-case bounds")
p("Because σ is monotone it suffices to bound η. For the regression base models the bound is direct: in "
  "sign-constrained linear regression each missing input moves η linearly, so its extremes lie at the ends of the "
  "admissible interval (or at one of the three marital categories); in spline regression each missing input was "
  "optimised separately over a 2,001-point grid. For the fuzzy additive model the bounds depend on the rule structure.")
p("*Main-effects rule bases (Algorithm 1).* When every rule involves one variable, η is a sum of univariate "
  "functions g_v(x_v) = Σ_{r on v} β_r μ_r(x_v), so its extremes over 𝒳_M are the sums of each missing variable's "
  "extremes. Each g_v is continuous and piecewise linear with breakpoints at the trapezoid corners of Eq. (2), and "
  "constant beyond the outermost corners, so its extremes on [l, u] lie in the finite set of range ends and corners "
  "inside the range (Box 1). The constant tails also mean that the width of the bound does not grow with the admissible "
  "range once the range covers the outermost corners, whereas a linear term, or a spline with linear tails, gives a "
  "bound that grows with the range. The point score, the development-mean imputation of each membership, always lies "
  "in [p_L, p_U] for a main-effects model, because η is linear in the memberships and the mean membership vector is a "
  "convex combination of admissible vectors.")
tab("tableAlg")
p("*Rule bases with conjunctions (Algorithm 2).* The conjunctive version 1 of the system, retained as a comparator, "
  "and the variant with 16 added pair rules contain two-concept rules min(μ_k(x_k), μ_m(x_m)), which couple variables. "
  "Missing variables are grouped into connected components (two missing variables are connected if a rule contains "
  "both); η is a constant plus a sum of functions of the components, so its extremes are sums of per-component "
  "extremes. For each component, Algorithm 2 (Box 2) enumerates the categories of its discrete variables and, for each "
  "combination, a finite candidate set for its continuous variables. The candidate sets are exact by the following "
  "two statements.")
p("*Statement 1 (one continuous missing variable).* Let x ∈ [l, u] be the only continuous missing variable of a "
  "component, with its discrete variables fixed. Then g(x) = Σ_r β_r min(μ_{k(r)}(x), c_r), where c_r ∈ [0, 1] is the "
  "observed partner membership of rule r (c_r = 1 for single-concept rules). Each μ_k is continuous and piecewise "
  "linear with breakpoints at its trapezoid corners, and min(μ_k(x), c_r) adds breakpoints only where μ_k(x) = c_r, "
  "i.e. at x = a + c_r(b − a) and x = d − c_r(d − c). Hence g is continuous and piecewise linear on [l, u] with "
  "breakpoints in the finite set 𝒞 = {l, u} ∪ {corners} ∪ {solutions of μ_k(x) = c_r}. A continuous piecewise-linear "
  "function on a closed interval attains its extremes at an endpoint or a breakpoint, so enumerating 𝒞 is exact. □")
p("*Statement 2 (two continuous missing variables).* Let (x, z) ∈ [l_x, u_x] × [l_z, u_z] be connected through pair "
  "rules min(μ_k(x), μ_m(z)). Partition each axis at its breakpoints from Statement 1. On each resulting rectangle "
  "every membership is affine, so each pair term equals one of two affine functions, switching along the line "
  "μ_k(x) = μ_m(z). These lines cut the rectangle into convex polygons on which η is affine; η is continuous, so its "
  "extremes over the rectangle lie at polygon vertices. Every vertex is a rectangle corner, an intersection of a "
  "switching line with a rectangle edge, or an intersection of two switching lines inside the rectangle, and "
  "enumerating these points on all rectangles is exact. □")
p("Components with more than two continuous missing variables fall back to the box bound obtained by letting each "
  "missing membership range over [0, 1]; this is a valid outer bound, not an exact one, and the trace flags it. For a "
  "component with m_c categorical combinations, n_b breakpoints per axis and q pair rules, the cost is O(m_c n_b) "
  "evaluations for one continuous variable and O(m_c n_b² q²) for two. Algorithm 2 was checked against a 41-point grid "
  "augmented with trapezoid corners and against dense brute force (Section 3.5).")
tab("tableAlgGen")

h2("2.8. Missingness scenarios")
p("*Natural missingness.* All models were applied to the three validation cohorts as observed. In 2005–2014 "
  "free-day sleep was not collected and sedentary time was not collected in 2005–2006; income-to-poverty ratio is "
  "missing for 7.5–13.3% of participants in every cohort.")
p("*Simulated missingness on the internal test set* (10 repetitions each, the same masks for every model and rule): "
  "(i) MCAR, each of the 12 predictors other than age and sex masked with probability 0.3 (other rates and the "
  "removal of whole domains are in Appendix A); (ii) MNAR, income-to-poverty ratio masked with probability 0.6 if "
  "below 1.3 and 0.1 otherwise, self-rated health masked with probability 0.6 if fair or poor and 0.1 otherwise, or "
  "both; (iii) stronger MNAR, the same selections with probabilities 0.9 and 0.05; (iv) outcome-dependent masking, "
  "self-rated health and income-to-poverty ratio each masked with probability 0.6 in participants with PHQ-9 ≥ 10 and "
  "0.1 otherwise.")
p("*Structural absence with a shifted conditional distribution.* Imputation spread draws a missing input from its "
  "conditional distribution given the observed inputs in the development data. To test the rules where that "
  "distribution is wrong for the target population, we removed one input for everyone, as free-day sleep was removed "
  "in 2005–2014, in target samples whose relation between that input and the other inputs departs from the "
  "development data. For self-rated health and for free-day sleep in turn, a linear model fitted on development data "
  "predicted the input from the other 13, and each internal test participant received the standardised residual "
  "z_i of their observed value. A target sample of the same size was drawn with replacement with probabilities "
  "proportional to exp(γ s z_i), where s = +1 for self-rated health (worse health than predicted) and s = −1 for "
  "free-day sleep (shorter sleep than predicted), and γ = 0, 0.5 or 1 (10 samples each). With γ = 1 the mean residual in "
  "the target sample was {{shm_srh_1}} SD for self-rated health and {{shm_we_1}} SD for free-day sleep, so an imputation "
  "model trained on development data systematically imputes healthier values and longer sleep than the truth. The "
  "input was then masked for every participant.")

h2("2.9. Comparison of abstention rules")
p("Rules were compared with the base model held fixed, on four outputs. (1) Coverage, the proportion of participants "
  "decided, split into decisions to refer and decisions not to refer. (2) Classification of decided participants at "
  "matched coverage: each rule ranks participants by its confidence (distance of the bounds from the threshold on the "
  "log-odds scale; distance of the imputation share from 0.5; for pattern submodels and complete cases, distance of "
  "the score from the threshold) and decides the most confident ones up to the coverage reached by worst-case bounds "
  "of the same base model; we report sensitivity, specificity and balanced accuracy among them. (3) The "
  "risk–coverage curve [@geifman2017], balanced error among the decided against the proportion decided over coverage "
  "0.05–1, summarised by the area under it (AURC; lower is better) with a percentile bootstrap interval (500 resamples "
  "of participants in the natural cohorts, 300 in 2005–2014 and in the first repetition of each simulated scenario) "
  "and a paired interval for the difference to worst-case bounds. (4) Decision consequences under refer-on-abstain, "
  "per 1,000 participants: participants referred, cases missed and questionnaires avoided (participants not "
  "referred), and net benefit [@vickers2006] at threshold probabilities 0.05 and 0.10. Balanced accuracy at matched "
  "coverage depends on which participants each rule chooses to decide, so the decision-consequence measures are the "
  "ones that correspond to the prescreening costs of Section 2.1.")

h2("2.10. Bound-guided input acquisition")
p("An abstention means that at least one missing input could change the decision. The acquisition rule (Algorithm 3, "
  "Box 3) asks for one missing input at a time: for each candidate v it estimates the probability ρ_{v} that observing "
  "v alone resolves the decision, evaluating the bounds at support points of the development-set distribution of v "
  "(all categories, or 25 quantile points for continuous inputs) with the other missing inputs still unknown, "
  "requests the input with the largest ρ_{v}, and stops when the decision is resolved or no requestable input remains. "
  "A cost-weighted variant requests the input with the largest ρ_{v}/c_{v}. We compared it with a fixed order by each "
  "input's contribution range, a cost order that requests structured record items first (diabetes, hypertension, "
  "chronic conditions, BMI, smoking, marital status, education), then income, then interview items, and a random "
  "order, on internal test records with 10% or 30% of inputs masked at random and later-period records with 30% "
  "masked. A requested masked input was revealed with its true value; inputs missing in the original data could not "
  "be requested, so in 2005–2014 free-day sleep can never be requested and acquisition was not evaluated there. "
  "Policies were compared on the number of inputs and on expected item cost per abstention, with illustrative costs "
  "of 1 for a record item, 2 for income and 3 (or 5) for an interview item. Differences are paired by record (2,000 "
  "bootstrap resamples). Every order resolves the same records with the same final decisions.")
tab("tableAlg2")

h2("2.11. Performance measures and statistical analysis")
p("Every model produced out-of-fold predictions in the development set by 5-fold cross-validation. These were used "
  "to fit a Platt recalibration map and to fix the referral threshold as the largest raw score reaching an out-of-fold "
  "sensitivity of at least 0.80. Models were then refitted on the whole development set and applied unchanged to all "
  "cohorts. We report AUROC, AUPRC, sensitivity, specificity, proportion flagged, Brier score, calibration slope and "
  "observed-to-expected ratio (O/E) [@vancalster2019], and net benefit across threshold probabilities 0.02–0.30 "
  "[@vickers2006,@vickers2016]. Confidence intervals came from 1,000 bootstrap resamples of participants (500 for "
  "AUROC in single cycles); AUROC differences used 2,000 paired resamples with percentile intervals and two-sided "
  "bootstrap p-values, which are not reported for differences smaller than 0.001. Because NHANES has a stratified "
  "cluster design, the primary comparison was repeated with a design-based bootstrap that resamples primary sampling "
  "units within strata ({{db_nstrata}} strata, {{db_npsu}} units in 2005–2014). AUROC weighted by interview weights is a "
  "sensitivity analysis. To include training variability, the fuzzy model, its variants and five comparators were "
  "refitted on {{spl_B}} repeated stratified 70/30 splits of the 2017–2020 release with hyperparameters fixed; we report "
  "the mean and SD of each AUROC difference. Recalibration-in-the-large was examined by estimating an intercept shift "
  "on a random half of the 2021–2023 or 2005–2014 cohort and evaluating it on the other half (200 repetitions). "
  "Subgroup AUROC, sensitivity and O/E were computed by sex, age band and race and ethnicity where at least 10 "
  "positive cases were available; because sensitivity differed by sex, we also examined sex-specific thresholds "
  "(target sensitivity ≥ 0.80 in each sex) and sex-specific recalibration-in-the-large.")
p("The PHQ-2 reference used items DPQ010 and DPQ020 with the usual cut-point ≥ 3 [@kroenke2003,@levis2020]. These two "
  "items are part of the nine summed in the reference standard, so their accuracy against PHQ-9 ≥ 10 is inflated by "
  "incorporation bias; against diagnostic interviews, PHQ-2 ≥ 3 had a pooled sensitivity of 0.72 and specificity of 0.85 "
  "in an individual participant data meta-analysis [@levis2020], which is the fairer reference. A record-typical "
  "model used only age, sex, BMI, diabetes, hypertension, chronic-condition count, smoking, education and marital "
  "status.")

h2("2.12. Development history, protocol and post hoc analyses")
p("The fuzzy model above is the second version of the system. Version 1 selected K = 32 rules of any arity by a "
  "greedy procedure with no forced main effects; {{v1_n_pairs}} of its 32 rules were two-concept conjunctions. On "
  "the internal and later-period sets it trailed logistic regression by {{v1old_d_lr_int_abs}} ({{v1old_d_lr_int_ci}}) "
  "and {{v1old_d_lr_tmp_abs}} ({{v1old_d_lr_tmp_ci}}) AUROC. We attributed the gap to the absence of main effects, "
  "tested that diagnosis with development-set cross-validation only, and wrote a protocol that fixed the revised "
  "structure, the K grid and selection rule, the comparators (logistic regression with missing indicators and "
  "gradient-boosted trees) and ablations, the 2005–2014 cohorts and their harmonisation, and the primary comparison: "
  "AUROC in 2005–2014 of the fuzzy model versus logistic regression with missing indicators. According to the "
  "protocol's addendum, the 2005–2014 files were downloaded after the protocol was saved; no independently verifiable "
  "record predates the download, and the earliest verifiable record of the protocol is its deposit with the code "
  "({{zenodo_doi_v3}}). The 2005–2014 comparison is therefore free of earlier results according to our records only. "
  "Every other comparator, every abstention rule other than worst-case bounds, every simulated missingness scenario "
  "other than MCAR, the acquisition rule, the repeated splits and the design-based analysis were specified after "
  "results were known; tables mark them with ‡.")

h2("2.13. Explanation trace, software and use of generative AI")
p("For each person the system exports a JSON trace with the inputs, missing-input flags, each concept's membership and "
  "provenance, each rule's contribution β_r μ_r (or its range when an input is missing), the intercept, the "
  "probability bounds and, for abstentions, the input requested by Algorithm 3. For complete inputs the contributions "
  "add exactly to η, as in any generalised linear model.")
p("Analyses used Python 3.13 with pandas 3.0.5, scikit-learn 1.9.1, SciPy 1.18.1, statsmodels and interpret 0.7.8. "
  "During the preparation of this work the authors used a generative AI tool (Claude, Anthropic) to help structure the "
  "manuscript, write the analysis code and edit its language. The code it helped write covers cohort construction, "
  "models, bounds, the acquisition rule, experiments and figures, and also the three verification tests, so the tests "
  "are not independent of the tool in authorship; their independence is in construction. "
  "tests/test_bounds_bruteforce.py compares the bounds with dense brute-force evaluation of the model on 4,001-point "
  "grids, which shares no code with Algorithms 1 and 2; tests/test_trace_fidelity.py recomputes probabilities from "
  "serialised traces without importing the model; tests/test_cohort_priors.py reproduces the cohort flow and the "
  "direction priors from the public files. The authors reviewed and edited all content, verified every reference and "
  "statistic against its source, and take full responsibility for the publication. The archived release was re-run "
  "from the raw files in a clean directory and reproduced every number in this manuscript except computing times. All "
  "numbers are written by the analysis code to one statistics file and inserted programmatically.")
