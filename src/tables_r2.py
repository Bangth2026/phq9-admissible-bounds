"""Tables of the revised manuscript, built from result files; cells may contain {{key}} placeholders."""
import json, pickle
import pandas as pd
R = "/home/claude/results/"; RV = R + "secondary/"
EHR_FLAG = {"age": "Yes", "female": "Yes", "pir": "Rarely", "educ": "Sometimes", "marital": "Sometimes", "bmi": "Yes",
            "sleep_wd": "No", "sleep_we": "No", "srh": "No", "smoker": "Yes", "diabetes": "Yes (coded)", "hypertension": "Yes (coded)",
            "sedentary": "No", "n_chronic": "Partly (coded)"}


def tables(V):
    sgn = {1: "+", -1: "−", 0: "0"}
    # ---------------------------------------------------------------- Table 1
    t1 = pd.read_csv(R + "table1.csv", index_col=0)
    keep = [r for r in t1.index if r != "Chronic conditions (count), median (IQR)"]
    cols = ["Development", "Internal test", "Temporal (2021-2023)", "External (2005-2014)"]
    cell = lambda x: "" if pd.isna(x) else ("not collected" if "nan" in str(x) else str(x))
    rows = [[r.replace(">=", "≥").replace("kg/m2", "kg/m²")] + [cell(t1.loc[r, c]) for c in cols] for r in keep]
    from experiment import LRModel, GBTModel  # noqa
    M = pickle.load(open(R + "models.pkl", "rb"))
    ch = ["≥ 1 chronic condition, %"] + [f"{100 * (d.n_chronic >= 1).sum() / d.n_chronic.notna().sum():.1f}" for d in (M["dev"], M["test"], M["L"], M["POOL"])]
    rows.insert([r[0] for r in rows].index("Hypertension, %") + 1, ch)
    T1 = dict(cap="Table 1. Characteristics of the development, internal test, temporal and external validation cohorts (unweighted unless stated).",
              head=["Characteristic", "Development (2017–2020)", "Internal test (2017–2020)", "Temporal (2021–2023)", "External (2005–2014)"], rows=rows,
              note="IQR, interquartile range; BMI, body mass index. Percentages of categorical variables use non-missing values as denominator. Weighted prevalence uses NHANES interview weights; it differs between development and internal test because the split was stratified on the unweighted outcome. Free-day sleep was not collected in 2005–2014; workday sleep was recorded in whole hours.",
              w=[2.2, 1.1, 1.1, 1.1, 1.1], small=True)
    # ---------------------------------------------------------------- Table 2
    pri = pd.read_csv(R + "priors_2015_2016.csv")
    from fofs import knowledge_concepts, VARIABLES
    C = knowledge_concepts(); rows2 = []
    for var, (dom, typ, rng, unit, src) in VARIABLES.items():
        cs = [k for k, v in C.items() if v["variable"] == var]
        if not cs: continue
        dirs = []
        for k in cs:
            d = pri.loc[pri.concept == k]
            mark = "†" if (len(d) and isinstance(d.iloc[0].get("note"), str)) else ""
            dirs.append((sgn[int(d.direction.iloc[0])] + mark) if len(d) else "")
        rows2.append([var, dom, src.replace("SLQ.", "SLQ.").replace("kg/m2", "kg/m²"), f"{rng[0]:g}–{rng[1]:g} {unit.replace('kg/m2', 'kg/m²') if unit != '-' else ''}".strip(),
                      ", ".join(f"{k} ({d})" for k, d in zip(cs, dirs)), C[cs[0]]["provenance"].replace(">=", "≥").replace("kg/m2", "kg/m²"), EHR_FLAG[var]])
    T2 = dict(cap="Table 2. Knowledge schema: observation and concept layers. Direction priors (+, −, 0) estimated on NHANES 2015–2016.",
              head=["Variable", "Domain", "NHANES source", "Admissible range", "Concepts (direction prior)", "Provenance of cut-points", "Typically in EHR"], rows=rows2,
              note="† Inherited from the corresponding workday concept, because free-day sleep was not asked in 2015–2016; never estimated. 'Typically in EHR' indicates whether the variable is usually held as structured electronic health record data (coded diagnoses for diabetes and hypertension; chronic conditions only partly coded); in NHANES all variables are survey self-reports or measurements. In 2005–2014 workday sleep comes from SLD010H, and before 2013–2014 the chronic-condition count uses nine items.",
              w=[0.85, 0.85, 0.95, 0.8, 2.3, 1.5, 0.75], small=True)
    # ---------------------------------------------------------------- Table 3
    rb = pd.read_csv(R + "rulebase_table.csv")
    from experiment import BASE as _B
    omitted = [k for k, v in C.items() if v["variable"] in _B and k not in set(rb.rule)]
    inh = {"ShortSleepFreeDay", "AdequateSleepFreeDay", "LongSleepFreeDay"}
    rows3 = [[r.rule + (" †" if r.rule in inh else ""), f"{r.beta:+.3f}", f"{r.odds_ratio:.2f}", f"{r.mean_firing:.3f}", sgn[int(r.sign)]] for r in rb.itertuples()]
    T3 = dict(cap="Table 3. Rule base of the proposed model ordered by average effect (|β| × mean firing strength in the development set); intercept β₀ = {{intercept}}.",
              head=["Rule (single concept)", "β", "exp(β)", "Mean firing", "Sign constraint"], rows=rows3,
              note="One reference concept per variable is omitted (" + ", ".join(omitted) + "); each β is a contrast with the reference concept of the same variable. exp(β) is the multiplicative change in odds at full firing. † Sign constraint from an inherited, never-estimated prior. No constrained coefficient lies at its zero bound.",
              w=[2.4, 0.8, 0.8, 0.9, 1.0], small=True)
    # ---------------------------------------------------------------- Table 4
    rows4 = []
    for lab, s in [("Proposed", "full"), ("Logistic regression (indicators)", "lr"), ("LR, no indicators", "lrni"),
                   ("LR, multiple imputation", "lrmi"), ("LR, restricted cubic splines", "rcs"), ("Sign-constrained LR", "slr"), ("GA²M (EBM)", "ebm"),
                   ("Gradient-boosted trees", "gbt"), ("Monotone GBT", "mgbt")]:
        for sl, setlab in [("int", "Internal"), ("tmp", "Temporal"), ("ext", "External")]:
            rows4.append([lab if sl == "int" else "", setlab, "{{%s_%s_AUROC}} ({{%s_%s_AUROC_ci}})" % (s, sl, s, sl),
                          "{{%s_%s_wAUROC}}" % (s, sl), "{{%s_%s_AUPRC}}" % (s, sl), "{{%s_%s_Sensitivity}}" % (s, sl),
                          "{{%s_%s_Specificity}}" % (s, sl), "{{%s_%s_flag}}" % (s, sl), "{{%s_%s_Brier}}" % (s, sl),
                          "{{%s_%s_OE}}" % (s, sl), "{{%s_%s_slope}}" % (s, sl)])
    T4 = dict(cap="Table 4. Discrimination, operating characteristics and calibration on internal testing (n = {{n_int}}), temporal validation (n = {{n_tmp}}) and external validation (n = {{n_ext}}).",
              head=["Model", "Set", "AUROC (95% CI)", "wAUROC", "AUPRC", "Sens.", "Spec.", "Flagged (%)", "Brier", "O/E", "Slope"], rows=rows4,
              note="wAUROC, AUROC weighted by NHANES interview weights. Sensitivity, specificity and proportion flagged at thresholds fixed on development out-of-fold predictions (target sensitivity ≥ 0.80). O/E, observed-to-expected ratio; slope, calibration slope; both after development-set Platt recalibration. LR, logistic regression; GA²M (EBM), explainable boosting machine with pairwise interactions; GBT, gradient-boosted trees.",
              w=[1.35, 0.65, 1.15, 0.55, 0.5, 0.45, 0.45, 0.55, 0.52, 0.4, 0.42], small=True)
    # ---------------------------------------------------------------- Table by cycle
    cyc = [("c0506", "2005–2006"), ("c0708", "2007–2008"), ("c0910", "2009–2010"), ("c1112", "2011–2012"), ("c1314", "2013–2014"), ("ext", "Pooled")]
    rowsC = [[lab, "{{n_%s}}" % c, "{{pos_%s}} ({{prev_%s}})" % (c, c), "{{full_%s_AUROC}} ({{full_%s_AUROC_ci}})" % (c, c),
              "{{lr_%s_AUROC}}" % c, "{{rcs_%s_AUROC}}" % c, "{{slr_%s_AUROC}}" % c, "{{ebm_%s_AUROC}}" % c, "{{gbt_%s_AUROC}}" % c,
              "{{d_lr_%s}} ({{d_lr_%s_ci}})" % (c, c), "{{full_%s_OE}} / {{lr_%s_OE}} / {{slr_%s_OE}}" % (c, c, c)] for c, lab in cyc]
    TC = dict(cap="Table 5. External validation by NHANES cycle (frozen models and thresholds).",
              head=["Cycle", "n", "PHQ-9 ≥ 10, n (%)", "AUROC proposed (95% CI)", "LR", "Spline LR", "Sign-constrained LR", "GA²M", "GBT", "ΔAUROC vs LR (95% CI)", "O/E proposed / LR / sign-constrained LR"],
              rows=rowsC, note="Free-day sleep was absent in every cycle and sedentary time in 2005–2006. ΔAUROC is proposed minus logistic regression (with indicators) from paired bootstrap resamples (1,000 per cycle, 2,000 pooled); the pooled row is the primary comparison.",
              w=[0.8, 0.5, 0.8, 1.15, 0.45, 0.5, 0.6, 0.45, 0.45, 1.25, 1.0], small=True)
    # ---------------------------------------------------------------- Table ablations / paired
    var = [("Logistic regression (indicators)", "lr", "lr"), ("LR, no indicators", "lrni", None), ("LR, multiple imputation", "lrmi", None), ("LR, restricted cubic splines", "rcs", None),
           ("Sign-constrained LR", "slr", "slr"), ("GA²M (EBM)", "ebm", "ebm"), ("Gradient-boosted trees", "gbt", "gbt"),
           ("Monotone GBT", "mgbt", None), ("Version 1 (conjunctive rules, K = 32)", "v1", "v1"), ("Main effects + 16 pair rules", "inter", "inter"),
           ("Normalised TSK aggregation (K = 32)", "tsk", None), ("No direction constraints", "nodir", "nodir"), ("Crisp memberships", "crisp", "crisp"),
           ("Fuzzy without schema (quantile memberships)", "noonto", "noonto"), ("Median imputation (no bounds)", "noint", None)]
    rowsA = []
    for lab, s, sp in var:
        r = [lab] + ["{{d_%s_%s}} ({{d_%s_%s_ci}}); p = {{d_%s_%s_p}}" % (s, sl, s, sl, s, sl) for sl in ("int", "tmp", "ext")]
        r.append("{{spl_%s_mean}} ({{spl_%s_lo}} to {{spl_%s_hi}}); {{spl_%s_frac}}%%" % (sp, sp, sp, sp) if sp else "–")
        rowsA.append(r)
    TA = dict(cap="Table 6. Paired AUROC differences between the proposed model and each comparator or variant (proposed minus comparator; positive values favour the proposed model).",
              head=["Comparator or variant", "Internal (95% CI)", "Temporal (95% CI)", "External (95% CI)", "Repeated splits: mean (2.5–97.5th pct); % of splits proposed higher"],
              rows=rowsA,
              note="2,000 paired bootstrap resamples of each test set; p-values are two-sided and unadjusted. Only the external comparison with logistic regression is primary. Repeated splits: {{spl_B}} stratified 70/30 splits of the 2017–2020 release with all models refitted at fixed hyperparameters. Box bounds leave the point score unchanged and are compared only on decisions (Table 8).",
              w=[1.9, 1.45, 1.45, 1.45, 1.5], small=True)
    # ---------------------------------------------------------------- Table PHQ-2 / operating points
    rowsP = []
    for sl, setlab in [("int", "Internal"), ("tmp", "Temporal"), ("ext", "External")]:
        rowsP.append([setlab, "PHQ-2 ≥ 3 (reference)", "{{phq2_%s_sens}}" % sl, "{{phq2_%s_spec}}" % sl, "{{phq2_%s_flag}}" % sl,
                      "{{phq2_%s_avoid}}" % sl, "{{phq2_%s_missed}}" % sl])
        for lab, s in [("Proposed", "full"), ("Logistic regression", "lr"), ("Gradient-boosted trees", "gbt")]:
            rowsP.append(["", lab, "{{op_%s_%s_sens_b}}" % (sl, s), "{{op_%s_%s_spec_b}}" % (sl, s), "{{op_%s_%s_flag_b}}" % (sl, s),
                          "{{op_%s_%s_avoid_b}}" % (sl, s), "{{op_%s_%s_missed_b}}" % (sl, s)])
            if sl != "int":
                rowsP.append(["", lab + ", after intercept update", "{{op_%s_%s_sens_a}}" % (sl, s), "{{op_%s_%s_spec_a}}" % (sl, s),
                              "{{op_%s_%s_flag_a}}" % (sl, s), "{{op_%s_%s_avoid_a}}" % (sl, s), "{{op_%s_%s_missed_a}}" % (sl, s)])
        rowsP.append(["", "Proposed, record-typical predictors (9)", "{{ehr9_full_%s_sens}}" % sl, "{{ehr9_full_%s_spec}}" % sl,
                      "{{ehr9_full_%s_flag}}" % sl, "–", "–"])
    TP = dict(cap="Table 7. PHQ-2 reference point and operating points of the non-symptom models per 1,000 adults.",
              head=["Set", "Strategy", "Sensitivity", "Specificity", "Flagged (%)", "Questionnaires avoided per 1,000", "Cases missed per 1,000"],
              rows=rowsP,
              note="PHQ-2 ≥ 3 uses items DPQ010 and DPQ020 and is shown as a reference, not as a competitor in the same input space. Model rows use thresholds fixed in development. 'After intercept update': intercept estimated on a random half of the cohort, metrics on the other half, mean of 200 repetitions, same probability threshold. Record-typical predictors: age, sex, BMI, diabetes, hypertension, chronic-condition count, smoking, education, marital status (AUROC {{ehr9_full_int}}, {{ehr9_full_tmp}} and {{ehr9_full_ext}}).",
              w=[0.75, 2.3, 0.75, 0.75, 0.75, 1.0, 0.9], small=True)
    # ---------------------------------------------------------------- Table missing stress
    rowsM = []
    for lvl, lab, src in [("0", "None", "st"), ("10", "10% MCAR", "st"), ("20", "20% MCAR", "st"), ("30", "30% MCAR", "st"), ("50", "50% MCAR", "st"),
                          ("incomelow", "MNAR: income, low-income records", "mn"), ("srhpoor", "MNAR: self-rated health, fair/poor", "mn"), ("both", "MNAR: both", "mn"),
                          ("Sleep", "Sleep domain removed", "st"), ("Socioeconomic", "Socioeconomic domain removed", "st"), ("Somatic", "Somatic domain removed", "st"),
                          ("Lifestyle", "Lifestyle domain removed", "st"), ("Selfratedhealth", "Self-rated health removed", "st")]:
        if src == "st":
            auc = ["{{st_%s_%s_AUROC}}" % (s, lvl) for s in ("full", "noint", "v1", "lr", "lrni", "lrmi", "slr")]
            auc.append("{{spmc_%s_rcs_AUROC}}" % lvl if lvl in ("0", "10", "20", "30", "50") else "–")
            auc += ["{{st_%s_%s_AUROC}}" % (s, lvl) for s in ("gbt", "ebm")]
            cov = ["{{st_%s_%s_Coverage}}" % (s, lvl) for s in ("full", "box", "slr")]
            cov.append("{{sbmc_%s_rcs_Coverage}}" % lvl if lvl in ("0", "10", "20", "30", "50") else "–")
            cov.append("{{st_cc_%s_Coverage}}" % lvl)
            dec = ["{{st_full_%s_Sens_covered}} / {{st_full_%s_Spec_covered}}" % (lvl, lvl), "{{st_full_%s_Sens_refer}} / {{st_full_%s_Spec_refer}}" % (lvl, lvl)]
        else:
            auc = ["{{spmn_%s_%s_AUROC}}" % (lvl, s) for s in ("full", "noint")] + ["–"] + ["{{spmn_%s_%s_AUROC}}" % (lvl, s) for s in ("lr", "lrni", "lrmi", "slr", "rcs", "gbt", "ebm")]
            cov = ["{{sbmn_%s_%s_Coverage}}" % (lvl, s) for s in ("full", "box", "slr", "rcs", "cc")]
            dec = ["{{sbmn_%s_full_Sens_covered}} / {{sbmn_%s_full_Spec_covered}}" % (lvl, lvl), "{{sbmn_%s_full_Sens_refer}} / {{sbmn_%s_full_Spec_refer}}" % (lvl, lvl)]
        rowsM.append([lab] + auc + cov + dec)
    rowsA_ = [r[:11] for r in rowsM]; rowsB_ = [[r[0]] + r[11:] for r in rowsM]
    TM = dict(cap="Table 9. Simulated missingness on the internal test set. (a) AUROC of point scores.",
              head=["Scenario", "Proposed", "Median imp.", "Version 1", "LR ind.", "LR no ind.", "LR MI", "Sign LR", "Spline LR", "GBT", "GA²M"], rows=rowsA_,
              note="Mean of 10 repetitions with the same masks for every model (spline LR at 20% and 50% MCAR: first 3 repetitions). MCAR, each of the 12 predictors other than age and sex masked completely at random; MNAR, income-to-poverty ratio (or self-rated health) masked with probability 0.6 in records below 130% of the poverty level (or with fair or poor health) and 0.1 otherwise. 'None' includes naturally occurring missingness. Version 1 does not use diabetes, so its effective masking rate is lower than nominal (e.g. {{st_v1_30_eff_mask_rate}}% at 30%).",
              w=[1.9] + [0.55] * 10, small=True)
    TM2 = dict(cap="Table 9 (continued). (b) Percentage of participants receiving a decision, and decisions of the proposed model.",
               head=["Scenario", "Admissible bounds", "Box bounds", "Sign LR bounds", "Spline LR bounds", "Complete case", "Proposed, covered: Sens. / Spec. (%)", "Proposed, refer-on-abstain: Sens. / Spec. (%)"], rows=rowsB_,
               note="Coverage, percentage of participants whose bounds do not straddle the threshold; complete case: decision only if every input is observed. Spline LR bounds were not computed for domain removal.",
               w=[1.9, 0.7, 0.6, 0.7, 0.7, 0.7, 1.1, 1.1], small=True)
    # ---------------------------------------------------------------- Table natural missingness
    rowsN = []
    for sl, lab in [("int", "Internal test (2017–2020)"), ("tmp", "Temporal (2021–2023)"), ("ext", "External (2005–2014)"), ("cD", "  2005–2006"), ("cE", "  2007–2008"), ("cF", "  2009–2010"), ("cG", "  2011–2012"), ("cH", "  2013–2014")]:
        rowsN.append([lab, "{{nat_%s_anymiss}}" % sl] + ["{{nat_%s_%s_Coverage}}" % (sl, s) for s in ("full", "box", "slr", "rcs", "cc")] +
                     ["{{nat_%s_full_Sens_covered}} / {{nat_%s_full_Spec_covered}}" % (sl, sl), "{{nat_%s_full_Sens_refer}} / {{nat_%s_full_Spec_refer}}" % (sl, sl),
                      "{{nat_%s_full_Flag_refer}}" % sl, "{{natm_%s_full_Sens}} / {{natm_%s_lrni_Sens}} / {{natm_%s_rcs_Sens}}" % (sl, sl, sl)])
    TN = dict(cap="Table 8. Bounds under natural missingness: coverage, decisions and sensitivity at matched workload.",
              head=["Cohort", "Any input missing (%)", "Cov. admissible", "Cov. box", "Cov. sign LR", "Cov. spline LR", "Cov. complete case",
                    "Prop. covered Sens./Spec. (%)", "Prop. refer-on-abstain Sens./Spec. (%)", "Referred (%)", "Matched workload Sens. (%): proposed / LR no ind. / spline LR"],
              rows=rowsN,
              note="Coverage (Cov.), percentage of participants whose probability bounds do not straddle the referral threshold (admissible-value and box bounds of the proposed model; exact linear bounds of sign-constrained LR; grid bounds of spline LR; complete case: decision only if every input is observed). Free-day sleep is missing for every external participant, so every external decision is made with it unknown. Matched workload: point-score threshold set so that each model flags the same proportion as the proposed refer-on-abstain policy ('Referred').",
              w=[1.35, 0.6, 0.55, 0.45, 0.5, 0.5, 0.55, 0.8, 0.8, 0.55, 1.2], tiny=True)
    # ---------------------------------------------------------------- Table acquisition
    rowsQ = []
    for key, lab in [("m10", "Internal, 10% masked"), ("m30", "Internal, 30% masked"), ("t30", "Temporal, 30% masked")]:
        for pol, pl in [("bound", "Algorithm 2 (bound-guided)"), ("fixed", "Fixed order (contribution range)"), ("cost", "Cost order (record items first)"), ("random", "Random order")]:
            d = "–" if pol == "bound" else "{{acq2_%s_d_%s}} ({{acq2_%s_d_%s_ci}}); {{acq2_%s_fewer_%s}} / {{acq2_%s_more_%s}}" % (key, pol, key, pol, key, pol, key, pol)
            rowsQ.append([lab if pol == "bound" else "", "{{acq_%s_bound_n}}" % key if pol == "bound" else "", "{{acq_%s_bound_resolved}}" % key if pol == "bound" else "",
                          pl, "{{acq2_%s_%s_mean}}" % (key, pol), d])
    TQ = dict(cap="Table 10. Bound-guided input acquisition (Algorithm 2) on records where the proposed model abstained.",
              head=["Scenario", "Abstentions", "Resolved (%)", "Order", "Mean inputs requested", "Extra inputs vs Algorithm 2 (95% CI); % of records with more / fewer inputs than Algorithm 2"],
              rows=rowsQ, note="Masked inputs are revealed with their true values when requested; inputs missing in the original data cannot be revealed, so every order resolves the same records with the same final decisions. Means are over all abstentions. Differences are paired by record (2,000 bootstrap resamples). Fixed order: {{acq_order}}. Cost order: {{cost_order}}.",
              w=[1.2, 0.7, 0.6, 1.7, 0.7, 2.2], small=True)
    # ---------------------------------------------------------------- Algorithm box
    steps = [["Input: rule base {(A_r, β_r)}, β_0; record with observed x_O and missing set M; admissible sets 𝒳_{v} (Table S5)."],
             ["1. Build the graph on M with an edge between two missing variables if some rule contains both; take connected components M_1, …, M_q."],
             ["2. η_L ← η_U ← β_0 + Σ over rules with no missing variable of β_r μ_r(x_O)."],
             ["3. For each component M_c with rule set ℛ_{c}: enumerate the categories of its discrete variables; for each combination:"],
             ["   (a) one continuous variable x: candidates 𝒞 = {l, u} ∪ trapezoid corners ∪ {a + c_r(b − a), d − c_r(d − c)} for every rule with observed partner membership c_r (Proposition 1);"],
             ["   (b) two continuous variables (x, z): on each rectangle of the breakpoint grid add corners, intersections of switching lines μ_k(x) = μ_m(z) with the edges, and pairwise intersections of switching lines (Proposition 2);"],
             ["   (c) more than two continuous variables: use the box bound for ℛ_{c} (outer bound; flagged in the trace)."],
             ["   Evaluate g_c = Σ β_r μ_r over the rules in ℛ_{c} on all candidates; keep the minimum and maximum over candidates and combinations."],
             ["4. η_L ← η_L + Σ_c min g_c,  η_U ← η_U + Σ_c max g_c;  return p_L = σ(η_L), p_U = σ(η_U)."],
             ["Algorithm 2 (bound-guided acquisition): while p_L < t ≤ p_U and a revealable missing input exists, request v* = argmax_{v} ρ_v (Section 2.10), enter its value and recompute the bounds with Algorithm 1."]]
    TAlg = dict(cap="Box 1. Algorithm 1 (exact admissible-value bounds) and Algorithm 2 (bound-guided acquisition).", head=["Steps"], rows=steps, note="", w=[1.0], small=True)
    # ---------------------------------------------------------------- Supplementary
    adj = pd.read_csv(RV + "priors_adjusted.csv").set_index("concept"); mv = pd.read_csv(RV + "priors_multivariable.csv").set_index("concept")
    rowsS1 = []
    for r in pri.itertuples():
        a = adj.loc[r.concept] if r.concept in adj.index else None
        aor = f"{a.OR:.2f} ({a.lo:.2f}–{a.hi:.2f})" if a is not None and pd.notna(a.OR) else "–"
        m_ = mv.loc[r.concept] if r.concept in mv.index else None
        mvd = ("ref." if (m_ is not None and isinstance(m_.get("note"), str)) else (sgn[int(m_.direction_mv)] if m_ is not None else "–"))
        rowsS1.append([r.concept, r.variable, f"{int(r.n_in)}" if pd.notna(r.n_in) else "–", f"{100*r.rate_in:.1f}" if pd.notna(r.rate_in) else "–",
                       f"{r.RR:.2f} ({r.lo:.2f}–{r.hi:.2f})" if pd.notna(r.RR) else "inherited", sgn[int(r.direction)], aor, mvd])
    TS1 = dict(cap="Table A1. Direction priors estimated on NHANES 2015–2016 (n = {{flow_pri_analytic}} adults): crude, age- and sex-adjusted, and mutually adjusted.",
               head=["Concept", "Variable", "n (μ ≥ 0.5)", "% PHQ-9 ≥ 10", "Risk ratio (95% CI)", "Direction", "Age–sex adjusted OR (95% CI)", "Mutually adjusted direction"], rows=rowsS1,
               note="Risk ratio compares membership ≥ 0.5 with < 0.5. Direction +1 if the lower confidence limit exceeds 1, −1 if the upper limit is below 1, 0 otherwise. Age–sex adjusted: logistic model with age, age² and sex (age concepts adjusted for sex only, sex concepts for age only); no direction changed. Mutually adjusted: one logistic model with every concept of the 14 base variables, one reference concept per variable (ref.). SLQ050 and SLQ120 concepts were used only in the extended-sleep sensitivity analysis.", small=True)
    sub = pd.read_csv(R + "subgroups.csv"); so = pd.read_csv(RV + "subgroup_oe.csv")
    sub = sub.merge(so, on=["set", "group"], how="left")
    setlab = {"internal_test": "Internal", "temporal": "Temporal", "external_pooled": "External"}
    rowsS2 = [[setlab[r.set], r.group.replace(">=", "≥ "), f"{r.n:,}", f"{r.positives:,}", f"{r['Proposed (full) AUROC']:.3f}", f"{r['Logistic regression AUROC']:.3f}",
               f"{r['Gradient-boosted trees AUROC']:.3f}", f"{100*r['Proposed (full) Sens']:.1f}", f"{100*r['Logistic regression Sens']:.1f}",
               f"{r['Proposed (full) OE']:.2f}", f"{r['Logistic regression OE']:.2f}"] for _, r in sub.iterrows()]
    TS2 = dict(cap="Table A2. Subgroup discrimination, sensitivity at development thresholds and calibration-in-the-large.",
               head=["Set", "Subgroup", "n", "Positive", "AUROC proposed", "AUROC LR", "AUROC GBT", "Sens. (%) proposed", "Sens. (%) LR", "O/E proposed", "O/E LR"],
               rows=rowsS2, note="Groups with fewer than 10 positive cases are omitted. Before 2011 NHANES did not distinguish non-Hispanic Asian participants, who are included in 'Other/multiracial' for those cycles.", small=True)
    v1rb = json.load(open(R + "v1/rulebase_full.json"))
    rowsS3 = [[r["rule"].replace(" AND ", " ∧ ") + (" †" if any(c in r["rule"] for c in ("FreeDay",)) else ""), f"{r['beta']:+.3f}"] for r in v1rb]
    TS3 = dict(cap="Table A3. Rule base of version 1 (conjunctive rules, K = 32), retained as a comparator.", head=["Rule antecedent", "β"], rows=rowsS3,
               note="The first row is the intercept. † Rule involving a free-day sleep concept whose prior was inherited, not estimated. Version 1 does not use diabetes or the Obese concept.", w=[4.0, 1.0], small=True)
    sel = pd.read_csv(RV + "selection.csv", index_col=0)
    TS4 = dict(cap="Table A4. Included and excluded adults (no depression screener record or an invalid PHQ-9 item) in the development and temporal releases.",
               head=["Characteristic"] + list(sel.columns), rows=[[i] + [str(x) for x in sel.loc[i]] for i in sel.index],
               note="Excluded adults have no outcome; their predictor values are summarised to judge selection.", small=True)
    rr = json.load(open(RV + "rev_missing.json"))["ranges"]
    rowsS5 = [[v, d["unit"].replace("kg/m2", "kg/m²"), "{{rng_%s}}" % v, "{{rng99_%s}}" % v] for v, d in rr.items()]
    for lvl, lab in [("0", "no masking"), ("10", "10% MCAR"), ("30", "30% MCAR")]:
        for s, ml in [("full", "proposed"), ("slr", "sign-constrained LR"), ("rcs", "spline LR")]:
            rowsS5.append([f"Coverage, {ml}, {lab}", "%", "{{sbmc_%s_%s_Coverage}}" % (lvl, s), "{{sbmc_%s_%s_Coverage_p99}}" % (lvl, s)])
    for s, ml in [("full", "proposed"), ("slr", "sign-constrained LR"), ("rcs", "spline LR")]:
        rowsS5.append([f"Coverage, {ml}, external cohort", "%", "{{nat_ext_%s_Coverage}}" % s, "{{nat_ext_%s_Coverage_p99}}" % s])
    TS5 = dict(cap="Table A5. Admissible ranges of continuous variables and the effect of narrowing them to the 1st–99th development percentiles on coverage.",
               head=["Variable or quantity", "Unit", "Admissible range (schema)", "1st–99th percentile"], rows=rowsS5,
               note="Coverage under masking is the mean over the same 10 masks used in Table 9. Discrete variables use their category sets (education 1–5, marital status 3 categories, self-rated health 1–5, chronic-condition count 0–10, binary items 0–1).", small=True)

    from tripod_r1 import ROWS
    TT = dict(cap="Table B1. TRIPOD+AI checklist (Collins et al. [@collins2024]): items and where they are reported.",
              head=["Item", "Section/topic", "Checklist item (abbreviated)", "Where reported"], rows=ROWS, note="", w=[0.5, 1.3, 3.0, 2.2], small=True)
    return dict(tableTRIPOD=TT, table1=T1, table2=T2, table3=T3, table4=T4, tableCycle=TC, tableAbl=TA, tablePHQ2=TP, tableMiss=TM,
                tableAcq=TQ, tableNat=TN, tableMiss2=TM2, tableAlg=TAlg, tableS1=TS1, tableS2=TS2, tableS3=TS3, tableS4=TS4, tableS5=TS5)
