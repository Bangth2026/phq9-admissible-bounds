"""Tables of version 4: reuses tables_r3 and adds the abstention-rule tables with the base model held fixed."""
import json
from tables_r3 import tables as tables_r3
R = "/home/claude/results/"; RV = R + "secondary/"
PH = " ‡"  # analyses specified after the original results were known
RULES = [("wc", "Worst-case bounds"), ("mi", "Imputation spread" + PH), ("ps", "Pattern submodels" + PH), ("sel", "Point-score selective" + PH), ("cc", "Complete case" + PH)]
MOD = [("fuzzy", "Fuzzy additive"), ("slr", "Sign-constrained LR" + PH), ("rcs", "Spline LR" + PH)]


def tables(V):
    T = tables_r3(V)
    has = lambda k: True
    # ---------------------------------------------------------------- Table 1: add row excluding the uncollected input
    t1 = T["table1"]; rows = t1["rows"]
    rows.append(["Any predictor missing, %, excluding free-day sleep", "{{amx_dev}}", "{{amx_int}}", "{{amx_tmp}}", "{{amx_ext}}"])
    t1["note"] += " The last row excludes free-day sleep, which was not collected in 2005–2014."
    # ---------------------------------------------------------------- Table 3 note
    T["table3"]["note"] += (" In 2005–2014 free-day sleep is missing for everyone; because the reference concept AdequateSleepFreeDay contributes 0, "
                            "ShortSleepFreeDay +{{we_beta_short}} and LongSleepFreeDay +{{we_beta_long}}, worst-case bounds there add 0 to the lower "
                            "bound and up to {{we_beta_short}} log-odds to the upper bound, while the point score adds {{we_fill}} (development-mean memberships).")
    # ---------------------------------------------------------------- Table 4: mark post hoc comparators
    lab_ph = {"LR, no indicators", "LR, multiple imputation", "LR, restricted cubic splines", "Sign-constrained LR", "GA²M (EBM)", "Monotone GBT"}
    for r in T["table4"]["rows"]:
        if r[0] in lab_ph: r[0] = r[0] + PH
    T["table4"]["note"] += " ‡ Comparator specified after the original results were known (Section 2.12)."
    T["table4"]["cap"] = T["table4"]["cap"].replace("Discrimination, operating characteristics and calibration", "Base models: discrimination, operating characteristics and calibration")
    # ---------------------------------------------------------------- Table 5: natural missingness, rule x model
    rows5 = []
    for sc, lab, models in [("int", "Internal test (2017–2020)", ["fuzzy"]), ("tmp", "Later period (2021–2023)", ["fuzzy"]),
                            ("ext", "Earlier period (2005–2014)", ["fuzzy", "slr", "rcs"])]:
        first = True
        for m in models:
            ml = dict(MOD)[m]
            for r, rl in RULES:
                k = f"a4_{sc}_{m}_{r}_"
                rows5.append([lab if first else "", ml if r == "wc" else "", rl,
                              "{{%scov}} ({{%scovr}} / {{%scovn}})" % (k, k, k),
                              "–" if r == "cc" and sc == "ext" else "{{%ssens}} / {{%sspec}}" % (k, k),
                              "{{%sats}} / {{%satp}}" % (k, k), "{{%satb}}" % k,
                              "{{%saurc}} ({{%saurcci}})" % (k, k), "–" if r == "wc" else "{{%sdaurc}} ({{%sdaurcci}})" % (k, k)])
                first = False
    rows5.append(["", "LR, multiple imputation" + PH, "Imputation spread" + PH, "{{a4_ext_lrmi_mi_cov}} ({{a4_ext_lrmi_mi_covr}} / {{a4_ext_lrmi_mi_covn}})",
                  "{{a4_ext_lrmi_mi_sens}} / {{a4_ext_lrmi_mi_spec}}", "{{a4_ext_lrmi_mi_ats}} / {{a4_ext_lrmi_mi_atp}}", "{{a4_ext_lrmi_mi_atb}}",
                  "{{a4_ext_lrmi_mi_aurc}} ({{a4_ext_lrmi_mi_aurcci}})", "–"])
    T5 = dict(cap="Table 5. Abstention rules compared with the base model held fixed: cohorts as observed.",
              head=["Cohort", "Base model", "Rule", "Decided, % (refer / do not refer)", "Decided: Sens. / Spec. (%)",
                    "At matched coverage: Sens. / Spec. (%)", "At matched coverage: balanced accuracy", "AURC (95% CI)", "ΔAURC vs worst-case bounds (95% CI)"],
              rows=rows5,
              note="Matched coverage: each rule decides the participants it is most confident about, up to the coverage of worst-case bounds of the same base model. "
                   "AURC, area under the balanced-error/coverage curve over coverage 0.05–1 (lower is better); ΔAURC negative favours the rule over worst-case bounds; "
                   "percentile bootstrap intervals (500 resamples; 300 in 2005–2014). Complete-case abstention decides nobody in 2005–2014, where free-day sleep was not "
                   "collected, and its AURC and accuracy at matched coverage there equal those of point-score selective prediction, which ranks by the same "
                   "distance of the point score from the threshold. Point-score selective prediction abstains when |logit p − logit t| < δ, with δ set on development "
                   "out-of-fold predictions to decide 90% of development participants. Last row: the imputation-spread rule of the "
                   "previous analysis, on the chained-equation logistic model. ‡ Specified after the original results were known.",
              w=[1.05, 1.0, 1.05, 1.0, 0.8, 0.85, 0.6, 0.95, 1.05], tiny=True)
    # ---------------------------------------------------------------- Table 6: simulated scenarios, rule x model
    SCN = [("mc30", "30% MCAR"), ("m6inc", "MNAR 0.6/0.1: income" + PH), ("m6srh", "MNAR 0.6/0.1: self-rated health" + PH), ("m6both", "MNAR 0.6/0.1: both" + PH),
           ("m9inc", "MNAR 0.9/0.05: income" + PH), ("m9srh", "MNAR 0.9/0.05: self-rated health" + PH),
           ("odep", "Outcome-dependent: health and income" + PH), ("sh0srh", "Self-rated health absent, γ = 0" + PH),
           ("sh05srh", "Self-rated health absent, γ = 0.5" + PH), ("sh1srh", "Self-rated health absent, γ = 1" + PH),
           ("sh0we", "Free-day sleep absent, γ = 0" + PH), ("sh05we", "Free-day sleep absent, γ = 0.5" + PH), ("sh1we", "Free-day sleep absent, γ = 1" + PH)]
    rows6 = []
    for sc, lab in SCN:
        for i, (m, ml) in enumerate(MOD):
            k = lambda r: f"a4_{sc}_{m}_{r}_"
            ps = (lambda f: "–") if sc == "mc30" else (lambda f: "{{%s%s}}" % (k("ps"), f))
            au = lambda r: "{{%saurc}} ({{%saurcsd}})" % (k(r), k(r))
            rows6.append([(lab + " ({{prv_%s}}%%)" % sc) if i == 0 else "", ml,
                          "{{%scov}}" % k("wc"), "{{%scov}}" % k("mi"), ps("cov"), "{{%scov}}" % k("sel"), "{{%scov}}" % k("cc"),
                          "{{%satb}}" % k("wc"), "{{%satb}}" % k("mi"), ps("atb"), "{{%satb}}" % k("sel"),
                          au("wc"), au("mi"), "–" if sc == "mc30" else au("ps"), au("sel")])
    T6 = dict(cap="Table 6. Abstention rules compared with the base model held fixed: the 13 simulated missingness scenarios on the internal test set (means of 10 repetitions).",
              head=["Scenario (prevalence of PHQ-9 ≥ 10)", "Base model", "Decided (%): worst-case", "imputation spread", "pattern submodels", "point-score selective", "complete case",
                    "Balanced accuracy at matched coverage: worst-case", "imputation spread", "pattern submodels", "point-score selective",
                    "AURC (SD): worst-case", "imputation spread", "pattern submodels", "point-score selective"],
              rows=rows6,
              note="The study has 16 missingness scenarios: the three cohorts as observed (Table 5) and the 13 simulated scenarios above. Matched coverage: coverage of "
                   "worst-case bounds of the same base model. MNAR 0.6/0.1 (or 0.9/0.05): input masked with the first probability in participants with "
                   "income-to-poverty ratio < 1.3 (or fair or poor self-rated health) and the second otherwise. Outcome-dependent: self-rated health and income each "
                   "masked with probability 0.6 if PHQ-9 ≥ 10 and 0.1 otherwise. Absent: the input is masked for everyone in a target sample drawn with probability "
                   "∝ exp(γ s z), where z is the standardised residual of the input given the other inputs in a development-data linear model (s = +1, worse health; "
                   "s = −1, shorter free-day sleep). Prevalence is the mean over the 10 samples. SD, standard deviation of AURC over the 10 repetitions. Pattern "
                   "submodels were not fitted under MCAR, where patterns are numerous. ‡ Specified after the original results were known.",
              w=[1.45, 0.85] + [0.42] * 5 + [0.47] * 4 + [0.55] * 4, tiny=True)
    # ---------------------------------------------------------------- Table 7: decision consequences, refer-on-abstain
    def cons_rows(m):
        rr = []
        for sc, lab in [("int", "Internal test, natural"), ("tmp", "2021–2023, natural"), ("ext", "2005–2014, natural"), ("mc30", "30% MCAR"),
                        ("m6srh", "MNAR 0.6/0.1: health" + PH), ("m9srh", "MNAR 0.9/0.05: health" + PH), ("odep", "Outcome-dependent" + PH),
                        ("sh0srh", "Health absent, γ = 0" + PH), ("sh1srh", "Health absent, γ = 1" + PH),
                        ("sh0we", "Free-day sleep absent, γ = 0" + PH), ("sh1we", "Free-day sleep absent, γ = 1" + PH)]:
            for j, (r, rl) in enumerate(RULES):
                if r == "ps" and sc == "mc30": continue
                k = f"a4_{sc}_{m}_{r}_"
                dn = "–" if r == "wc" else "{{dnb_%s_%s_%s_10}}" % (sc, m, r)
                rr.append([(lab + " ({{prv_%s}}%%)" % sc) if j == 0 else "", rl, "{{%sref}}" % k, "{{%smis}}" % k, "{{%savd}}" % k, "{{%ssensr}}" % k,
                           "{{%snb05}}" % k, "{{%snb10}}" % k, dn])
        return rr
    T7 = dict(cap="Table 7. Decision consequences of each abstention rule under refer-on-abstain, fuzzy additive base model, per 1,000 participants.",
              head=["Scenario (prevalence, %)", "Rule", "Referred", "Cases missed", "Questionnaires avoided", "Sensitivity (%)", "Net benefit at 0.05", "Net benefit at 0.10",
                    "ΔNB at 0.10 vs worst-case (95% CI)"],
              rows=cons_rows("fuzzy"),
              note="Refer-on-abstain: every abstention is referred, and decided participants follow the rule's decision (pattern submodels decide everyone). "
                   "Questionnaires avoided = participants not referred. Net benefit = (true positives − false positives × p_t/(1 − p_t))/n at threshold probability p_t; "
                   "referring no one gives 0; under refer-on-abstain, complete-case abstention in 2005–2014 and with an input absent refers everyone. ΔNB: paired bootstrap "
                   "difference to worst-case bounds (1,000 resamples; 500 in 2005–2014; first repetition for simulated scenarios; intervals at 0.05 in the statistics "
                   "file); n.c., not computed. Simulated scenarios are means of 10 repetitions. The same quantities for the regression base models are in Table A6. "
                   "‡ Specified after the original results were known.",
              w=[1.5, 1.35, 0.6, 0.6, 0.75, 0.65, 0.65, 0.65, 1.15], tiny=True)
    TA6 = dict(cap="Table A6. Decision consequences under refer-on-abstain for the sign-constrained (upper block) and spline (lower block) logistic base models, per 1,000 participants.",
               head=T7["head"], rows=[["Sign-constrained LR ‡"] + [""] * 8] + cons_rows("slr") + [["Spline LR ‡"] + [""] * 8] + cons_rows("rcs"),
               note=T7["note"].replace(" The same quantities for the regression base models are in Table A6.", ""), w=T7["w"], tiny=True)
    # ---------------------------------------------------------------- PHQ-2 table: incorporation bias
    T["tablePHQ2"]["cap"] = "Table 9. PHQ-2 reference point and operating points of the non-symptom models per 1,000 participants."
    T["tablePHQ2"]["note"] = T["tablePHQ2"]["note"].replace(
        "PHQ-2 ≥ 3 uses items DPQ010 and DPQ020 and is shown as a reference, not as a competitor in the same input space.",
        "PHQ-2 ≥ 3 uses items DPQ010 and DPQ020 and is shown as a reference, not as a competitor in the same input space. Both items are part of the PHQ-9 "
        "total that defines the outcome, so the PHQ-2 rows are inflated by incorporation bias; against diagnostic interviews, PHQ-2 ≥ 3 had a pooled "
        "sensitivity of 0.72 and specificity of 0.85 [@levis2020].")
    # ---------------------------------------------------------------- acquisition with item costs
    q = T["tableAcq"]
    q["cap"] = "Table 8. Bound-guided input acquisition (Algorithm 3) on records where the fuzzy additive model abstained ‡."
    q["note"] += (" Expected item cost, internal test with 30% masked (cost 1 per record item, 2 for income, 3 per interview item): Algorithm 3 {{aqc_bound_3}}, "
                  "cost-weighted Algorithm 3 {{aqc_bound_cost_3}}, fixed order {{aqc_fixed_3}}, cost order {{aqc_cost_3}}, random {{aqc_random_3}}; with interview "
                  "items costing 5: {{aqc_bound_5}}, {{aqc_bound_cost_5}}, {{aqc_fixed_5}}, {{aqc_cost_5}} and {{aqc_random_5}}. ‡ Specified after the original results were known.")
    q["note"] = q["note"].replace("Algorithm 2", "Algorithm 3")
    for r in q["rows"]:
        r[3] = r[3].replace("Algorithm 2", "Algorithm 3")
    q["head"] = [h.replace("Algorithm 2", "Algorithm 3") for h in q["head"]]
    # ---------------------------------------------------------------- algorithm boxes renumbered
    T["tableAlg"]["cap"] = "Box 1. Algorithm 1: worst-case bounds and abstention for a main-effects rule base."
    T["tableAlgGen"]["cap"] = "Box 2. Algorithm 2: exact worst-case bounds for rule bases with one- and two-concept antecedents."
    for r in T["tableAlgGen"]["rows"]:
        r[0] = r[0].replace("Statement C1", "Statement 1").replace("Statement C2", "Statement 2")
    T["tableAlg2"]["cap"] = "Box 3. Algorithm 3: bound-guided input acquisition."
    for r in T["tableAlg2"]["rows"]:
        r[0] = r[0].replace("Algorithm 1(", "Bounds(").replace("Algorithm 1;", "Algorithm 1 or 2 (the bounds);")
    T["tableAlg2"]["rows"].insert(-1, ["   Cost-weighted variant: replace step 7 by v* ← argmax_{v ∈ Q} ρ_{v} / c_{v}, with c_{v} the cost of obtaining v."])
    # ---------------------------------------------------------------- appendix renumbering of former main tables
    T["tableCycle"]["cap"] = T["tableCycle"]["cap"].replace("Table 5.", "Table A7.")
    ta = T["tableAbl"]; ta["cap"] = ta["cap"].replace("Table 6.", "Table A8.")
    ta["head"][-1] = "Repeated splits: mean (SD); % of splits proposed higher"
    ph_rows = {"LR, no indicators", "LR, multiple imputation", "LR, restricted cubic splines", "Sign-constrained LR", "GA²M (EBM)", "Monotone GBT"}
    for r in ta["rows"]:
        if r[0] in ph_rows: r[0] += PH
    var = {"Logistic regression (indicators)": "lr", "Sign-constrained LR ‡": "slr", "GA²M (EBM) ‡": "ebm", "Gradient-boosted trees": "gbt",
           "Version 1 (conjunctive rules, K = 32)": "v1", "Main effects + 16 pair rules": "inter", "No direction constraints": "nodir",
           "Crisp memberships": "crisp", "Fuzzy without schema (quantile memberships)": "noonto"}
    for r in ta["rows"]:
        s = var.get(r[0]); r[-1] = ("{{spl_%s_mean}} ({{spl_%s_sd}}); {{spl_%s_frac}}%%" % (s, s, s)) if s else "–"
    ta["note"] = ("2,000 paired bootstrap resamples of each test set; p-values are two-sided and unadjusted, and are omitted (n.r.) where the difference is "
                  "smaller than 0.001 in absolute value, where they have no practical meaning. Only the 2005–2014 comparison with logistic regression (indicators) is "
                  "primary. Repeated splits: {{spl_B}} stratified 70/30 splits of the 2017–2020 release with all models refitted at fixed hyperparameters. "
                  "‡ Specified after the original results were known.")
    T["tableMiss"]["cap"] = T["tableMiss"]["cap"].replace("Table 9.", "Table A9.")
    T["tableMiss2"]["cap"] = T["tableMiss2"]["cap"].replace("Table 9 (continued).", "Table A9 (continued).")
    T["tableNat"]["cap"] = "Table A10. Worst-case bounds of each base model under natural missingness, by cycle, with sensitivity at matched workload."
    for t in ("tableMiss2", "tableNat"):
        T[t]["head"] = [h.replace("Admissible bounds", "Worst-case (fuzzy)").replace("Cov. admissible", "Cov. worst-case (fuzzy)") for h in T[t]["head"]]
    T.update(tableAbst5=T5, tableAbst6=T6, tableCons=T7, tableA6=TA6)
    return T
