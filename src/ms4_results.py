# Results of version 4.
from ms4_helpers import *  # noqa

h1("3. Results")
h2("3.1. Cohorts")
p("Table 1 summarises the cohorts. Prevalence of PHQ-9 ≥ 10 was {{prev_ext}}% in 2005–2014, "
  "{{flow_dev_prevalence_unweighted}}% in 2017–2020 and {{prev_tmp}}% in 2021–2023. The 2021–2023 cohort was older and "
  "more educated, with fewer participants rating their health as fair or poor but more sedentary time and more chronic "
  "conditions. The 2005–2014 cohort was younger, had more current smokers and reported about half an hour less workday "
  "sleep, partly because sleep was recorded in whole hours. Every 2005–2014 participant lacked free-day sleep; "
  "excluding that input, {{amx_ext}}% had at least one missing predictor, against {{amx_int}}% in the internal test set "
  "and {{amx_tmp}}% in 2021–2023. Excluded adults of the development release were slightly older and more often women "
  "than included adults (Table A4).")
tab("table1")

h2("3.2. Knowledge schema and rule base")
p("Direction priors from 2015–2016 agreed with established epidemiology: low income (RR {{rr_LowIncome}}), current "
  "smoking ({{rr_CurrentSmoker}}), poor self-rated health ({{rr_PoorHealth}}), multimorbidity ({{rr_Multimorbidity}}), "
  "short ({{rr_ShortSleepWorkday}}) and long ({{rr_LongSleepWorkday}}) workday sleep were positive; college education "
  "({{rr_CollegeGraduate}}) and being married or partnered ({{rr_MarriedOrPartnered}}) were negative. Of the "
  "{{n_rules}} rules, {{n_rules_signed}} carry a sign constraint; none of the constrained coefficients sits at its zero "
  "bound, and refitting without constraints changed no sign, so the constraints were not binding. Table 3 lists the "
  "rules. Each coefficient is a contrast with the reference concept of its variable; the negative coefficients of "
  "NormalWeight and Overweight, for example, are contrasts with Obese. One coefficient shapes the 2005–2014 bounds: "
  "within free-day sleep the reference concept contributes 0, ShortSleepFreeDay +{{we_beta_short}} and "
  "LongSleepFreeDay +{{we_beta_long}}, so with free-day sleep unknown the lower bound adds nothing and the upper bound "
  "adds up to {{we_beta_short}} log-odds. The sign constraint on ShortSleepFreeDay comes from an inherited prior that "
  "was never estimated. The point score instead adds {{we_fill}} log-odds, the development-mean memberships of the "
  "two concepts weighted by their coefficients, to every 2005–2014 participant.")

h2("3.3. Base models: discrimination and calibration")
p("The pre-specified primary comparison was AUROC in 2005–2014: {{full_ext_AUROC}} ({{full_ext_AUROC_ci}}) for the "
  "fuzzy model against {{lr_ext_AUROC}} for logistic regression with missing indicators, a difference of {{d_lr_ext}} "
  "({{d_lr_ext_ci}}; design-based interval {{db_lr_ci}}). This contrast is uninformative about the fuzzy model or its "
  "schema. Free-day sleep was not collected in 2005–2014; the indicator model applies a learnt indicator coefficient to "
  "everyone and over-predicts (O/E {{lr_ext_OE}}), whereas logistic regression without indicators reached "
  "{{lrni_ext_AUROC}}, the same as the fuzzy model ({{d_lrni_ext}}, {{d_lrni_ext_ci}}; design-based {{db_lrni_ci}}). "
  "Refitted without free-day sleep, the fuzzy model reached {{nowe_full_ext}} and regression without indicators "
  "{{nowe_lrni_ext}}. Spline regression ranked best in the later and earlier periods ({{rcs_tmp_AUROC}} and "
  "{{rcs_ext_AUROC}}; difference to the fuzzy model in 2005–2014 {{d_rcs_ext}}, design-based {{db_rcs_ci}}). Internally "
  "the fuzzy model was the lowest of the main models ({{full_int_AUROC}} against {{int_min_oth}}–{{int_max_oth}}). Over "
  "{{spl_B}} repeated splits of 2017–2020 it averaged {{spl_full_mean}} (SD {{spl_full_sd}}), and its mean difference "
  "to linear logistic regression was {{spl_lr_mean}} (SD {{spl_lr_sd}}). Design-based intervals were close to "
  "participant-level intervals and changed no conclusion (Table 4; Fig. 3). The base models therefore discriminated "
  "about equally well, and none should be read as superior for prediction. Differences of 0.01 AUROC are small in "
  "operational terms: with each model's threshold set to reach 80% sensitivity in 2005–2014, the fuzzy model flagged "
  "{{op80_ext_full_flag}} participants per 1,000, regression without indicators {{op80_ext_lrni_flag}} and spline "
  "regression {{op80_ext_rcs_flag}}, so the better-ranking spline model would save about {{op80_ext_d_rcs}} referrals per "
  "1,000 at the same sensitivity. The differences that matter for this study are those between abstention rules, not "
  "between base models.")
tab("table4")
fig("fig_roc.png", "Fig. 3. ROC curves on internal testing (2017–March 2020), later-period validation (2021–2023) and "
    "earlier-period validation (2005–2014). AUROC in parentheses.", 6.6)
p("Calibration differed more than ranking (Fig. 4). Internally O/E was close to 1 for every model; the calibration "
  "slope of logistic regression with indicators was {{lr_int_slope}}, i.e. its predictions were too moderate. Before "
  "recalibration the slope was already {{calm_lr_raw}}: the L2 penalty chosen to maximise cross-validated AUROC shrank "
  "the coefficients, and the Platt map fitted on out-of-fold scores corrected part of this. Fitting the map on the "
  "refit's in-sample scores gave a slope of {{calm_lr_platt_insample}} and isotonic recalibration on out-of-fold scores "
  "{{calm_lr_isotonic_oof}}; for the fuzzy model the corresponding slopes were {{calm_full_platt_oof}}, "
  "{{calm_full_platt_insample}} and {{calm_full_isotonic_oof}}. In 2021–2023 every model under-predicted (O/E "
  "{{full_tmp_OE}} for the fuzzy model), and an intercept update estimated on half the cohort brought O/E on the other "
  "half to {{rec_tmp_full_OE_after}}. In 2005–2014 the fuzzy model ({{full_ext_OE}}) and the regression models without "
  "indicators were calibrated in the large, whereas models that learnt a separate value for missingness over-predicted. "
  "Results by cycle, paired differences for all comparators and ablations are in Tables A7–A8. Of the design elements "
  "of the fuzzy model, rule structure mattered most: the conjunctive version 1 was {{d_v1_ext_abs}} lower in 2005–2014 "
  "({{d_v1_ext_ci}}). Fuzzy memberships gave a small advantage over crisp memberships at the same cut-points in "
  "2017–2023 data ({{d_crisp_int}} internally, {{d_crisp_tmp}} in 2021–2023, {{spl_crisp_mean}} on average over repeated "
  "splits with SD {{spl_crisp_sd}}) that was not seen in 2005–2014 ({{d_crisp_ext}}, {{d_crisp_ext_ci}}). Removing the schema "
  "or the sign constraints had no detectable effect on discrimination.")
fig("fig_calibration.png", "Fig. 4. Calibration by deciles of predicted probability after development-set Platt "
    "recalibration. All models under-predict in 2021–2023.", 6.6)

h2("3.4. Abstention rules with the base model held fixed")
p("Throughout this section, a decision made by worst-case bounds is one that holds for every admissible value of the "
  "missing inputs *under the model*. It is not a guarantee that the decision is correct: where the model is "
  "miscalibrated, as all models were in 2021–2023, a decision that holds for every completion can still be wrong.")
p("*Cohorts as observed.* Where few inputs were missing (internal test and 2021–2023), the five rules differed little: "
  "with the fuzzy model, worst-case bounds decided {{a4_int_fuzzy_wc_cov}}% of internal test participants, imputation "
  "spread {{a4_int_fuzzy_mi_cov}}%, point-score selective prediction {{a4_int_fuzzy_sel_cov}}% and complete cases "
  "{{a4_int_fuzzy_cc_cov}}%, balanced accuracy at matched coverage ranged from {{a4_int_fuzzy_cc_atb}} to "
  "{{a4_int_fuzzy_mi_atb}}, and every AURC interval for the difference to worst-case bounds included zero (Table 5). In "
  "2005–2014, where free-day sleep was missing for everyone, the rules separated. With the fuzzy model, worst-case "
  "bounds decided {{a4_ext_fuzzy_wc_cov}}% of participants ({{a4_ext_fuzzy_wc_covr}}% referred, {{a4_ext_fuzzy_wc_covn}}% "
  "not referred), imputation spread {{a4_ext_fuzzy_mi_cov}}% ({{a4_ext_fuzzy_mi_covr}}% and {{a4_ext_fuzzy_mi_covn}}%), "
  "point-score selective prediction {{a4_ext_fuzzy_sel_cov}}%, pattern submodels everyone and complete cases nobody; "
  "weighted by interview weights, worst-case coverage was {{a4w_ext_fuzzy_wc_cov}}% and imputation-spread coverage "
  "{{a4w_ext_fuzzy_mi_cov}}%. The asymmetry of the worst-case decisions follows from the free-day sleep coefficients "
  "(Section 3.2): with the upper bound raised by up to {{we_beta_short}} log-odds, 'do not refer' required a low risk "
  "from every other input. Among decided participants, worst-case bounds had sensitivity {{a4_ext_fuzzy_wc_sens}}% and "
  "specificity {{a4_ext_fuzzy_wc_spec}}%, against {{a4_ext_fuzzy_mi_sens}}% and {{a4_ext_fuzzy_mi_spec}}% for imputation "
  "spread on the same model. At the coverage of worst-case bounds, balanced accuracy was {{a4_ext_fuzzy_wc_atb}} for "
  "worst-case bounds, {{a4_ext_fuzzy_mi_atb}} for imputation spread, {{a4_ext_fuzzy_ps_atb}} for pattern submodels and "
  "{{a4_ext_fuzzy_sel_atb}} for point-score selective prediction, and AURC was {{a4_ext_fuzzy_wc_aurc}}, "
  "{{a4_ext_fuzzy_mi_aurc}}, {{a4_ext_fuzzy_ps_aurc}} and {{a4_ext_fuzzy_sel_aurc}} (Fig. 5). Ranking participants by the "
  "distance of the fuzzy model's point score from the threshold thus gave the lowest AURC in 2005–2014, lower than "
  "pattern submodels by {{selps_fuzzy_daurc}} ({{selps_fuzzy_daurc_ci}}); with sign-constrained regression the two did "
  "not differ ({{selps_slr_daurc_ci}}), and with spline regression the point score was again better "
  "({{selps_rcs_daurc_ci}}). The same ordering of worst-case bounds against the other rules held with the regression "
  "base models, by a wider margin for spline regression, whose linear tails widen the bounds: worst-case coverage "
  "{{a4_ext_rcs_wc_cov}}% and balanced accuracy {{a4_ext_rcs_wc_atb}}, against {{a4_ext_rcs_mi_cov}}% and "
  "{{a4_ext_rcs_mi_atb}} for imputation spread.")
tab("tableAbst5")
p("*Random and non-random missingness.* Under 30% MCAR, worst-case bounds decided {{a4_mc30_fuzzy_wc_cov}}% of "
  "participants, imputation spread {{a4_mc30_fuzzy_mi_cov}}% and point-score selective prediction "
  "{{a4_mc30_fuzzy_sel_cov}}% with the fuzzy model; at matched coverage balanced accuracy was {{a4_mc30_fuzzy_wc_atb}}, "
  "{{a4_mc30_fuzzy_mi_atb}} and {{a4_mc30_fuzzy_sel_atb}}, and the regression models showed the same gap (Table 6). "
  "Non-random missingness of income changed little. Non-random missingness of self-rated health, the strongest "
  "predictor, narrowed the differences. With masking probabilities 0.9 in fair or poor health and 0.05 otherwise, "
  "worst-case bounds decided {{a4_m9srh_fuzzy_wc_cov}}%, imputation spread {{a4_m9srh_fuzzy_mi_cov}}% and pattern "
  "submodels everyone; balanced accuracy at matched coverage was {{a4_m9srh_fuzzy_wc_atb}}, {{a4_m9srh_fuzzy_mi_atb}} and "
  "{{a4_m9srh_fuzzy_ps_atb}}; and the fuzzy model's AURC was lowest for worst-case bounds ({{a4_m9srh_fuzzy_wc_aurc}}, SD "
  "{{a4_m9srh_fuzzy_wc_aurcsd}} over repetitions, against {{a4_m9srh_fuzzy_mi_aurc}} and {{a4_m9srh_fuzzy_ps_aurc}}). With "
  "the regression base models the rules were within 0.01 AURC of each other. Under outcome-dependent masking, "
  "worst-case bounds had a slightly lower AURC than imputation spread and pattern submodels for every base model, by "
  "at most {{odep_maxgap}} (SD over repetitions {{a4_odep_fuzzy_ps_aurcsd}}–{{a4_odep_fuzzy_wc_aurcsd}} for the fuzzy "
  "model), with first-repetition intervals that included zero.")
p("*Structural absence with a shifted conditional distribution.* When self-rated health was absent for everyone, "
  "worst-case bounds of the fuzzy model decided {{a4_sh0srh_fuzzy_wc_cov}}% of participants without shift and "
  "{{a4_sh1srh_fuzzy_wc_cov}}% with γ = 1, and with γ = 1 those of sign-constrained and spline regression "
  "{{a4_sh1srh_slr_wc_cov}}% and {{a4_sh1srh_rcs_wc_cov}}%, because the full range of self-rated health spans most of the "
  "linear predictor. The shift also raised the prevalence of PHQ-9 ≥ 10 in the target samples, from {{prv_sh0srh}}% "
  "(γ = 0) to {{prv_sh05srh}}% and {{prv_sh1srh}}% (Table 6). Measured by sensitivity under refer-on-abstain, the shift "
  "degraded imputation and pattern submodels modestly: from {{a4_sh0srh_fuzzy_mi_sensr}}% to {{a4_sh1srh_fuzzy_mi_sensr}}% "
  "for imputation spread and from {{a4_sh0srh_fuzzy_ps_sensr}}% to {{a4_sh1srh_fuzzy_ps_sensr}}% for pattern submodels, "
  "against {{a4_sh0srh_fuzzy_wc_sensr}}% and {{a4_sh1srh_fuzzy_wc_sensr}}% for worst-case bounds; most of the rise in cases "
  "missed per 1,000 (from {{a4_sh0srh_fuzzy_mi_mis}} to {{a4_sh1srh_fuzzy_mi_mis}} for imputation spread) reflects the "
  "larger number of cases. At matched coverage the decided records were still classified better by imputation spread "
  "({{a4_sh1srh_fuzzy_mi_atb}} against {{a4_sh1srh_fuzzy_wc_atb}}), pattern submodels ({{a4_sh1srh_fuzzy_ps_atb}}) and "
  "point-score selective prediction ({{a4_sh1srh_fuzzy_sel_atb}}). Repeating the γ = 1 shift with ordinal residuals, which "
  "correlated {{ord_corr}} with the linear residuals, gave the same picture (prevalence {{prv_ordsrh}}%; balanced "
  "accuracy {{ord_wc_atb}}, {{ord_mi_atb}} and {{ord_ps_atb}}; sensitivity {{ord_wc_sensr}}%, {{ord_mi_sensr}}% and "
  "{{ord_ps_sensr}}%). With free-day sleep absent and shifted, the ordering of 2005–2014 was reproduced (balanced "
  "accuracy {{a4_sh1we_fuzzy_wc_atb}}, {{a4_sh1we_fuzzy_mi_atb}}, {{a4_sh1we_fuzzy_ps_atb}} and {{a4_sh1we_fuzzy_sel_atb}}).")
fig("fig_riskcov4.png", "Fig. 5. Risk–coverage curves of the abstention rules with the fuzzy additive model as the "
    "fixed base model: balanced accuracy among decided participants against the percentage decided, with participants "
    "ranked by each rule's confidence. Open circles mark each rule's own operating point. The grey curve ranks by the "
    "distance of the point score from the threshold, the confidence used by point-score selective prediction (and by "
    "complete cases among complete records). Simulated panels show the first of 10 repetitions; pattern submodels were "
    "not fitted under MCAR.", 6.8)

h2("3.5. Summary across scenarios")
p("Table @SUM@ condenses the 16 scenarios for the fuzzy model, and separates results that held across scenarios from "
  "those confined to a few. Three results held in every scenario with substantial missingness. Worst-case bounds "
  "decided the fewest participants, missed the fewest cases and avoided the fewest questionnaires; their net benefit at "
  "0.10 was never above that of the best other rule except in 2021–2023; and when an input was absent for everyone, "
  "pattern submodels or point-score selective prediction did best on both AURC and net benefit (worst-case AURC up to "
  "{{sum_wc_gap_max}} higher). Two results were confined to particular mechanisms. Worst-case bounds had the lowest "
  "AURC in {{sum_n_wc_aurc_best}} of the 16 scenarios, all with non-random or outcome-dependent masking or in "
  "2021–2023, but by at most {{sum_wc_lead_max}}. Their net benefit at 0.10 was lower than that of the best rule in "
  "{{sum_n_wc_nb_lower}} scenarios, similar or equivalent in {{sum_n_wc_nb_eq}} and highest in {{sum_n_wc_nb_high}} "
  "(2021–2023).")
tab("tableSum")

h2("3.6. Decision consequences")
p("Table 7 converts each rule into the prescreening currency of Section 2.1, under refer-on-abstain. In 2005–2014, per "
  "1,000 participants, worst-case bounds of the fuzzy model referred {{a4_ext_fuzzy_wc_ref}}, missed "
  "{{a4_ext_fuzzy_wc_mis}} cases and avoided {{a4_ext_fuzzy_wc_avd}} questionnaires; imputation spread referred "
  "{{a4_ext_fuzzy_mi_ref}}, missed {{a4_ext_fuzzy_mi_mis}} and avoided {{a4_ext_fuzzy_mi_avd}}; pattern submodels referred "
  "{{a4_ext_fuzzy_ps_ref}}, missed {{a4_ext_fuzzy_ps_mis}} and avoided {{a4_ext_fuzzy_ps_avd}}; point-score selective "
  "prediction referred {{a4_ext_fuzzy_sel_ref}}, missed {{a4_ext_fuzzy_sel_mis}} and avoided {{a4_ext_fuzzy_sel_avd}}. Net "
  "benefit at a threshold probability of 0.10 was {{a4_ext_fuzzy_wc_nb10}} for worst-case bounds; the other rules were "
  "higher by {{dnb_ext_fuzzy_mi_10}} (imputation spread), {{dnb_ext_fuzzy_ps_10}} (pattern submodels) and "
  "{{dnb_ext_fuzzy_sel_10}} (point-score selective prediction). At 0.10, point-score selective prediction minus pattern "
  "submodels was {{selps_fuzzy_dnb10}} ({{selps_fuzzy_dnb10_ci}}); at 0.05 the two were equivalent "
  "({{selps_fuzzy_dnb05}}, {{selps_fuzzy_dnb05_ci}}), so in 2005–2014 the better ranking of the point score did not carry over to "
  "the operating point.")
p("Across scenarios, worst-case bounds always missed the fewest cases and avoided the fewest questionnaires. Their net "
  "benefit at 0.10 was lower than that of the other rules in 2005–2014, under 30% MCAR ({{a4_mc30_fuzzy_wc_nb10}} against "
  "{{a4_mc30_fuzzy_mi_nb10}} and {{a4_mc30_fuzzy_sel_nb10}}; not bootstrapped) and with an input absent and shifted "
  "(differences {{dnb_sh1srh_fuzzy_mi_10}} and {{dnb_sh1we_fuzzy_mi_10}} for imputation spread). It was equivalent in the "
  "internal test set ({{dnb_int_fuzzy_mi_10}} for imputation spread), under 0.9/0.05 masking of self-rated health "
  "({{dnb_m9srh_fuzzy_mi_10}}) and under outcome-dependent masking ({{dnb_odep_fuzzy_mi_10}}). In 2021–2023, where every "
  "model under-predicted, worst-case bounds had the highest net benefit at 0.05 ({{a4_tmp_fuzzy_wc_nb05}} against "
  "{{a4_tmp_fuzzy_mi_nb05}} and {{a4_tmp_fuzzy_ps_nb05}}; differences {{dnb_tmp_fuzzy_mi_05}} and "
  "{{dnb_tmp_fuzzy_ps_05}}), and under outcome-dependent masking their net benefit at 0.05 exceeded that of pattern "
  "submodels ({{dnb_odep_fuzzy_ps_05}}) and point-score selective prediction ({{dnb_odep_fuzzy_sel_05}}). Elsewhere "
  "the differences at 0.05 were equivalent or favoured the other rules. Pattern submodels missed the most cases whenever "
  "missingness was related to the outcome ({{a4_odep_fuzzy_ps_mis}} per 1,000 under outcome-dependent masking, against "
  "{{a4_odep_fuzzy_wc_mis}} for worst-case bounds). Results for the regression base models are in Table A6.")
tab("tableCons")

h2("3.7. Worst-case bounds for conjunctive rule bases")
p("Algorithm 1 agreed with dense brute force on every record. For the conjunctive version 1, a 41-point grid with "
  "trapezoid corners under-covered the exact range of Algorithm 2 in {{ex_v1_nshort}} of {{n_int}} records with 30% of "
  "inputs masked, by at most {{ex_v1_short}} log-odds, and changed no decision; {{ex_v1_fbrows}} records "
  "({{ex_v1_fbrows_pct}}%) needed the outer bound. Against dense brute force, Algorithm 2 never returned a range that "
  "excluded a brute-force value (largest excess {{ex_max_outside}}; Fig. A2).")
h2("3.8. Bound-guided acquisition")
p("Among records on which the fuzzy model abstained under 30% masking, Algorithm 3 requested {{acq2_m30_bound_mean}} "
  "inputs per abstention, against {{acq2_m30_fixed_mean}} for the fixed order (paired difference {{acq2_m30_d_fixed}}, "
  "{{acq2_m30_d_fixed_ci}}; {{acq_per1000_fixed}} fewer inputs per 1,000 abstentions), {{acq2_m30_cost_mean}} for the cost "
  "order ({{acq_per1000_cost}} fewer per 1,000) and {{acq2_m30_random_mean}} for a random order (Table 8). With item "
  "costs of 1 for record items, 2 for income and 3 for interview items, expected cost per abstention was "
  "{{aqc_bound_cost_3}} for the cost-weighted variant, {{aqc_bound_3}} for Algorithm 3, {{aqc_cost_3}} for the cost order "
  "and {{aqc_fixed_3}} for the fixed order; the cost order asked first for cheap record items that often could not settle "
  "the decision, needed more inputs ({{aqc_cost_n}} per abstention) and did not minimise cost ({{aqd_cost_3}} more than the cost-weighted variant, {{aqd_cost_3_ci}}). With interview items "
  "costing 5 the ordering was the same ({{aqc_bound_cost_5}} against {{aqc_cost_5}}). These results come from masking "
  "at random with true values revealed on request; free-day sleep in 2005–2014 cannot be requested.")
tab("tableAcq")

h2("3.9. PHQ-2 reference, record-typical predictors and subgroups")
p("PHQ-2 ≥ 3 had sensitivity {{phq2_int_sens}} and specificity {{phq2_int_spec}} internally and flagged "
  "{{phq2_int_flag}}% of participants; the fuzzy model at its development threshold had sensitivity "
  "{{full_int_Sensitivity}} at specificity {{full_int_Specificity}} and flagged {{full_int_flag}}% (Table 9). The PHQ-2 "
  "figures are inflated by incorporation, because its two items are part of the outcome; against diagnostic "
  "interviews its pooled sensitivity was 0.72 at specificity 0.85 [@levis2020], which is still higher than that of any "
  "non-symptom model here at a far lower referral rate. Internally the fuzzy model's net benefit was "
  "{{nbabs_int_05_full}} at a threshold probability of 0.05, {{nbabs_int_10_full}} at 0.10 and {{nbabs_int_20_full}} at "
  "0.20 (referring everyone: {{nbabs_int_05_all}}, {{nbabs_int_10_all}} and {{nbabs_int_20_all}}; Fig. 7). Restricting "
  "inputs to nine record-typical variables reduced AUROC to {{ehr9_full_int}} internally and {{ehr9_full_tmp}} in "
  "2021–2023. Removing self-rated health had the largest effect of any input ({{sens_nosrh_full_int}} internally and "
  "{{sens_nosrh_full_tmp}} in 2021–2023), removing sleep variables a small one ({{sens_nosleep_full_int}} and "
  "{{sens_nosleep_full_tmp}}).")
tab("tablePHQ2")
fig("fig_dca.png", "Fig. 7. Decision curves of the base models. Net benefit of referring for questionnaire completion "
    "when the predicted probability exceeds the threshold, compared with referring everyone or no one. Dotted: fuzzy "
    "model after recalibration-in-the-large on the same cohort (illustrative upper reference).", 6.6)
p("Sensitivity at the common threshold differed by sex: {{sex_ext_male_sens_common}}% in men and "
  "{{sex_ext_female_sens_common}}% in women in 2005–2014 (logistic regression {{sex_ext_male_sens_lr}}% and "
  "{{sex_ext_female_sens_lr}}%). Men had lower prevalence and were predicted lower risk, so a common threshold flagged "
  "fewer of them. Sex-specific thresholds chosen on development data ({{thr_sex_m}} for men and {{thr_sex_f}} for women, "
  "against the common {{thr_full}}, all on the raw probability scale) brought sensitivity to {{sex_ext_male_sens_sex}}% "
  "and {{sex_ext_female_sens_sex}}% in 2005–2014, at the cost of flagging {{sex_ext_male_flag_sex}}% of men and "
  "{{sex_ext_female_flag_sex}}% of women. Discrimination was lowest in adults aged 18–39 in every set, and low for all "
  "models among Mexican American participants in 2021–2023 ({{sub_tmp_MexicanAmerican_full}}; Table A2). Fig. 8 "
  "shows the explanation trace of one internal-test participant (Section 2.13).")
fig("fig_explanation.png", "Fig. 8. Trace of one internal-test participant with PHQ-9 ≥ 10. Bars show each rule's "
    "contribution to the log-odds (β_r × firing strength), labelled with its signed value; colour repeats the sign. The "
    "title gives the intercept, the sum of the bars and the resulting log-odds and probability.", 5.8)
