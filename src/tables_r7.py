"""Tables of version 7: tables_r6 with net-benefit differences at 0.05 and 0.10 in the decision-consequence tables,
and intervals that pool all repetitions for simulated scenarios."""
from tables_r6 import tables as tables_r6
from tables_r4 import RULES, PH

SCN7 = [("int", "Internal test, natural"), ("tmp", "2021–2023, natural"), ("ext", "2005–2014, natural"), ("mc30", "30% MCAR"),
        ("m6srh", "MNAR 0.6/0.1: health" + PH), ("m9srh", "MNAR 0.9/0.05: health" + PH), ("odep", "Outcome-dependent" + PH),
        ("sh0srh", "Health absent, γ = 0" + PH), ("sh1srh", "Health absent, γ = 1" + PH),
        ("sh0we", "Free-day sleep absent, γ = 0" + PH), ("sh1we", "Free-day sleep absent, γ = 1" + PH)]


def tables(V):
    T = tables_r6(V)

    def cons_rows(m):
        rr = []
        for sc, lab in SCN7:
            for j, (r, rl) in enumerate(RULES):
                if r == "ps" and sc == "mc30": continue
                k = f"a4_{sc}_{m}_{r}_"
                d05 = "–" if r == "wc" else "{{dnb_%s_%s_%s_05}}" % (sc, m, r)
                d10 = "–" if r == "wc" else "{{dnb_%s_%s_%s_10}}" % (sc, m, r)
                rr.append([(lab + " ({{prv_%s}}%%)" % sc) if j == 0 else "", rl, "{{%sref}}" % k, "{{%smis}}" % k, "{{%savd}}" % k,
                           "{{%ssensr}}" % k, "{{%snb05}}" % k, d05, "{{%snb10}}" % k, d10])
        return rr
    head = ["Scenario (prevalence, %)", "Rule", "Referred", "Cases missed", "Questionnaires avoided", "Sensitivity (%)",
            "Net benefit at 0.05", "ΔNB at 0.05 vs worst-case (95% CI)", "Net benefit at 0.10", "ΔNB at 0.10 vs worst-case (95% CI)"]
    note = ("Refer-on-abstain: every abstention is referred, and decided participants follow the rule's decision (pattern submodels decide everyone). "
            "Questionnaires avoided = participants not referred. Net benefit = (true positives − false positives × p_t/(1 − p_t))/n at threshold "
            "probability p_t; referring no one gives 0; complete-case abstention refers everyone when an input is absent. ΔNB: difference to worst-case "
            "bounds with 95% percentile bootstrap interval (cohorts: 1,000 paired resamples, 500 in 2005–2014; simulated scenarios: 200 resamples within "
            "each of the 10 repetitions, pooled, for the fuzzy model; regression base models: first repetition); n.c., not computed. Simulated "
            "scenarios are means of 10 repetitions. ‡ Specified after the original results were known.")
    T["tableCons"].update(head=head, rows=cons_rows("fuzzy"), note=note + " The same quantities for the regression base models are in Table A6.",
                          w=[1.35, 1.2, 0.5, 0.5, 0.65, 0.55, 0.55, 1.05, 0.55, 1.05], tiny=True)
    T["tableA6"].update(head=head, rows=[["Sign-constrained LR ‡"] + [""] * 9] + cons_rows("slr") + [["Spline LR ‡"] + [""] * 9] + cons_rows("rcs"),
                        note=note, w=T["tableCons"]["w"], tiny=True)
    T["tableSum"]["note"] = ("Rules compared: worst-case bounds, imputation spread, pattern submodels (not under MCAR) and point-score selective "
                             "prediction; complete cases are omitted because they decide nobody when an input is absent. 'Lower' and 'equivalent': the 95% "
                             "bootstrap interval for the difference in net benefit between the best rule and worst-case bounds excludes or includes zero "
                             "(cohorts: paired resamples; simulated scenarios: resamples within each of the 10 repetitions, pooled). Where worst-case bounds "
                             "had the highest point estimate, the comparison is with the next rule. Simulated scenarios are means of 10 repetitions. Details "
                             "for all base models are in Tables @T3@, @T5@, A6 and A13.")
    T["tableSum"]["head"][-1] = "Worst-case net benefit at 0.10 vs best other rule"
    return T
