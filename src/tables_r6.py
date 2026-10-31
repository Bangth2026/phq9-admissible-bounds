"""Tables of version 6: tables_r4 plus a summary of the abstention rules across the 16 missingness scenarios."""
from tables_r4 import tables as tables_r4

SCN = [("int", "Internal test, as observed", "stable"), ("tmp", "2021–2023, as observed", "stable"), ("ext", "2005–2014, as observed (free-day sleep never collected)", "stable"),
       ("mc30", "30% MCAR", "stable"), ("m6inc", "MNAR 0.6/0.1: income", "MNAR"), ("m6srh", "MNAR 0.6/0.1: self-rated health", "MNAR"),
       ("m6both", "MNAR 0.6/0.1: both", "MNAR"), ("m9inc", "MNAR 0.9/0.05: income", "MNAR"), ("m9srh", "MNAR 0.9/0.05: self-rated health", "MNAR"),
       ("odep", "Outcome-dependent", "MNAR"), ("sh0srh", "Self-rated health absent, γ = 0", "absent"), ("sh05srh", "Self-rated health absent, γ = 0.5", "absent"),
       ("sh1srh", "Self-rated health absent, γ = 1", "absent"), ("sh0we", "Free-day sleep absent, γ = 0", "absent"),
       ("sh05we", "Free-day sleep absent, γ = 0.5", "absent"), ("sh1we", "Free-day sleep absent, γ = 1", "absent")]
GROUP = {"stable": "", "MNAR": "", "absent": ""}


def tables(V):
    T = tables_r4(V)
    T["tableS5"]["note"] = T["tableS5"]["note"].replace("used in Table 9.", "used in Table A9.")
    rows = []
    for sc, lab, g in SCN:
        rows.append([lab, "{{sum_%s_ba}} ({{sum_%s_ba_v}})" % (sc, sc), "{{sum_%s_wc_a}}" % sc,
                     "{{sum_%s_bn}} ({{sum_%s_bn_v}})" % (sc, sc), "{{sum_%s_wc_n}}" % sc, "{{sum_%s_verdict}}" % sc])
    T["tableSum"] = dict(
        cap="Table @SUM@. Summary across the 16 missingness scenarios, fuzzy additive base model: the rule with the lowest AURC and the rule with the highest net benefit at 0.10, and how worst-case bounds compared.",
        head=["Scenario", "Lowest AURC: rule (value)", "AURC, worst-case bounds", "Highest net benefit at 0.10: rule (value)", "Net benefit at 0.10, worst-case bounds",
              "Worst-case net benefit vs best rule"],
        rows=rows,
        note="Rules compared: worst-case bounds, imputation spread, pattern submodels (not under MCAR) and point-score selective prediction; complete cases are omitted "
             "because they decide nobody when an input is absent. 'Equivalent' and 'lower': paired bootstrap interval for the difference includes or excludes zero "
             "(cohorts and the first repetition of four simulated scenarios); 'no interval': no bootstrap was run, and differences above 0.005 are called lower. "
             "Simulated scenarios are means of 10 repetitions. Details for all base models are in Tables @T3@, @T5@, A6 and A13.",
        w=[2.2, 1.45, 0.75, 1.55, 0.8, 1.15], small=True)
    return T
