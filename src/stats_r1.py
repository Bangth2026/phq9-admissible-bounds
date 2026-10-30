"""Statistics, part 2: starts from paper_stats.json (stats.py) and adds the secondary analyses."""
import json, re
import numpy as np, pandas as pd
R = "/home/claude/results/"; RV = R + "secondary/"
S = json.load(open(R + "paper_stats.json"))
RM = json.load(open(RV + "rev_models.json")); RX = json.load(open(RV + "rev_missing.json"))
RE = json.load(open(RV + "rev_exact.json")); SP = json.load(open(RV + "rev_splits.json"))
PM = json.load(open(RV + "rev_priors_mv.json")); PA = json.load(open(RV + "rev_priors.json"))
main = json.load(open(R + "main_results.json")); rec = json.load(open(R + "recalibration.json"))
f3 = lambda x: f"{x:.3f}"; pct = lambda x, d=1: f"{100 * x:.{d}f}"
def pval(p): return "1.000" if p >= 0.9995 else (f"{p:.3f}" if p >= 0.001 else "<0.001")
SETS = {"internal_test": "int", "temporal": "tmp", "external_pooled": "ext", "external_2005-2006": "c0506",
        "external_2007-2008": "c0708", "external_2009-2010": "c0910", "external_2011-2012": "c1112", "external_2013-2014": "c1314"}
NEW = {"Sign-constrained logistic regression": "slr", "Unconstrained linear LR (no indicators)": "ulr",
       "LR, median imputation without indicators": "lrni", "LR, multiple imputation": "lrmi",
       "Monotone gradient-boosted trees": "mgbt", "Explainable boosting machine (GA2M)": "ebm"}
short = {"Proposed (full)": "full", "Logistic regression": "lr", "Gradient-boosted trees": "gbt",
         "Version 1 (conjunctive rules)": "v1", "Main effects + interactions": "inter", "No direction constraints": "nodir",
         "No interval missing handling": "noint", "Crisp ontology-guided": "crisp", "Fuzzy without ontology": "noonto",
         "Box missing bounds": "box", "Normalized TSK aggregation": "tsk"} | NEW

# ---- paired differences with more decimals: overwrite base keys
def dfmt(x): return f"{x:+.4f}" if abs(x) < 0.0005 else f"{x:+.3f}"
for src in (main["paired_auroc"], RM["new_paired"]):
    for k, v in src.items():
        st, m = k.split("|"); s = short[m]; sl = SETS[st]
        S[f"d_{s}_{sl}"] = dfmt(v["diff"]); S[f"d_{s}_{sl}_ci"] = f"{v['lo']:+.4f} to {v['hi']:+.4f}"
        S[f"d_{s}_{sl}_p"] = pval(v["p_two_sided"]); S[f"d_{s}_{sl}_abs"] = f"{abs(v['diff']):.3f}"
# ---- new comparators' performance
for m, s in NEW.items():
    o = RM["new_models"][m]; S[f"thr_{s}"] = f"{o['threshold']:.3f}"
    for st, sl in SETS.items():
        v = o["sets"][st]; S[f"{s}_{sl}_AUROC"] = f3(v["AUROC"]); lo, hi = v["CI"]["AUROC"]; S[f"{s}_{sl}_AUROC_ci"] = f"{lo:.3f}–{hi:.3f}"
        for k in ["AUPRC", "Sensitivity", "Specificity", "PPV", "NPV"]: S[f"{s}_{sl}_{k}"] = f3(v[k])
        S[f"{s}_{sl}_Brier"] = f"{v['Brier']:.4f}"; S[f"{s}_{sl}_OE"] = f"{v['OE']:.2f}"; S[f"{s}_{sl}_slope"] = f"{v['cal_slope']:.2f}"
        S[f"{s}_{sl}_wAUROC"] = f3(v["AUROC_weighted"]); S[f"{s}_{sl}_flag"] = pct(v["Flagged"])
S["slr_lambda"] = f"{RM['new_models']['Sign-constrained logistic regression']['lambda']:g}"
S["mgbt_params"] = ", ".join(f"{k}={v}" for k, v in RM["new_models"]["Monotone gradient-boosted trees"]["params"].items())
ext_d = [RM["new_paired"][f"external_pooled|{m}"]["diff"] for m in NEW if m != "Unconstrained linear LR (no indicators)"] + \
        [main["paired_auroc"]["external_pooled|Gradient-boosted trees"]["diff"]]
S["dext_min"] = f"{min(ext_d):+.3f}"; S["dext_max"] = f"{max(ext_d):+.3f}"
mains = ["Proposed (full)", "Logistic regression", "Gradient-boosted trees"] + [m for m in NEW if m != "Unconstrained linear LR (no indicators)"]
def auc(m, st): return (RM["new_models"][m] if m in NEW else main[m])["sets"][st]["AUROC"]
oth = [m for m in mains if m != "Proposed (full)"]
S["int_min_oth"] = f3(min(auc(m, "internal_test") for m in oth)); S["int_max_oth"] = f3(max(auc(m, "internal_test") for m in oth))
S["tmp_min_auc"] = f3(min(auc(m, "temporal") for m in mains)); S["tmp_max_auc"] = f3(max(auc(m, "temporal") for m in mains))
gaps = [v for d in RM["wgap"].values() for v in d.values()]
S["wgap_min"] = f"{min(gaps):.3f}"; S["wgap_max"] = f"{max(gaps):.3f}"
# ---- repeated splits
S["spl_B"] = "20"
S["spl_full_mean"] = f3(SP["Proposed (full)"]["mean_auc"]); S["spl_full_sd"] = f3(SP["Proposed (full)"]["sd"])
for m, s in {"Logistic regression": "lr", "Version 1 (conjunctive rules)": "v1", "Main effects + interactions": "inter",
             "No direction constraints": "nodir", "Crisp ontology-guided": "crisp", "Fuzzy without ontology": "noonto",
             "Sign-constrained logistic regression": "slr", "Gradient-boosted trees": "gbt", "Explainable boosting machine (GA2M)": "ebm"}.items():
    v = SP[m]
    S[f"spl_{s}_mean"] = f"{v['mean_diff']:+.3f}"; S[f"spl_{s}_lo"] = f"{v['lo']:+.3f}"; S[f"spl_{s}_hi"] = f"{v['hi']:+.3f}"
    S[f"spl_{s}_frac"] = f"{100 * v['frac_pos']:.0f}"; S[f"spl_{s}_frac_neg"] = f"{100 * (1 - v['frac_pos']):.0f}"
    S[f"spl_{s}_auc"] = f3(v["mean_auc"]); S[f"spl_{s}_abs"] = f"{abs(v['mean_diff']):.3f}"
# ---- constraints at same K, priors
S["v1_con_int"] = f3(RM["constrained_same_K|Version 1 (conjunctive rules)"]["internal_test"])
S["v1_unc_int"] = f3(RM["unconstrained_same_K|Version 1 (conjunctive rules)"]["internal_test"])
S["v1_con_ext"] = f3(RM["constrained_same_K|Version 1 (conjunctive rules)"]["external_pooled"])
S["v1_unc_ext"] = f3(RM["unconstrained_same_K|Version 1 (conjunctive rules)"]["external_pooled"])
ch = [c for c in PM["changes"] if c["concept"] not in PM["references"].values()]
S["mvpri_n_changed"] = str(len(ch))
sg = {1: "+", -1: "−", 0: "0"}
S["mvpri_changed_list"] = "; ".join(f"{c['concept']} {sg[c['direction_crude']]} to {sg[c['direction_mv']]}" for c in ch)
for lab, s in (("Proposed", "full"), ("Version 1", "v1")):
    for st, sl in (("internal_test", "int"), ("temporal", "tmp"), ("external_pooled", "ext")):
        S[f"mvpri_{s}_{sl}"] = f3(PM[lab][st])
S["adjpri_n_changed"] = str(len(PA["prior_changes"]))
# ---- PHQ-2 and operating points
for st, sl in (("internal_test", "int"), ("temporal", "tmp"), ("external_pooled", "ext")):
    v = RM["phq2"][st]
    S[f"phq2_{sl}_sens"] = f3(v["Sensitivity"]); S[f"phq2_{sl}_spec"] = f3(v["Specificity"]); S[f"phq2_{sl}_flag"] = pct(v["Flagged"])
    S[f"phq2_{sl}_ppv"] = f3(v["PPV"]); S[f"phq2_{sl}_auc"] = f3(v["AUROC_phq2"])
    prev = main["pos"][st] / main["n"][st]
    S[f"phq2_{sl}_avoid"] = f"{1000 * (1 - v['Flagged']):.0f}"; S[f"phq2_{sl}_missed"] = f"{1000 * prev * (1 - v['Sensitivity']):.0f}"
for k, v in RM["operating"].items():
    st, m = k.split("|"); s = short[m]; sl = SETS[st]
    b = v["before"]; S[f"op_{sl}_{s}_sens_b"] = f3(b[0]); S[f"op_{sl}_{s}_spec_b"] = f3(b[1]); S[f"op_{sl}_{s}_flag_b"] = pct(b[2])
    S[f"op_{sl}_{s}_avoid_b"] = f"{b[3]:.0f}"; S[f"op_{sl}_{s}_missed_b"] = f"{b[4]:.0f}"
    if sl == "int": S[f"op_int_{s}_avoid"] = S[f"op_int_{s}_avoid_b"]; S[f"op_int_{s}_missed"] = S[f"op_int_{s}_missed_b"]
    if "after" in v:
        a = v["after"]; S[f"op_{sl}_{s}_sens_a"] = f3(a[0]); S[f"op_{sl}_{s}_spec_a"] = f3(a[1]); S[f"op_{sl}_{s}_flag_a"] = pct(a[2])
        S[f"op_{sl}_{s}_avoid_a"] = f"{a[3]:.0f}"; S[f"op_{sl}_{s}_missed_a"] = f"{a[4]:.0f}"
# ---- decision curves: threshold range where proposed beats both defaults
for st, sl in (("internal_test", "int"), ("temporal", "tmp"), ("external_pooled", "ext")):
    d = RM["dca"][st]; t = np.array(d["thresholds"]); nb = np.array(d["Proposed (full)"]); ta = np.array(d["treat_all"])
    ok = (nb > ta + 1e-9) & (nb > 1e-9)
    S[f"dca_{sl}_full_lo"] = f"{t[ok].min():.2f}" if ok.any() else "–"; S[f"dca_{sl}_full_hi"] = f"{t[ok].max():.2f}" if ok.any() else "–"
# ---- record-typical models
for lab, key in (("EHR9", "ehr9"), ("EHR7", "ehr7")):
    for m, s in (("Proposed", "full"), ("Logistic regression", "lr"), ("Gradient-boosted trees", "gbt")):
        for st, sl in (("internal_test", "int"), ("temporal", "tmp"), ("external_pooled", "ext")):
            v = RM["record_typical"][lab][m][st]
            S[f"{key}_{s}_{sl}"] = f3(v["AUROC"]); S[f"{key}_{s}_{sl}_ci"] = f"{v['CI'][0]:.3f}–{v['CI'][1]:.3f}"
            S[f"{key}_{s}_{sl}_sens"] = f3(v["Sensitivity"]); S[f"{key}_{s}_{sl}_spec"] = f3(v["Specificity"]); S[f"{key}_{s}_{sl}_flag"] = pct(v["Flagged"])
# ---- exactness
ex = RX["exactness"]
S["ex_full_short"] = f"{RE['Proposed (full)']['max_short']:.1e}".replace("e-16", " × 10⁻¹⁶").replace("e-15", " × 10⁻¹⁵")
S["ex_v1_nshort"] = str(RE["Version 1 (conjunctive rules)"]["n_short"]); S["ex_v1_short"] = f"{RE['Version 1 (conjunctive rules)']['max_short']:.4f}"
S["ex_v1_dec"] = str(RE["Version 1 (conjunctive rules)"]["dec_changed"])
S["ex_v1_fb"] = f"{100 * RE['Version 1 (conjunctive rules)']['fallback_components'] / RE['Version 1 (conjunctive rules)']['components']:.1f}"
S["ex_v1_fbrows"] = str(RE["Version 1 (conjunctive rules)"]["n_fallback_rows"])
S["ex_inter_fbrows"] = str(RE["Main effects + interactions"]["n_fallback_rows"]); S["ex_inter_nshort"] = str(RE["Main effects + interactions"]["n_short"])
S["ex_max_outside"] = f"{max(v['brute_max_outside'] for v in ex.values()):.1e}".replace("e-16", " × 10⁻¹⁶").replace("e-15", " × 10⁻¹⁵")
S["ex_brute_max"] = f"{max(v['brute_max_abs_logodds'] for v in ex.values()):.4f}"
S["ex_full_sec"] = f"{ex['Proposed (full)']['seconds_exact']:.1f}"; S["ex_v1_sec"] = f"{ex['Version 1 (conjunctive rules)']['seconds_exact']:.1f}"
S["ex_inter_sec"] = f"{ex['Main effects + interactions']['seconds_exact']:.1f}"
# ---- ranges
for v, d in RX["ranges"].items():
    S[f"rng_{v}"] = f"{d['admissible'][0]:g}–{d['admissible'][1]:g}"; S[f"rng99_{v}"] = f"{d['p01_p99'][0]:.1f}–{d['p01_p99'][1]:.1f}"
for lvl, d in RX["narrowed"].items():
    k = str(int(float(lvl) * 100)); S[f"nar_{k}_adm"] = pct(d["admissible"]["coverage"]); S[f"nar_{k}_p01"] = pct(d["p01_p99"]["coverage"])
    S[f"nar_{k}_admw"] = f3(d["admissible"]["width"]); S[f"nar_{k}_p01w"] = f3(d["p01_p99"]["width"])
NS = json.load(open(RV + "narrowed_slr.json"))
for lvl, d in NS.items():
    k = str(int(float(lvl) * 100)); S[f"slrnar_{k}_adm"] = pct(d["admissible"]["coverage"]); S[f"slrnar_{k}_p01"] = pct(d["p01_p99"]["coverage"])
    S[f"slrnar_{k}_admw"] = f3(d["admissible"]["width"]); S[f"slrnar_{k}_p01w"] = f3(d["p01_p99"]["width"])
# ---- stress tests
stc = {"Abstain if any input missing": "cc"} | short
for r in RX["stress"]:
    lvl = str(r["level"]).replace(" ", "").replace("-", ""); s = stc[r["model"]]
    for k in ["AUROC", "AUROC_unclipped", "MeanWidth"]:
        if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k])): S[f"st_{s}_{lvl}_{k}"] = f3(r[k])
    for k in ["Coverage", "Sens_refer", "Spec_refer", "Sens_covered", "Spec_covered", "eff_mask_rate", "Sens_point", "Spec_point"]:
        if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k])): S[f"st_{s}_{lvl}_{k}"] = pct(r[k])
    if r.get("clip_changed") is not None and not np.isnan(r["clip_changed"]): S[f"st_{s}_{lvl}_clip"] = pct(r["clip_changed"])
# ---- acquisition
for scen, key in (("masked10_internal", "m10"), ("masked30_internal", "m30"), ("masked30_temporal", "t30")):
    for pol, v in RX["acquisition"][scen].items():
        S[f"acq_{key}_{pol}_n"] = f"{v['n']:,}"; S[f"acq_{key}_{pol}_resolved"] = pct(v["resolved"])
        S[f"acq_{key}_{pol}_mean"] = f"{v['mean_asked']:.2f}"; S[f"acq_{key}_{pol}_one"] = pct(v["one_question"])
        S[f"acq_{key}_{pol}_max"] = str(v["max_asked"]); S[f"acq_{key}_{pol}_sens"] = pct(v["sens_after"]); S[f"acq_{key}_{pol}_spec"] = pct(v["spec_after"])
S["acq_order"] = ", ".join(RX["acq_fixed_order"])
# ---- interpretability
it = RM["interp"]
S["full_terms"] = str(it["full_terms"]); S["full_active_med"] = f"{it['full_active_median']:.0f}"; S["full_active_p90"] = f"{it['full_active_p90']:.0f}"
S["lr_terms"] = str(it["lr_terms"]); S["lr_n_ind"] = str(sum(n.startswith("missingindicator") for n in it["lr_names"]))
S["ebm_terms"] = str(RM["ebm_terms"]); S["ebm_n_pairs"] = str(len(RM["ebm_pairs"]))
ex_ = json.load(open(R + "example_case.json"))
S["ex_diab"] = f"{[r for r in ex_['trace']['rules'] if r['rule'] == 'Diabetes'][0].get('logit_contribution', 0):+.3f}"
# ---- subgroup O/E
so = pd.read_csv(RV + "subgroup_oe.csv")
r_ = so[(so.set == "temporal") & (so.group == "Race/ethnicity: Mexican American")].iloc[0]
S["oe_tmp_MexicanAmerican_full"] = f"{r_['Proposed (full) OE']:.2f}"; S["oe_tmp_all_full"] = S["full_tmp_OE"]
S["sub_oe_desc"] = "larger than overall" if r_["Proposed (full) OE"] > main["Proposed (full)"]["sets"]["temporal"]["OE"] else "no larger than overall"
S["repo_url"] = "[GitHub repository URL]"; S["zenodo_doi"] = "[Zenodo DOI]"
for k in list(S):
    if isinstance(S[k], str): S[k] = re.sub(r"(?<![\w.])-(?=\d)", "−", S[k])
json.dump(S, open(R + "paper_stats_r1.json", "w"), indent=0, ensure_ascii=False)
print(len(S))
