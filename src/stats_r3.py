"""Statistics, part 3: starts from paper_stats_r1.json (stats_r1.py) and adds the second set of analyses."""
import json, re
import numpy as np, pandas as pd
R = "/home/claude/results/"; RV = R + "secondary/"
S = json.load(open(R + "paper_stats_r1.json"))
M2 = json.load(open(RV + "sec2_models.json")); X2 = json.load(open(RV + "sec2_missing.json"))
RE = json.load(open(RV + "rev_exact.json")); SP = json.load(open(RV + "rev_splits.json")); main = json.load(open(R + "main_results.json"))
RM = json.load(open(RV + "rev_models.json")); pri = pd.read_csv(R + "priors_2015_2016.csv")
f3 = lambda x: f"{x:.3f}"; pct = lambda x, d=1: f"{100 * x:.{d}f}"
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
def sci(x):
    m_, e_ = f"{x:.1e}".split("e"); return f"{m_} × 10" + str(int(e_)).translate(SUP)
def small(x, nd=4):
    """Signed difference; values that would print as zero are given in scientific notation."""
    if x == 0: return "0"
    if abs(x) < 0.5 * 10 ** (-nd): return ("+" if x > 0 else "−") + sci(abs(x))
    return f"{x:+.{nd}f}"
def pval(p): return "1.000" if p >= 0.9995 else (f"{p:.3f}" if p >= 0.001 else "<0.001")
SETS = {"internal_test": "int", "temporal": "tmp", "external_pooled": "ext", "external_2005-2006": "c0506",
        "external_2007-2008": "c0708", "external_2009-2010": "c0910", "external_2011-2012": "c1112", "external_2013-2014": "c1314"}
short = {"Proposed (full)": "full", "Logistic regression": "lr", "Gradient-boosted trees": "gbt",
         "Version 1 (conjunctive rules)": "v1", "Main effects + interactions": "inter", "No direction constraints": "nodir",
         "No interval missing handling": "noint", "Crisp ontology-guided": "crisp", "Fuzzy without ontology": "noonto",
         "Box missing bounds": "box", "Normalized TSK aggregation": "tsk", "Sign-constrained logistic regression": "slr",
         "Unconstrained linear LR (no indicators)": "ulr", "LR, median imputation without indicators": "lrni",
         "LR, multiple imputation": "lrmi", "Monotone gradient-boosted trees": "mgbt", "Explainable boosting machine (GA2M)": "ebm",
         "LR with restricted cubic splines": "rcs", "Abstain if any input missing": "cc"}

# ---- paired differences: never print ±0.0000 (minor comment)
for src in (main["paired_auroc"], RM["new_paired"]):
    for k, v in src.items():
        st, m = k.split("|"); s = short[m]; sl = SETS[st]
        S[f"d_{s}_{sl}"] = small(v["diff"], 3) if abs(v["diff"]) >= 0.0005 else small(v["diff"], 4)
        S[f"d_{s}_{sl}_ci"] = f"{small(v['lo'], 4)} to {small(v['hi'], 4)}"
for m, s in [(k, short[k]) for k in SP if k != "Proposed (full)"]:
    v = SP[m]; S[f"spl_{s}_mean"] = small(v["mean_diff"], 3) if abs(v["mean_diff"]) >= 0.0005 else small(v["mean_diff"], 4)
    S[f"spl_{s}_lo"] = small(v["lo"], 3) if abs(v["lo"]) >= 0.0005 else small(v["lo"], 4)
    S[f"spl_{s}_hi"] = small(v["hi"], 3) if abs(v["hi"]) >= 0.0005 else small(v["hi"], 4)

v1o = json.load(open(R + "v1/main_results.json"))["paired_auroc"]
for k, sl in (("internal_test|Logistic regression", "int"), ("temporal|Logistic regression", "tmp")):
    v = v1o[k]; S[f"v1old_d_lr_{sl}_ci"] = f"{small(v['lo'], 4)} to {small(v['hi'], 4)}"
# ---- protocol time stamps (from the protocol addendum written at the time; file times were not re-read)
S["t_protocol"] = "13:45:10 UTC on 28 October 2026"; S["t_first_download"] = "13:45:18 UTC on 28 October 2026"

# ---- priors: estimated vs inherited counts
est = pri[pri.note.isna()] if "note" in pri else pri
S["n_pri_est_pos"] = str(int((est.direction == 1).sum())); S["n_pri_est_neg"] = str(int((est.direction == -1).sum()))
S["n_pri_est_zero"] = str(int((est.direction == 0).sum())); S["n_pri_inh"] = str(int(pri.note.notna().sum()))
pr = M2["prior_refcontrast"]
S["refpri_n_changed"] = str(len(pr["changes"]))
sg = {1: "+", -1: "−", 0: "0"}
S["refpri_list"] = "; ".join(f"{c['concept']} versus {c['reference']} {sg[c['direction_crude']]} to {sg[c['direction_reference_contrast']]}" for c in pr["changes"])
S["refpri_maxbeta"] = f"{pr['max_abs_beta_change']:.4f}"
for st, sl in (("internal_test", "int"), ("temporal", "tmp"), ("external_pooled", "ext")): S[f"refpri_{sl}"] = f3(pr["auc"][st])

# ---- spline LR
o = M2["rcs"]
for st, sl in SETS.items():
    v = o["sets"][st]; S[f"rcs_{sl}_AUROC"] = f3(v["AUROC"]); lo, hi = v["CI"]["AUROC"]; S[f"rcs_{sl}_AUROC_ci"] = f"{lo:.3f}–{hi:.3f}"
    for k in ["AUPRC", "Sensitivity", "Specificity"]: S[f"rcs_{sl}_{k}"] = f3(v[k])
    S[f"rcs_{sl}_Brier"] = f"{v['Brier']:.4f}"; S[f"rcs_{sl}_OE"] = f"{v['OE']:.2f}"; S[f"rcs_{sl}_slope"] = f"{v['cal_slope']:.2f}"
    S[f"rcs_{sl}_wAUROC"] = f3(v["AUROC_weighted"]); S[f"rcs_{sl}_flag"] = pct(v["Flagged"])
    p = M2["rcs_paired"][st]
    S[f"d_rcs_{sl}"] = small(p["diff"], 3); S[f"d_rcs_{sl}_ci"] = f"{small(p['lo'], 4)} to {small(p['hi'], 4)}"; S[f"d_rcs_{sl}_p"] = pval(p["p_two_sided"])
    S[f"d_rcs_{sl}_abs"] = f"{abs(p['diff']):.3f}"
# ---- refit without free-day sleep
for nm, d in M2["no_sleep_we"].items():
    s = short[nm]
    for st, sl in (("internal_test", "int"), ("temporal", "tmp"), ("external_pooled", "ext")):
        S[f"nowe_{s}_{sl}"] = f3(d[st]["AUROC"]); S[f"nowe_{s}_{sl}_OE"] = f"{d[st]['OE']:.2f}"
for k, v in M2["no_sleep_we_paired"].items():
    st, nm = k.split("|"); S[f"dnowe_{short[nm]}_{SETS[st]}"] = small(v["diff"], 3); S[f"dnowe_{short[nm]}_{SETS[st]}_ci"] = f"{small(v['lo'], 4)} to {small(v['hi'], 4)}"

# ---- natural missingness
lab = {"internal_test": "int", "temporal": "tmp", "external_pooled": "ext"} | {f"external_{k}": f"c{k}" for k in "DEFGH"}
for st, sl in lab.items():
    S[f"nat_{sl}_anymiss"] = pct(X2["natural"][st + "_missing_any"]); S[f"nat_{sl}_pirmiss"] = pct(X2["natural"][st + "_pir_missing"])
    for r in X2["natural"][st]:
        s = short[r["model"]]; kind = r["kind"]
        if kind == "bounds":
            for k in ("Coverage", "Abstain", "Sens_refer", "Spec_refer", "Flag_refer", "Sens_covered", "Spec_covered", "Coverage_p99", "Share_decided_with_we_missing"):
                if r.get(k) is not None: S[f"nat_{sl}_{s}_{k}"] = pct(r[k])
            for k in ("NB_refer_0.05", "NB_refer_0.1"):
                if r.get(k) is not None: S[f"nat_{sl}_{s}_{k.replace('.', '')}"] = f3(r[k])
        elif kind == "point":
            S[f"natp_{sl}_{s}_AUROC"] = f3(r["AUROC"]); S[f"natp_{sl}_{s}_Sens"] = pct(r["Sens"]); S[f"natp_{sl}_{s}_Spec"] = pct(r["Spec"]); S[f"natp_{sl}_{s}_Flag"] = pct(r["Flag"])
            for k in ("NB_0.05", "NB_0.1"): S[f"natp_{sl}_{s}_{k.replace('.', '')}"] = f3(r[k])
        elif kind == "matched":
            S[f"natm_{sl}_{s}_Sens"] = pct(r["Sens"]); S[f"natm_{sl}_{s}_Spec"] = pct(r["Spec"]); S[f"natm_{sl}_{s}_Flag"] = pct(r["Flag"])
# ---- stress (MCAR on the same masks, MNAR)
for r in X2["stress"]:
    lv = str(r["level"]).replace("_", ""); s = short[r["model"]]; kind = r["kind"]; sc = "mc" if r["scenario"] == "MCAR" else "mn"
    pre = {"bounds": "sb", "point": "sp", "matched": "sm"}[kind] + sc
    for k, v in r.items():
        if k in ("scenario", "level", "model", "kind") or v is None or (isinstance(v, float) and np.isnan(v)): continue
        kk = k.replace(".", "")
        S[f"{pre}_{lv}_{s}_{kk}"] = f3(v) if (k == "AUROC" or k.startswith("NB") or k == "MeanWidth") else pct(v)
st_ = pd.DataFrame(X2["stress"]); mm = st_[(st_.scenario == "MCAR") & (st_.level == "30") & (st_.kind == "matched") & (st_.model != "Proposed (full)")]
S["smmc_30_min_Sens"] = pct(mm.Sens.min()); S["smmc_30_max_Sens"] = pct(mm.Sens.max())
mm = st_[(st_.scenario == "MCAR") & (st_.level == "30") & (st_.kind == "matched")]
S["smmc_30_minall_Sens"] = pct(mm.Sens.min()); S["smmc_30_maxall_Sens"] = pct(mm.Sens.max())
# ---- acquisition (paired)
for scen, key in (("masked10_internal", "m10"), ("masked30_internal", "m30"), ("masked30_temporal", "t30")):
    a = X2["acquisition2"][scen]
    for pol in ("bound", "fixed", "cost", "random"): S[f"acq2_{key}_{pol}_mean"] = f"{a[pol + '_mean_all']:.2f}"
    for o_ in ("fixed", "cost", "random"):
        d, lo, hi = a[f"diff_{o_}"]; S[f"acq2_{key}_d_{o_}"] = f"{d:.2f}"; S[f"acq2_{key}_d_{o_}_ci"] = f"{lo:.2f} to {hi:.2f}"
        S[f"acq2_{key}_fewer_{o_}"] = pct(a[f"frac_fewer_{o_}"]); S[f"acq2_{key}_more_{o_}"] = pct(a[f"frac_more_{o_}"])
S["cost_order"] = ", ".join(X2["cost_order"])
# ---- decision curves at selected thresholds
for st, sl in (("internal_test", "int"), ("temporal", "tmp"), ("external_pooled", "ext")):
    d = RM["dca"][st]; t = np.array(d["thresholds"])
    for pt in (0.05, 0.10, 0.15, 0.20):
        i = int(np.argmin(np.abs(t - pt))); k = f"{int(round(pt * 100)):02d}"
        S[f"dca_{sl}_{k}_full"] = f3(d["Proposed (full)"][i]); S[f"dca_{sl}_{k}_all"] = f3(d["treat_all"][i])
        S[f"dca_{sl}_{k}_lr"] = f3(d["Logistic regression"][i]); S[f"dca_{sl}_{k}_ebm"] = f3(d["Explainable boosting machine (GA2M)"][i])
        S[f"dca_{sl}_{k}_dall"] = small(d["Proposed (full)"][i] - d["treat_all"][i], 3)
        S[f"dca_{sl}_{k}_dlr"] = small(d["Proposed (full)"][i] - d["Logistic regression"][i], 3)
# ---- exactness denominators
S["ex_v1_fbrows_pct"] = pct(RE["Version 1 (conjunctive rules)"]["n_fallback_rows"] / RE["Version 1 (conjunctive rules)"]["n"])
S["ex_inter_fbrows_pct"] = pct(RE["Main effects + interactions"]["n_fallback_rows"] / RE["Main effects + interactions"]["n"])
CR = json.load(open(RV + "contrib_ranges.json"))
for v in CR["fuzzy"]: S[f"crf_{v}"] = f"{CR['fuzzy'][v]:.2f}"; S[f"crl_{v}"] = f"{CR['linear'].get(v, 0):.2f}"
S["repo_url"] = "https://github.com/{{GITHUB}}"; S["zenodo_doi"] = "{{ZENODO}}"
for k in list(S):
    if isinstance(S[k], str): S[k] = re.sub(r"(?<![\w.])-(?=\d)", "−", S[k])
X3 = json.load(open(RV + "sec3.json"))
scen = {"natural|internal_test": "int", "natural|temporal": "tmp", "natural|external_pooled": "ext", "mcar30": "mc30",
        "mnar|income_low": "mninc", "mnar|srh_poor": "mnsrh", "mnar|both": "mnboth"}
for k, sk in scen.items():
    v = X3["mi_abstention"][k]
    for meth in ("bounds", "linear", "mi"):
        for part in ("own", "at_bounds_cov"):
            for m_, val in v[meth][part].items():
                S[f"ab_{sk}_{meth}_{'own' if part == 'own' else 'at'}_{m_}"] = f3(val) if m_ in ("bacc", "err") else pct(val)
for st, sl in (("internal_test", "int"), ("temporal", "tmp"), ("external_pooled", "ext")):
    for sx in ("male", "female"):
        for m_, val in X3["sex"][st][sx].items():
            if val is None: continue
            S[f"sex_{sl}_{sx}_{m_}"] = (f"{val:,}" if isinstance(val, int) else (f3(val) if m_ in ("auroc",) else (f"{val:.2f}" if m_ == "oe" else pct(val))))
S["thr_sex_m"] = f3(X3["sex"]["thr_sex"]["0"]); S["thr_sex_f"] = f3(X3["sex"]["thr_sex"]["1"])
for k, v in X3["rcs_full10"].items(): S[f"spmc_{k}_rcs_AUROC"] = f3(v["AUROC"]); S[f"sbmc_{k}_rcs_Coverage"] = pct(v["Coverage"])
S["crl3_srh"] = f"{CR['linear']['srh']:.3f}"; S["crl3_n_chronic"] = f"{CR['linear']['n_chronic']:.3f}"
d_ = RM["dca"]["internal_test"]; import numpy as _np
for st, sl in (("internal_test", "int"), ("temporal", "tmp"), ("external_pooled", "ext")):
    d_ = RM["dca"][st]; tt = _np.array(d_["thresholds"])
    for pt in (0.05, 0.10, 0.20):
        i = int(_np.argmin(_np.abs(tt - pt))); k = f"{int(round(pt * 100)):02d}"
        S[f"nbabs_{sl}_{k}_full"] = f"{d_['Proposed (full)'][i]:.3f}"; S[f"nbabs_{sl}_{k}_all"] = f"{d_['treat_all'][i]:.3f}"
for k in list(S):
    if isinstance(S[k], str): S[k] = re.sub(r"(?<![\w.])-(?=\d)", "−", S[k])
json.dump(S, open(R + "paper_stats_r3.json", "w"), indent=0, ensure_ascii=False)
print(len(S))
