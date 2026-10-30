"""Collect every number used in the manuscript into one file (single source of truth). Version 2."""
import json, re, pickle, sys
import pandas as pd, numpy as np
R = "/home/claude/results/"
sys.path.insert(0, "/home/claude/src")
from experiment import LRModel, GBTModel  # noqa: needed for unpickling

flow = json.load(open(R + "flow.json")); main = json.load(open(R + "main_results.json"))
v1 = json.load(open(R + "v1/main_results.json"))
sec = json.load(open(R + "secondary_results.json")); rec = json.load(open(R + "recalibration.json"))
pri = pd.read_csv(R + "priors_2015_2016.csv"); rb = pd.read_csv(R + "rulebase_table.csv")
ex = json.load(open(R + "example_case.json")); exm = json.load(open(R + "example_missing.json"))
S = {}
def f3(x): return f"{x:.3f}"
def pct(x, d=1): return f"{100*x:.{d}f}"
def sci(x):
    m_, e_ = f"{x:.1e}".split("e")
    return f"{m_} × 10" + str(int(e_)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
def pval(p): return "1.00" if p >= 0.995 else (f"{p:.3f}" if p >= 0.001 else "<0.001")

SETS = {"internal_test": "int", "temporal": "tmp", "external_pooled": "ext", "external_2005-2006": "c0506",
        "external_2007-2008": "c0708", "external_2009-2010": "c0910", "external_2011-2012": "c1112",
        "external_2013-2014": "c1314"}
short = {"Proposed (full)": "full", "Logistic regression": "lr", "Gradient-boosted trees": "gbt",
         "Version 1 (conjunctive rules)": "v1", "Main effects + interactions": "inter",
         "No direction constraints": "nodir", "No interval missing handling": "noint", "Crisp ontology-guided": "crisp",
         "Fuzzy without ontology": "noonto", "Box missing bounds": "box", "Normalized TSK aggregation": "tsk"}

# ---- flow
for c, lab in [("P", "dev"), ("L", "tmp"), ("I", "pri"), ("D", "c0506"), ("E", "c0708"), ("F", "c0910"), ("G", "c1112"), ("H", "c1314")]:
    f_ = flow[c]
    for k, v in f_.items():
        S[f"flow_{lab}_{k}"] = f"{v:,}" if isinstance(v, int) else pct(v)
    S[f"flow_{lab}_noDPQ"] = f"{f_['adults'] - f_['adults_with_DPQ_record']:,}"
    S[f"flow_{lab}_invalid"] = f"{f_['adults_with_DPQ_record'] - f_['analytic']:,}"
ext_adults = sum(flow[c]["adults"] for c in "DEFGH"); ext_dpq = sum(flow[c]["adults_with_DPQ_record"] for c in "DEFGH")
S["ext_adults"] = f"{ext_adults:,}"; S["ext_noDPQ"] = f"{ext_adults - ext_dpq:,}"
S["ext_invalid"] = f"{ext_dpq - sum(flow[c]['analytic'] for c in 'DEFGH'):,}"
S["n_dev"] = f"{main['n']['dev']:,}"; S["pos_dev"] = f"{main['pos']['dev']:,}"
for st, sl in SETS.items():
    S[f"n_{sl}"] = f"{main['n'][st]:,}"; S[f"pos_{sl}"] = f"{main['pos'][st]:,}"
    S[f"prev_{sl}"] = pct(main["pos"][st] / main["n"][st])
S["n_test"] = S["n_int"]; S["pos_test"] = S["pos_int"]

# ---- model performance
for m, s in short.items():
    o = main[m]
    S[f"K_{s}"] = str(o.get("K", "–")); S[f"nr_{s}"] = str(o.get("n_rules", "–"))
    S[f"thr_{s}"] = f"{o['threshold']:.3f}"; S[f"oof_{s}"] = f3(o["oof_auroc"])
    for st, sl in SETS.items():
        v = o["sets"][st]
        S[f"{s}_{sl}_AUROC"] = f3(v["AUROC"])
        lo, hi = v["CI"]["AUROC"]; S[f"{s}_{sl}_AUROC_ci"] = f"{lo:.3f}–{hi:.3f}"
        for k in ["AUPRC", "Sensitivity", "Specificity", "PPV", "NPV", "Brier"]:
            S[f"{s}_{sl}_{k}"] = f"{v[k]:.4f}" if k == "Brier" else f3(v[k])
            if k in v["CI"]:
                lo, hi = v["CI"][k]; S[f"{s}_{sl}_{k}_ci"] = f"{lo:.4f}–{hi:.4f}" if k == "Brier" else f"{lo:.3f}–{hi:.3f}"
        S[f"{s}_{sl}_OE"] = f"{v['OE']:.2f}"; S[f"{s}_{sl}_slope"] = f"{v['cal_slope']:.2f}"
        S[f"{s}_{sl}_wAUROC"] = f3(v["AUROC_weighted"]); S[f"{s}_{sl}_flag"] = pct(v["Flagged"])
        if "Coverage" in v:
            for k in ["Coverage", "Sens_covered", "Spec_covered", "Sens_refer", "Spec_refer", "AbstainRate_pos", "AbstainRate_neg"]:
                S[f"{s}_{sl}_{k}"] = pct(v[k])
for k, v in main["paired_auroc"].items():
    st, m = k.split("|"); s = short[m]; sl = SETS[st]
    S[f"d_{s}_{sl}"] = f"{v['diff']:+.3f}"; S[f"d_{s}_{sl}_ci"] = f"{v['lo']:+.3f} to {v['hi']:+.3f}"
    S[f"d_{s}_{sl}_p"] = pval(v["p_two_sided"])
for m, t in main["tuning"].items():
    if t["cv"]:
        for K, a in t["cv"].items(): S[f"cv_{short[m]}_K{K}"] = f3(a)
S["gbt_params"] = ", ".join(f"{k}={v}" for k, v in main["Gradient-boosted trees"]["params"].items())
# version-1 analysis (archived) for the development history
for k in ["internal_test|Logistic regression", "temporal|Logistic regression"]:
    v = v1["paired_auroc"][k]; sl = SETS[k.split("|")[0]]
    S[f"v1old_d_lr_{sl}"] = f"{v['diff']:+.3f}"; S[f"v1old_d_lr_{sl}_ci"] = f"{v['lo']:+.3f} to {v['hi']:+.3f}"
cvs = json.load(open(R + "cv_structure_v2.json"))
for K in (0, 8, 16, 32, 64): S[f"cvmain_K{K}"] = f3(cvs[f"K{K}"])
import os, datetime as _dt
_t = lambda p: _dt.datetime.fromtimestamp(os.path.getmtime(p), _dt.timezone.utc)
S["t_protocol"] = _dt.datetime.fromisoformat(open(R + "protocol_lock_time.txt").read().split()[0].replace("Z", "+00:00")).strftime("%H:%M:%S UTC on %d %B %Y")
_first = min(_t(f"/home/claude/data/{c}_{x}.xpt") for c in ["DEMO", "DPQ", "SLQ", "BMX", "HUQ", "SMQ", "PAQ", "MCQ", "DIQ", "BPQ"] for x in "DEFGH")
S["t_first_download"] = _first.strftime("%H:%M:%S UTC")
S["miss_we_dev"] = pct(M_dev_we) if (M_dev_we := pd.read_pickle("/home/claude/data/cohort_P.pkl").sleep_we.isna().mean()) is not None else ""
S["cv_lr_dev"] = f3(main["Logistic regression"]["oof_auroc"])

# ---- missing-data stress tests
mi = pd.DataFrame(sec["missing"])
for _, r in mi.iterrows():
    lvl = r.level if isinstance(r.level, str) else f"{int(round(float(r.level) * 100))}"
    key = f"mis_{short[r.model]}_{str(lvl).replace(' ', '').replace('-', '')}"
    for k in ["AUROC", "Coverage", "Sens_refer", "Spec_refer", "Sens_covered", "Spec_covered", "MeanWidth"]:
        if k in r and pd.notna(r[k]):
            S[f"{key}_{k}"] = f3(r[k]) if k in ("AUROC", "MeanWidth") else pct(r[k])
fi = sec["fidelity"]
S["fid_n"] = f"{fi['n']:,}"; S["fid_max"] = sci(fi["max_abs_error"]); S["fid_med_rules"] = f"{fi['median_rules_firing']:.0f}"
S["fid_p90_rules"] = f"{fi['p90_rules_firing']:.0f}"
for lab, d in sec["sensitivity"].items():
    key = {"Without sleep predictors": "nosleep", "Without self-rated health": "nosrh", "Extended sleep items (2017-2020 only)": "ext"}[lab]
    for m, v in d.items():
        for st, a in v.items(): S[f"sens_{key}_{short[m]}_{SETS[st]}"] = f3(a)
for k, v in rec.items():
    st, m = k.split("|")
    for q, (a, lo, hi) in v.items():
        S[f"rec_{SETS[st]}_{short[m]}_{q}"] = f"{a:.4f}" if "Brier" in q else f"{a:.2f}"
        S[f"rec_{SETS[st]}_{short[m]}_{q}_ci"] = f"{lo:.4f}–{hi:.4f}" if "Brier" in q else f"{lo:.2f}–{hi:.2f}"
sub = pd.read_csv(R + "subgroups.csv")
for _, r in sub.iterrows():
    g = r.group.replace("Race/ethnicity: ", "").replace("Sex: ", "").replace(" ", "").replace(">=", "ge").replace("-", "").replace("/", "")
    sl = SETS[r.set]
    for m in ["Proposed (full)", "Logistic regression", "Gradient-boosted trees"]:
        S[f"sub_{sl}_{g}_{short[m]}"] = f3(r[m + " AUROC"]); S[f"sub_{sl}_{g}_{short[m]}_sens"] = pct(r[m + " Sens"])
    S[f"sub_{sl}_{g}_n"] = f"{int(r.n):,}"; S[f"sub_{sl}_{g}_pos"] = f"{int(r.positives):,}"

# ---- ontology and rule base
S["n_priors_pos"] = str(int((pri.direction == 1).sum())); S["n_priors_neg"] = str(int((pri.direction == -1).sum()))
S["n_priors_zero"] = str(int((pri.direction == 0).sum())); S["n_concepts"] = str(len(pri) - 5)
for _, r in pri.iterrows():
    if pd.notna(r.get("RR")): S[f"rr_{r.concept}"] = f"{r.RR:.2f} ({r.lo:.2f}–{r.hi:.2f})"
S["n_rules"] = str(len(rb)); S["n_rules_neg"] = str(int((rb.beta < 0).sum())); S["n_rules_pos"] = str(int((rb.beta > 0).sum()))
S["n_rules_signed"] = str(int((rb.sign != 0).sum())); S["n_rules_free"] = str(int((rb.sign == 0).sum()))
S["intercept"] = f"{json.load(open(R + 'rulebase_full.json'))[0]['beta']:.2f}"
M_ = pickle.load(open(R + "models.pkl", "rb")); prj = json.load(open(R + "priors.json"))
mnd = M_["models"]["No direction constraints"]; viol = tot = 0
for r_, b_ in zip(mnd.rules, mnd.beta):
    d_ = {prj.get(mnd.cnames[j], 0) for j in r_} - {0}
    if d_:
        tot += 1; viol += int(d_.pop() * b_ < 0)
S["nodir_signed_rules"] = str(tot); S["nodir_violations"] = str(viol)
mv1 = M_["models"]["Version 1 (conjunctive rules)"]
S["v1_n_pairs"] = str(sum(len(r_) == 2 for r_ in mv1.rules))

# ---- examples, missingness
S["ex_p"] = f3(ex["trace"]["prob_lower"]); S["ex_thr"] = f3(ex["threshold"])
S["exm_lo"] = f3(exm["trace"]["prob_lower"]); S["exm_hi"] = f3(exm["trace"]["prob_upper"]); S["exm_pt"] = f3(exm["point"])
S["exm_missing"] = " and ".join(exm["trace"]["missing_inputs"])
ex_in = ex["trace"]["inputs"]
S["ex_age"] = f"{ex_in['age']:.0f}"; S["ex_sex"] = "woman" if ex_in["female"] == 1 else "man"
S["ex_sleep"] = f"{ex_in['sleep_wd']:.1f}"; S["ex_pir"] = f"{ex_in['pir']:.2f}"; S["ex_nchronic"] = f"{ex_in['n_chronic']:.0f}"
exm_in = exm["trace"]["inputs"]; S["exm_age"] = f"{exm_in['age']:.0f}"; S["exm_sleep"] = f"{exm_in['sleep_wd']:.0f}"
S["exm_decision"] = "refer" if exm["trace"]["prob_lower"] >= main["Proposed (full)"]["threshold"] else ("no-refer" if exm["trace"]["prob_upper"] < main["Proposed (full)"]["threshold"] else "abstain")
S["ex_n_fired"] = str(sum(abs(r.get("logit_contribution", 0)) > 1e-9 for r in ex["trace"]["rules"]))
t1 = pd.read_csv(R + "table1.csv", index_col=0)
S["any_missing_dev"] = t1.loc["Any predictor missing, %", "Development"]; S["pir_missing_dev"] = t1.loc["Income-to-poverty ratio missing, %", "Development"]
mf = M_["models"]["Proposed (full)"]; te = M_["test"]
zl_, zu_ = mf.bounds(te); thr_ = main["Proposed (full)"]["threshold"]; ab_ = (zl_ < thr_) & (zu_ >= thr_)
S["abst_n"] = str(int(ab_.sum())); S["abst_pir"] = f"{100 * te.pir[ab_].isna().mean():.0f}"
S["abst_educ"] = f"{100 * te.educ[ab_].isna().mean():.0f}"; S["abst_marital"] = f"{100 * te.marital[ab_].isna().mean():.0f}"
bc_ = json.load(open(R + "bounds_check.json"))
S["bc_max"] = sci(max(v["max_abs_diff"] for v in bc_.values())); S["bc_n"] = str(sum(v["n"] for v in bc_.values()))

for _k in [k for k in list(S) if (k.startswith("d_") or k.startswith("v1old_d_")) and not k.endswith("_ci") and not k.endswith("_p")]:
    S[_k + "_abs"] = S[_k].lstrip("+-−")
for _k in list(S):
    if isinstance(S[_k], str):
        S[_k] = re.sub(r"(?<![\w.])-(?=\d)", "−", S[_k])
json.dump(S, open(R + "paper_stats.json", "w"), indent=0, ensure_ascii=False)
print(len(S))
