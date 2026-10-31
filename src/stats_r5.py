"""Statistics, part 5: starts from paper_stats_r4.json and adds sec5 (point-score selective prediction, net-benefit
intervals, prevalence of target samples, ordinal-residual shift) and sec5b."""
import json
import numpy as np
R = "/home/claude/results/"; RV = R + "secondary/"
S = json.load(open(R + "paper_stats_r4.json"))
A4 = json.load(open(RV + "sec4.json")); A5 = json.load(open(RV + "sec5.json")); B5 = json.load(open(RV + "sec5b.json"))
f3 = lambda x: f"{x:.3f}"; pct = lambda x, d=1: f"{100 * x:.{d}f}"
def sg(x, nd=4): return f"{x:+.{nd}f}".replace("-", "−")
def ci(lo, hi, nd=4): return f"{sg(lo, nd)} to {sg(hi, nd)}"
SC = {"natural|internal_test": "int", "natural|temporal": "tmp", "natural|external_pooled": "ext", "mcar30": "mc30",
      "mnar|income_low": "m6inc", "mnar|srh_poor": "m6srh", "mnar|both": "m6both", "mnar_strong|srh": "m9srh",
      "mnar_strong|pir": "m9inc", "outcome_dep": "odep"}
for var, vk in (("srh", "srh"), ("sleep_we", "we")):
    for g, gk in (("0.0", "0"), ("0.5", "05"), ("1.0", "1")): SC[f"shift|{var}|{g}"] = f"sh{gk}{vk}"
MODELS = ("fuzzy", "slr", "rcs")
for key, sc in SC.items():
    O = A5[key]
    S[f"prv_{sc}"] = pct(A4[key]["prev"]) if key in A4 else pct(O["prev"])
    for m in MODELS:
        v = O[m]["sel"] if "sel" in O[m] else O[m]
        k = f"a4_{sc}_{m}_sel_"
        S[k + "cov"] = pct(v["coverage"]); S[k + "covr"] = pct(v["cov_refer"]); S[k + "covn"] = pct(v["cov_norefer"])
        S[k + "sens"] = pct(v["sens"]); S[k + "spec"] = pct(v["spec"]); S[k + "sensr"] = pct(v["sens_refer"])
        S[k + "ref"] = f"{v['per1000_referred']:.0f}"; S[k + "mis"] = f"{v['per1000_missed']:.1f}"; S[k + "avd"] = f"{v['per1000_avoided']:.0f}"
        S[k + "nb05"] = f"{v['nb05']:.4f}".replace("-", "−"); S[k + "nb10"] = f"{v['nb10']:.4f}".replace("-", "−"); S[k + "aurc"] = f3(v["aurc"])
        S[k + "atb"] = f3(v["at"]["bacc"] if "at" in v else v["atb"])
        if "at" in v: S[k + "ats"] = pct(v["at"]["sens"]); S[k + "atp"] = pct(v["at"]["spec"])
        if "aurc_sd" in v: S[k + "aurcsd"] = f3(v["aurc_sd"])
        S[k + "daurc"] = sg(v["aurc"] - A4[key][m]["wc"]["aurc"], 3)
    if "boot_aurc_sel" in O:
        for m in MODELS:
            b = O["boot_aurc_sel"][f"{m}|sel"]; S[f"a4_{sc}_{m}_sel_aurcci"] = f"{b['lo']:.3f}–{b['hi']:.3f}"
            S[f"a4_{sc}_{m}_sel_daurcci"] = ci(b["d_lo"], b["d_hi"], 3)
# net-benefit differences against worst-case bounds
for key, sc in SC.items():
    bn = A5.get(key, {}).get("boot_nb") or A5.get("boot_nb_rep0", {}).get(key)
    for m in MODELS:
        for r in ("mi", "ps", "sel", "cc"):
            for pt, pk in ((0.05, "05"), (0.10, "10")):
                kk = f"dnb_{sc}_{m}_{r}_{pk}"
                v = bn.get(f"{m}|{r}|{pt}") if bn else None
                if v: S[kk] = f"{sg(v['d'])} ({ci(v['lo'], v['hi'])})"; S[kk + "_eq"] = "yes" if v["lo"] <= 0 <= v["hi"] else "no"
                else: S[kk] = "n.c."
# AURC SD for mcar etc. already in r4 keys; ensure present for all simulated
for key, sc in SC.items():
    if key.startswith("natural"): continue
    for m in MODELS:
        for r in ("wc", "mi", "ps", "cc"):
            if r in A4[key][m] and "aurc_sd" in A4[key][m][r]: S[f"a4_{sc}_{m}_{r}_aurcsd"] = f3(A4[key][m][r]["aurc_sd"])
# ordinal-residual shift
so = A5["shift_ord_full"]; S["ord_corr"] = f"{A5['ord_corr_lin']:.2f}"; S["prv_ordsrh"] = pct(A5["shift_ord_meta"]["prev"])
for r in ("wc", "mi", "ps"):
    v = so["fuzzy"][r]; S[f"ord_{r}_cov"] = pct(v["coverage"]); S[f"ord_{r}_atb"] = f3(v["at"]["bacc"]); S[f"ord_{r}_sensr"] = pct(v["sens_refer"])
    S[f"ord_{r}_nb10"] = f"{v['nb10']:.4f}"; S[f"ord_{r}_mis"] = f"{v['per1000_missed']:.1f}"
S["delta_fuzzy"] = f"{A5['delta']['fuzzy']:.3f}"
for m in MODELS:
    b = B5[m]
    S[f"selps_{m}_daurc"] = sg(b["d_aurc"], 3); S[f"selps_{m}_daurc_ci"] = ci(*b["d_aurc_ci"], 3)
    S[f"selps_{m}_dnb10"] = sg(b["d_nb10"]); S[f"selps_{m}_dnb10_ci"] = ci(*b["d_nb10_ci"])
    S[f"selps_{m}_dnb05"] = sg(b["d_nb05"]); S[f"selps_{m}_dnb05_ci"] = ci(*b["d_nb05_ci"])
S["zenodo_doi_v4"] = "https://doi.org/10.5281/zenodo.23829099"
S.setdefault("zenodo_doi_v5", "[Zenodo DOI of release v5.0.0, pending]")
import re
S = {k: (re.sub(r"−0\.0+(?![0-9])", lambda m: m.group(0)[1:], v) if isinstance(v, str) else v) for k, v in S.items()}
json.dump(S, open(R + "paper_stats_r5.json", "w"), indent=1, ensure_ascii=False)
print(len(S))
