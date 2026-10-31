"""Statistics, part 4: starts from paper_stats_r3.json and adds the fourth set of analyses (sec4, sec4b, 100 splits)."""
import json, os
import numpy as np, pandas as pd
R = "/home/claude/results/"; RV = R + "secondary/"
S = json.load(open(R + "paper_stats_r3.json"))
A = json.load(open(RV + "sec4.json")); Bx = json.load(open(RV + "sec4b.json"))
f3 = lambda x: f"{x:.3f}"; pct = lambda x, d=1: f"{100 * x:.{d}f}"
def sg(x, nd=3):
    s = f"{x:+.{nd}f}"; return s.replace("-", "−")
def ci(lo, hi, nd=3): return f"{sg(lo, nd)} to {sg(hi, nd)}"

SC = {"natural|internal_test": "int", "natural|temporal": "tmp", "natural|external_pooled": "ext", "mcar30": "mc30",
      "mnar|income_low": "m6inc", "mnar|srh_poor": "m6srh", "mnar|both": "m6both", "mnar_strong|srh": "m9srh",
      "mnar_strong|pir": "m9inc", "outcome_dep": "odep"}
for var, vk in (("srh", "srh"), ("sleep_we", "we")):
    for g, gk in (("0.0", "0"), ("0.5", "05"), ("1.0", "1")): SC[f"shift|{var}|{g}"] = f"sh{gk}{vk}"
for key, sc in SC.items():
    if key not in A: continue
    O = A[key]; boot = O.get("boot") or O.get("boot_rep0") or {}
    for m in ("fuzzy", "slr", "rcs", "lrmi"):
        for r, v in O[m].items():
            k = f"a4_{sc}_{m}_{r}_"
            S[k + "cov"] = pct(v["coverage"]); S[k + "covr"] = pct(v["cov_refer"]); S[k + "covn"] = pct(v["cov_norefer"])
            S[k + "sens"] = pct(v["sens"]); S[k + "spec"] = pct(v["spec"]); S[k + "bacc"] = f3(v["bacc"])
            S[k + "atb"] = f3(v["at"]["bacc"]); S[k + "ats"] = pct(v["at"]["sens"]); S[k + "atp"] = pct(v["at"]["spec"])
            S[k + "aurc"] = f3(v["aurc"]); S[k + "sensr"] = pct(v["sens_refer"]); S[k + "flagr"] = pct(v["flag_refer"])
            S[k + "ref"] = f"{v['per1000_referred']:.0f}"; S[k + "mis"] = f"{v['per1000_missed']:.1f}"; S[k + "avd"] = f"{v['per1000_avoided']:.0f}"
            S[k + "nb05"] = f"{v['nb05']:.4f}".replace("-", "−"); S[k + "nb10"] = f"{v['nb10']:.4f}".replace("-", "−")
            if "aurc_sd" in v: S[k + "aurcsd"] = f3(v["aurc_sd"])
            b = boot.get(f"{m}|{r}")
            if b:
                S[k + "aurcci"] = f"{b['lo']:.3f}–{b['hi']:.3f}"
                if "d_lo" in b: S[k + "daurcci"] = ci(b["d_lo"], b["d_hi"])
            if "aurc" in v and r != "wc" and "wc" in O[m]:
                S[k + "daurc"] = sg(v["aurc"] - O[m]["wc"]["aurc"])
# weighted coverage in 2005-2014
for k, v in A["natural|external_pooled"].get("weighted_coverage", {}).items():
    m, r = k.split("|"); S[f"a4w_ext_{m}_{r}_cov"] = pct(v)
for k, v in A.get("shift_meta", {}).items():
    var, g = k.split("|"); vk = "srh" if var == "srh" else "we"; gk = {"0.0": "0", "0.5": "05", "1.0": "1"}[g]
    S[f"shm_{vk}_{gk}"] = f"{v['mean_resid']:+.2f}".replace("-", "−"); S[f"shmt_{vk}_{gk}"] = f"{v['mean_true']:.2f}"
    S[f"shmd_{vk}"] = f"{v['mean_dev']:.2f}"
S["n_patterns_ps"] = str(len(A.get("pattern_submodels", {})))

# ---------------------------------------------------------------- sec4b
d = Bx["design_boot"]
for k in ("lr", "lrni", "rcs"):
    S[f"db_{k}_d"] = sg(d[k]["d"], 4) if abs(d[k]["d"]) >= 5e-5 else "0.0000"; S[f"db_{k}_ci"] = ci(d[k]["lo"], d[k]["hi"], 4)
S["db_nstrata"] = str(d["n_strata"]); S["db_npsu"] = str(d["n_psu"])
rl = Bx["riley"]
S.update(rl_P=str(rl["P"]), rl_phi=f"{rl['phi']:.3f}", rl_R2=f"{rl['R2cs']:.3f}", rl_n1=f"{np.ceil(rl['n1']):,.0f}",
         rl_n2=f"{np.ceil(rl['n2']):,.0f}", rl_n3=f"{np.ceil(rl['n3']):,.0f}", rl_n=f"{rl['n_required']:,}", rl_events=str(rl["events_required"]),
         rl_epp=f"{rl['epp_required']:.1f}", rl_epp_dev=f"{rl['epp_dev']:.1f}", rl_events_dev=str(rl["events_dev"]))
c = Bx["calibration_maps"]
for m in ("lr", "full"):
    for k, v in c[m].items(): S[f"calm_{m}_{k}"] = f"{v:.2f}"
fl = Bx["sleep_we_fill"]; S["we_fill"] = f"{fl['contrib']:.3f}"
S["we_beta_short"] = f"{fl['terms']['ShortSleepFreeDay'][0]:.3f}"; S["we_mu_short"] = f"{fl['terms']['ShortSleepFreeDay'][1]:.3f}"
S["we_beta_long"] = f"{fl['terms']['LongSleepFreeDay'][0]:.3f}"; S["we_mu_long"] = f"{fl['terms']['LongSleepFreeDay'][1]:.3f}"
for k, v in Bx["anymiss_excl_we"].items(): S[f"amx_{ {'dev':'dev','internal_test':'int','temporal':'tmp','external_pooled':'ext'}[k]}"] = pct(v)
aq = Bx["acq_cost"]["masked30_internal"]
for pol in ("bound", "bound_cost", "fixed", "cost", "random"):
    for ic in ("3", "5"): S[f"aqc_{pol}_{ic}"] = f"{aq[f'{pol}_cost{ic}']:.2f}"
    S[f"aqc_{pol}_n"] = f"{aq[f'{pol}_n']:.2f}"
for other in ("bound", "fixed", "cost", "random"):
    for ic in ("3", "5"):
        v = aq[f"d_{other}_cost{ic}"]; S[f"aqd_{other}_{ic}"] = f"{v[0]:.2f}"; S[f"aqd_{other}_{ic}_ci"] = f"{v[1]:.2f} to {v[2]:.2f}"
S["aqc_n"] = str(aq["n"])
S["acq_per1000_fixed"] = f"{1000 * float(S['acq2_m30_d_fixed']):.0f}"; S["acq_per1000_cost"] = f"{1000 * float(S['acq2_m30_d_cost']):.0f}"

# ---------------------------------------------------------------- 100 repeated splits (mean and SD of differences)
if os.path.exists(RV + "repeated_splits100.csv"):
    df = pd.read_csv(RV + "repeated_splits100.csv")
    short = {"Logistic regression": "lr", "Sign-constrained logistic regression": "slr", "Gradient-boosted trees": "gbt",
             "Explainable boosting machine (GA2M)": "ebm", "Version 1 (conjunctive rules)": "v1", "Main effects + interactions": "inter",
             "No direction constraints": "nodir", "Crisp ontology-guided": "crisp", "Fuzzy without ontology": "noonto"}
    S["spl_B"] = str(len(df))
    S["spl_full_mean"] = f3(df["Proposed (full)"].mean()); S["spl_full_sd"] = f3(df["Proposed (full)"].std())
    for nm, s in short.items():
        if nm not in df: continue
        dd = df["Proposed (full)"] - df[nm]
        S[f"spl_{s}_mean"] = sg(dd.mean(), 4) if abs(dd.mean()) < 0.0005 else sg(dd.mean(), 3); S[f"spl_{s}_sd"] = f"{dd.std():.3f}"
        S[f"spl_{s}_frac"] = f"{100 * (dd > 0).mean():.0f}"; S[f"spl_{s}_abs"] = f"{abs(dd.mean()):.3f}"
        S[f"spl_{s}_lo"] = sg(dd.quantile(.025)); S[f"spl_{s}_hi"] = sg(dd.quantile(.975))
O = A["outcome_dep"]
S["odep_maxgap"] = f"{max(O[m][r]['aurc'] - O[m]['wc']['aurc'] for m in ('fuzzy','slr','rcs') for r in ('mi','ps')):.3f}"
S["zenodo_doi_v3"] = "https://doi.org/10.5281/zenodo.23827787"
S.setdefault("zenodo_doi_v4", "[Zenodo DOI of release v4.0.0, pending]")
json.dump(S, open(R + "paper_stats_r4.json", "w"), indent=1, ensure_ascii=False)
print(len(S))
