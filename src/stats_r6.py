"""Statistics, part 6: starts from paper_stats_r5.json; adds the operational meaning of AUROC differences (sec6) and
the summary of abstention rules across the 16 missingness scenarios (fuzzy additive base model)."""
import json, re
R = "/home/claude/results/"; RV = R + "secondary/"
S = json.load(open(R + "paper_stats_r5.json"))
A4 = json.load(open(RV + "sec4.json")); A5 = json.load(open(RV + "sec5.json")); A6 = json.load(open(RV + "sec6.json"))
for k, v in A6.items():
    st, m = k.split("|"); sk = "ext" if st == "external_pooled" else "tmp"
    S[f"op80_{sk}_{m}_flag"] = f"{v['flag_per1000']:.0f}"; S[f"op80_{sk}_{m}_fp"] = f"{v['fp_per1000']:.0f}"
S["op80_ext_d_rcs"] = f"{A6['external_pooled|full']['flag_per1000'] - A6['external_pooled|rcs']['flag_per1000']:.0f}"
S["op80_ext_d_lrni"] = f"{A6['external_pooled|full']['flag_per1000'] - A6['external_pooled|lrni']['flag_per1000']:.0f}"

SC = [("natural|internal_test", "int"), ("natural|temporal", "tmp"), ("natural|external_pooled", "ext"), ("mcar30", "mc30"),
      ("mnar|income_low", "m6inc"), ("mnar|srh_poor", "m6srh"), ("mnar|both", "m6both"), ("mnar_strong|pir", "m9inc"),
      ("mnar_strong|srh", "m9srh"), ("outcome_dep", "odep")]
for var, vk in (("srh", "srh"), ("sleep_we", "we")):
    for g, gk in (("0.0", "0"), ("0.5", "05"), ("1.0", "1")): SC.append((f"shift|{var}|{g}", f"sh{gk}{vk}"))
NAME = {"wc": "worst-case", "mi": "imputation spread", "ps": "pattern submodels", "sel": "point-score selective"}
best_aurc, best_nb, wc_rank = {}, {}, {}
for key, sc in SC:
    vals_a, vals_n = {}, {}
    for r in ("wc", "mi", "ps"):
        if r in A4[key]["fuzzy"]: vals_a[r] = A4[key]["fuzzy"][r]["aurc"]; vals_n[r] = A4[key]["fuzzy"][r]["nb10"]
    sel = A5[key]["fuzzy"]["sel"] if key.startswith("natural") else A5[key]["fuzzy"]
    vals_a["sel"] = sel["aurc"]; vals_n["sel"] = sel["nb10"]
    ba = min(vals_a, key=vals_a.get); bn = max(vals_n, key=vals_n.get)
    S[f"sum_{sc}_ba"] = NAME[ba]; S[f"sum_{sc}_ba_v"] = f"{vals_a[ba]:.3f}"; S[f"sum_{sc}_wc_a"] = f"{vals_a['wc']:.3f}"
    S[f"sum_{sc}_bn"] = NAME[bn]; S[f"sum_{sc}_bn_v"] = f"{vals_n[bn]:.4f}".replace("-", "−"); S[f"sum_{sc}_wc_n"] = f"{vals_n['wc']:.4f}".replace("-", "−")
    others = sorted(v for r, v in vals_a.items() if r != "wc")
    S[f"sum_{sc}_lead_a"] = f"{others[0] - vals_a['wc']:.3f}"
    S[f"sum_{sc}_gap_a"] = f"{vals_a['wc'] - vals_a[ba]:.3f}"; S[f"sum_{sc}_gap_n"] = f"{vals_n[bn] - vals_n['wc']:.4f}"
    # verdict on worst-case net benefit at 0.10 relative to the best other rule, from bootstrap intervals where available
    bn_boot = A5.get(key, {}).get("boot_nb") or A5.get("boot_nb_rep0", {}).get(key)
    if bn == "wc":
        verdict = "highest"
    elif bn_boot and f"fuzzy|{bn}|0.1" in bn_boot:
        b = bn_boot[f"fuzzy|{bn}|0.1"]; verdict = "equivalent" if b["lo"] <= 0 <= b["hi"] else "lower"
    else:
        verdict = "lower (no interval)" if vals_n[bn] - vals_n["wc"] > 0.005 else "similar (no interval)"
    S[f"sum_{sc}_verdict"] = verdict
    best_aurc[sc] = ba; best_nb[sc] = bn; wc_rank[sc] = verdict
S["sum_n_wc_aurc_best"] = str(sum(v == "wc" for v in best_aurc.values()))
S["sum_n_wc_nb_lower"] = str(sum(v.startswith("lower") for v in wc_rank.values()))
S["sum_n_wc_nb_eq"] = str(sum(v.startswith("equivalent") or v.startswith("similar") for v in wc_rank.values()))
S["sum_n_wc_nb_high"] = str(sum(v == "highest" for v in wc_rank.values()))
leads = [float(S[f"sum_{sc}_lead_a"]) for key, sc in SC if best_aurc[sc] == "wc"]
S["sum_wc_lead_max"] = f"{max(leads):.3f}"; S["sum_wc_lead_min"] = f"{min(leads):.3f}"
gaps = [float(S[f"sum_{sc}_gap_a"]) for key, sc in SC if best_aurc[sc] != "wc"]
S["sum_wc_gap_max"] = f"{max(gaps):.3f}"
S["zenodo_doi_v5"] = "https://doi.org/10.5281/zenodo.23829450"
S.setdefault("zenodo_doi_v6", "[Zenodo DOI of release v6.0.0, pending]")
json.dump(S, open(R + "paper_stats_r6.json", "w"), indent=1, ensure_ascii=False)
print({k: (best_aurc[k], best_nb[k], wc_rank[k]) for k in best_aurc})
print(S["sum_n_wc_aurc_best"], S["sum_n_wc_nb_lower"], S["sum_n_wc_nb_eq"], S["sum_n_wc_nb_high"])
