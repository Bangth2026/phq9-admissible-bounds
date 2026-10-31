"""Statistics, part 8: starts from paper_stats_r7.json; adds the configuration sensitivity of imputation spread and the
2021-2023 comparison after recalibration (sec8); verdict labels say 'no clear difference' instead of 'equivalent'."""
import json
R = "/home/claude/results/"; RV = R + "secondary/"
S = json.load(open(R + "paper_stats_r7.json")); A8 = json.load(open(RV + "sec8.json"))
pct = lambda x: f"{100 * x:.1f}"
def sg(x, nd=4):
    s = f"{x:+.{nd}f}".replace("-", "−")
    return "0.0000" if s in ("+0.0000", "−0.0000") else s
for k in list(S):
    if k.startswith("sum_") and k.endswith("_verdict") and S[k] == "equivalent": S[k] = "no clear difference"
SCN = {"natural|internal_test": "int", "natural|temporal": "tmp", "natural|external_pooled": "ext", "mcar30": "mc30", "shift|srh|1.0": "sh1srh"}
for key, sc in SCN.items():
    for cfg, v in A8[key].items():
        n, c = cfg.split("|")
        S[f"s8_{sc}_{n}_{c}_cov"] = pct(v["coverage"]); S[f"s8_{sc}_{n}_{c}_atb"] = f"{v['atb']:.3f}"
        S[f"s8_{sc}_{n}_{c}_nb10"] = f"{v['nb10']:.4f}"; S[f"s8_{sc}_{n}_{c}_mis"] = f"{v['per1000_missed']:.1f}"
for key, sc in SCN.items():
    nbs = [v["nb10"] for v in A8[key].values()]; covs = [v["coverage"] for v in A8[key].values()]
    S[f"s8_{sc}_nbmin"] = f"{min(nbs):.4f}"; S[f"s8_{sc}_nbmax"] = f"{max(nbs):.4f}"
    S[f"s8_{sc}_covmin"] = pct(min(covs)); S[f"s8_{sc}_covmax"] = pct(max(covs))
rc = A8["recal_tmp"]["fuzzy"]; S["rc_shift"] = f"{rc['shift']:.2f}"; S["rc_t"] = f"{rc['t_recal']:.3f}"
RR = A8["recal_tmp_rules"]["fuzzy"]
for r, v in RR.items():
    S[f"rc_{r}_nb05"] = f"{v['nb05']:.4f}"; S[f"rc_{r}_nb10"] = f"{v['nb10']:.4f}"; S[f"rc_{r}_mis"] = f"{v['per1000_missed']:.1f}"
    S[f"rc_{r}_avd"] = f"{v['per1000_avoided']:.0f}"; S[f"rc_{r}_cov"] = pct(v["coverage"])
B = A8["recal_tmp_bootnb"]
for r in ("mi", "ps", "sel"):
    for pt, pk in ((0.05, "05"), (0.1, "10")):
        v = B[f"fuzzy|{r}|{pt}"]; S[f"rc_d_{r}_{pk}"] = f"{sg(v['d'])} ({sg(v['lo'])} to {sg(v['hi'])})"
        S[f"rc_d_{r}_{pk}_clear"] = "no" if v["lo"] <= 0 <= v["hi"] else "yes"
S["zenodo_concept"] = "https://doi.org/10.5281/zenodo.23827786"; S["zenodo_doi_v7"] = "https://doi.org/10.5281/zenodo.23835010"
json.dump(S, open(R + "paper_stats_r8.json", "w"), indent=1, ensure_ascii=False)
print({k: S[k] for k in S if k.startswith("rc_")})
print({k: S[k] for k in S if k.startswith("s8_ext") or k.startswith("s8_mc30")})
