"""Statistics, part 7: starts from paper_stats_r6.json; pooled intervals for simulated scenarios (sec7), verdicts of
the scenario summary based on intervals only, and sensitivity of the point-score band."""
import json, os
R = "/home/claude/results/"; RV = R + "secondary/"
S = json.load(open(R + "paper_stats_r6.json"))
A4 = json.load(open(RV + "sec4.json")); A5 = json.load(open(RV + "sec5.json")); A7 = json.load(open(RV + "sec7.json"))
def sg(x, nd=4):
    s = f"{x:+.{nd}f}".replace("-", "−")
    return "0.0000" if s in ("+0.0000", "−0.0000") else s
SC = [("natural|internal_test", "int"), ("natural|temporal", "tmp"), ("natural|external_pooled", "ext"), ("mcar30", "mc30"),
      ("mnar|income_low", "m6inc"), ("mnar|srh_poor", "m6srh"), ("mnar|both", "m6both"), ("mnar_strong|pir", "m9inc"),
      ("mnar_strong|srh", "m9srh"), ("outcome_dep", "odep")]
for var, vk in (("srh", "srh"), ("sleep_we", "we")):
    for g, gk in (("0.0", "0"), ("0.5", "05"), ("1.0", "1")): SC.append((f"shift|{var}|{g}", f"sh{gk}{vk}"))
NAME = {"wc": "worst-case", "mi": "imputation spread", "ps": "pattern submodels", "sel": "point-score selective"}
# pooled intervals for the fuzzy model in simulated scenarios
for key, sc in SC:
    if key.startswith("natural"): continue
    for r in ("mi", "ps", "sel"):
        for mt, pk in (("nb05", "05"), ("nb10", "10")):
            v = A7[key].get(f"{r}|{mt}")
            if v: S[f"dnb_{sc}_fuzzy_{r}_{pk}"] = f"{sg(v['mean'])} ({sg(v['lo'])} to {sg(v['hi'])})"


def interval(key, r, pt):
    if key.startswith("natural"):
        b = A5[key]["boot_nb"].get(f"fuzzy|{r}|{pt}")
        return (b["lo"], b["hi"]) if b else None
    v = A7[key].get(f"{r}|nb{'10' if pt == 0.1 else '05'}")
    return (v["lo"], v["hi"]) if v else None


def vals(key, sel_key=None):
    va, vn = {}, {}
    for r in ("wc", "mi", "ps"):
        if r in A4[key]["fuzzy"]: va[r] = A4[key]["fuzzy"][r]["aurc"]; vn[r] = A4[key]["fuzzy"][r]["nb10"]
    sel = A5[key]["fuzzy"]["sel"] if key.startswith("natural") else A5[key]["fuzzy"]
    va["sel"], vn["sel"] = sel["aurc"], sel["nb10"]
    if sel_key:
        st = A7[key]["sel_targets"][sel_key]; va["sel"], vn["sel"] = st["aurc"], st["nb10"]
    return va, vn


cnt = {"lower": 0, "equivalent": 0, "highest": 0}
for key, sc in SC:
    va, vn = vals(key); bn = max(vn, key=vn.get)
    comp = bn if bn != "wc" else max((r for r in vn if r != "wc"), key=vn.get)
    lo, hi = interval(key, comp, 0.1)
    if bn == "wc": verdict = "equivalent" if lo <= 0 <= hi else "highest"
    else: verdict = "equivalent" if lo <= 0 <= hi else "lower"
    S[f"sum_{sc}_verdict"] = verdict; cnt[verdict] += 1
S["sum_n_wc_nb_lower"] = str(cnt["lower"]); S["sum_n_wc_nb_eq"] = str(cnt["equivalent"]); S["sum_n_wc_nb_high"] = str(cnt["highest"])
# sensitivity of the point-score band (80% and 95% development coverage instead of 90%)
ch = {}
for tgt in ("0.8", "0.95"):
    n = 0
    for key, sc in SC:
        va0, vn0 = vals(key); va1, vn1 = vals(key, tgt)
        if max(vn0, key=vn0.get) != max(vn1, key=vn1.get) or min(va0, key=va0.get) != min(va1, key=va1.get): n += 1
    ch[tgt] = n
S["dsel_sentence"] = (f"Setting the point-score band to decide 80% or 95% of development participants instead of 90% changed the best rule "
                      f"(by AURC or net benefit) in {ch['0.8']} and {ch['0.95']} of the 16 scenarios.")
S["dsel_80"] = str(ch["0.8"]); S["dsel_95"] = str(ch["0.95"])
S.setdefault("phone", "[telephone number]")
S["zenodo_doi_v6"] = "https://doi.org/10.5281/zenodo.23830412"
S.setdefault("zenodo_doi_v7", "[Zenodo DOI of release v7.0.0, pending]")
RR = RV + "rerun_compare.json"
if os.path.exists(RR):
    c = json.load(open(RR)); S["rerun_statement"] = c["statement"]
else:
    S["rerun_statement"] = "[clean re-run in progress]"
json.dump(S, open(R + "paper_stats_r7.json", "w"), indent=1, ensure_ascii=False)
print(cnt, ch)
