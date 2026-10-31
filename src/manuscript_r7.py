"""Version 7: main manuscript (CMPB format, about 3,500 words) and Supplementary Material.
Main text from ms7_main.py; the Supplementary Material is built from the long-form manuscript (manuscript_full7.json),
renumbered with S labels. Every number comes from results/paper_stats_r7.json."""
import json, re, sys
sys.path.insert(0, "/home/claude/src")
from refs import REFS
import ms4_helpers as H
import ms7_main  # noqa  (appends the main-text blocks to H.B)

R = "/home/claude/results/"
S = json.load(open(R + "paper_stats_r7.json"))


def V(k):
    if k not in S: raise KeyError(f"missing statistic: {k}")
    return S[k]


def fill(t): return re.sub(r"\{\{(\w+)\}\}", lambda m: m.group(0) if m.group(1) == "wordcount" else V(m.group(1)), t)


def make_citer():
    order = []
    def cites(text):
        def rep(m):
            keys = [k.strip().lstrip("@") for k in m.group(1).split(",")]
            nums = []
            for k in keys:
                if k not in REFS: raise KeyError("unknown reference " + k)
                if k not in order: order.append(k)
                nums.append(order.index(k) + 1)
            nums = sorted(nums); parts, i = [], 0
            while i < len(nums):
                j = i
                while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
                parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
                i = j + 1
            return "[" + ", ".join(parts) + "]"
        return re.sub(r"\[(@[^\]]+)\]", rep, text)
    return cites, order


A_TO_S = lambda n: f"S{int(n) + 3}"   # Appendix tables A1-A13 -> Supplementary Tables S4-S16

# ======================================================================================== main manuscript
from tables_r7 import tables
TT = tables(V)
CAP = {"tableAbst5": ("Table 5.", "Table 2."), "tableSum": ("Table @SUM@.", "Table 3."), "tableCons": ("Table 7.", "Table 4.")}
cites, order = make_citer()
out = []
for b in H.B:
    b = dict(b)
    if b["t"] in TT:
        T = dict(TT[b["t"]])
        if b["t"] in CAP: T["cap"] = T["cap"].replace(*CAP[b["t"]])
        out.append({"t": "table", **T, "cap": cites(fill(T["cap"])), "note": cites(fill(T.get("note", ""))),
                    "rows": [[cites(fill(str(c))) for c in r] for r in T["rows"]]}); continue
    for k in ("x", "cap"):
        if k in b: b[k] = [cites(fill(s)) for s in b[k]] if isinstance(b[k], list) else cites(fill(b[k]))
    out.append(b)
js = json.dumps(out, ensure_ascii=False)
for a, c in [("Tables @T3@, @T5@, A6 and A13", "Tables 2 and 4 and Supplementary Tables S9 and S16"), ("p = <0.001", "p < 0.001"),
             ("the proposed model", "the fuzzy additive model"), ("admissible-value", "worst-case")]:
    js = js.replace(a, c)
js = re.sub(r"(?<!Supplementary )Table A(\d+)", lambda m: "Supplementary Table " + A_TO_S(m.group(1)), js)
out = json.loads(js)
# word count of the main text (sections 1-5)
sec, wc = None, 0
for b in out:
    if b["t"] == "h1": sec = b["x"]
    elif b["t"] == "p" and sec and sec[:2] in ("1.", "2.", "3.", "4.", "5."): wc += len(b["x"].split())
out = json.loads(json.dumps(out, ensure_ascii=False).replace("{{wordcount}}", f"{wc:,}"))
for i, b in enumerate(out):
    if b["t"] == "refs": out[i] = {"t": "refs", "x": [f"[{n + 1}] {REFS[k]}" for n, k in enumerate(order)]}
# Elsevier: Appendix (supplementary material statement) after the references
k = [i for i, b in enumerate(out) if b["t"] == "h1" and b["x"].startswith("Appendix A")][0]
app = out[k:k + 2]; del out[k:k + 2]; out += app
json.dump(out, open(R + "manuscript_r7.json", "w"), ensure_ascii=False, indent=0)
left = re.findall(r"\{\{\w+\}\}", json.dumps(out, ensure_ascii=False))
print("MAIN blocks", len(out), "words", wc, "refs", len(order), "unfilled", left)
json.dump({"highlights": [fill(h) for h in ["Five abstention rules for missing inputs were compared with the base model fixed.",
                                            "Pattern submodels and point-score selective prediction did best in 2005–2014.",
                                            "Worst-case bounds decided fewest participants, least accurately at equal coverage.",
                                            "Worst-case bounds missed fewer cases only by referring more people.",
                                            "PHQ-2 flagged {{phq2_int_flag}}% (sensitivity {{phq2_int_sens}}); the model flagged {{full_int_flag}}% ({{full_int_Sensitivity}})."]]},
          open(R + "highlights_r7.json", "w"), ensure_ascii=False)

# ======================================================================================== Supplementary Material
F = json.load(open(R + "manuscript_full7.json"))
def idx(pred): return [i for i, b in enumerate(F) if pred(b)][0]
i_m = idx(lambda b: b["t"] == "h1" and b["x"].startswith("2. "))
i_d = idx(lambda b: b["t"] == "h1" and b["x"].startswith("4. "))
i_c = idx(lambda b: b["t"] == "h1" and b["x"].startswith("5. "))
i_r = idx(lambda b: b["t"] == "h1" and b["x"] == "References")
i_a = idx(lambda b: b["t"] == "h1" and b["x"].startswith("Appendix A"))
i_b = idx(lambda b: b["t"] == "h1" and b["x"].startswith("Appendix B"))
MAIN_TABLE_CAPS = ("Table 1.", "Table 3.", "Table 4.", "Table 5.")
MAIN_FIGS = ("Fig. 1.", "Fig. 2.", "Fig. 5.")
def keep(b):
    if b["t"] == "table" and (b["cap"].startswith(MAIN_TABLE_CAPS) or b["cap"].startswith(("Box 1.", "Box 2."))): return False
    if b["t"] == "fig" and b["cap"].startswith(MAIN_FIGS): return False
    return True
body = [b for b in F[i_m:i_d] if keep(b)]
disc = []; take = False
for b in F[i_d:i_c]:
    if b["t"] == "h2": take = b["x"].startswith(("4.3.", "4.4.", "4.6."))
    if take: disc.append(b)
apx = [b for b in F[i_a + 1:i_b] if keep(b)]
refs = F[i_r + 1]
SUP = [{"t": "title", "x": "Supplementary Material"},
       {"t": "meta", "x": "Abstaining when inputs are missing: worst-case bounds, imputation spread and pattern submodels compared for prescreening elevated depressive symptoms in NHANES"},
       {"t": "meta", "x": "Truong Hai Bang, Van An Phan Thi"},
       {"t": "p", "x": "This file gives the extended methods (Section S1), extended results (Section S2), additional discussion (Section S3), "
                       "additional tables and figures (Section S4) and the TRIPOD+AI checklist (Section S5). It is the full-length version from which "
                       "the main text was condensed; references to 'main text' point to the article. References are numbered for this file."}]
for b in body:
    b = dict(b)
    if b["t"] == "h1" and b["x"].startswith("2. "): b["x"] = "S1. Extended methods"
    elif b["t"] == "h2": b["x"] = re.sub(r"^2\.(\d+)\.", r"S1.\1.", b["x"])
    SUP.append(b)
res_i = [i for i, b in enumerate(SUP) if b["t"] == "h1" and b["x"] == "S1. Extended methods"]
# results block: the long form puts results after methods; relabel 3.x headings
fixed = []
for b in SUP:
    b = dict(b)
    if b["t"] == "h2" and b["x"].startswith("S1.") is False and re.match(r"^3\.\d+\.", b["x"]): b["x"] = re.sub(r"^3\.(\d+)\.", r"S2.\1.", b["x"])
    fixed.append(b)
SUP = fixed
for k, b in enumerate(SUP):
    if b["t"] == "h1" and b["x"].startswith("3. "): SUP[k] = {"t": "h1", "x": "S2. Extended results"}
SUP.append({"t": "h1", "x": "S3. Additional discussion"})
mapd = {"4.3.": "S3.1.", "4.4.": "S3.2.", "4.6.": "S3.3."}
for b in disc:
    b = dict(b)
    if b["t"] == "h2": b["x"] = mapd[b["x"][:4]] + b["x"][4:]
    SUP.append(b)
SUP.append({"t": "h1", "x": "S4. Additional tables and figures"})
SUP += apx
SUP.append({"t": "h1", "x": "S5. TRIPOD+AI checklist"})
TRIPOD_FINAL = [
 ["1", "Title", "Identify the study as developing/evaluating a prediction model", "Title"], ["2", "Abstract", "Structured summary", "Abstract"],
 ["3a", "Background", "Healthcare context and rationale", "Main 1"], ["3b", "Background", "Target population, intended purpose and users", "Main 1, 2.1; S1.1"],
 ["3c", "Background", "Known health inequalities", "S1.11, S2.9; Table S5"], ["4", "Objectives", "Study objectives", "Main 1"],
 ["5a", "Data", "Sources of data", "Main 2.1; S1.2"], ["5b", "Data", "Dates of data collection", "Main 2.1; S1.2"],
 ["6a", "Participants", "Study setting", "S1.2"], ["6b", "Participants", "Eligibility criteria", "Main 2.1; S1.3"],
 ["6c", "Participants", "Treatments received", "Not applicable"], ["7", "Data preparation", "Pre-processing and quality checking", "S1.3, S1.4"],
 ["8a", "Outcome", "Definition and assessment", "Main 2.1; S1.3"], ["8b", "Outcome", "Outcome assessors", "Not applicable (self-report)"],
 ["8c", "Outcome", "Blinding", "S1.1"], ["9a", "Predictors", "Choice of predictors", "Main 2.1; S1.4"],
 ["9b", "Predictors", "Definition and measurement", "S1.4, S1.5; Table S14"], ["9c", "Predictors", "Predictor assessors", "Not applicable"],
 ["10", "Sample size", "How the study size was arrived at", "Main 2.1; S1.3"], ["11", "Missing data", "How missing data were handled", "Main 2.3–2.5; S1.5–S1.8"],
 ["12a", "Analytical methods", "How data were used", "Main 2.1, 2.6; S1.11"], ["12b", "Analytical methods", "How predictors were handled", "S1.5"],
 ["12c", "Analytical methods", "Model type, training, tuning", "Main 2.2; S1.5, S1.11"], ["12d", "Analytical methods", "Heterogeneity across clusters", "S1.11"],
 ["12e", "Analytical methods", "Performance measures and comparisons", "Main 2.6; S1.9, S1.11"], ["12f", "Analytical methods", "Model updating", "S1.11"],
 ["12g", "Analytical methods", "Calculation of predicted values", "Main 2.2–2.4; S1.5–S1.7"], ["13", "Class imbalance", "Handling of class imbalance", "S1.11"],
 ["14", "Fairness", "Approaches to model fairness", "S1.11, S2.9; Table S5"], ["15", "Model output", "Output and thresholds", "Main 2.2, 2.3"],
 ["16", "Training vs evaluation", "Differences between data sets", "Main 3.1; Table 1"], ["17", "Ethical approval", "Ethics and consent", "Declarations"],
 ["18a", "Funding", "Funding", "Declarations"], ["18b", "Conflicts of interest", "Declarations", "Declarations"],
 ["18c", "Protocol", "Protocol access", "S1.12; code repository"], ["18d", "Registration", "Registration", "S1.12 (not registered)"],
 ["18e", "Data sharing", "Data availability", "Declarations"], ["18f", "Code sharing", "Code availability", "Declarations"],
 ["19", "Patient and public involvement", "Involvement", "None undertaken"], ["20a", "Participants", "Flow of participants", "Main 2.1, Fig. 1"],
 ["20b", "Participants", "Characteristics, missing data", "Main 3.1; Table 1; Table S7"], ["20c", "Participants", "Development vs evaluation data", "Main 3.1; Table 1"],
 ["21", "Model development", "Participants and events per analysis", "Main 2.1; Tables 1–2"], ["22", "Model specification", "Full model", "Table S15; knowledge schema file"],
 ["23a", "Model performance", "Performance with intervals", "Main 3; Tables 2–4; S2; Table S1"], ["23b", "Model performance", "Heterogeneity", "S2.3, S2.9; Table S10"],
 ["24", "Model updating", "Results of updating", "S2.3"], ["25", "Interpretation", "Interpretation, fairness", "Main 4.1–4.2"],
 ["26", "Limitations", "Limitations", "Main 4.3; S3.3"], ["27a", "Usability", "Handling of unavailable inputs", "Main 2.3–2.5, 3.2–3.4; Boxes 1–2; Box S1"],
 ["27b", "Usability", "User interaction and expertise", "Main 1, 2.1"], ["27c", "Implications", "Future research and generalisability", "Main 4.2–4.3"]]
SUP.append({"t": "table", "cap": "Table S17. TRIPOD+AI checklist (Collins et al., BMJ 2024): items and where they are reported ('Main' = article; S = this file).",
            "head": ["Item", "Section/topic", "Checklist item (abbreviated)", "Where reported"], "rows": TRIPOD_FINAL, "note": "", "w": [0.5, 1.3, 3.0, 2.2], "small": True})
SUP.append({"t": "h1", "x": "References"}); SUP.append(refs)
js = json.dumps(SUP, ensure_ascii=False)
TMAP = {"1": "1 (main text)", "3": "2 (main text)", "4": "3 (main text)", "5": "4 (main text)", "2": "S1", "6": "S2", "7": "S3"}
FMAP = {"1": "1 (main text)", "2": "2 (main text)", "5": "3 (main text)", "3": "S1", "4": "S2", "6": "S3", "7": "S4"}
js = re.sub(r"Table A(\d+)", lambda m: "Table " + A_TO_S(m.group(1)), js)
js = re.sub(r"Tables A(\d+)–A(\d+)", lambda m: f"Tables {A_TO_S(m.group(1))}–{A_TO_S(m.group(2))}", js)
js = re.sub(r"Tables A(\d+) and A(\d+)", lambda m: f"Tables {A_TO_S(m.group(1))} and {A_TO_S(m.group(2))}", js)
js = re.sub(r"(?<![A-Za-z])Table (\d+)(?![0-9])", lambda m: "Table " + TMAP.get(m.group(1), m.group(1)), js)
js = re.sub(r"Fig\. A(\d+)", lambda m: "Fig. S" + str(int(m.group(1)) + 4), js)
js = re.sub(r"Figs\. A(\d+)–A(\d+)", lambda m: f"Figs. S{int(m.group(1)) + 4}–S{int(m.group(2)) + 4}", js)
js = re.sub(r"Fig\. (\d+)(?![0-9])", lambda m: "Fig. " + FMAP.get(m.group(1), m.group(1)), js)
js = js.replace("Box 3", "Box S1").replace("Box 1", "Box 1 (main text)").replace("Box 2", "Box 2 (main text)").replace("Boxes 1–3", "Boxes 1–2 (main text) and Box S1")
js = re.sub(r"Section(s?) 2\.(\d+)", r"Section\1 S1.\2", js); js = re.sub(r"Section(s?) 3\.(\d+)", r"Section\1 S2.\2", js)
js = re.sub(r"Sections S1\.(\d+)–2\.(\d+)", r"Sections S1.\1–S1.\2", js); js = re.sub(r"Sections S2\.(\d+)–3\.(\d+)", r"Sections S2.\1–S2.\2", js)
js = js.replace("Appendix A", "Section S4").replace("Appendix B", "Section S5")
js = js.replace("Table S1 (main text)", "Table S1")
SUP = json.loads(js)
json.dump(SUP, open(R + "supplement_r7.json", "w"), ensure_ascii=False, indent=0)
print("SUPP blocks", len(SUP))
