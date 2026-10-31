"""Manuscript, version 4. Every number is injected from results/paper_stats_r4.json via {{key}};
a missing key raises an error."""
import json, re, sys
sys.path.insert(0, "/home/claude/src")
from refs import REFS
import ms4_helpers as H
import ms4_intro, ms4_methods, ms4_results, ms4_back  # noqa  (each appends blocks to H.B in order)

R = "/home/claude/results/"
S = json.load(open(R + "paper_stats_r4.json"))
B = H.B


def V(k):
    if k not in S:
        raise KeyError(f"missing statistic: {k}")
    return S[k]


def fill(text):
    return re.sub(r"\{\{(\w+)\}\}", lambda m: V(m.group(1)), text)


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


def drop_tiny_p(cell):
    """Paired-difference cells 'd (ci); p = x': omit p when |d| < 0.001."""
    m = re.match(r"^([+−\-]?[0-9.]+(?: × 10\S+)?) \((.*)\); p = (\S+)$", cell)
    if not m: return cell
    d = m.group(1)
    tiny = "×" in d or abs(float(d.replace("−", "-").replace("+", ""))) < 0.001
    return f"{d} ({m.group(2)}); p n.r." if tiny else cell


TRIPOD = [
 ["1", "Title", "Identify the study as developing/evaluating a prediction model, target population and outcome", "Title"],
 ["2", "Abstract", "Structured summary", "Abstract"],
 ["3a", "Background", "Healthcare context and rationale, including existing models", "1"],
 ["3b", "Background", "Target population and intended purpose, intended users", "1; 2.1"],
 ["3c", "Background", "Known health inequalities between groups", "2.11; 3.8; Table A2"],
 ["4", "Objectives", "Study objectives, development and/or evaluation", "1 (RQ1–RQ3)"],
 ["5a", "Data", "Sources of data and rationale", "2.2"],
 ["5b", "Data", "Dates of data collection", "2.2"],
 ["6a", "Participants", "Study setting", "2.2"],
 ["6b", "Participants", "Eligibility criteria", "2.3"],
 ["6c", "Participants", "Details of any treatments received", "Not applicable (cross-sectional survey)"],
 ["7", "Data preparation", "Pre-processing and quality checking", "2.3, 2.4"],
 ["8a", "Outcome", "Definition of the outcome, timing, assessment method", "2.3, Eq. (1)"],
 ["8b", "Outcome", "Qualifications and demographics of outcome assessors", "Not applicable (self-report questionnaire); mode of administration in 4.4"],
 ["8c", "Outcome", "Blinding of outcome assessment", "2.1 (predictors and outcome from the same survey)"],
 ["9a", "Predictors", "Choice of initial predictors and pre-selection", "2.4"],
 ["9b", "Predictors", "Definition of all predictors and measurement", "2.4, 2.5; Table 2"],
 ["9c", "Predictors", "Qualifications of predictor assessors", "Not applicable (survey and examination data)"],
 ["10", "Sample size", "How the study size was arrived at", "2.3 (all eligible participants; minimum sample size per Riley et al.)"],
 ["11", "Missing data", "How missing data were handled", "2.5–2.8"],
 ["12a", "Analytical methods", "How data were used (development, evaluation)", "2.3, 2.11"],
 ["12b", "Analytical methods", "How predictors were handled", "2.5"],
 ["12c", "Analytical methods", "Model type, rationale, training, hyperparameter tuning", "2.5, 2.11"],
 ["12d", "Analytical methods", "Heterogeneity across clusters", "2.11 (cycles, design-based bootstrap, subgroups)"],
 ["12e", "Analytical methods", "Performance measures and comparison of models", "2.9, 2.11"],
 ["12f", "Analytical methods", "Model updating", "2.11 (recalibration-in-the-large)"],
 ["12g", "Analytical methods", "Calculation of predicted values for individuals", "2.5–2.7"],
 ["13", "Class imbalance", "Methods to handle class imbalance, recalibration", "2.11 (no resampling; Platt recalibration)"],
 ["14", "Fairness", "Approaches to address model fairness", "2.11, 3.8; Table A2"],
 ["15", "Model output", "Output of the model and rationale for thresholds", "2.6 (Eq. 4), 2.11"],
 ["16", "Training versus evaluation", "Differences between development and evaluation data", "2.4, 3.1; Table 1"],
 ["17", "Ethical approval", "Ethics committee approval and consent", "Declarations"],
 ["18a", "Funding", "Sources of funding and role of funders", "Declarations"],
 ["18b", "Conflicts of interest", "Declarations", "Declarations"],
 ["18c", "Protocol", "Where the protocol can be accessed", "2.12; repository (protocol_v2.md)"],
 ["18d", "Registration", "Registration information", "2.12 (not registered)"],
 ["18e", "Data sharing", "Details of data availability", "Declarations (data availability)"],
 ["18f", "Code sharing", "Details of code availability", "Declarations (code availability)"],
 ["19", "Patient and public involvement", "Details of involvement", "None undertaken"],
 ["20a", "Participants", "Flow of participants, numbers with and without the outcome", "2.3, 3.1; Fig. 1"],
 ["20b", "Participants", "Characteristics of participants, missing data", "3.1; Table 1; Table A4"],
 ["20c", "Participants", "Comparison of development and evaluation data", "3.1; Table 1"],
 ["21", "Model development", "Number of participants and outcome events in each analysis", "2.3; Tables 1, 4, 5"],
 ["22", "Model specification", "Full model to allow predictions for individuals", "Table 3 (intercept and coefficients); knowledge_schema.jsonld"],
 ["23a", "Model performance", "Performance estimates with confidence intervals, key subgroups", "3.3–3.5, 3.8; Tables 4–8; Table A2"],
 ["23b", "Model performance", "Heterogeneity across clusters", "3.3, 3.8; Table A7"],
 ["24", "Model updating", "Results of model updating", "3.3"],
 ["25", "Interpretation", "Overall interpretation of the main results, fairness", "4.1–4.5"],
 ["26", "Limitations", "Limitations of the study", "4.6"],
 ["27a", "Usability", "How poor quality or unavailable input data are handled", "2.6–2.10, 3.4–3.7; Tables 5–7, 9; Boxes 1–3"],
 ["27b", "Usability", "Whether users need to interact with input data or expertise required", "2.1, 4.5"],
 ["27c", "Implications", "Future research steps and generalisability", "4.2, 4.5, 4.6, 5"],
]

if __name__ == "__main__":
    from tables_r4 import tables
    TT = tables(V); TT["tableTRIPOD"]["rows"] = TRIPOD
    out = []
    for b in B:
        b = dict(b)
        if b["t"] in TT:
            T = TT[b["t"]]
            rows = [[drop_tiny_p(cites(fill(str(c)))) for c in r] for r in T["rows"]]
            out.append({"t": "table", **T, "cap": cites(fill(T["cap"])), "note": cites(fill(T.get("note", ""))), "rows": rows}); continue
        for k in ("x", "cap"):
            if k in b:
                b[k] = [cites(fill(s)) for s in b[k]] if isinstance(b[k], list) else cites(fill(b[k]))
        out.append(b)
    js = json.dumps(out, ensure_ascii=False).replace("p = <0.001", "p < 0.001")
    for a, b_ in [("the proposed model", "the fuzzy additive model"), ('"Proposed"', '"Fuzzy additive"'), ('"Proposed, ', '"Fuzzy additive, '),
                  ("proposed minus", "fuzzy model minus"), ("proposed higher", "fuzzy model higher"), ("admissible-value", "worst-case"), ("the proposed refer-on-abstain", "the fuzzy model's refer-on-abstain"),
                  ("positive values favour the fuzzy additive model", "positive values favour the fuzzy model")]:
        js = js.replace(a, b_)
    out = json.loads(js)
    for i, b in enumerate(out):
        if b["t"] == "refs":
            out[i] = {"t": "refs", "x": [f"[{n + 1}] {REFS[k]}" for n, k in enumerate(order)]}
    unused = [k for k in REFS if k not in order]
    json.dump(out, open(R + "manuscript_r4.json", "w"), ensure_ascii=False, indent=0)
    left = re.findall(r"\{\{\w+\}\}", json.dumps(out, ensure_ascii=False))
    print("blocks", len(out), "refs cited", len(order), "unused", unused, "unfilled", left)
