"""Tables of version 8: tables_r7 plus a box of terms (Box 1); algorithm boxes renumbered 2-3; 'no clear difference'
instead of 'equivalent'; fuller notes on matched coverage and AURC."""
from tables_r7 import tables as tables_r7

TERMS = [
    ["Decide", "The rule returns a decision for the participant: refer or do not refer."],
    ["Refer", "Recommend that the participant complete the full PHQ-9 questionnaire."],
    ["Do not refer", "Recommend that the PHQ-9 is not needed for now."],
    ["Abstain", "The rule returns no decision, because the missing inputs leave the decision uncertain by the rule's criterion."],
    ["Coverage", "Proportion of participants the rule decides; split into the proportions referred and not referred."],
    ["Refer-on-abstain", "Policy that refers every participant on whom the rule abstains, so that everyone receives a referral decision; used for cases missed, questionnaires avoided and net benefit."],
    ["Selective prediction", "Deciding only the participants the model is most confident about; here, point-score selective prediction abstains when the predicted probability lies in a band around the threshold."],
    ["Matched coverage", "Each rule decides the participants it ranks most confidently, up to the coverage of worst-case bounds with the same base model, so that rules are compared on the same number of decisions."],
    ["AURC", "Area under the curve of balanced error against coverage, from 0.05 to 1, when participants are added in the rule's order of confidence; lower is better."],
    ["Worst-case guarantee", "The predicted probability stays on one side of the threshold for every admissible value of the missing inputs. It concerns the model's prediction, not the participant's true status."],
]


def tables(V):
    T = tables_r7(V)
    T["tableTerms"] = {"cap": "Box 1. Terms used for the outputs of an abstention rule.", "head": ["Term", "Meaning"], "rows": TERMS,
                       "note": "", "w": [1.5, 5.3], "small": True}
    T["tableAlg"]["cap"] = T["tableAlg"]["cap"].replace("Box 1.", "Box 2.")
    T["tableAlgGen"]["cap"] = T["tableAlgGen"]["cap"].replace("Box 2.", "Box 3.")
    T["tableSum"]["note"] = T["tableSum"]["note"].replace(
        "'Lower' and 'equivalent': the 95% bootstrap interval for the difference in net benefit between the best rule and worst-case bounds "
        "excludes or includes zero",
        "'Lower': the 95% bootstrap interval for the difference in net benefit between the best rule and worst-case bounds excludes zero; "
        "'no clear difference': it includes zero, which shows a lack of evidence of a difference, not equivalence")
    assert "no clear difference" in T["tableSum"]["note"]
    n = T["tableAbst5"]["note"]
    T["tableAbst5"]["note"] = n.replace("AURC, area under the balanced-error/coverage curve over coverage 0.05–1 (lower is better);",
        "AURC, area under the curve of balanced error against coverage over coverage 0.05–1, with participants added in each rule's order of "
        "confidence (lower is better);")
    assert T["tableAbst5"]["note"] != n
    return T
