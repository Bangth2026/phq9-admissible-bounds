"""Export the knowledge schema (observation, concept, rule and provenance layers) as JSON-LD using SKOS for the
concept scheme and a small namespace for membership functions, priors and rules."""
import json, pickle
import pandas as pd
from fofs import VARIABLES, knowledge_concepts
from experiment import LRModel, GBTModel  # noqa

R = "/home/claude/results/"
pri = pd.read_csv(R + "priors_2015_2016.csv").set_index("concept")
S = pickle.load(open(R + "models.pkl", "rb")); m = S["models"]["Proposed (full)"]
NS = "urn:phq-prescreen:schema:"
ctx = {"@vocab": NS, "skos": "http://www.w3.org/2004/02/skos/core#", "dct": "http://purl.org/dc/terms/",
       "xsd": "http://www.w3.org/2001/XMLSchema#", "prov": "http://www.w3.org/ns/prov#",
       "broader": {"@id": "skos:broader", "@type": "@id"}, "inScheme": {"@id": "skos:inScheme", "@type": "@id"},
       "prefLabel": "skos:prefLabel", "definition": "skos:definition", "source": "dct:source",
       "wasDerivedFrom": "prov:wasDerivedFrom"}
graph = [{"@id": "scheme", "@type": "skos:ConceptScheme", "prefLabel": "Non-symptom predictors of PHQ-9 >= 10 (NHANES)",
          "dct:created": "2026-10-28", "dct:modified": "2026-10-30"}]
for v, (dom, typ, rng, unit, src) in VARIABLES.items():
    graph.append({"@id": f"var/{v}", "@type": "ObservationVariable", "prefLabel": v, "domain": dom, "valueType": typ,
                  "admissibleMin": rng[0], "admissibleMax": rng[1], "unit": unit, "source": src})
for c, spec in knowledge_concepts(extended=True).items():
    mf = spec["mf"]
    node = {"@id": f"concept/{c}", "@type": ["skos:Concept", "LinguisticConcept"], "inScheme": "scheme",
            "prefLabel": c, "boundTo": {"@id": f"var/{spec['variable']}"},
            "cutPointProvenance": spec["provenance"]}
    if mf[0] == "trap":
        node["membership"] = {"@type": "TrapezoidalMembership", "a": mf[1], "b": mf[2], "c": mf[3], "d": mf[4],
                              "partition": "Ruspini"}
    else:
        node["membership"] = {"@type": "CrispSetMembership", "values": list(mf[1])}
    if c in pri.index:
        r = pri.loc[c]
        node["directionPrior"] = {"@type": "DirectionPrior", "value": int(r.direction),
                                  "estimatedOn": "NHANES 2015-2016 (cohort_I)",
                                  "riskRatio": None if pd.isna(r.get("RR")) else float(r.RR),
                                  "ciLower": None if pd.isna(r.get("lo")) else float(r.lo),
                                  "ciUpper": None if pd.isna(r.get("hi")) else float(r.hi),
                                  "note": None if pd.isna(r.get("note")) else r.note}
    graph.append(node)
graph.append({"@id": "rulebase/v2", "@type": "AdditiveRuleBase", "intercept": float(m.b0), "link": "logit",
              "aggregation": "additive", "tnorm": "minimum",
              "rules": [{"@type": "Rule", "antecedent": [{"@id": f"concept/{m.cnames[j]}"} for j in r],
                         "coefficient": float(b), "signConstraint": int(s)}
                        for r, b, s in zip(m.rules, m.beta, m.signs)]})
json.dump({"@context": ctx, "@graph": graph}, open(R + "secondary/knowledge_schema.jsonld", "w"), indent=1)
print("concepts", sum(1 for g in graph if g.get("@type") and "LinguisticConcept" in g["@type"]))
