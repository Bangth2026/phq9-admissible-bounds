"""Reproduces the development-set CV used to lock the version-2 structure (protocol_v2.md)."""
import json
from sklearn.metrics import roc_auc_score
from experiment import dev, BASE, PRIORS, cv_scores, FOFS, LRModel
out = {f"K{K}": roc_auc_score(dev.y, cv_scores(lambda K=K: FOFS(BASE, n_rules=K, priors=PRIORS, structure="main"), dev)) for K in (0, 8, 16, 32, 64)}
out["LR"] = roc_auc_score(dev.y, cv_scores(lambda: LRModel(BASE), dev))
json.dump(out, open("/home/claude/results/cv_structure_v2.json", "w"), indent=1); print(out)
