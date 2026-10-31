"""Analyses, part 7: intervals for the simulated scenarios that pool all 10 repetitions (paired bootstrap within each
repetition, 200 resamples, pooled over repetitions) for the difference of each rule to worst-case bounds in AURC and
net benefit; sensitivity of the point-score band to the development coverage target (80%, 90%, 95%)."""
import json, time
import numpy as np
exec(open("/home/claude/src/sec5.py").read().split("# ---------------------------------------------------------------- natural cohorts")[0])
t0 = time.time(); OUT7 = {}
DELTAS = {}
for tgt in (0.80, 0.95):
    oof = cv_scores(lambda: MAKE["fuzzy"](BASE), dev); m, t = MODELS["fuzzy"]
    DELTAS[tgt] = float(np.quantile(np.abs(lmargin(oof, t)), 1 - tgt))
OUT7["deltas_fuzzy"] = {str(k): v for k, v in DELTAS.items()} | {"0.9": DELTA["fuzzy"]}


def sel_at(d, delta):
    m, t = MODELS["fuzzy"]; s = m.score(d); mg = np.abs(lmargin(s, t))
    return mg >= delta, s >= t, mg


def pooled(key, B=200):
    diffs = {}; extra = {}
    for rep in range(10):
        d = SCEN[key](rep); y = d.y.values
        r, a = evaluate(d, with_ps=(key != "mcar30"), keep=True)
        rules = dict(a["fuzzy"]); rules["sel"] = ps_sel(d, "fuzzy")
        for tgt, dl in DELTAS.items():
            v = summarise(y, *sel_at(d, dl), rules["wc"][0].mean()); extra.setdefault(str(tgt), []).append((v["aurc"], v["nb10"]))
        ref = {k: (dec | ~cov) for k, (cov, dec, conf) in rules.items()}
        rg = np.random.default_rng(SEED + 700 + rep)
        for b in range(B):
            i = rg.integers(0, len(y), len(y))
            wa = aurc(y[i], rules["wc"][2][i], rules["wc"][1][i]); w05 = nb(y[i], ref["wc"][i], .05); w10 = nb(y[i], ref["wc"][i], .10)
            for k in ("mi", "ps", "sel"):
                if k not in rules: continue
                cov, dec, conf = rules[k]
                diffs.setdefault((k, "aurc"), []).append(aurc(y[i], conf[i], dec[i]) - wa)
                diffs.setdefault((k, "nb05"), []).append(nb(y[i], ref[k][i], .05) - w05)
                diffs.setdefault((k, "nb10"), []).append(nb(y[i], ref[k][i], .10) - w10)
    o = {f"{k}|{mt}": dict(mean=float(np.mean(v)), lo=float(np.quantile(v, .025)), hi=float(np.quantile(v, .975))) for (k, mt), v in diffs.items()}
    o["sel_targets"] = {t: dict(aurc=float(np.mean([x[0] for x in v])), nb10=float(np.mean([x[1] for x in v]))) for t, v in extra.items()}
    return o


for key in SCEN:
    if key.startswith("shift_ord"): continue
    OUT7[key] = pooled(key); print(key, round(time.time() - t0), flush=True)
    json.dump(OUT7, open(RV + "sec7.json", "w"), indent=1)
# point-score band targets in the three cohorts
for nm, d in [("natural|internal_test", test), ("natural|temporal", L), ("natural|external_pooled", POOL)]:
    m, t = MODELS["fuzzy"]; zl, zu = m.bounds(d); cref = float(((zl >= t) | (zu < t)).mean())
    OUT7.setdefault(nm, {})["sel_targets"] = {str(tgt): (lambda v: dict(aurc=v["aurc"], nb10=v["nb10"]))(summarise(d.y.values, *sel_at(d, dl), cref))
                                               for tgt, dl in DELTAS.items()}
json.dump(OUT7, open(RV + "sec7.json", "w"), indent=1)
print("done sec7", round(time.time() - t0))
