"""Fig. 7 of version 4: risk-coverage curves of four abstention rules with the base model held fixed
(fuzzy additive model), in four scenarios. Recomputes the curves with the functions of sec4.py."""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
exec(open("/home/claude/src/sec4.py").read().split("OUT = {}")[0])


def residual_z(var):
    from sklearn.linear_model import LinearRegression
    oth = [v for v in BASE if v != var]
    Xd = ml_matrix(dev, oth); med = Xd.median(); ok = dev[var].notna().values
    lm = LinearRegression().fit(Xd.fillna(med).values[ok], dev[var].values[ok])
    sd = np.std(dev[var].values[ok] - lm.predict(Xd.fillna(med).values[ok]))
    return lambda d: np.nan_to_num((d[var].values - lm.predict(ml_matrix(d, oth).fillna(med).values)) / sd)


def scen_data():
    out = [("2005–2014, natural", POOL)]
    rng = np.random.default_rng(SEED); MC = None
    for rate in [0.0, 0.1, 0.2, 0.3, 0.5]:
        for rep in range(10 if rate > 0 else 1):
            d = test.copy(); mask = rng.random((len(d), len(MASKABLE))) < rate
            for j, f in enumerate(MASKABLE): d.loc[mask[:, j], f] = np.nan
            if rate == 0.3 and MC is None: MC = d
    out.append(("Internal test, 30% MCAR", MC))
    g = np.random.default_rng(SEED + 202); d = test.copy(); u = g.random(len(d))
    d.loc[u < np.where(d.srh.values >= 4, 0.9, 0.05), "srh"] = np.nan
    out.append(("MNAR 0.9/0.05: self-rated health", d))
    zt = residual_z("srh")(test); w = np.exp(1.0 * zt); w /= w.sum()
    i = np.random.default_rng(SEED + 404).choice(len(test), len(test), replace=True, p=w)
    d = test.iloc[i].reset_index(drop=True).copy(); d["srh"] = np.nan
    out.append(("Self-rated health absent, shifted (γ = 1)", d))
    return out


LAB = {"wc": "Worst-case bounds", "mi": "Imputation spread", "ps": "Pattern submodels", "cc": "Complete case / point-score margin"}
COL = {"wc": "#c0504d", "mi": "#4f81bd", "ps": "#2e8b57", "cc": "#7f7f7f"}
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6), sharey=True)
store = {}
for ax, (lab, d) in zip(axes, scen_data()):
    res, arr = evaluate(d, with_ps="MCAR" not in lab, keep=True)
    y = d.y.values; rules = arr["fuzzy"]; store[lab] = {}
    for r in ("wc", "mi", "ps", "cc"):
        if r not in rules: continue
        cov, dec, conf = rules[r]
        ba = rc_curve(y, conf, dec)
        ax.plot(100 * GRID, ba, color=COL[r], lw=1.8, label=LAB[r])
        oc = cov.mean()
        if 0.05 <= oc <= 1.0 and r != "cc":
            ax.plot(100 * oc, np.interp(oc, GRID, ba), "o", mfc="white", mec=COL[r], ms=6)
        store[lab][r] = dict(curve=[float(x) for x in ba], own=float(oc))
    ax.set_title(lab, fontsize=9.5); ax.set_xlabel("Participants decided (%)"); ax.grid(alpha=0.3)
axes[0].set_ylabel("Balanced accuracy among decided")
axes[0].legend(fontsize=7.5, loc="lower left")
plt.tight_layout(); plt.savefig("/home/claude/results/fig_riskcov4.png", dpi=200)
json.dump(store, open("/home/claude/results/secondary/fig_riskcov4.json", "w"), indent=1)
print("saved")
