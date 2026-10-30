import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
S = json.load(open("/home/claude/results/paper_stats.json"))
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(10, 5.2)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 60)
def box(x, y, w, h, t, fc="#eef4fa", ec="#1b6ca8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.4)
def arr(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9))
# development column
box(2, 50, 30, 7, f"NHANES 2017–March 2020\n{S['flow_dev_demo_records']} participants")
box(2, 38, 30, 7, f"Adults aged ≥18 years\nn = {S['flow_dev_adults']}")
box(36, 38, 27, 7, f"Excluded: no PHQ-9 record n = {S['flow_dev_noDPQ']}\nincomplete/invalid item(s) n = {S['flow_dev_invalid']}", "#fbf3e6", "#e09f3e")
box(2, 26, 30, 7, f"Analytic cohort n = {S['flow_dev_analytic']}\nPHQ-9 ≥ 10: {S['flow_dev_positives']} ({S['flow_dev_prevalence_unweighted']}%)")
box(2, 10, 14, 9, f"Development\nn = {S['n_dev']}\n({S['pos_dev']} positive)")
box(18, 10, 14, 9, f"Internal test\nn = {S['n_test']}\n({S['pos_test']} positive)")
arr(17, 50, 17, 45.6); arr(17, 38, 17, 33.6); arr(32, 41.5, 35.6, 41.5); arr(10, 26, 9, 19.6); arr(24, 26, 25, 19.6)
ax.text(17, 22.6, "stratified 70/30 split", ha="center", fontsize=7, style="italic")
box(2, 1, 30, 6, f"5-fold CV within development: tuning,\nthreshold, Platt calibration", "#f3f3f3", "#888888")
# temporal column
box(68, 50, 30, 7, f"NHANES August 2021–August 2023\n{S['flow_tmp_demo_records']} participants")
box(68, 38, 30, 7, f"Adults aged ≥18 years\nn = {S['flow_tmp_adults']}")
box(68, 26, 30, 7, f"Excluded: no PHQ-9 record n = {S['flow_tmp_noDPQ']}\nincomplete/invalid item(s) n = {S['flow_tmp_invalid']}", "#fbf3e6", "#e09f3e")
box(68, 10, 30, 9, f"Temporal validation cohort\nn = {S['flow_tmp_analytic']}\nPHQ-9 ≥ 10: {S['flow_tmp_positives']} ({S['flow_tmp_prevalence_unweighted']}%)\nfrozen models and thresholds")
arr(83, 50, 83, 45.6); arr(83, 38, 83, 33.6); arr(83, 26, 83, 19.6)
box(36, 50, 27, 7, f"NHANES 2015–2016 (n = {S['flow_pri_analytic']} adults)\ndirection priors only", "#f3f3f3", "#888888")
box(36, 1, 27, 9, f"External validation NHANES 2005–2014\n5 cycles, n = {S['n_ext']}\nPHQ-9 ≥ 10: {S['pos_ext']} ({S['prev_ext']}%)\nfrozen models and thresholds")
fig.tight_layout(); fig.savefig("/home/claude/results/fig_flow.png", dpi=300)
