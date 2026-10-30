"""Build analytic cohorts from NHANES 2017-Mar 2020 (P_) and Aug 2021-Aug 2023 (_L)."""
import numpy as np, pandas as pd, json, sys

D = "/home/claude/data/"


def rd(cycle, comp):
    f = {"P": f"P_{comp}", "L": f"{comp}_L"}.get(cycle, f"{comp}_{cycle}")
    df = pd.read_sas(D + f + ".xpt")
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(6)  # SAS transport stores 0 as ~5e-79
    return df


def nanify(s, bad):
    s = s.copy()
    s[s.isin(bad)] = np.nan
    return s


def yes(s):  # 1=yes,2=no, others missing
    return s.map({1.0: 1.0, 2.0: 0.0})


def build(cycle):
    demo = rd(cycle, "DEMO"); dpq = rd(cycle, "DPQ"); slq = rd(cycle, "SLQ")
    bmx = rd(cycle, "BMX"); huq = rd(cycle, "HUQ"); smq = rd(cycle, "SMQ")
    paq = rd(cycle, "PAQ"); mcq = rd(cycle, "MCQ"); diq = rd(cycle, "DIQ"); bpq = rd(cycle, "BPQ")
    wt = "WTINTPRP" if cycle == "P" else "WTINT2YR"
    if cycle in "DEFGHI":  # 2005-2016: harmonise marital status coding
        demo["DMDMARTZ"] = demo.DMDMARTL.map({1.0: 1.0, 6.0: 1.0, 2.0: 2.0, 3.0: 2.0, 4.0: 2.0, 5.0: 3.0})
    if "RIDRETH3" not in demo:  # before 2011: no separate Asian category
        demo["RIDRETH3"] = demo.RIDRETH1.map({1.0: 1.0, 2.0: 2.0, 3.0: 3.0, 4.0: 4.0, 5.0: 7.0})
    df = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR", "DMDEDUC2", "DMDMARTZ",
               wt, "SDMVSTRA", "SDMVPSU", "RIDSTATR"]].rename(columns={wt: "wt"})
    flow = {"demo_records": len(df)}
    df = df[df.RIDAGEYR >= 18]; flow["adults"] = len(df)
    items = [f"DPQ0{k}0" for k in range(1, 10)]
    df = df.merge(dpq, on="SEQN", how="left")
    flow["adults_with_DPQ_record"] = int(df[items].notna().any(axis=1).sum())
    valid = df[items].isin([0, 1, 2, 3]).all(axis=1)
    df["excluded_outcome"] = ~valid
    df["phq9"] = df[items].where(df[items].isin([0, 1, 2, 3])).sum(axis=1, min_count=9)
    df["y"] = (df.phq9 >= 10).astype(float)
    df.loc[~valid, "y"] = np.nan
    for t in [slq, bmx[["SEQN", "BMXBMI"]], huq, smq, paq, mcq, diq, bpq]:
        df = df.merge(t, on="SEQN", how="left", suffixes=("", "_dup"))
    X = pd.DataFrame({"SEQN": df.SEQN})
    X["age"] = df.RIDAGEYR
    X["female"] = (df.RIAGENDR == 2).astype(float)
    X["pir"] = df.INDFMPIR
    X["educ"] = nanify(df.DMDEDUC2, [7, 9])
    m = nanify(df.DMDMARTZ, [77, 99])
    X["married"] = (m == 1).astype(float).where(m.notna())
    X["prev_married"] = (m == 2).astype(float).where(m.notna())
    X["bmi"] = df.BMXBMI
    X["sleep_wd"] = df.SLD012 if "SLD012" in df else nanify(df.SLD010H, [77, 99])  # 2005-2014: SLD010H
    X["sleep_we"] = df.SLD013 if "SLD013" in df else np.nan
    X["srh"] = nanify(df.HUQ010, [7, 9])
    smk = df.SMQ040.map({1.0: 1.0, 2.0: 1.0, 3.0: 0.0})
    smk[df.SMQ020 == 2] = 0.0
    X["smoker"] = smk
    X["diabetes"] = df.DIQ010.map({1.0: 1.0, 2.0: 0.0, 3.0: 0.0})
    X["hypertension"] = yes(df.BPQ020)
    X["sedentary"] = (nanify(df.PAD680, [7777, 9999]) / 60.0) if "PAD680" in df else np.nan  # hours/day
    cond = ["MCQ010", "MCQ160A", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F",
            "MCQ160L", "MCQ160P", "MCQ220"]
    if cycle in "HI":
        cond = [c if c != "MCQ160P" else "MCQ160O" for c in cond]  # COPD item renamed
    elif cycle in "DEFG":
        cond = [c for c in cond if c != "MCQ160P"]  # COPD item not fielded before 2013
    cm = pd.concat([yes(df[c]) for c in cond], axis=1)
    X["n_chronic"] = cm.sum(axis=1, min_count=1)
    if cycle in ("P", "I"):  # extended sleep items, not fielded in 2021-2023
        X["told_sleep_trouble"] = yes(df.SLQ050)
        X["daytime_sleepy"] = nanify(df.SLQ120, [7, 9])
    X["y"] = df.y.values; X["phq9"] = df.phq9.values
    X["phq2"] = (df.DPQ010.where(df.DPQ010.isin([0,1,2,3])) + df.DPQ020.where(df.DPQ020.isin([0,1,2,3]))).values
    for c in ["wt", "SDMVSTRA", "SDMVPSU", "RIDRETH3"]:
        X[c] = df[c].values
    X["excluded_outcome"] = df.excluded_outcome.values
    flow["excluded_missing_or_invalid_PHQ9"] = int(df.excluded_outcome.sum())
    coh = X[~X.excluded_outcome].drop(columns="excluded_outcome").reset_index(drop=True)
    flow["analytic"] = len(coh); flow["positives"] = int(coh.y.sum())
    flow["prevalence_unweighted"] = float(coh.y.mean())
    flow["prevalence_weighted"] = float(np.average(coh.y, weights=coh.wt)) if coh.wt.gt(0).all() else float(np.average(coh.y[coh.wt > 0], weights=coh.wt[coh.wt > 0]))
    excl = X[X.excluded_outcome]
    return coh, excl, flow


if __name__ == "__main__":
    out = {}
    for cyc in ["P", "L", "I", "D", "E", "F", "G", "H"]:
        coh, excl, flow = build(cyc)
        coh.to_pickle(f"/home/claude/data/cohort_{cyc}.pkl")
        excl.to_pickle(f"/home/claude/data/excluded_{cyc}.pkl")
        out[cyc] = flow
    json.dump(out, open("/home/claude/results/flow.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
