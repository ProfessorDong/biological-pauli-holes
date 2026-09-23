#!/usr/bin/env python3
"""
analysis.py -- test whether the static confinement descriptor (crowd_Ht) predicts
the SELF-CONSISTENT single-mutant KIE series of SLO (Hu et al., JBC 2019, Table 1).

Descriptor: crowd_Ht = protein exchange wall (sum exp(-r/lambda), 2-6 A shell) at
the transferring H in the substrate-bound near-attack model (dock_substrate.py),
evaluated for WT and each truncation model (make_mutants.py).

Honest expectation: SLO single-mutant KIEs are gating-dominated (the largest KIE,
I553A=148, is a DISTAL mutant whose crowd_Ht is unchanged), so a STATIC descriptor
is expected to predict poorly. A null/weak result here is itself the finding: it
says the confinement axis is not the dominant determinant of KIE variation among
SLO single mutants -> motivates the MD (gating) pipeline.

Run:  conda activate pauli ; python scripts/analysis.py
"""
import os, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLO_DIR = os.path.join(ROOT, "pauli_data", "slo")
KIE_CSV = os.path.join(SLO_DIR, "slo_kie.csv")
OUT_DIR = os.path.join(ROOT, "results"); os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, "slo_descriptors.csv")
TS_NPZ  = os.path.join(OUT_DIR, "_transfer_site.npz")
LAMBDA, SHELL = 1.5, (2.0, 6.0)

def heavy(u):
    al = getattr(u.atoms, "altLocs", np.array([""]*len(u.atoms)))
    ag = u.atoms[np.isin(al, ["", " ", "A"])]
    return ag.select_atoms("not name H* and not name FE FE2 FE3 and not resname HOH WAT")

def crowd_Ht(pdb, Ht):
    import MDAnalysis as mda
    pos = heavy(mda.Universe(pdb)).positions
    r = np.linalg.norm(pos - Ht, axis=1); m = (r > SHELL[0]) & (r < SHELL[1])
    return float(np.sum(np.exp(-r[m]/LAMBDA)))

def regress(df, resp, pred):
    import statsmodels.api as sm
    from sklearn.model_selection import LeaveOneOut
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from scipy.stats import spearmanr
    y = df[resp].values; n = len(df)
    X = sm.add_constant(df[[pred]]); ols = sm.OLS(y, X).fit()
    Xn = df[[pred]].values; yp = np.zeros(n)
    for tr, te in LeaveOneOut().split(Xn):
        yp[te] = LinearRegression().fit(Xn[tr], y[tr]).predict(Xn[te])
    q2 = 1 - np.sum((y-yp)**2)/np.sum((y-y.mean())**2)
    rmse = float(np.sqrt(mean_squared_error(y, yp)))
    rho, prho = spearmanr(df[pred], y)
    print(f"    {resp:11s}~{pred:9s} n={n} R2={ols.rsquared:5.2f} slope={ols.params.iloc[1]:+.3f} "
          f"p={ols.pvalues.iloc[1]:.3f} LOO-Q2={q2:+.2f} | Spearman rho={rho:+.2f} p={prho:.3f}")

def main():
    kie = pd.read_csv(KIE_CSV, comment="#"); kie.columns=[c.strip() for c in kie.columns]
    if not os.path.exists(TS_NPZ):
        print("[!] run scripts/dock_substrate.py first (need transfer site)"); return
    Ht = np.load(TS_NPZ)["Ht"]

    rows=[]
    for _,r in kie.iterrows():
        p=os.path.join(SLO_DIR, str(r["pdb_id"])+".pdb")
        c=crowd_Ht(p,Ht) if os.path.exists(p) else np.nan
        rows.append(c)
    kie["crowd_Ht"]=rows
    for col in ("KIE_10C","KIE_40C","dHt_A"):
        if col in kie: kie[col]=pd.to_numeric(kie[col],errors="coerce")
    kie.to_csv(OUT_CSV,index=False)

    cols=[c for c in ["variant","pdb_id","site_class","dHt_A","crowd_Ht","KIE_10C","KIE_40C"] if c in kie]
    print("\n", kie[cols].to_string(index=False), "\n", sep="")
    print(f"wrote {OUT_CSV}")

    for T in ("KIE_10C","KIE_40C"):
        if T not in kie: continue
        d=kie.dropna(subset=[T,"crowd_Ht"]).copy(); d["ln"+T]=np.log(d[T])
        print(f"\n=== response ln({T}) vs crowd_Ht ===")
        print("  [all variants -- distal mutants have flat crowd_Ht, so they test the null]")
        if len(d)>=4 and d["crowd_Ht"].nunique()>1: regress(d,"ln"+T,"crowd_Ht")
        sub=d[d["site_class"].isin(["reference","intermediate","contact"])]
        print("  [subset where crowd_Ht actually varies: reference+intermediate+contact]")
        if len(sub)>=4 and sub["crowd_Ht"].nunique()>1: regress(sub,"ln"+T,"crowd_Ht")
        else: print(f"    (only {len(sub)} such variants)")

    print("\nInterpretation: a weak/insignificant crowd_Ht effect (esp. that the "
          "largest KIE, I553A, is a DISTAL mutant with unchanged crowd_Ht) shows SLO "
          "single-mutant KIEs are GATING-dominated -- the static confinement descriptor "
          "is not the controlling variable here. That motivates the MD pipeline (Part B), "
          "which samples the donor-acceptor distance distribution (the gating term).")

if __name__ == "__main__":
    main()
