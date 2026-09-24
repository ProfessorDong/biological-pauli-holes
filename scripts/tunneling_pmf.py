#!/usr/bin/env python3
"""Tunneling-weighted KIE from the DAD potential of mean force (the RIGHT observable).

Vibronically-nonadiabatic H-transfer (Kuznetsov-Ulstrup / Soudackov-Hammes-Schiffer, the
standard framework for SLO). At fixed T the isotope-independent Marcus factor cancels in the
KIE, leaving the donor-acceptor-distance (DAD) average of the isotope-dependent vibronic
coupling |V_X(R)|^2 = |V0|^2 exp(-2 a_X (R-R0)), with a_D > a_H (the heavier D overlap decays
faster with R):

     KIE(T) = <|V_H|^2>_P(R;T) / <|V_D|^2>_P(R;T)
            = Integral P(R;T) exp(-2 a_H (R-R0)) dR  /  Integral P(R;T) exp(-2 a_D (R-R0)) dR

P(R;T) = exp(-W(R)/kT)/Z from the umbrella PMF W(R). Reweighting the PMF to two temperatures
gives the KIE temperature dependence (the gating signature). The SYSTEM-relative KIE (each
normalized to WT) is independent of R0 and of the intrinsic prefactor, so it is the robust
quantity to regress against the measured ladder. a_H,a_D scanned for sensitivity.
Usage: tunneling_pmf.py"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import glob, numpy as np
trapz = np.trapezoid if hasattr(np,"trapezoid") else np.trapz
kB=0.0019872041  # kcal/mol/K
SYS=['WT','L754A','I553A','DM']
KIE_EXP={'WT':(66,52),'L754A':(106,82),'I553A':(148,77),'DM':(537,None)}  # (10C,40C); DM diff. series
A_PAIRS=[(20,28),(25,35),(30,42)]   # (a_H,a_D) in A^-1; middle = default, sensitivity scan
R0=2.70                              # reference reactive DAD (A); cancels in system-relative KIE

def load_pmf(tag):
    f=_REPO + f"/results/umbrella/{tag}/pmf.dat"
    d=np.loadtxt(f); return d[:,0], d[:,1]                # r (A), W (kcal/mol)

def kie(tag, aH, aD, T):
    r,W=load_pmf(tag)
    lnP = -W/(kB*T)                                       # ln unnormalized P(R;T)
    # integrands in log space (avoid underflow): ln[P * exp(-2a(r-R0))]
    def integ(a):
        g = lnP - 2*a*(r-R0); g -= g.max()
        return trapz(np.exp(g), r)                     # (max factored out cancels in ratio? keep per-isotope)
    # keep the max-shift per isotope but restore the relative offset between H and D
    gH = lnP - 2*aH*(r-R0); gD = lnP - 2*aD*(r-R0)
    mx = max(gH.max(), gD.max())
    IH = trapz(np.exp(gH-mx), r); ID = trapz(np.exp(gD-mx), r)
    return IH/ID

print("  DAD tunneling-weighted KIE (system-relative, normalized to WT)\n")
for aH,aD in A_PAIRS:
    tag_default = (aH,aD)==(25,35)
    print(f"  === a_H={aH}, a_D={aD} A^-1 {'(default)' if tag_default else '(sensitivity)'} ===")
    print(f"    {'sys':<7}{'KIE_pred(10C)':>14}{'rel/WT':>9}{'KIE_exp':>9}{'rel_exp':>9}{'Tdep_pred':>11}{'Tdep_exp':>10}")
    pred={}
    for tag in SYS:
        try:
            k10=kie(tag,aH,aD,283.15); k40=kie(tag,aH,aD,313.15)
        except Exception as e:
            print(f"    {tag}: PMF missing ({e})"); continue
        pred[tag]=(k10,k40)
    if 'WT' not in pred: continue
    kwt=pred['WT'][0]
    for tag in SYS:
        if tag not in pred: continue
        k10,k40=pred[tag]; e10,e40=KIE_EXP[tag]
        relp=k10/kwt; rele=e10/KIE_EXP['WT'][0]
        tdp=k10/k40; tde=(e10/e40) if e40 else float('nan')
        print(f"    {tag:<7}{k10:>14.3f}{relp:>9.2f}{e10:>9}{rele:>9.2f}{tdp:>11.3f}{tde:>10.2f}")
    # correlation of relative-KIE (pred vs exp) over the self-consistent JBC set
    fitset=['WT','L754A','I553A']
    if all(t in pred for t in fitset):
        xp=np.log([pred[t][0]/kwt for t in fitset]); xe=np.log([KIE_EXP[t][0]/66 for t in fitset])
        R=np.corrcoef(xp,xe)[0,1]
        print(f"    -> ln(rel KIE) pred vs exp, JBC set {fitset}: R={R:+.3f}  R^2={R*R:.3f}")
    print()
print("  Note: single-replica PMFs -> descriptors carry sampling error; run replicas (Path A)")
print("  for error bars before interpreting the correlation. a_H,a_D from H/D vibronic overlaps;")
print("  the SYSTEM-RELATIVE KIE is independent of R0 and the intrinsic prefactor.")
