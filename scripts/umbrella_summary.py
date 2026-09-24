#!/usr/bin/env python3
"""Assemble the DAD-PMF gating descriptors for all systems and regress them against the
KIE ladder. Self-consistency (project rule): the PRIMARY regression uses ONLY the JBC-2019
10 C series (WT, L754A, I553A); the L546A/L754A double mutant KIE (537, Hu JACS 2014, 30 C)
is a DIFFERENT series and is shown as a cross-reference point, NOT in the fit.

Gating theory (Klinman/Hammes-Schiffer): larger + more T-dependent KIE <-> longer r_DA^0 and/or
softer k_gate. So expect KIE to ANTI-correlate with k_gate and correlate with r_DA^0 / <dR^2>.
Usage: umbrella_summary.py"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import subprocess, re, numpy as np
PY=_os.environ.get('SLOMD_PYTHON', '/home/liang/anaconda3/envs/slomd/bin/python')
PMF=_REPO + "/scripts/umbrella_pmf.py"

# JBC-2019 self-consistent series (10 C); DM is a cross-ref (different series/temperature)
KIE={'WT':(66,52,'JBC2019'),'L754A':(106,82,'JBC2019'),'I553A':(148,77,'JBC2019'),
     'DM':(537,None,'JACS2014_xref')}
PRIMARY=['WT','L754A','I553A']                 # self-consistent fit set

def descr(tag, half=None):
    cmd=[PY,PMF,tag]+(['--half',half] if half else [])
    out=subprocess.run(cmd,capture_output=True,text=True).stdout
    m=re.search(r"r_DA\^0=([\d.]+).*k_gate=([-\d.]+).*<dR\^2>\^1/2=([\d.]+).*dW\(3.1A\)=([-\d.]+)",out)
    return tuple(float(x) for x in m.groups()) if m else None

rows={}
for tag in ['WT','L754A','I553A','DM']:
    d=descr(tag)
    if d is None: print(f"  {tag}: PMF not ready (windows incomplete)"); continue
    f,s=descr(tag,'first'),descr(tag,'second')
    conv=abs(f[0]-s[0]) if (f and s) else float('nan')     # r_DA^0 block spread (convergence)
    rows[tag]=d+(conv,)
print(f"\n  {'system':<7}{'KIE_10C':>8}{'r_DA0(A)':>10}{'k_gate':>9}{'<dR2>^.5':>10}{'dW(3.1)':>9}{'conv(A)':>9}")
for tag in ['WT','L754A','I553A','DM']:
    if tag in rows:
        r0,kg,rms,dwn,cv=rows[tag]
        print(f"  {tag:<7}{KIE[tag][0]:>8}{r0:>10.2f}{kg:>9.1f}{rms:>10.3f}{dwn:>9.1f}{cv:>9.3f}")

# regression on the self-consistent primary set
fit=[t for t in PRIMARY if t in rows]
if len(fit)>=3:
    y=np.log(np.array([KIE[t][0] for t in fit]))           # ln KIE (10 C)
    print(f"\n  === regression vs ln(KIE_10C), self-consistent JBC set {fit} ===")
    for j,name in [(0,'r_DA^0'),(1,'k_gate'),(2,'<dR^2>^.5'),(3,'dW(3.1A)')]:
        x=np.array([rows[t][j] for t in fit])
        if np.std(x)==0: continue
        r=np.corrcoef(x,y)[0,1]; sl=np.polyfit(x,y,1)[0]
        print(f"    ln(KIE) vs {name:<10}: R={r:+.3f}  R^2={r*r:.3f}  slope={sl:+.3f}")
    # T-dependence check (KIE_10/KIE_40)
    td=np.array([KIE[t][0]/KIE[t][1] for t in fit])
    for j,name in [(0,'r_DA^0'),(1,'k_gate'),(2,'<dR^2>^.5')]:
        x=np.array([rows[t][j] for t in fit])
        if np.std(x)==0: continue
        r=np.corrcoef(x,td)[0,1]
        print(f"    KIE_10/KIE_40 vs {name:<10}: R={r:+.3f}")
    if 'DM' in rows: print(f"  (DM cross-ref: descriptors above; KIE 537 @30C, different series -> not in fit)")
else:
    print(f"\n  regression pending: need WT+L754A+I553A PMFs ({len(fit)}/3 ready)")
