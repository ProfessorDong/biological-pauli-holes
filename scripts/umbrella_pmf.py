#!/usr/bin/env python3
"""MBAR potential of mean force W(r_DA) from the umbrella windows of ONE system, and the
gating descriptors that vibronically-nonadiabatic SLO rate theory connects to the KIE:
  r_DA^0        equilibrium donor-acceptor distance (PMF minimum)
  k_gate        gating force constant = d2W/dr2 at r_DA^0   (kcal/mol/A^2)
  <dR^2>^1/2    DAD thermal fluctuation = sqrt(kT/k_gate)   (A)
  dW(near)      W at the near-attack (3.1 A) above the minimum (kcal/mol)
Convergence: each window is block-split (1st vs 2nd half) and the PMF recomputed for a spread.
Usage: umbrella_pmf.py <tag> [--half first|second]"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, glob, numpy as np
from pymbar import FES

tag=sys.argv[1]
half = sys.argv[sys.argv.index('--half')+1] if '--half' in sys.argv else None
d=_REPO + f"/results/umbrella/{tag}"
kT=0.5961                                                  # kcal/mol at 300 K
K_BIAS=12.0                                                # kcal/mol/A^2 (matches driver)

files=sorted(glob.glob(f"{d}/win_*_colvar.dat"), key=lambda f:float(f.split('win_')[1].split('_')[0]))
centers=[]; samples=[]
for f in files:
    r0=float(f.split('win_')[1].split('_')[0]); x=np.loadtxt(f)
    x=x[len(x)//5:]                                        # drop first 20% (equilibration within prod)
    if half=='first':  x=x[:len(x)//2]
    if half=='second': x=x[len(x)//2:]
    centers.append(r0); samples.append(x)
centers=np.array(centers); N_k=np.array([len(x) for x in samples]); x_n=np.concatenate(samples)
K=len(centers); N=len(x_n)
# reduced bias potential of every sample in every window
u_kn=np.zeros((K,N))
for k in range(K):
    u_kn[k,:]=(0.5*K_BIAS*(x_n-centers[k])**2)/kT
# MBAR-based FES (histogram) along r_DA
counts,edges=np.histogram(x_n, bins=45)                   # evaluate only populated bins (robust)
allc=0.5*(edges[:-1]+edges[1:]); ctr=allc[counts>10]
fes=FES(u_kn, N_k, verbose=False)
fes.generate_fes(u_kn, x_n, fes_type='histogram', histogram_parameters={'bin_edges':edges})
res=fes.get_fes(ctr, reference_point='from-lowest', uncertainty_method='analytical')
W=res['f_i']*kT; dW=res['df_i']*kT                        # kcal/mol
good=np.isfinite(W)
ctr,W,dW=ctr[good],W[good],dW[good]
W=W-W.min()

# descriptors
i0=int(np.argmin(W)); r0=ctr[i0]
sel=np.abs(ctr-r0)<0.5                                     # parabolic fit near the minimum
c=np.polyfit(ctr[sel],W[sel],2); kgate=2*c[0]             # kcal/mol/A^2
rms=np.sqrt(kT/kgate) if kgate>0 else float('nan')
i31=int(np.argmin(np.abs(ctr-3.1))); dWnear=W[i31]-W.min()
if half is None:
    np.savetxt(f"{d}/pmf.dat", np.c_[ctr,W,dW], header="r_DA_A  W_kcal/mol  dW")
print(f"  {tag}{'' if half is None else ' ['+half+']'}: r_DA^0={r0:.2f} A  k_gate={kgate:.1f} kcal/mol/A^2  "
      f"<dR^2>^1/2={rms:.3f} A  dW(3.1A)={dWnear:.1f} kcal/mol  [K={K} windows, N={N}]")
