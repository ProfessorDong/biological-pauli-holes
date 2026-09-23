#!/usr/bin/env python3
"""Does the sign of the transverse exchange curvature survive thermal sampling?

The static panel (native_donor_eigen_uncertainty.py) gives one geometry per system with
uncertainties propagated from the quadratic fit. That answers "is the fit resolved", NOT
"is the sign a property of the thermal ensemble". This aggregates the stage-5 campaign:
independent configurations drawn from restrained MD at the compressed clamp, spaced by the
measured integrated autocorrelation time of r_HW, each with its own SAPT0 transverse Hessian.

Reported per system: the distribution of the SMALLER K_perp^exch eigenvalue, the fraction of
configurations in which it is negative, and the ensemble mean with a standard error over
configurations. Configurations are tau-spaced, so they are treated as independent; that is an
assumption inherited from the extraction, not established here.

POST HOC. The pre-registered endpoints are on K_sep and stay there.
"""
import json, numpy as np
from pathlib import Path

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
CFG  = ROOT / 'results/stage5_configs'
SYS  = ['L754A', 'I552A', 'I538A', 'L546A', 'I553A', 'V750A', 'WT']   # ascending methane K_sep
KIE  = {'WT': 66, 'V750A': 62, 'I552A': 66, 'I538A': 100,
        'L754A': 106, 'L546A': 131, 'I553A': 148}

print('STAGE-5 ENSEMBLE, transverse exchange curvature, native donor, h = 0.15 A')
print('independent MD configurations at the compressed clamp\n')
hdr = (f"{'system':7s} {'KIE':>4s} {'n':>3s} | {'<smaller eig>':>16s} {'SEM':>8s} "
       f"| {'frac < 0':>9s} | {'min':>8s} {'max':>8s} | {'<r_DA>':>7s}")
print(hdr); print('-' * len(hdr))

out = {}
for t in SYS:
    files = sorted((CFG / t).glob(f'{t}_cfg*_w015_transverse.json'))
    if not files:
        print(f'{t:7s} {KIE[t]:4d}   0 | no results'); continue
    d = [json.load(open(f)) for f in files]
    lo = np.array([min(x['K_perp_exch_eigs']) for x in d])
    hi = np.array([max(x['K_perp_exch_eigs']) for x in d])
    rda = np.array([x['r_DA'] for x in d])
    sem = lo.std(ddof=1) / np.sqrt(len(lo)) if len(lo) > 1 else np.nan
    frac = (lo < 0).mean()
    print(f'{t:7s} {KIE[t]:4d} {len(lo):3d} | {lo.mean():+16.5f} {sem:8.5f} '
          f'| {frac*100:8.0f}% | {lo.min():+8.4f} {lo.max():+8.4f} | {rda.mean():7.3f}')
    out[t] = dict(n=len(lo), mean_smaller=float(lo.mean()), sem=float(sem),
                  frac_negative=float(frac), min=float(lo.min()), max=float(lo.max()),
                  mean_larger=float(hi.mean()), mean_r_DA=float(rda.mean()),
                  all_smaller=[float(v) for v in lo])

print()
neg_mean = [t for t in out if out[t]['mean_smaller'] < 0]
print(f'ensemble-mean smaller eigenvalue negative: {len(neg_mean)} of {len(out)}  '
      f'({", ".join(neg_mean)})')
res = [t for t in out if out[t]['n'] > 1 and
       abs(out[t]['mean_smaller']) > 2 * out[t]['sem']]
negres = [t for t in res if out[t]['mean_smaller'] < 0]
print(f'resolved at 2 SEM over configurations: {len(res)} of {len(out)}; '
      f'of those negative: {len(negres)}')
print(f'unresolved at 2 SEM: {sorted(set(out) - set(res))}')

# total interaction, same grid
print('\nTOTAL INTERMOLECULAR INTERACTION K_perp^int, same configurations')
for t in SYS:
    files = sorted((CFG / t).glob(f'{t}_cfg*_w015_transverse.json'))
    if not files: continue
    d = [json.load(open(f)) for f in files]
    lo = np.array([min(x['K_perp_int_eigs']) for x in d])
    hi = np.array([max(x['K_perp_int_eigs']) for x in d])
    nd = ((lo < 0) & (hi < 0)).mean()
    print(f'  {t:7s} n={len(lo):2d}  <smaller>={lo.mean():+8.4f}  <larger>={hi.mean():+8.4f}  '
          f'negative definite in {nd*100:3.0f}% of configurations')

(ROOT / 'results/native_donor_validation').mkdir(parents=True, exist_ok=True)
p = ROOT / 'results/native_donor_validation/stage5_ensemble_panel.json'
json.dump(out, open(p, 'w'), indent=1)
print(f'\nwrote {p}')
