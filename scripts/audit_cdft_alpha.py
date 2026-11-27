#!/usr/bin/env python3
"""Constrained-DFT diabats on the exact scan coordinate.

Audit producer, 2026-11-25. Reads results/diabat_analysis.json and the four scan files
results/cdft_{reactant,product}_{diabat,extend}/*.json, and writes
results/audit_2026_11/cdft_alpha.json only.

scripts/analyze_diabats.py takes the fractional proton position alpha of the original 15-point
grid from the output file names (pt059 -> 0.59), which round the grid values 0.2 + k(0.6/14) to two
decimals; the continuation points carry four. This script keeps the validity filter of that
analysis (the valid points and energies listed in diabat_analysis.json), assigns each valid point
its exact alpha by matching its energy in the scan files, and repeats the same steps: a cubic spline
for each diabat, the crossing as the first sign change of E_reactant - E_product on 4001 points of
the overlap, and each minimum as the lowest valid point. It also reports the continuation steps
that were attempted, against the original grid spacing.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

R = Path(_REPO) / 'results'
OUT = R / 'audit_2026_11' / 'cdft_alpha.json'
Ha = 627.5094740631


def exact(points, valid):
    """Exact alpha of each valid point, matched on its energy (unique match required)."""
    alphas = []
    for e in valid:
        hits = [p['alpha'] for p in points if p.get('E_diabat') is not None and abs(p['E_diabat'] - e) < 1e-9]
        assert len(set(round(h, 10) for h in hits)) == 1, f'energy {e} matched {hits}'
        alphas.append(float(hits[0]))
    return np.array(alphas)


def main():
    A = json.loads((R / 'diabat_analysis.json').read_text())
    scans = {k: json.loads((R / f'cdft_{k}' / ('extend_scan.json' if k.endswith('extend') else 'diabat_scan.json')).read_text())
             for k in ('reactant_diabat', 'reactant_extend', 'product_diabat', 'product_extend')}
    rDA = scans['reactant_diabat']['r_DA']
    out = dict(r_DA_A=rDA, grid_spacing=0.6 / 14)
    sp = {}
    for side in ('reactant', 'product'):
        pts = scans[f'{side}_diabat']['points'] + scans[f'{side}_extend']['points']
        E = np.array(A[side]['E'])
        a = exact(pts, E)
        o = np.argsort(a); a, E = a[o], E[o]
        sp[side] = (a, E, CubicSpline(a, E))
        imin = int(np.argmin(E))
        out[side] = dict(alpha_exact=a.tolist(), alpha_as_analyzed=A[side]['alphas'], n_valid=int(len(a)),
                         domain=[float(a.min()), float(a.max())], alpha_min=float(a[imin]),
                         r_CH_at_min_A=float(a[imin] * rDA), r_HO_at_min_A=float((1 - a[imin]) * rDA),
                         E_min_Ha=float(E[imin]))
    (ar, Er, csr), (ap, Ep, csp) = sp['reactant'], sp['product']
    lo, hi = max(ar.min(), ap.min()), min(ar.max(), ap.max())
    g = np.linspace(lo, hi, 4001); diff = csr(g) - csp(g)
    idx = np.where(np.diff(np.sign(diff)) != 0)[0]
    ax = float(np.interp(0.0, [diff[idx[0]], diff[idx[0] + 1]], [g[idx[0]], g[idx[0] + 1]])) if len(idx) else None
    out['overlap'] = [float(lo), float(hi)]
    out['crossing'] = None if ax is None else dict(
        alpha=ax, r_CH_A=ax * rDA, r_HO_A=(1 - ax) * rDA,
        bracketing_valid_reactant=[float(ar[ar < ax].max()), float(ar[ar > ax].min())],
        bracketing_valid_product=[float(ap[ap < ax].max()), float(ap[ap > ax].min())],
        alpha_as_analyzed=A.get('crossing_alpha'))
    out['dG_between_minima_kcal'] = float((out['product']['E_min_Ha'] - out['reactant']['E_min_Ha']) * Ha)
    steps = {k: sorted({p['step'] for p in scans[k]['points']}) for k in ('reactant_extend', 'product_extend')}
    smin = min(min(v) for v in steps.values())
    out['continuation'] = dict(steps_attempted=steps, start_alpha={k: scans[k]['start'] for k in steps},
                               target_alpha={k: scans[k]['target'] for k in steps},
                               smallest_step=smin, smallest_step_over_grid_spacing=smin / (0.6 / 14))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('overlap', 'crossing', 'dG_between_minima_kcal', 'continuation')}, indent=1))
    for side in ('reactant', 'product'):
        s = out[side]
        print(side, 'domain', s['domain'], 'min alpha', round(s['alpha_min'], 4), 'r_CH', round(s['r_CH_at_min_A'], 3),
              'r_HO', round(s['r_HO_at_min_A'], 3))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
