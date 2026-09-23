#!/usr/bin/env python3
"""Fit a stage-4 sensitivity run from its per-point files and report as pre-registered.

Reports BOTH eigenvalues and the fitted gradient at the setting, and labels the residual a fit
residual. Uses the same quad2d as the published panel rather than a reimplementation.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
P = ROOT / 'results/native_donor_validation/stage4'
sys.path.insert(0, str(ROOT / 'scripts'))
from transverse_native_donor import quad2d

for label in sys.argv[1:]:
    pts = sorted((P / 'points').glob(f'{label}_??_??.json'))
    if not pts:
        print(f'{label}: no points'); continue
    d = [json.loads(p.read_text()) for p in pts]
    npa = max(max(x['i'] for x in d), max(x['j'] for x in d)) + 1
    if len(d) != npa * npa:
        print(f'{label}: INCOMPLETE {len(d)}/{npa*npa} points'); continue
    rows = [(x['d1'], x['d2'], x['exch'], x['total']) for x in d]
    Hx, rx, gx = quad2d([(r[0], r[1], r[2]) for r in rows], 2)
    Ht, rt, gt = quad2d([(r[0], r[1], r[3]) for r in rows], 2)
    wx, wt = np.linalg.eigvalsh(Hx), np.linalg.eigvalsh(Ht)
    h = max(abs(x['d1']) for x in d)
    res = dict(label=label, method=d[0]['method'], basis=d[0]['basis'], donor=d[0]['donor'],
               wall=d[0]['wall'], n_donor=d[0]['n_donor'], n_wall=d[0]['n_wall'],
               half=h, n_per_axis=npa, n_points=len(d),
               K_perp_exch_eigs=[float(v) for v in wx],
               K_perp_exch_grad_kcal_per_A=gx, K_perp_exch_grad_norm=float(np.hypot(*gx)),
               K_perp_exch_rms_kcal=rx,
               K_perp_int_eigs=[float(v) for v in wt],
               K_perp_int_grad_kcal_per_A=gt, K_perp_int_grad_norm=float(np.hypot(*gt)),
               K_perp_int_rms_kcal=rt)
    (P / f'{label}.json').write_text(json.dumps(res, indent=2))
    print(f'{label:34s} {d[0]["method"]:7s}/{d[0]["basis"]:14s} {npa}x{npa} h={h:.2f}  '
          f'exch {wx[0]:+.5f} {wx[1]:+.5f}  grad {np.hypot(*gx):.4f}  rms {rx:.1e}')
