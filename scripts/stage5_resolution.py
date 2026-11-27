#!/usr/bin/env python3
"""Which thermal-ensemble (stage-5) signs are resolved at two standard errors.

Each stage-5 configuration stores its two transverse exchange eigenvalues and the RMS residual of the
quadratic fit on the 5x5 grid of half-width 0.15 A (results/stage5_configs/<tag>/*_w015_transverse.json),
but not the grid or the fitted matrix. The static panel propagates standard errors as in
native_donor_eigen_uncertainty.py: s^2 = RSS/19, Cov = s^2 (A^T A)^-1, and
se(lambda) = sqrt(g^T Cov_quad g) with g = (v0^2, 2 v0 v1, v1^2) for eigenvector v. The eigenvector is
not stored, so this script brackets se over all directions in the transverse plane and counts the
configurations whose smaller eigenvalue is resolved under the largest (conservative) and the smallest
standard error. As a check, the bracket built from each static grid's RMS residual must contain the
se that the full-grid algebra gives for that static configuration. Writes
results/audit_2026_11/stage5_resolution.json.
Usage: stage5_resolution.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'results/stage5_configs'
TAGS = ('I538A', 'I552A', 'I553A', 'L546A', 'L754A', 'V750A', 'WT')
CONV = 0.694770                      # kcal mol^-1 A^-2 -> N/m
HALF, NPA, NPAR = 0.15, 5, 6


def variance_factor_range():
    g1 = np.linspace(-HALF, HALF, NPA)
    A = np.array([[1, x, y, 0.5 * x * x, x * y, 0.5 * y * y] for x in g1 for y in g1], float)
    sub = np.linalg.inv(A.T @ A)[3:6, 3:6]
    f = []
    for th in np.linspace(0, np.pi, 3601):
        v = np.array([np.cos(th), np.sin(th)])
        g = np.array([v[0] ** 2, 2 * v[0] * v[1], v[1] ** 2])
        f.append(float(g @ sub @ g))
    return min(f), max(f), len(A)


def static_check(fmin, fmax, n):
    """The bracket must contain the published static-panel se, which uses the stored grid."""
    res = {}
    for t in TAGS:
        g = json.loads((ROOT / f'results/native_donor_validation/{t}_r255_w015_native_transverse.json')
                       .read_text())['grid']
        A = np.array([[1, r['d1'], r['d2'], .5 * r['d1'] ** 2, r['d1'] * r['d2'], .5 * r['d2'] ** 2] for r in g])
        z = np.array([r['exch'] for r in g], float)
        c, *_ = np.linalg.lstsq(A, z, rcond=None)
        resid = A @ c - z
        cov = float(resid @ resid) / (len(z) - NPAR) * np.linalg.inv(A.T @ A)[3:6, 3:6]
        w, V = np.linalg.eigh(np.array([[c[3], c[4]], [c[4], c[5]]]) * CONV)
        v = V[:, 0]
        gv = np.array([v[0] ** 2, 2 * v[0] * v[1], v[1] ** 2]) * CONV
        se = float(np.sqrt(gv @ cov @ gv))
        s = float(np.sqrt(np.mean(resid ** 2))) * np.sqrt(n / (n - NPAR))
        lo, hi = s * np.sqrt(fmin) * CONV, s * np.sqrt(fmax) * CONV
        assert lo - 1e-12 <= se <= hi + 1e-12, (t, se, lo, hi)
        res[t] = dict(lambda1=float(w[0]), se_published_algebra=se, se_bracket=[float(lo), float(hi)])
    return res


def main():
    fmin, fmax, n = variance_factor_range()
    out = dict(rule='smaller eigenvalue resolved if |lambda_1| >= 2 se', half_width_A=HALF,
               variance_factor=[fmin, fmax], static_panel_check=static_check(fmin, fmax, n), systems={})
    for t in TAGS:
        rows = []
        for f in sorted((CFG / t).glob(f'{t}_cfg*_w015_transverse.json')):
            d = json.loads(f.read_text())
            l1 = float(sorted(d['K_perp_exch_eigs'])[0])
            s = float(d['rms_exch']) * np.sqrt(n / (n - NPAR))
            se_lo, se_hi = s * np.sqrt(fmin) * CONV, s * np.sqrt(fmax) * CONV
            rows.append(dict(config=d['config'], lambda1=l1, se_bracket=[float(se_lo), float(se_hi)],
                             resolved_conservative=bool(abs(l1) >= 2 * se_hi), resolved_lenient=bool(abs(l1) >= 2 * se_lo)))
        cons = [r for r in rows if r['resolved_conservative']]
        out['systems'][t] = dict(n=len(rows),
                                 n_negative=int(sum(r['lambda1'] < 0 for r in rows)),
                                 n_resolved_conservative=len(cons),
                                 n_resolved_negative_conservative=sum(r['lambda1'] < 0 for r in cons),
                                 n_resolved_positive_conservative=sum(r['lambda1'] > 0 for r in cons),
                                 n_resolved_lenient=int(sum(r['resolved_lenient'] for r in rows)),
                                 min_abs_lambda1_over_se_conservative=min(abs(r['lambda1']) / r['se_bracket'][1]
                                                                          for r in rows),
                                 configs=rows)
    dest = ROOT / 'results/audit_2026_11/stage5_resolution.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    for t, v in out['systems'].items():
        print(f"{t:6s} n {v['n']:2d}  lambda1<0 {v['n_negative']:2d}  resolved at 2 se: "
              f"{v['n_resolved_conservative']} (largest se; {v['n_resolved_negative_conservative']} negative, "
              f"{v['n_resolved_positive_conservative']} positive) to {v['n_resolved_lenient']} (smallest se); "
              f"min |lambda1|/se {v['min_abs_lambda1_over_se_conservative']:.1f}")
    print(f'variance factor {fmin:.1f} to {fmax:.1f}; wrote {dest}')


if __name__ == '__main__':
    main()
