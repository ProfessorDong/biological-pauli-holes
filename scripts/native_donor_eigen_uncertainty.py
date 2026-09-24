#!/usr/bin/env python3
"""Standard errors on the fitted transverse eigenvalues, propagated from the grid residuals.

The audit asked for fit stability on the near-zero eigenvalues specifically. A quadratic form
fitted by least squares to 25 grid energies has a covariance that follows from the residual, and
an eigenvalue of a 2x2 symmetric matrix perturbs as dlambda = v^T dH v for its own eigenvector.
That gives a per-eigenvalue standard error directly, and hence an honest sign classification.

Usage: native_donor_eigen_uncertainty.py   (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path

import numpy as np

D = Path(_REPO + '/results/native_donor_validation')
SYS = ['L754A', 'I552A', 'I538A', 'L546A', 'I553A', 'V750A', 'WT']
CONV = 0.694770


def analyse(path, key='exch'):
    d = json.loads(path.read_text())
    g = d['grid']
    A = np.array([[1, r['d1'], r['d2'], 0.5 * r['d1'] ** 2,
                   r['d1'] * r['d2'], 0.5 * r['d2'] ** 2] for r in g], float)
    z = np.array([r[key if key != 'exch' else 'exch'] for r in g], float)
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    resid = A @ c - z
    dof = len(z) - A.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)                       # covariance of the coefficients
    H = np.array([[c[3], c[4]], [c[4], c[5]]]) * CONV
    w, V = np.linalg.eigh(H)
    idx = {3: (0, 0), 4: (0, 1), 5: (1, 1)}                 # coefficient -> Hessian entry
    se = []
    for k in range(2):
        v = V[:, k]
        # dlambda = v^T dH v, with dH built from the three independent coefficients
        gvec = np.array([v[0] * v[0], 2 * v[0] * v[1], v[1] * v[1]]) * CONV
        sub = cov[np.ix_([3, 4, 5], [3, 4, 5])]
        se.append(float(np.sqrt(gvec @ sub @ gvec)))
    return w, np.array(se), float(np.sqrt(s2))


def main():
    print('TRANSVERSE EIGENVALUES WITH PROPAGATED STANDARD ERRORS, native donor, h = 0.15 A\n')
    print(f'{"system":7s} | {"smaller eigenvalue":>26s} | {"larger eigenvalue":>26s} | sign of smaller')
    print('-' * 92)
    counts = {'negative': [], 'positive': [], 'unresolved': []}
    for t in SYS:
        w, se, s = analyse(D / f'{t}_r255_w015_native_transverse.json')
        cls = ('negative' if w[0] + 2 * se[0] < 0 else
               'positive' if w[0] - 2 * se[0] > 0 else 'unresolved')
        counts[cls].append(t)
        print(f'{t:7s} | {w[0]:+11.5f} +/- {se[0]:8.5f} | {w[1]:+11.5f} +/- {se[1]:8.5f} | '
              f'{cls:>10s}  ({abs(w[0]) / se[0]:4.1f} sigma)')
    print('\nclassification at 2 sigma:')
    for k, v in counts.items():
        print(f'  {k:11s} {len(v)}  {", ".join(v) if v else "-"}')

    print('\nGRID-WIDTH STABILITY, V750A and I552A where available')
    for t in ['V750A', 'I552A']:
        for h in ['010', '015', '020']:
            p = D / f'{t}_r255_w{h}_native_transverse.json'
            if not p.exists():
                continue
            w, se, s = analyse(p)
            cls = ('negative' if w[0] + 2 * se[0] < 0 else
                   'positive' if w[0] - 2 * se[0] > 0 else 'unresolved')
            print(f'  {t:6s} h=0.{h[1:]}  smaller {w[0]:+9.5f} +/- {se[0]:7.5f}  -> {cls}')
    json.dump({t: dict(zip(('eigs', 'se'), (list(map(float, analyse(
        D / f"{t}_r255_w015_native_transverse.json")[0])),
        list(map(float, analyse(D / f"{t}_r255_w015_native_transverse.json")[1])))))
        for t in SYS}, open(D / 'eigen_uncertainty.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
