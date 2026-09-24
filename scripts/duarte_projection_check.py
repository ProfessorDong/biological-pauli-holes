"""Additive per-mode exchange shares, the construction used in the IQA
force-constant decomposition literature, compared against the generalized
eigenvalue extrema reported in the manuscript.

Duarte and Bruns (J. Phys. Chem. A 126, 8945 (2022)) decompose a force constant
by projecting each energy component's Hessian onto the normal modes of the TOTAL
Hessian, so that the component contributions along a fixed mode add to the total
force constant of that mode. This script performs the same projection on the
seven native-donor transverse Hessians and checks the identity that makes the
manuscript's measure conservative: because the generalized eigenvalues of
(K_exch, K_tot) are the extrema of the Rayleigh quotient v'K_exch v / v'K_tot v
over all directions v, every additive share along any mode of K_tot must lie
between them.

Mass weighting is a uniform scalar here: both matrices are 2x2 blocks for the
displacement of a single atom, so 1/m_H multiplies both and cancels from every
ratio, and it leaves the eigenvectors of K_tot unchanged.

Read-only. Writes results/native_donor_validation/duarte_projection_check.json.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigvalsh

D = Path(_REPO) / 'results/native_donor_validation'
TAGS = ['L754A', 'I552A', 'I538A', 'L546A', 'I553A', 'V750A', 'WT']
CONV = 0.694770  # kcal/mol/A^2 -> N/m


def hessian(grid, key):
    xy = np.array([[r['d1'], r['d2']] for r in grid])
    x, y = xy.T
    A = np.column_stack([np.ones(len(x)), x, y, x * x / 2, x * y, y * y / 2])
    z = np.array([r[key] for r in grid])
    c = np.linalg.lstsq(A, z - z.mean(), rcond=None)[0]
    return np.array([[c[3], c[4]], [c[4], c[5]]]) * CONV


out = {'scope': __doc__.strip().splitlines()[0], 'systems': {}}
for tag in TAGS:
    n = json.loads((D / f'{tag}_r255_w015_native_transverse.json').read_text())
    t = json.loads((D / f'{tag}_r255_w015_total_restoring.json').read_text())
    Ke = hessian(n['grid'], 'exch')
    assert np.allclose(np.linalg.eigvalsh(Ke), n['K_perp_exch_eigs'], atol=1e-10)
    rec = {}
    for m in ['HF', 'B3LYP']:
        Kt = hessian(t['grid'], m + '_complex')
        assert np.allclose(np.linalg.eigvalsh(Kt), t[m]['K_tot_complex_eigs'], atol=1e-6)
        lam, V = np.linalg.eigh(Kt)
        shares = np.array([V[:, i] @ Ke @ V[:, i] / lam[i] for i in range(2)])
        gen = eigvalsh(Ke, Kt)
        rec[m] = {'K_tot_eigs_Nm': lam.tolist(),
                  'additive_share_percent': (100 * shares).tolist(),
                  'generalized_extrema_percent': (100 * gen).tolist(),
                  'shares_inside_extrema': bool(gen.min() - 1e-12 <= shares.min()
                                                and shares.max() <= gen.max() + 1e-12)}
    out['systems'][tag] = rec

sh = {m: np.array([out['systems'][t][m]['additive_share_percent'] for t in TAGS]).ravel()
      for m in ['HF', 'B3LYP']}
ge = {m: np.array([out['systems'][t][m]['generalized_extrema_percent'] for t in TAGS]).ravel()
      for m in ['HF', 'B3LYP']}
out['summary'] = {m: {'additive_share_span_percent': [float(sh[m].min()), float(sh[m].max())],
                      'generalized_extrema_span_percent': [float(ge[m].min()), float(ge[m].max())],
                      'negative_shares': int((sh[m] < 0).sum()), 'n_shares': int(sh[m].size),
                      'systems_with_a_negative_share':
                          [t for t in TAGS
                           if min(out['systems'][t][m]['additive_share_percent']) < 0]}
                  for m in ['HF', 'B3LYP']}
out['all_shares_inside_extrema'] = all(
    out['systems'][t][m]['shares_inside_extrema'] for t in TAGS for m in ['HF', 'B3LYP'])

(D / 'duarte_projection_check.json').write_text(json.dumps(out, indent=2) + '\n')

print(f"{'system':8s}{'method':7s}{'additive shares %':>24s}{'generalized extrema %':>24s}  inside")
for tag in TAGS:
    for m in ['HF', 'B3LYP']:
        r = out['systems'][tag][m]
        s = '[{:+.4f}, {:+.4f}]'.format(*r['additive_share_percent'])
        g = '[{:+.4f}, {:+.4f}]'.format(*r['generalized_extrema_percent'])
        print(f"{tag:8s}{m:7s}{s:>24s}{g:>24s}  {'yes' if r['shares_inside_extrema'] else 'NO'}")
print()
for m in ['HF', 'B3LYP']:
    s = out['summary'][m]
    print(f"{m}: shares {s['additive_share_span_percent'][0]:+.4f}% to "
          f"{s['additive_share_span_percent'][1]:+.4f}%, "
          f"{s['negative_shares']}/{s['n_shares']} negative in "
          f"{', '.join(s['systems_with_a_negative_share'])}")
print('every additive share lies inside its generalized bracket:',
      out['all_shares_inside_extrema'])
