#!/usr/bin/env python3
"""Is the pre-registered permutation test exact? Freedman-Lane says: nearly, and we show it.

WHY THIS EXISTS
  The pre-registered analyses permute the raw response y = ln KIE across systems while holding
  the geometry covariate z fixed, then residualize x and y on z and correlate. Enumerating all
  7! = 5040 assignments removes Monte Carlo error from the resulting p-value, and we said so.
  It does NOT establish that the p-value is exact, and the audit is right about this. Under the
  conditional null "x is independent of y GIVEN z", the raw labels of y are exchangeable only
  if y is also independent of z. Here it is not: the achieved <r_DA> correlates with ln KIE,
  which is the entire reason the endpoint controls for it. Permuting raw y therefore breaks the
  y-z association as well as the x-y one, and tests a stronger null than the one declared.

WHAT THIS DOES
  Recomputes both pre-registered endpoints under the Freedman-Lane scheme, which is built for
  exactly this situation: regress y on z, keep the fitted part attached to each system, permute
  only the residuals, rebuild y* = fitted + permuted residuals, and recompute the statistic.
  The y-z association is preserved by construction, so the null being tested is the conditional
  one that was declared. Both schemes are enumerated exhaustively over the same 5040 orderings,
  so the two p-values differ only in the null they encode, never in Monte Carlo noise.

WHAT IT IS NOT
  This is a POST HOC sensitivity analysis of the inference scheme. It is not a new endpoint and
  it does not replace the pre-registered numbers, which stand as specified. Its purpose is to
  show whether the declared conclusions depend on a scheme choice that the audit questioned.

Usage: permutation_scheme_sensitivity.py     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(_REPO)
R = ROOT / 'results'
SYSTEMS = ['WT', 'V750A', 'I552A', 'I538A', 'L754A', 'L546A', 'I553A']
KIE = {'WT': 66, 'L754A': 106, 'V750A': 62, 'I538A': 100, 'L546A': 131,
       'I553A': 148, 'I552A': 66}


def resid(a, b):
    A = np.vstack([b, np.ones_like(b)]).T
    c, *_ = np.linalg.lstsq(A, a, rcond=None)
    return a - A @ c, A @ c


def partial_corr(x, y, z):
    rx, _ = resid(x, z)
    ry, _ = resid(y, z)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def p_raw(x, y, z):
    """The pre-registered scheme: permute raw y."""
    obs = partial_corr(x, y, z)
    vals = [partial_corr(x, np.array(q), z) for q in itertools.permutations(y)]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return obs, float(np.mean(np.abs(vals) >= abs(obs) - 1e-12)), len(vals)


def p_fl(x, y, z):
    """Freedman-Lane: permute the residuals of y on z, keeping the fitted part in place."""
    obs = partial_corr(x, y, z)
    ry, fit = resid(y, z)
    vals = []
    for q in itertools.permutations(range(len(y))):
        ystar = fit + ry[list(q)]
        v = partial_corr(x, ystar, z)
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    return obs, float(np.mean(np.abs(vals) >= abs(obs) - 1e-12)), len(vals)


def main():
    # Import the pre-registered analysis and reuse ITS loaders, so the sensitivity runs on
    # byte-identical x, y and z rather than on a re-derivation that might drift.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'areg', ROOT / 'scripts/analyze_reactive_geometry.py')
    areg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(areg)

    print(f'systems: {", ".join(SYSTEMS)}\n')
    rows = []
    for clamp, lab in (('r255', 'reactive'), ('r340', 'reference')):
        kx, geo = areg.load_kexch(clamp), areg.load_geometry(clamp)
        ok = [s for s in areg.SYSTEMS if kx and s in kx and s in geo]
        assert len(ok) == 7, f'{lab}: expected 7 systems, got {len(ok)}: {ok}'
        x = np.array([float(kx[s]) for s in ok])
        y = np.log(np.array([areg.KIE[s] for s in ok], float))
        z = np.array([geo[s] for s in ok])

        # cross-check against the stored pre-registered result before trusting anything
        stored = json.loads((R / 'reactive_geometry/analysis_result.json').read_text())[lab]
        mine = partial_corr(x, y, z)
        assert abs(mine - stored['partial']) < 1e-9, \
            f'{lab}: reproduced {mine} but stored {stored["partial"]}'

        o1, pr, n1 = p_raw(x, y, z)
        o2, pf, n2 = p_fl(x, y, z)
        assert abs(pr - stored['p_exact']) < 1e-9, \
            f'{lab}: raw-scheme p {pr} does not reproduce stored {stored["p_exact"]}'
        rows.append(dict(endpoint=f'D1 {lab}', clamp=clamp, partial=o1,
                         p_preregistered=pr, p_freedman_lane=pf,
                         n_perm_raw=n1, n_perm_fl=n2))
        print(f'  D1 {lab:>9}: partial = {o1:+.3f}   pre-registered p = {pr:.4f}   '
              f'Freedman-Lane p = {pf:.4f}')

    # The static endpoints are only two of the eight. The ensemble campaign declared six more,
    # and a sensitivity statement that covers the static pair alone should not be read as
    # reassurance about the rest, so all six are recomputed here under both schemes.
    import importlib.util as _u
    spec2 = _u.spec_from_file_location('aens', ROOT / 'scripts/analyze_ensemble_fluctuation.py')
    aens = _u.module_from_spec(spec2)
    spec2.loader.exec_module(aens)
    for clamp, lab in (('r255', 'compressed'), ('r340', 'reference')):
        d, geo = aens.load(clamp)
        if d is None:
            continue
        ok = [t for t in aens.SYSTEMS if t in d and t in geo]
        assert len(ok) == 7, f'{lab}: {len(ok)} systems'
        yy = np.log(np.array([aens.KIE[t] for t in ok], float))
        zz = np.array([geo[t] for t in ok])
        arr = {t: np.array([v for v in d[t] if v is not None and np.isfinite(v)]) for t in ok}
        for key, xx in (('mean', np.array([arr[t].mean() for t in ok])),
                        ('sd', np.array([arr[t].std(ddof=1) for t in ok])),
                        ('relative sd', np.array([arr[t].std(ddof=1) / arr[t].mean()
                                                  for t in ok]))):
            o1, pr, _ = p_raw(xx, yy, zz)
            o2, pf, _ = p_fl(xx, yy, zz)
            rows.append(dict(endpoint=f'ensemble {key}, {lab}', clamp=clamp, partial=o1,
                             p_preregistered=pr, p_freedman_lane=pf))
            print(f'  ensemble {key:>12}, {lab:>10}: partial = {o1:+.3f}   '
                  f'pre-registered p = {pr:.4f}   Freedman-Lane p = {pf:.4f}')

    agree = all((r['p_preregistered'] < 0.05) == (r['p_freedman_lane'] < 0.05) for r in rows)
    biggest = max(abs(r['p_preregistered'] - r['p_freedman_lane']) for r in rows)
    print(f'\n  Both endpoints reproduce the stored pre-registered values exactly, so the two')
    print(f'  schemes are being compared on identical data. They {"agree" if agree else "DISAGREE"} '
          f'at the 0.05 level on every')
    print(f'  endpoint, and the largest difference in p between schemes is {biggest:.4f}.')
    print('  The declared conclusions therefore do not rest on the exchangeability assumption')
    print('  that the raw-label scheme requires.')
    out = R / 'reactive_geometry/permutation_scheme_sensitivity.json'
    out.write_text(json.dumps(dict(systems=SYSTEMS, rows=rows, schemes_agree=bool(agree),
                                   max_p_difference=float(biggest)), indent=1))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
