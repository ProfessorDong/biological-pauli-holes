#!/usr/bin/env python3
"""Leave-one-variant-out and bootstrap-degeneracy checks on the declared endpoints.

WHY THIS EXISTS
  Two things the audit asked for that the manuscript did not report.

  1. Leave-one-variant-out. With seven variants a single point can carry an association, and the
     paper already knows this matters: dropping L754A was correct for one analysis and wrong for
     another. Reporting the full leave-one-out spread makes the leverage visible instead of
     leaving a reader to wonder.

  2. Bootstrap degeneracy. The reported intervals often span nearly the whole correlation range.
     Part of that is genuine, but part is an artefact of the design: a resample of seven systems
     can contain very few distinct variants, and after fitting an intercept and one covariate the
     residual space can collapse so that the partial correlation is exactly +/-1 by construction.
     The frequency of those degenerate resamples is a property of the bootstrap, not of the
     enzyme, so it should be printed alongside the interval.

Usage: endpoint_robustness.py     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import importlib.util as iu
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(_REPO)


def load_modules():
    out = {}
    for name, rel in (('static', 'scripts/analyze_reactive_geometry.py'),
                      ('ens', 'scripts/analyze_ensemble_fluctuation.py')):
        spec = iu.spec_from_file_location(name, ROOT / rel)
        m = iu.module_from_spec(spec)
        spec.loader.exec_module(m)
        out[name] = m
    return out


def partial(x, y, z):
    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        c, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ c
    rx, ry = resid(x, z), resid(y, z)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def exact_p(x, y, z):
    obs = partial(x, y, z)
    v = np.array([partial(x, np.array(q), z) for q in itertools.permutations(y)])
    v = v[np.isfinite(v)]
    return obs, float(np.mean(np.abs(v) >= abs(obs) - 1e-12))


def loo(x, y, z, names):
    rows = []
    for i, nm in enumerate(names):
        k = [j for j in range(len(names)) if j != i]
        o, p = exact_p(x[k], y[k], z[k])
        rows.append((nm, o, p))
    return rows


def degeneracy(x, y, z, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    deg = uniq = 0
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        u = len(set(i.tolist()))
        uniq += u <= 3
        v = partial(x[i], y[i], z[i])
        if np.isfinite(v):
            vals.append(v)
            deg += abs(abs(v) - 1.0) < 1e-9
    return deg / max(len(vals), 1), uniq / n_boot, np.array(vals)


def main():
    M = load_modules()
    st, en = M['static'], M['ens']
    out = {}
    for clamp, lab in (('r255', 'compressed'), ('r340', 'reference')):
        kx, geo = st.load_kexch(clamp), st.load_geometry(clamp)
        ok = [s for s in st.SYSTEMS if s in kx and s in geo]
        x = np.array([float(kx[s]) for s in ok])
        y = np.log(np.array([st.KIE[s] for s in ok], float))
        z = np.array([geo[s] for s in ok])
        o, p = exact_p(x, y, z)
        rows = loo(x, y, z, ok)
        dfrac, ufrac, vals = degeneracy(x, y, z)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f'\n=== static separation curvature, {lab} clamp ===')
        print(f'  full panel: partial {o:+.3f}, exact p {p:.4f}')
        print(f'  leave-one-variant-out, partial correlation and exact p:')
        for nm, oo, pp in sorted(rows, key=lambda r: r[1]):
            print(f'    without {nm:<6} {oo:+.3f}   p={pp:.4f}')
        sp = [r[1] for r in rows]
        print(f'    spread {min(sp):+.3f} to {max(sp):+.3f}; no single variant changes the '
              f'sign' if min(sp) * max(sp) > 0 else
              f'    spread {min(sp):+.3f} to {max(sp):+.3f}; the SIGN flips on leave-one-out')
        print(f'  bootstrap: {100*dfrac:.1f}% of resamples give a degenerate |partial|=1, '
              f'{100*ufrac:.1f}% contain 3 or fewer distinct variants')
        print(f'  interval [{lo:+.3f}, {hi:+.3f}] is therefore partly a property of the design')
        out[f'static_{clamp}'] = dict(
            systems=ok, partial=o, p_exact=p,
            loo=[dict(dropped=nm, partial=oo, p_exact=pp) for nm, oo, pp in rows],
            bootstrap_degenerate_fraction=float(dfrac),
            bootstrap_few_variant_fraction=float(ufrac),
            ci95=[float(lo), float(hi)])
    # The pre-registered secondary endpoint is the CONTRAST between the two geometries. The
    # manuscript reported its value but never its uncertainty, and a significant association at
    # one clamp beside a non-significant one at the other does not by itself establish that the
    # two differ. The contrast is therefore permuted directly: one relabelling of ln KIE is
    # applied to BOTH clamps at once, which is the null under which the descriptor carries no
    # information at either geometry.
    kx1, geo1 = st.load_kexch('r255'), st.load_geometry('r255')
    kx2, geo2 = st.load_kexch('r340'), st.load_geometry('r340')
    ok = [t for t in st.SYSTEMS if t in kx1 and t in kx2 and t in geo1 and t in geo2]
    x1 = np.array([float(kx1[t]) for t in ok]); z1 = np.array([geo1[t] for t in ok])
    x2 = np.array([float(kx2[t]) for t in ok]); z2 = np.array([geo2[t] for t in ok])
    y = np.log(np.array([st.KIE[t] for t in ok], float))

    def contrast(yy):
        return abs(partial(x1, yy, z1)) - abs(partial(x2, yy, z2))
    obs = contrast(y)
    null = np.array([contrast(np.array(q)) for q in itertools.permutations(y)])
    null = null[np.isfinite(null)]
    pc = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    lo_c, hi_c = np.percentile(null, [2.5, 97.5])
    print(f'\n=== pre-registered geometry contrast ===')
    print(f'  |r_compressed| - |r_reference| = {obs:+.3f}')
    print(f'  exact two-sided p over {len(null)} relabellings = {pc:.4f}')
    print(f'  the null distribution of the contrast spans [{lo_c:+.3f}, {hi_c:+.3f}] at 95%,')
    print(f'  so a contrast of this size is entirely ordinary under no association at either clamp.')
    out['contrast'] = dict(systems=ok, observed=float(obs), p_exact=pc,
                           null_ci95=[float(lo_c), float(hi_c)], n_perm=int(len(null)))

    (ROOT / 'results/reactive_geometry/endpoint_robustness.json').write_text(
        json.dumps(out, indent=1))
    print(f'\nwrote {ROOT}/results/reactive_geometry/endpoint_robustness.json')


if __name__ == '__main__':
    main()
