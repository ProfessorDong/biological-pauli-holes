"""Pre-registered analysis: does the Pauli wall explain the KIE ladder at the
reactive geometry but not at the equilibrium geometry?

WRITTEN BEFORE ANY k_exch VALUE FROM THE CLAMPED SIMULATIONS WAS INSPECTED.
The plan it implements is fixed in results/reactive_geometry/PREREGISTRATION.md,
which was committed before the reference clamp series existed.

PRIMARY endpoint : partial correlation of k_exch with ln KIE, CONTROLLING for the
                   achieved <r_DA> of each clamped run. The raw bivariate
                   correlation is not the endpoint, because the clamp leaves a
                   0.065 A residual spread in r_DA that is itself associated with
                   ln KIE (Pearson +0.519, n=7), and our own Morse Franck-Condon
                   calculation shows 0.048 A of r_DA spans the whole KIE ladder.
                   Residual geometry could therefore manufacture a correlation.
SECONDARY        : the same quantity at the r_DA = 3.40 A reference clamp. The
                   framework's claim is the CONTRAST between the two geometries,
                   not the reactive clamp alone.
ABORT            : if variance inflation between k_exch and achieved r_DA exceeds
                   5, the contributions cannot be separated at n=7 and the
                   reported conclusion is that the design cannot resolve it.

Inference: exhaustive permutation over all 7! label assignments for an exact p,
bootstrap confidence intervals, and no treatment of nominal p<0.05 as decisive.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, math, itertools, sys
from pathlib import Path
import numpy as np

ROOT = Path(_REPO)
RG = ROOT / 'results' / 'reactive_geometry'

KIE = {'WT': 66, 'I839A': 62, 'L754A': 106, 'V750A': 62,
       'I538A': 100, 'L546A': 131, 'I553A': 148, 'I552A': 66}
SYSTEMS = ['WT', 'V750A', 'I552A', 'I538A', 'L754A', 'L546A', 'I553A']
VIF_ABORT = 5.0


def partial_corr(x, y, z):
    """Correlation of x and y after linearly removing z from both."""
    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    rx, ry = resid(x, z), resid(y, z)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def vif(x, z):
    """Variance inflation factor for x regressed on z (2-variable case)."""
    r = np.corrcoef(x, z)[0, 1]
    return float(1.0 / max(1e-12, (1.0 - r * r)))


def exact_permutation_p(stat_fn, x, y, z):
    """Exhaustive permutation over all orderings of the target labels."""
    obs = stat_fn(x, y, z)
    n = len(y)
    cnt = tot = 0
    for perm in itertools.permutations(range(n)):
        s = stat_fn(x, y[list(perm)], z)
        if not math.isnan(s):
            tot += 1
            if abs(s) >= abs(obs) - 1e-12:
                cnt += 1
    return obs, cnt / tot, tot


def bootstrap_ci(stat_fn, x, y, z, n_boot=20000, seed_vals=None):
    """Percentile CI by resampling systems with replacement."""
    n = len(x)
    rs = np.random.default_rng(12345)
    out = []
    for _ in range(n_boot):
        idx = rs.integers(0, n, n)
        if len(set(idx.tolist())) < 3:
            continue
        s = stat_fn(x[idx], y[idx], z[idx])
        if not math.isnan(s):
            out.append(s)
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def load_kexch(clamp):
    """k_exch per system for a clamp ('r255' or 'r340'), from the SAPT stage."""
    f = RG / f'kexch_{clamp}.json'
    if not f.exists():
        return None
    return json.loads(f.read_text())


def load_geometry(clamp):
    out = {}
    for s in SYSTEMS:
        f = RG / f'{s}_{clamp}_colvar.dat'
        if f.exists():
            out[s] = float(np.loadtxt(f).mean())
    return out


def analyze(clamp, label):
    kx = load_kexch(clamp)
    geo = load_geometry(clamp)
    if kx is None:
        print(f'\n=== {label} ===\n  k_exch not yet computed ({RG}/kexch_{clamp}.json absent)')
        return None
    sys_ok = [s for s in SYSTEMS if s in kx and s in geo]
    if len(sys_ok) < 5:
        print(f'\n=== {label} ===\n  only {len(sys_ok)} systems available; not analysed')
        return None
    x = np.array([float(kx[s]) for s in sys_ok])
    y = np.log(np.array([KIE[s] for s in sys_ok], float))
    z = np.array([geo[s] for s in sys_ok])

    print(f'\n=== {label} (n={len(sys_ok)}) ===')
    print(f'  {"system":<8}{"KIE":>5}{"k_exch":>12}{"<r_DA>":>9}')
    for s in sys_ok:
        print(f'  {s:<8}{KIE[s]:>5}{float(kx[s]):>12.4g}{geo[s]:>9.3f}')

    v = vif(x, z)
    raw = float(np.corrcoef(x, y)[0, 1])
    print(f'\n  raw Pearson(k_exch, ln KIE)      = {raw:+.3f}   [NOT the endpoint]')
    print(f'  VIF(k_exch vs achieved r_DA)     = {v:.2f}')
    if v > VIF_ABORT:
        print(f'  ABORT per pre-registration: VIF > {VIF_ABORT}. k_exch and residual')
        print(f'  geometry cannot be separated at this n. Design cannot resolve it.')
        return dict(clamp=clamp, n=len(sys_ok), raw=raw, vif=v, aborted=True)

    pc, p_exact, n_perm = exact_permutation_p(partial_corr, x, y, z)
    lo, hi = bootstrap_ci(partial_corr, x, y, z)
    print(f'  PRIMARY partial corr | r_DA       = {pc:+.3f}')
    print(f'    exact permutation p             = {p_exact:.4f}  ({n_perm} permutations)')
    print(f'    bootstrap 95% CI                = [{lo:+.3f}, {hi:+.3f}]')
    if lo < 0 < hi:
        print(f'    CI spans zero: not resolved at n={len(sys_ok)}')
    return dict(clamp=clamp, n=len(sys_ok), raw=raw, vif=v,
                partial=pc, p_exact=p_exact, ci=[lo, hi], aborted=False)


def main():
    print('Pre-registered analysis (plan fixed in PREREGISTRATION.md before data)')
    a = analyze('r255', 'REACTIVE clamp, target r_DA = 2.55 A')
    b = analyze('r340', 'REFERENCE clamp, target r_DA = 3.40 A')
    if a and b and not a['aborted'] and not b['aborted']:
        print('\n=== SECONDARY endpoint: reactive vs reference ===')
        print(f'  partial corr reactive  = {a["partial"]:+.3f}  (p={a["p_exact"]:.4f})')
        print(f'  partial corr reference = {b["partial"]:+.3f}  (p={b["p_exact"]:.4f})')
        d = abs(a['partial']) - abs(b['partial'])
        print(f'  |reactive| - |reference| = {d:+.3f}')
        print('  framework predicts a POSITIVE difference (descriptor works at the')
        print('  geometry where the reaction occurs, not at the equilibrium mean)')
    out = RG / 'analysis_result.json'
    out.write_text(json.dumps(dict(reactive=a, reference=b), indent=2, default=str))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
