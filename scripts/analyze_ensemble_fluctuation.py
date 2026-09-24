"""Pre-registered analysis: does the FLUCTUATION of the Pauli wall carry the KIE ladder?

WRITTEN BEFORE ANY MULTI-FRAME k_exch VALUE WAS INSPECTED. The plan it implements is
fixed in results/ensemble_fluctuation/PREREGISTRATION.md, committed before the frame
campaign produced output.

Every descriptor tested previously was a STATIC property of one configuration, and all
failed. The literature on distal control of enzymatic hydrogen transfer attributes the
effect to protein FLEXIBILITY (Nagel 2013; Offenbacher 2017), and the exact rate
decomposition identifies isotope-specific reweighting of a DISTRIBUTION as the only
possible origin of a KIE. This tests the distribution rather than a representative point.

DESCRIPTORS, declared in advance, no others to be added:
  D1  <k_exch>            mean over frames        (secondary; sanity link to static test)
  D2  sd(k_exch)          fluctuation             (PRIMARY)
  D3  sd/<k_exch>         relative fluctuation    (secondary)

PRIMARY endpoint: partial correlation of D2 with ln KIE controlling for achieved <r_DA>,
at the REACTIVE clamp. Everything else is secondary and reported regardless of outcome.

MULTIPLICITY: 3 descriptors x 2 clamps = 6 tests. Only D2-at-reactive is interpreted at
face value. Secondary results are compared against a Bonferroni threshold of 0.05/6 =
0.0083 and labelled exploratory.

ABORT: VIF > 5 on any endpoint means the descriptor and residual geometry cannot be
separated at n=7, and that endpoint is reported as unresolvable rather than as a result.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, math, itertools, sys
from pathlib import Path
import numpy as np

ROOT = Path(_REPO)
EF = ROOT / 'results' / 'ensemble_fluctuation'
KIE = {'WT': 66, 'L754A': 106, 'V750A': 62, 'I538A': 100,
       'L546A': 131, 'I553A': 148, 'I552A': 66}
SYSTEMS = ['WT', 'V750A', 'I552A', 'I538A', 'L754A', 'L546A', 'I553A']
VIF_ABORT = 5.0
BONFERRONI = 0.05 / 6


def partial_corr(x, y, z):
    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        c, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ c
    rx, ry = resid(x, z), resid(y, z)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def vif(x, z):
    r = np.corrcoef(x, z)[0, 1]
    return float(1.0 / max(1e-12, 1.0 - r * r))


def exact_perm_p(x, y, z):
    obs = partial_corr(x, y, z)
    cnt = tot = 0
    for perm in itertools.permutations(range(len(y))):
        s = partial_corr(x, y[list(perm)], z)
        if not math.isnan(s):
            tot += 1
            if abs(s) >= abs(obs) - 1e-12:
                cnt += 1
    return obs, cnt / tot, tot


def boot_ci(x, y, z, n_boot=20000):
    rs = np.random.default_rng(2024)
    out = []
    for _ in range(n_boot):
        i = rs.integers(0, len(x), len(x))
        if len(set(i.tolist())) < 3:
            continue
        s = partial_corr(x[i], y[i], z[i])
        if not math.isnan(s):
            out.append(s)
    o = np.array(out)
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def load(clamp):
    """Per-system arrays of per-frame k_exch, plus achieved geometry."""
    f = EF / f'kexch_frames_{clamp}.json'
    if not f.exists():
        return None, None
    d = json.loads(f.read_text())
    geo = {}
    for s in SYSTEMS:
        c = EF / f'{s}_{clamp}_colvar.dat'
        if c.exists():
            geo[s] = float(np.loadtxt(c).mean())
    return d, geo


def endpoint(x, y, z, name, primary=False):
    v = vif(x, z)
    raw = float(np.corrcoef(x, y)[0, 1])
    tag = 'PRIMARY' if primary else 'secondary'
    print(f'    {name:<22} raw={raw:+.3f}  VIF={v:.2f}   [{tag}]')
    if v > VIF_ABORT:
        print(f'      ABORT: VIF > {VIF_ABORT}; not separable at n={len(x)}')
        return dict(name=name, vif=v, aborted=True)
    pc, p, n = exact_perm_p(x, y, z)
    lo, hi = boot_ci(x, y, z)
    flag = ''
    if not primary:
        flag = '  (exceeds Bonferroni 0.0083)' if p < BONFERRONI else '  (n.s. after multiplicity)'
    print(f'      partial={pc:+.3f}  exact p={p:.4f}{flag}')
    print(f'      bootstrap 95% CI = [{lo:+.3f}, {hi:+.3f}]'
          + ('  spans zero' if lo < 0 < hi else '  EXCLUDES ZERO'))
    return dict(name=name, vif=v, raw=raw, partial=pc, p_exact=p,
                ci=[lo, hi], aborted=False, excludes_zero=not (lo < 0 < hi))


def analyze(clamp, label):
    kx, geo = load(clamp)
    if kx is None:
        print(f'\n=== {label} ===\n  frames not yet processed ({clamp})')
        return None
    ok = [s for s in SYSTEMS if s in kx and s in geo]
    if len(ok) < 5:
        print(f'\n=== {label} ===\n  only {len(ok)} systems; not analysed')
        return None
    print(f'\n=== {label} (n={len(ok)}) ===')
    print(f'    {"system":<8}{"KIE":>5}{"<k_exch>":>11}{"sd":>10}{"sd/mean":>9}{"nfr":>5}{"<r_DA>":>9}')
    D1, D2, D3, Y, Z = [], [], [], [], []
    for s in ok:
        v = np.asarray(kx[s], float)
        v = v[np.isfinite(v)]
        m, sd = float(v.mean()), float(v.std(ddof=1))
        D1.append(m); D2.append(sd); D3.append(sd / m if m != 0 else np.nan)
        Y.append(math.log(KIE[s])); Z.append(geo[s])
        print(f'    {s:<8}{KIE[s]:>5}{m:>11.4g}{sd:>10.4g}{sd/m if m else float("nan"):>9.3f}'
              f'{len(v):>5}{geo[s]:>9.3f}')
    D1, D2, D3 = map(lambda a: np.array(a, float), (D1, D2, D3))
    Y, Z = np.array(Y), np.array(Z)
    res = {}
    res['D2_sd'] = endpoint(D2, Y, Z, 'D2 sd(k_exch)', primary=(clamp == 'r255'))
    res['D1_mean'] = endpoint(D1, Y, Z, 'D1 <k_exch>')
    res['D3_rel'] = endpoint(D3, Y, Z, 'D3 sd/<k_exch>')
    return res


def main():
    print('Pre-registered ensemble-fluctuation analysis')
    print('(plan fixed in results/ensemble_fluctuation/PREREGISTRATION.md before data)')
    a = analyze('r255', 'REACTIVE clamp 2.55 A')
    b = analyze('r340', 'REFERENCE clamp 3.40 A')
    if a and b:
        pa, pb = a['D2_sd'], b['D2_sd']
        if not pa.get('aborted') and not pb.get('aborted'):
            print('\n=== SECONDARY: reactive vs reference (D2) ===')
            d = abs(pa['partial']) - abs(pb['partial'])
            print(f'    |reactive| - |reference| = {d:+.3f}'
                  '   (framework predicts positive)')
    (EF / 'analysis_result.json').write_text(
        json.dumps(dict(reactive=a, reference=b), indent=2, default=str))
    print(f'\nwrote {EF/"analysis_result.json"}')


if __name__ == '__main__':
    main()
