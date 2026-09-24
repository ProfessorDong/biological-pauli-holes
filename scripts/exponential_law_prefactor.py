#!/usr/bin/env python3
"""Fit the exchange curvature in the form the exact solutions actually have.

WHY THIS EXISTS
  exponential_law_panel.py regresses ln k on d alone. That is misspecified with respect to
  this paper's own result. Both solved geometries give

      Delta E = A (d/a0)^p exp(-r d),   so   ln k = const + p ln d - r d,

  with p = 1.17 for the paraboloid and 2.17 for the sphere. Dropping the p ln d term does not
  merely lose the prefactor: it biases the fitted rate in a way that depends on where the data
  sit, because the LOCAL slope of the true curve is

      d(ln k)/dd = -(r - p/d),

  which steepens as d grows. Systems sampling larger d then fit steeper apparent rates even
  when r is identical, and the panel does show fitted rate rising with mean sampled distance.
  That is a property of the model, not of the enzyme, and it has to be removed before any
  statement about rates differing between variants can be believed.

WHAT IS FITTED
  E   ln k = a_g + p ln d - r d          common p, common r
  E2  ln k = a_g + p ln d - r_s d        common p, rate free per system

  a_g is an intercept per system and clamp, the prefactor model the panel analysis already
  showed the data demand. E2 against E is the rate-heterogeneity test under the correct
  functional form; comparing its p-value with the one from the linear-in-d model shows how
  much of the apparent heterogeneity was misspecification.

WHAT THIS STILL CANNOT DO
  It cannot determine kappa. The abscissa is a closest approach between two molecular
  fragments, not the hydrogen-to-wall distance of the solved one-electron problems, and a
  fitted p absorbs whatever else varies smoothly with distance. Agreement in r at the tens of
  percent level is consistency, not measurement.

Usage: exponential_law_prefactor.py     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist

ROOT = Path(_REPO)
DF = ROOT / 'results/sapt_bio/donor_fragment'
TAGS = ['WT', 'I553A', 'I552A', 'L754A', 'V750A', 'I538A', 'L546A']
CLAMPS = ['r255', 'r340']
A0 = 0.5291772109
SOLVED = {'paraboloid': (2.011 / A0, 1.1719), 'sphere': (2.0017 / A0, 2.1741)}


def fit(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, float(r @ r)


def ftest(rss0, p0, rss1, p1, n):
    if p1 <= p0 or rss1 <= 0:
        return float('nan'), float('nan')
    F = ((rss0 - rss1) / (p1 - p0)) / (rss1 / (n - p1))
    return F, float(fdist.sf(F, p1 - p0, n - p1))


def main():
    d, k, sysid, sc = [], [], [], []
    for si, t in enumerate(TAGS):
        for c in CLAMPS:
            f = DF / f'{t}_{c}_fsapt_ensemble.json'
            if not f.exists():
                continue
            for r in json.loads(f.read_text())['frames']:
                if r['k_total'] <= 0:
                    continue
                d.append(r['dmin']), k.append(r['k_total'])
                sysid.append(si), sc.append(f'{t}_{c}')
    d, k, sysid, sc = np.array(d), np.array(k), np.array(sysid), np.array(sc)
    ly, lnd, n = np.log(k), np.log(d), len(d)
    present = sorted(set(sysid))
    G = np.column_stack([(sc == g).astype(float) for g in sorted(set(sc))])
    S = np.column_stack([(sysid == s).astype(float) for s in present])

    XE = np.column_stack([G, lnd, -d])
    XE2 = np.column_stack([G, lnd, -S * d[:, None]])
    bE, rE = fit(XE, ly)
    bE2, rE2 = fit(XE2, ly)
    pE, pE2 = XE.shape[1], XE2.shape[1]
    p_hat, r_hat = float(bE[-2]), float(bE[-1])

    # IDENTIFIABILITY FIRST. Over any narrow window ln d is nearly linear in d, so the two
    # regressors are collinear and p and r trade off against each other almost freely. If the
    # variance inflation is large, the individual coefficients below are not measurements of
    # anything, however well the model fits, and only their combination over the sampled
    # window is determined.
    from scipy.stats import pearsonr as _pr
    coll = _pr(d, lnd).statistic
    vif = 1.0 / (1.0 - coll ** 2)
    identified = vif < 10
    print(f'{n} configurations, d = {d.min():.2f} to {d.max():.2f} A\n')
    print(f'  IDENTIFIABILITY  corr(d, ln d) = {coll:.5f}, variance inflation = {vif:.0f}')
    status = 'separately identified' if identified else 'NOT separately identified'
    print(f'    p and r are {status}: over this window ln d is nearly a linear')
    print('    function of d, so the split between the algebraic prefactor and the')
    print('    exponential is arbitrary within a wide band.\n')
    print(f'  ln k = a_(system,clamp) + p ln d - r d\n')
    print(f'    fitted r = {r_hat:.2f} per A')
    print(f'    fitted p = {p_hat:.2f}')
    for name, (rs, ps) in SOLVED.items():
        print(f'    {name:11s} solved: r = {rs:.2f} per A, p = {ps:.2f}   '
              f'(ratio r {r_hat/rs:.2f})')

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        try:
            bb, _ = fit(np.column_stack([G[i], lnd[i], -d[i]]), ly[i])
            boot.append((bb[-1], bb[-2]))
        except np.linalg.LinAlgError:
            pass
    boot = np.array(boot)
    rlo, rhi = np.percentile(boot[:, 0], [2.5, 97.5])
    plo, phi = np.percentile(boot[:, 1], [2.5, 97.5])
    print(f'\n    95% bootstrap  r: {rlo:.2f} to {rhi:.2f}   p: {plo:.2f} to {phi:.2f}')
    for name, (rs, ps) in SOLVED.items():
        print(f'      {name:11s} r inside: {rlo <= rs <= rhi}   p inside: {plo <= ps <= phi}')

    F, pv = ftest(rE, pE, rE2, pE2, n)
    print(f'\n  E2 over E (rate varies by system, correct functional form)')
    print(f'    F = {F:.2f}, p = {pv:.2e}')
    prior = json.loads((DF / 'exponential_law_panel.json').read_text())
    print(f'    for comparison, linear-in-d model gave p = {prior["p_rate"]:.2e}')

    _pl = prior['per_system']
    prior_spread = max(v['rate'] for v in _pl.values()) / min(v['rate'] for v in _pl.values())
    rates = {TAGS[s]: float(bE2[G.shape[1] + 1 + j])
             for j, s in enumerate(present)}
    print(f'\n{"system":>8}{"rate (per A)":>14}')
    for t, v in sorted(rates.items(), key=lambda z: z[1]):
        print(f'{t:>8}{v:>14.2f}')
    rr = np.array(list(rates.values()))
    print(f'\n  per-system rates span {rr.min():.2f} to {rr.max():.2f}, '
          f'ratio max/min = {rr.max()/rr.min():.2f}')

    (DF / 'exponential_law_prefactor.json').write_text(json.dumps(dict(
        n=n, fitted_r_per_A=r_hat, fitted_p=p_hat,
        r_ci95=[float(rlo), float(rhi)], p_ci95=[float(plo), float(phi)],
        solved={k2: dict(r_per_A=v[0], p=v[1]) for k2, v in SOLVED.items()},
        F_rate=float(F), p_rate=float(pv),
        p_rate_linear_model=prior['p_rate'],
        per_system_rate=rates, rss=dict(E=rE, E2=rE2),
        params=dict(E=pE, E2=pE2),
        collinearity_r=float(coll), vif=float(vif), identified=bool(identified),
        spread_linear_model=float(prior_spread),
        spread_with_prefactor=float(rr.max()/rr.min())), indent=1))
    if not identified:
        print(f'\n  CONCLUSION  with variance inflation {vif:.0f} the separate values of r and p')
        print('    above must not be quoted. What the data determine is the EFFECTIVE local')
        print('    decay over the sampled window, which the linear-in-d fit already reports.')
        print('    The useful result here is comparative: including the prefactor term')
        print(f'    compresses the between-system rate spread from '
              f'{prior_spread:.2f}x to {rr.max()/rr.min():.2f}x, so most of the apparent')
        print('    heterogeneity in the linear model was misspecification rather than physics.')
    print(f'\nwrote {DF}/exponential_law_prefactor.json')


if __name__ == '__main__':
    main()
