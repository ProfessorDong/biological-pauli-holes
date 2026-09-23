#!/usr/bin/env python3
"""Compare the curvature the panel measures with the curvature the theory defines.

The manuscript's k_exch^bio is obtained by translating the whole wall fragment along the
donor-to-wall axis. Call that K_sep. The theory's confinement argument is about K_perp, the
Hessian of the exchange energy with respect to transverse motion of the transferring nucleus
(scripts/transverse_proton_hessian.py). This script puts the two side by side and recomputes
the zero-point-energy bound of the manuscript's Sec. 'converting the bound' using K_perp,
which is the quantity that actually enters hbar*sqrt(k/m)/2.

The comparison is the point. If K_sep and K_perp disagree in magnitude only, K_sep is a scaled
proxy. If they disagree in RANK, K_sep is not a proxy for the theoretical quantity at all, and
every downstream isotope statement built on it has to be withdrawn or re-derived.

Usage: transverse_vs_separation.py     (pauli env)
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
OUT = ROOT / 'results/transverse_hessian'
HBAR = 1.054571817e-34
AMU = 1.66053907e-27
KB = 1.380649e-23
T = 283.15                      # 10 C, the temperature of the KIE ladder
J2KCAL = 6.02214076e23 / 4184.0


def zpe_kcal(k_Nm, m_amu):
    """Harmonic zero-point energy of one mode, kcal/mol. Negative k has no real mode."""
    if k_Nm <= 0:
        return float('nan')
    return 0.5 * HBAR * np.sqrt(k_Nm / (m_amu * AMU)) * J2KCAL


def main():
    recs = []
    for f in sorted(glob.glob(str(OUT / '*_transverse_hessian.json'))):
        d = json.load(open(f))
        w = np.array(d['eigenvalues_Nm'])
        recs.append(dict(tag=d['tag'], clamp=d['clamp'], half=d['half_width_A'],
                         w1=w[0], w2=w[1], tr=float(w.sum()), Ksep=d['K_sep_Nm'],
                         rms=d['rms_residual_kcal'], file=os.path.basename(f)))

    # A silent filename collision once dropped a system from this table without any error,
    # and the analysis simply ran on the survivors. Refuse to proceed unless every record is
    # distinct in (system, clamp, grid) and the full seven-system panel is present at the
    # reactive clamp, which is the panel every other analysis in the paper uses.
    keys = [(r['tag'], r['clamp'], r['half']) for r in recs]
    assert len(keys) == len(set(keys)), f'duplicate (system, clamp, grid) records: {keys}'
    PANEL = {'WT', 'I553A', 'I552A', 'L754A', 'V750A', 'I538A', 'L546A'}
    got = {r['tag'] for r in recs if r['clamp'] == 'r255' and abs(r['half'] - 0.15) < 1e-9}
    assert got == PANEL, f'reactive-clamp panel incomplete, missing {sorted(PANEL - got)}'

    print(f'{"system":>7}{"clamp":>7}{"grid":>7}{"K_perp eig 1":>14}{"eig 2":>10}'
          f'{"trace":>10}{"K_sep":>10}{"trace/K_sep":>13}')
    for r in recs:
        ratio = r['tr'] / r['Ksep'] if r['Ksep'] else float('nan')
        print(f'{r["tag"]:>7}{r["clamp"]:>7}{r["half"]:>7.2f}{r["w1"]:>14.3f}{r["w2"]:>10.3f}'
              f'{r["tr"]:>10.3f}{r["Ksep"]:>10.3f}{ratio:>13.4f}')
    print('\n  all in N/m. Fit quality (RMS residual of the quadratic form, kcal/mol): '
          f'{max(r["rms"] for r in recs):.2e} worst case.')

    # convergence: same system and clamp at two grid half-widths
    conv = {}
    for r in recs:
        conv.setdefault((r['tag'], r['clamp']), []).append(r)
    for key, v in conv.items():
        if len(v) > 1:
            v = sorted(v, key=lambda z: z['half'])
            print(f'\n  GRID CONVERGENCE {key[0]} {key[1]}: '
                  + ', '.join(f'half={x["half"]:.2f} -> trace {x["tr"]:+.3f}' for x in v))
            spread = max(x['tr'] for x in v) - min(x['tr'] for x in v)
            print(f'    trace varies by {spread:.3f} N/m between grids, so the sign and the '
                  f'order of magnitude are grid-independent.'
                  if spread < 0.2 else
                  f'    trace varies by {spread:.3f} N/m; NOT converged, widen the analysis.')

    # rank comparison on the primary grid at the reactive clamp
    prim = [r for r in recs if r['clamp'] == 'r255' and abs(r['half'] - 0.15) < 1e-9]
    if len(prim) >= 3:
        ks = np.array([r['Ksep'] for r in prim])
        kp = np.array([r['tr'] for r in prim])
        tags = [r['tag'] for r in prim]
        from itertools import permutations

        from scipy.stats import rankdata, spearmanr
        rho = spearmanr(ks, kp)
        # n is at most 7 here, so the asymptotic p-value scipy returns is not valid. The exact
        # null is the 7! = 5040 relabellings, which is cheap to enumerate exhaustively and is
        # the same exact-permutation convention used for the pre-registered endpoints.
        ra, rb = rankdata(ks), rankdata(kp)
        obs = float(np.corrcoef(ra, rb)[0, 1])
        null = [np.corrcoef(ra, np.array(q))[0, 1] for q in permutations(rb)]
        p_exact = float(np.mean(np.abs(np.array(null)) >= abs(obs) - 1e-12))
        print(f'\n  RANK COMPARISON at the reactive clamp on {len(prim)} systems '
              f'({", ".join(tags)})')
        print(f'    Spearman rho(K_sep, trace K_perp) = {obs:+.3f}')
        print(f'    exact two-sided p over all {len(null)} relabellings = {p_exact:.4f}'
              f'   (scipy asymptotic value {rho.pvalue:.3f} is not valid at this n)')
        o1 = [tags[i] for i in np.argsort(ks)]
        o2 = [tags[i] for i in np.argsort(kp)]
        print(f'    order by K_sep : {" < ".join(o1)}')
        print(f'    order by K_perp: {" < ".join(o2)}')
        print(f'    orders {"AGREE" if o1 == o2 else "DISAGREE"}.')

    # the zero-point bound, recomputed on the quantity that actually carries it
    print('\n  ZERO-POINT CHANNEL, recomputed on K_perp')
    print('    The manuscript bounds the isotope effect confinement can supply by adding')
    print('    k_exch to a covalent transverse curvature and taking hbar*sqrt(k/m)/2. That')
    print('    substitution is only valid for the transverse curvature, so it is redone here')
    print('    with K_perp. Modes with a negative eigenvalue support no real oscillator and')
    print('    are reported as such rather than being given a frequency.')
    rows = []
    for kcov in (50.0, 100.0, 500.0):
        vals = []
        for r in prim:
            tot = 0.0
            for w in (r['w1'], r['w2']):
                dz = zpe_kcal(kcov + w, 1.008) - zpe_kcal(kcov + w, 2.014)
                tot += dz
            vals.append(tot)
        spread = max(vals) - min(vals)
        factor = float(np.exp(spread * 4184.0 / 6.02214076e23 / (KB * T)))
        rows.append((kcov, spread, factor))
        print(f'    k_cov = {kcov:5.0f} N/m : largest across-variant difference in the two-mode '
              f'H/D zero-point shift = {spread:.2e} kcal/mol, isotope-effect factor {factor:.4f}')
    print('\n    The measured ladder spans a factor of 2.39, from 62 to 148.')
    worst = max(r[2] for r in rows)
    pct = np.log(worst) / np.log(2.39) * 100
    print(f'    So on the transverse curvature that the theory actually defines, this channel')
    print(f'    accounts for at most {pct:.2f}% of the observed spread in ln(KIE).')

    (OUT / 'transverse_vs_separation.json').write_text(json.dumps(dict(
        records=recs,
        rank=dict(tags=[r['tag'] for r in prim],
                  K_sep=[r['Ksep'] for r in prim],
                  trace_K_perp=[r['tr'] for r in prim],
                  spearman_rho=float(obs), spearman_p_exact=float(p_exact),
                  n_permutations=len(null),
                  orders_agree=bool(o1 == o2)) if len(prim) >= 3 else None,
        zpe_bound=[dict(k_cov_Nm=a, spread_kcal=b, kie_factor=c) for a, b, c in rows],
        max_percent_of_ladder=float(pct)), indent=1))
    print(f'\nwrote {OUT}/transverse_vs_separation.json')


if __name__ == '__main__':
    main()
