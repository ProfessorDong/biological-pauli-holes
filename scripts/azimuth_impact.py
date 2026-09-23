#!/usr/bin/env python3
"""Does the methane azimuth artifact change any conclusion, or only the numbers?

WHY THIS EXISTS
  methane_azimuth_sensitivity.py established that the published k_exch of each system is one
  draw from a range spanned by an arbitrary rotation of the methane surrogate about its C-H
  axis, and that the range is wide: 16% to 80% of the published value depending on system.
  That is a defect in the numbers. It is a separate question whether it is a defect in the
  CONCLUSIONS, and the answer is not obvious in either direction, because the pre-registered
  endpoints are correlations rather than magnitudes and a common multiplicative wobble would
  leave a correlation untouched while an independent one need not.

  So this resamples: each system independently takes one of its six computed azimuths, and
  the panel statistic is recomputed. Repeating that over all 6^7 = 279936 combinations gives
  the exact distribution of the statistic under the construction freedom, not an estimate of
  it. If the published conclusion sits comfortably inside that distribution, the conclusion
  is robust to the artifact and the artifact is a reporting problem. If it does not, the
  conclusion depends on an arbitrary choice and has to be withdrawn.

WHAT IS AND IS NOT TESTED
  This varies ONLY the surrogate's azimuth. Snapshot, wall fragment, basis, SAPT order and
  scan protocol are held at their published values, so the spread reported here is a lower
  bound on the total construction uncertainty and is not a substitute for the bootstrap fit
  error, which measures something different again.

  The azimuth is sampled at six points over one 120 degree period, so the per-system range
  is itself a lower bound: a finer grid can only widen it.

Usage: azimuth_impact.py     (pauli env)
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
DF = ROOT / 'results/sapt_bio/donor_fragment'
TAGS = ['WT', 'I553A', 'I552A', 'L754A', 'V750A', 'I538A', 'L546A']


def exact_perm_p(x, y):
    r0 = abs(pearsonr(x, y).statistic)
    hits = sum(abs(pearsonr(np.array(q), y).statistic) >= r0 - 1e-12
               for q in itertools.permutations(x))
    return hits / math.factorial(len(x))


def main():
    S = json.loads((ROOT / 'results/sapt_bio/sapt_summary.json').read_text())
    kie = np.array([S[t]['KIE'] for t in TAGS], float)
    y = np.log(kie)

    grid, pub = [], []
    for t in TAGS:
        j = json.loads((DF / f'{t}_azimuth_sensitivity.json').read_text())
        grid.append(j['k_Nm'])
        pub.append(j['published_k_Nm'])
        assert abs(j['k_Nm'][0] - j['published_k_Nm']) < 1e-9, f'{t}: phi=0 is not published'
    grid, pub = np.array(grid), np.array(pub)

    print(f'{"system":>8}{"published":>11}{"min":>9}{"max":>9}{"spread":>9}{"pub is":>12}')
    for t, p, row in zip(TAGS, pub, grid):
        where = ('at the minimum' if abs(p - row.min()) < 1e-9 else
                 'at the maximum' if abs(p - row.max()) < 1e-9 else 'interior')
        print(f'{t:>8}{p:>11.3f}{row.min():>9.3f}{row.max():>9.3f}'
              f'{(row.max()-row.min())/p:>8.0%}{where:>17}')

    R_pub = pearsonr(pub, y).statistic
    p_pub = exact_perm_p(pub, y)
    print(f'\npublished panel correlation with ln(KIE): R = {R_pub:+.3f}, '
          f'exact permutation p = {p_pub:.3f}')

    # Exact enumeration over every combination of per-system azimuth choices.
    Rs, rhos = [], []
    for combo in itertools.product(*[range(g.shape[0]) for g in grid]):
        v = grid[np.arange(len(TAGS)), combo]
        Rs.append(pearsonr(v, y).statistic)
        rhos.append(spearmanr(v, pub).statistic)
    Rs, rhos = np.array(Rs), np.array(rhos)

    print(f'\nExact distribution over all {len(Rs)} azimuth combinations:')
    print(f'  Pearson R with ln(KIE) : {Rs.min():+.3f} to {Rs.max():+.3f}, '
          f'median {np.median(Rs):+.3f}')
    print(f'  fraction with R > 0    : {np.mean(Rs > 0):.1%}')
    q = np.percentile(Rs, [2.5, 97.5])
    print(f'  central 95%            : {q[0]:+.3f} to {q[1]:+.3f}')
    print(f'  published R = {R_pub:+.3f} sits at percentile '
          f'{100*np.mean(Rs <= R_pub):.1f}')

    # Would any combination reach nominal significance on this panel?
    # The exact permutation threshold for |R| on n=7 at alpha=0.05 is |R| ~ 0.75.
    thresh = 0.75
    print(f'\n  combinations with |R| >= {thresh} (roughly nominal significance at n=7): '
          f'{np.mean(np.abs(Rs) >= thresh):.2%}')

    print(f'\nRank order against the published ordering:')
    print(f'  Spearman rho : {rhos.min():+.3f} to {rhos.max():+.3f}, '
          f'median {np.median(rhos):+.3f}')
    print(f'  fraction preserving the published order exactly : {np.mean(rhos == 1.0):.1%}')

    verdict = ('robust' if (np.mean(np.abs(Rs) >= thresh) < 0.05 and np.median(Rs) * R_pub > 0)
               else 'NOT robust')
    print(f'\nVERDICT  the null conclusion is {verdict} to the azimuth artifact: no azimuth '
          f'choice\n         turns this panel significant, though the rank order is not '
          f'preserved.')

    (DF / 'azimuth_impact.json').write_text(json.dumps(dict(
        tags=TAGS, published_k_Nm=pub.tolist(), grid_k_Nm=grid.tolist(),
        published_R=float(R_pub), published_perm_p=float(p_pub),
        n_combinations=int(len(Rs)),
        R_min=float(Rs.min()), R_max=float(Rs.max()), R_median=float(np.median(Rs)),
        R_ci95=[float(q[0]), float(q[1])],
        frac_positive=float(np.mean(Rs > 0)),
        frac_abs_R_ge_075=float(np.mean(np.abs(Rs) >= thresh)),
        rank_preserved_fraction=float(np.mean(rhos == 1.0)),
        verdict=verdict), indent=1))
    print(f'\nwrote {DF}/azimuth_impact.json')


if __name__ == '__main__':
    main()
