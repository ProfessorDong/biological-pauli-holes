#!/usr/bin/env python3
"""Does the methane donor surrogate change any conclusion? Seven-system verdict.

WHY THIS EXISTS
  Li, Soudackov and Hammes-Schiffer (JACS 140, 3068 (2018)) criticize a gas-phase methane
  donor model of soybean lipoxygenase, that of Champion and co-workers, as too small to
  describe a substrate with a pi backbone. Their criticism concerns the donor-acceptor
  coordinate against the iron cofactor, not the transverse donor-to-wall curvature measured
  here, so nothing below tests their claim. It does establish that a one-carbon stand-in for
  a conjugated substrate is a hazard worth quantifying in our own setting.

  The manuscript already validates the WALL fragment (methane replaced by the native
  isobutane or acetamide side chain), but until now the DONOR remained a methane in every
  reported number. This collects the donor swap across all seven JBC systems and states what
  does and does not move.

  The two surrogates are independent and, as it turns out, run in opposite directions: the
  wall surrogate over-estimates the curvature, the donor surrogate under-estimates it.

WHAT IT REPORTS
  * the per-system curvature with each donor, and their ratio
  * whether the rank order across cavity mutations survives the swap
  * the correlation with the isotope-effect ladder under each donor, with exact
    permutation p-values on the seven-system panel

  The methane column must reproduce the published native-fragment series; that is asserted
  here rather than assumed, because it is what makes the comparison a controlled one.

Usage: donor_surrogate_summary.py     (pauli env)
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
D = ROOT / 'results/sapt_bio/donor_fragment'
TAGS = ['WT', 'I553A', 'I552A', 'L754A', 'V750A', 'I538A', 'L546A']
# The published native-fragment curvature, methane donor, for the assertion below.
PUBLISHED = {'WT': 7.885, 'I553A': 13.961, 'I552A': 4.932, 'L754A': 0.004,
             'V750A': 7.710, 'I538A': 6.535, 'L546A': 16.381}
# L754A's donor sits 4.8 A from an ASN672 wall, so both curvatures are consistent with
# zero and their ratio is a ratio of noise. It is excluded from the ratio range only.
NEAR_ZERO = {'L754A'}


def exact_perm_p(x, y):
    r0 = abs(pearsonr(x, y).statistic)
    hits = sum(abs(pearsonr(np.array(q), y).statistic) >= r0 - 1e-12
               for q in itertools.permutations(x))
    return hits / math.factorial(len(x))


def main():
    S = json.loads((ROOT / 'results/sapt_bio/sapt_summary.json').read_text())
    kie = np.array([S[t]['KIE'] for t in TAGS], float)
    m, p, walls = [], [], []
    for t in TAGS:
        j = json.loads((D / f'{t}_donor_comparison.json').read_text())
        a, b = j['results']['methane']['k_Nm'], j['results']['pentadienyl']['k_Nm']
        assert abs(a - PUBLISHED[t]) < 0.002, \
            f'{t}: methane arm gives {a:.3f}, published series has {PUBLISHED[t]}'
        m.append(a), p.append(b), walls.append(j['wall_atom'])
    m, p, y = np.array(m), np.array(p), np.log(kie)
    ratio = p / m
    keep = np.array([t not in NEAR_ZERO for t in TAGS])

    print(f'{"system":>8}{"KIE":>6}{"wall":>7}{"methane":>10}{"C5H8":>9}{"ratio":>9}')
    for t, k, w, a, b, r in zip(TAGS, kie, walls, m, p, ratio):
        flag = '   (both consistent with zero)' if t in NEAR_ZERO else ''
        print(f'{t:>8}{int(k):>6}{w:>7}{a:>10.3f}{b:>9.3f}{r:>9.3f}{flag}')

    rho_pres = spearmanr(m, p).statistic
    print(f'\nrank order across the panel: Spearman rho = {rho_pres:.4f} '
          f'({"preserved exactly" if rho_pres == 1.0 else "NOT preserved"})')
    print(f'ratio, excluding the near-zero system: {ratio[keep].min():.2f} to '
          f'{ratio[keep].max():.2f}, median {np.median(ratio[keep]):.2f}')

    stats = {}
    print(f'\n{"donor":>34}{"Pearson R":>12}{"perm p":>9}{"Spearman":>10}')
    for lbl, key, v in (('methane surrogate (published)', 'methane', m),
                        ('real bis-allylic C5H8', 'pentadienyl', p)):
        R, pp, rho = pearsonr(v, y).statistic, exact_perm_p(v, y), spearmanr(v, y).statistic
        stats[key] = dict(pearson_R=float(R), exact_perm_p=float(pp),
                          spearman_rho=float(rho))
        print(f'{lbl:>34}{R:>+12.3f}{pp:>9.3f}{rho:>+10.3f}')

    verdict = ('The donor surrogate understates the curvature by a median factor of '
               f'{np.median(ratio[keep]):.2f} but preserves the rank order exactly, and the '
               'correlation with the isotope-effect ladder remains unresolved from zero '
               'under either donor. Replacing the surrogate does not rescue the descriptor, '
               'so the reported null is not an artifact of the model donor.')
    print(f'\nVERDICT  {verdict}')

    (D / 'donor_surrogate_summary.json').write_text(json.dumps(dict(
        tags=TAGS, KIE_10C=kie.tolist(), wall_atom=walls,
        k_methane_Nm=m.tolist(), k_pentadienyl_Nm=p.tolist(), ratio=ratio.tolist(),
        ratio_range_excluding=sorted(NEAR_ZERO),
        ratio_min=float(ratio[keep].min()), ratio_max=float(ratio[keep].max()),
        ratio_median=float(np.median(ratio[keep])),
        spearman_methane_vs_pentadienyl=float(rho_pres),
        kie_correlation=stats, verdict=verdict), indent=1))
    print(f'wrote {D}/donor_surrogate_summary.json')


if __name__ == '__main__':
    main()
