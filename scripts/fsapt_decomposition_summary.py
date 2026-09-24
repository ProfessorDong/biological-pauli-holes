#!/usr/bin/env python3
"""Is the donor-surrogate correction electronic in every system, or only in wild type?

WHY THIS EXISTS
  Replacing the methane donor with the real bis-allylic C5H8 unit raises the transverse
  exchange curvature by a median factor of 1.31 across the seven JBC systems. On wild type
  the F-SAPT group partition attributes 96.5% of that curvature to the bis-allylic CH2
  itself rather than to the two vinyl flanks, which makes the correction electronic (a
  delocalized pentadienyl pi system is a harder wall partner than a closed-shell sp3
  carbon) rather than steric (more atoms happening to sit near the wall).

  One system is an anecdote. If the same partition holds across all seven, the reading is
  a property of the substrate rather than of one snapshot's packing, and the manuscript can
  say so. This collects the seven and checks that.

WHAT TO WATCH FOR
  L754A is not comparable to the other six. Its donor faces an open Asn672 wall at 4.5 A
  instead of a Leu732 wall at 3.4 A, so its total exchange curvature is 0.042 N/m, near
  enough to zero that the split between groups is a split of noise. It is reported and
  excluded from the summary statistics, in the same way and for the same reason as in
  donor_surrogate_summary.py.

Usage: fsapt_decomposition_summary.py     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(_REPO)
D = ROOT / 'results/sapt_bio/donor_fragment'
TAGS = ['WT', 'I553A', 'I552A', 'L754A', 'V750A', 'I538A', 'L546A']
NEAR_ZERO = {'L754A'}          # open wall; both donors give a curvature consistent with zero


def main():
    S = json.loads((ROOT / 'results/sapt_bio/sapt_summary.json').read_text())
    rows, missing = [], []
    for t in TAGS:
        f = D / f'{t}_fsapt_decomposition.json'
        if not f.exists():
            missing.append(t)
            continue
        j = json.loads(f.read_text())
        k = j['k_by_group_Nm']
        bi = k['biallylic']
        vi = sum(v for g, v in k.items() if g != 'biallylic')
        rows.append(dict(tag=t, k_total=j['k_total_Nm'], k_bi=bi, k_vi=vi,
                         k_meth=j['k_methane_Nm'], share=bi / j['k_total_Nm'],
                         vs_meth=bi / j['k_methane_Nm'] if j['k_methane_Nm'] else np.nan,
                         KIE=S[t]['KIE'], wall=j['wall_atom']))
    if missing:
        print(f'NOT YET COMPUTED: {", ".join(missing)}\n')

    print(f'{"system":>8}{"wall":>6}{"methane":>9}{"C5H8 tot":>10}{"bis-allylic":>13}'
          f'{"vinyls":>9}{"bis share":>11}{"bis/methane":>13}')
    for r in rows:
        flag = '  *' if r['tag'] in NEAR_ZERO else ''
        print(f'{r["tag"]:>8}{r["wall"]:>6}{r["k_meth"]:>9.3f}{r["k_total"]:>10.3f}'
              f'{r["k_bi"]:>13.3f}{r["k_vi"]:>9.3f}{r["share"]:>10.1%}'
              f'{r["vs_meth"]:>12.3f}x{flag}')
    if any(r['tag'] in NEAR_ZERO for r in rows):
        print('  * open wall, curvature consistent with zero: excluded from the statistics '
              'below, its group split is a split of noise')

    keep = [r for r in rows if r['tag'] not in NEAR_ZERO]
    if not keep:
        print('\nnothing to summarize yet')
        return
    share = np.array([r['share'] for r in keep])
    vs = np.array([r['vs_meth'] for r in keep])

    # Two independent questions that the WT result alone could not separate, because there
    # they happened to answer together.
    #
    #   LOCALIZATION  what fraction of the curvature sits on the reactive CH unit rather
    #                 than on the vinyl flanks. If this is high everywhere, the flanks are
    #                 spectators and the surrogate's missing atoms are not the issue.
    #   FIDELITY      how well a methane reproduces that unit. This is the electronic
    #                 question, and it is the one that can vary with how the pentadienyl
    #                 pi system happens to be oriented toward the wall.
    localized = bool((share > 0.85).all())
    print(f'\nLOCALIZATION  bis-allylic share of the total : {share.min():.1%} to '
          f'{share.max():.1%}, median {np.median(share):.1%}   (n={len(keep)})')
    print(f'              flanks are spectators in every system: {localized}')
    print(f'\nFIDELITY      bis-allylic vs methane donor    : {vs.min():.2f}x to '
          f'{vs.max():.2f}x, median {np.median(vs):.2f}x')
    if vs.max() - vs.min() > 0.15:
        lo = min(keep, key=lambda r: r['vs_meth'])['tag']
        hi = max(keep, key=lambda r: r['vs_meth'])['tag']
        print(f'              NOT uniform: methane is faithful in {lo} '
              f'({vs.min():.2f}x) and understates in {hi} ({vs.max():.2f}x), so the '
              'electronic correction is geometry-dependent, not a constant of the substrate')
    else:
        print('              uniform across the panel')
    uniform = localized and bool((vs > 1.15).all())

    # No correlation with the isotope-effect ladder is reported here, deliberately. L754A is
    # excluded above because its GROUP SPLIT is noise, which is a statement about the
    # decomposition and not about its curvature: k_exch = 0.004 N/m is a legitimate
    # measurement and belongs in the panel. Correlating on the remaining six would drop the
    # panel's most influential point for a reason that does not apply to the regression, and
    # it happens to raise R substantially, which is precisely the leverage artifact the
    # manuscript already reports for this dataset. The panel correlation, on all seven and
    # under both donors, is in donor_surrogate_summary.py.
    print('\nno KIE correlation is computed here; see donor_surrogate_summary.py, which '
          'uses all seven')

    (D / 'fsapt_decomposition_summary.json').write_text(json.dumps(dict(
        rows=rows, excluded=sorted(NEAR_ZERO), n_comparable=len(keep),
        bis_share_range=[float(share.min()), float(share.max())],
        bis_share_median=float(np.median(share)),
        bis_vs_methane_range=[float(vs.min()), float(vs.max())],
        bis_vs_methane_median=float(np.median(vs)),
        flanks_are_spectators_everywhere=localized,
        electronic_in_every_system=uniform), indent=1))
    print(f'\nwrote {D}/fsapt_decomposition_summary.json')


if __name__ == '__main__':
    main()
