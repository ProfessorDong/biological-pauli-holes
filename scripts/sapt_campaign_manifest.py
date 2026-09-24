#!/usr/bin/env python3
"""One authoritative record of what every SAPT campaign in this paper actually computed.

WHY THIS EXISTS
  The Methods described a donor methane against a WALL METHANE, while the headline numbers
  come from a native side-chain wall, and several campaigns share the symbol
  k_exch^bio while differing in fragment, geometry and displacement coordinate. A reader
  cannot tell them apart from the symbol, and neither could we without opening the files.
  This script reads each campaign's actual output and prints the table, so the manuscript
  quotes a generated record rather than a remembered one.

  Each row names the file the number came from. Nothing here is typed in by hand except the
  fragment and coordinate descriptions, and each of those is annotated with the script that
  builds the geometry so it can be checked in one grep.

Usage: sapt_campaign_manifest.py [--latex]     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import sys
from pathlib import Path

ROOT = Path(_REPO)
R = ROOT / 'results'

CAMPAIGNS = [
    dict(name='Equilibrium, native wall',
         donor='constructed methane at C11', wall='native side chain',
         geom='near-attack umbrella frame, r_DA approx 3.35 A',
         coord='wall translated along donor-wall axis, 5 points over +/-0.20 A',
         basis='jun-cc-pVDZ', order='SAPT0', builder='sapt_native_fragment.py',
         file='sapt_bio/native_fragment/summary.json',
         get=lambda d: {k: v['k_exch_Nm'] for k, v in d['native'].items()}),
    dict(name='Equilibrium, methane wall',
         donor='constructed methane at C11', wall='constructed methane at wall atom',
         geom='same frame as the row above',
         coord='wall translated along donor-wall axis, 5 points over +/-0.20 A',
         basis='jun-cc-pVDZ', order='SAPT0', builder='sapt_bio_curvature.py',
         file='sapt_bio/sapt_summary_corrected.json',
         get=lambda d: {k: v['k_Nm'] for k, v in d['per_system_corrected'].items()}),
    dict(name='Clamped reactive geometry',
         donor='constructed methane at C11', wall='native side chain',
         geom='r_DA clamped to 2.55 A',
         coord='wall translated along donor-wall axis, 5 points over +/-0.20 A',
         basis='jun-cc-pVDZ', order='SAPT0', builder='sapt_clamped_fragment.py',
         file='reactive_geometry/kexch_r255.json', get=lambda d: d),
    dict(name='Clamped reference geometry',
         donor='constructed methane at C11', wall='native side chain',
         geom='r_DA clamped to 3.40 A',
         coord='wall translated along donor-wall axis, 5 points over +/-0.20 A',
         basis='jun-cc-pVDZ', order='SAPT0', builder='sapt_clamped_fragment.py',
         file='reactive_geometry/kexch_r340.json', get=lambda d: d),
    dict(name='Transverse proton Hessian',
         donor='constructed methane at C11', wall='native side chain',
         geom='r_DA clamped to 2.55 A',
         coord='TRANSFERRING H displaced on a 5x5 grid over +/-0.15 A transverse to r_DA',
         basis='jun-cc-pVDZ', order='SAPT0', builder='transverse_proton_hessian.py',
         file='transverse_hessian/transverse_vs_separation.json',
         get=lambda d: {t: v for t, v in zip(d['rank']['tags'], d['rank']['trace_K_perp'])}),
]


def main():
    latex = '--latex' in sys.argv
    rows = []
    for c in CAMPAIGNS:
        f = R / c['file']
        if not f.exists():
            print(f'  MISSING {c["file"]}', file=sys.stderr)
            continue
        vals = c['get'](json.loads(f.read_text()))
        wt = vals.get('WT')
        finite = [v for v in vals.values() if v is not None]
        rows.append((c, wt, min(finite), max(finite), len(finite)))

    if not latex:
        print(f'{"campaign":>28}{"n":>4}{"WT":>9}{"min":>9}{"max":>9}   file')
        for c, wt, lo, hi, n in rows:
            w = f'{wt:.3f}' if wt is not None else 'n/a'
            print(f'{c["name"]:>28}{n:>4}{w:>9}{lo:>9.3f}{hi:>9.3f}   {c["file"]}')
        print('\n  All values N/m. Note the last row is a different second derivative from')
        print('  every row above it: the others displace the WALL along the donor-wall axis,')
        print('  it displaces the transferring HYDROGEN transverse to the reaction axis.')
        print('\n  fragment and coordinate detail')
        for c, *_ in rows:
            print(f'\n  {c["name"]}  [{c["builder"]}]')
            print(f'    donor : {c["donor"]}')
            print(f'    wall  : {c["wall"]}')
            print(f'    geom  : {c["geom"]}')
            print(f'    coord : {c["coord"]}')
            print(f'    theory: {c["order"]}/{c["basis"]}')
        return

    print('\\begin{tabular}{p{2.6cm}p{2.1cm}p{2.1cm}p{3.5cm}cc}')
    print('\\toprule')
    print('Campaign & Donor & Wall & Displacement coordinate & $n$ & WT value \\\\')
    print('\\midrule')
    for c, wt, lo, hi, n in rows:
        w = f'${wt:.2f}$' if wt is not None else 'n/a'
        print(f'{c["name"]} & {c["donor"]} & {c["wall"]} & {c["coord"]} & {n} & {w} \\\\')
    print('\\bottomrule')
    print('\\end{tabular}')


if __name__ == '__main__':
    main()
