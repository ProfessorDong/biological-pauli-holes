#!/usr/bin/env python3
"""What actually surrounds the transferring hydrogen? The measurement behind xi0.

WHY THIS EXISTS
  The placement of soybean lipoxygenase on the exponential confinement curve rests on one
  number, the closest approach of the transferring hydrogen to an external closed-shell
  heavy atom, quoted as 2.4 to 3.0 A. Everything downstream is exponential in it, so a
  referee is entitled to ask how it was measured and what would change it. A co-author
  proposed instead that the pocket is a benzene-like ring of six carbons at 1.31 A, which
  would raise the induced dipole 45-fold and the field 280-fold. That is a question about
  geometry, and geometry is measurable.

  This measures the full coordination environment over the MD ensemble, separating three
  populations that a naive nearest-neighbour search conflates:

    the substrate's own atoms   the donor carbon is covalently bonded at 1.10 A, and the
                                1,3-carbons sit near 2.1 A purely by bond geometry. Neither
                                is a confining wall; they are the molecule the H belongs to.
    the acceptor oxygen         the reaction partner, not a wall. Including it would count
                                the reaction coordinate as confinement.
    the external environment    protein, cofactor and solvent. This is the wall.

  Only the third population is what the confinement model means by a wall, and it is the
  one the manuscript quotes.

Usage: coordination_shell.py [TAG [clamp]]     (slomd env: parmed)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import parmed

ROOT = Path(_REPO)
NF = ROOT / 'results/sapt_bio/native_fragment'
ENS = ROOT / 'results/ensemble_fluctuation'
OUT = ROOT / 'results/coordination_shell.json'
SHELLS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'WT'
    clamp = sys.argv[2] if len(sys.argv) > 2 else 'r255'
    prm = parmed.load_file(str(ROOT / 'md/mcpb/SLO_sub_solv.prmtop'))
    # element from MASS, not element_name: parmed reports MCPB metal-site atoms as yttrium
    mass = np.array([a.mass for a in prm.atoms])
    heavy = mass >= 2.0
    resn = np.array([a.residue.name for a in prm.atoms])
    substrate, acceptor = resn == 'LIG', resn == 'OH1'
    external = heavy & ~substrate & ~acceptor

    idx = json.loads((NF / f'{tag}_fragment_indices.json').read_text())
    xh = [h['idx'] for h in idx['hydrogens'] if h['transferring']][0]
    X = np.load(ENS / f'{tag}_{clamp}_frames.npy')
    n = X.shape[0]

    pops = {'substrate_own_atoms': substrate & heavy,
            'acceptor_oxygen': acceptor & heavy,
            'external_wall': external}
    res, who = {}, Counter()
    counts = {s: [] for s in SHELLS}
    for lbl, m in pops.items():
        dd = []
        for f in range(n):
            P = X[f].astype(float)
            d = np.linalg.norm(P - P[xh], axis=1)
            d[xh] = 1e9
            d[~m] = 1e9
            dd.append(d.min())
            if lbl == 'external_wall':
                a = prm.atoms[int(np.argmin(d))]
                who[f'{a.residue.name}{a.residue.number}:{a.name}'] += 1
        dd = np.array(dd)
        res[lbl] = dict(mean=float(dd.mean()), sd=float(dd.std()),
                        min=float(dd.min()), max=float(dd.max()))
    for f in range(n):
        P = X[f].astype(float)
        d = np.linalg.norm(P - P[xh], axis=1)
        d[xh] = 1e9
        d[~heavy] = 1e9
        for s in SHELLS:
            counts[s].append(int((d < s).sum()))

    print(f'{tag} {clamp}, {n} configurations. Transferring H, heavy-atom environment.\n')
    for lbl, v in res.items():
        print(f'  nearest {lbl:22s} {v["mean"]:.3f} +/- {v["sd"]:.3f} A  '
              f'[{v["min"]:.2f}-{v["max"]:.2f}]')
    print('\n  identity of the external wall atom:')
    for k, c in who.most_common(4):
        print(f'    {k:20s} {c}/{n}')
    print(f'\n  {"shell (A)":>10}{"mean # heavy atoms":>22}')
    for s in SHELLS:
        print(f'  {s:>10.1f}{np.mean(counts[s]):>22.2f}')

    c15 = np.array(counts[1.5])
    six = int((c15 >= 6).sum())
    print(f'\n  A benzene-ring pocket requires six heavy atoms near 1.31 A.')
    print(f'  configurations with >= 6 heavy atoms within 1.5 A : {six}/{n}')
    print(f'  most heavy atoms ever within 1.5 A                : {int(c15.max())} '
          f'(the covalently bonded donor carbon)')
    assert six == 0, 'a six-coordinate sub-1.5 A cage was found; the model must be revisited'

    OUT.write_text(json.dumps(dict(
        tag=tag, clamp=clamp, n_configurations=n, populations=res,
        wall_identity=dict(who.most_common()),
        mean_counts={str(s): float(np.mean(counts[s])) for s in SHELLS},
        frames_with_six_within_1p5A=six,
        max_heavy_within_1p5A=int(c15.max())), indent=1))
    print(f'\nwrote {OUT}')
    os._exit(0)


if __name__ == '__main__':
    main()
