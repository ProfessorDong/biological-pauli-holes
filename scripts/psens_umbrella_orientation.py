#!/usr/bin/env python3
"""Which C11 hydrogen faces the acceptor in the umbrella windows and in the L754A starting structures.

For every variant and replica, the final frame of each umbrella window gives the C-H...O angle at
each C11 hydrogen (H25 = pro-S, H26 = pro-R; checked by CIP in psens_proS_summary.py's notes) and its
distance to the acceptor oxygen. Also reported: the L754A tleap start (WT pose) and the equilibrated
structure that seeds all three L754A replicas.
Writes results/protonation_sensitivity/proS/umbrella_final_frames_orientation.json.
Usage: psens_umbrella_orientation.py      (slomd env)
"""
import glob
import json
from pathlib import Path

import numpy as np
import parmed
from parmed.amber import Rst7

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/protonation_sensitivity/proS/umbrella_final_frames_orientation.json'
TOP = {'L754A': 'SLO_L754A_sub_solv', 'WT': 'SLO_sub_solv', 'I553A': 'SLO_I553A_sub_solv',
       'I538A': 'SLO_I538A_sub_solv', 'I552A': 'SLO_I552A_sub_solv', 'L546A': 'SLO_L546A_sub_solv',
       'V750A': 'SLO_V750A_sub_solv'}


def atoms(prm):
    s = parmed.load_file(str(prm))
    lig = [r for r in s.residues if r.name == 'LIG'][0]
    A = {a.name: a.idx for a in lig.atoms}
    O = [a.idx for a in s.atoms if a.residue.name == 'OH1' and a.name == 'O'][0]
    return A, O


def geom(X, A, O):
    g = {}
    for h, lab in (('H25', 'proS'), ('H26', 'proR')):
        u, v = X[A['C14']] - X[A[h]], X[O] - X[A[h]]
        g[lab] = [float(np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))), float(np.linalg.norm(v))]
    return g


def main():
    out = {}
    for tag, base in TOP.items():
        A, O = atoms(ROOT / f'md/mcpb/{base}.prmtop')
        for rep in ('', '_rep2', '_rep3'):
            files = sorted(glob.glob(str(ROOT / f'results/umbrella/{tag}{rep}/win_*_final.rst7')),
                           key=lambda p: -float(p.split('win_')[1].split('_')[0]))
            out[f'{tag}{rep or "_rep1"}'] = [[f.split('win_')[1].split('_')[0],
                                              geom(Rst7.open(f).coordinates.reshape(-1, 3), A, O)] for f in files]
    A, O = atoms(ROOT / 'md/mcpb/SLO_L754A_sub_solv.prmtop')
    out['_L754A_structures'] = {
        'tleap_start': geom(Rst7.open(str(ROOT / 'md/mcpb/SLO_L754A_sub_solv.inpcrd')).coordinates.reshape(-1, 3), A, O),
        'equilibrated': geom(Rst7.open(str(ROOT / 'md/mcpb/SLO_L754A_sub_eq.rst7')).coordinates.reshape(-1, 3), A, O)}
    OUT.write_text(json.dumps(out, indent=1))
    for k, rows in out.items():
        if k.startswith('_'):
            print(k, json.dumps(rows))
            continue
        short = [g for r0, g in rows if float(r0) <= 3.21]
        print(f'{k:12s} {len(rows):2d} windows; pro-R nearer in {sum(g["proR"][1] < g["proS"][1] for _, g in rows):2d}; '
              f'shortest three: pro-S facing {sum(g["proS"][0] > 120 for g in short)}, pro-R facing {sum(g["proR"][0] > 120 for g in short)}')


if __name__ == '__main__':
    main()
