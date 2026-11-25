#!/usr/bin/env python3
"""Which C11 hydrogen faces the acceptor? Orientation of H25 (pro-S) and H26 (pro-R), every variant.

For each variant, model (S, P) and clamp (r255, r340), over the 40 ensemble frames: the C-H...O angle
at the hydrogen (180 deg = pointing at the acceptor oxygen), its median, the fraction of frames above
120 deg, and the mean H...O distance. Writes results/protonation_sensitivity/proS/orientation.json.
Usage: psens_hydrogen_orientation.py      (slomd env)
"""
import json
from pathlib import Path

import numpy as np
import parmed

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / 'results/protonation_sensitivity'
TAGS = ('WT', 'I538A', 'I552A', 'I553A', 'L546A', 'L754A', 'V750A')
out = {}
for t in TAGS:
    for m in ('S', 'P'):
        prm = ROOT / 'md/mcpb' / (('SLO_sub_solv' if t == 'WT' else f'SLO_{t}_sub_solv') + '.prmtop') if m == 'S' \
            else PS / 'build' / t / 'P_solv.prmtop'
        s = parmed.load_file(str(prm))
        lig = [r for r in s.residues if r.name == 'LIG'][0]
        A = {a.name: a.idx for a in lig.atoms}
        O = [a.idx for a in s.atoms if a.residue.name == 'OH1' and a.name == 'O'][0]
        for c in ('r255', 'r340'):
            X = np.load((ROOT / 'results/ensemble_fluctuation' if m == 'S' else PS / 'ensemble_fluctuation') / f'{t}_{c}_frames.npy',
                        mmap_mode='r')
            rec = {}
            for h in ('H25', 'H26'):
                th, r = [], []
                for f in range(X.shape[0]):
                    P = np.asarray(X[f], float)
                    u, v = P[A['C14']] - P[A[h]], P[O] - P[A[h]]
                    th.append(float(np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))))
                    r.append(float(np.linalg.norm(v)))
                rec[h] = dict(median_angle=float(np.median(th)), frac_above_120=float(np.mean(np.array(th) > 120)),
                              mean_r_HO=float(np.mean(r)), n=len(th))
            out[f'{t}_{m}_{c}'] = rec
(PS / 'proS/orientation.json').write_text(json.dumps(out, indent=1))
for k, v in out.items():
    if k.endswith('r340'):
        print(f'{k:14s} H25 {v["H25"]["median_angle"]:5.0f} deg {v["H25"]["mean_r_HO"]:.2f} A | H26 {v["H26"]["median_angle"]:5.0f} deg {v["H26"]["mean_r_HO"]:.2f} A')
