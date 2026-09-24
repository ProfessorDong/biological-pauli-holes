#!/usr/bin/env python3
"""Pre-flight the ensemble F-SAPT campaign: check every frame before spending hours on it.

WHY THIS EXISTS
  The campaign is seven systems by two clamps by ten configurations by five displacements,
  which is hundreds of quantum-chemistry calls and hours of wall clock. Every failure mode
  worth worrying about is detectable in milliseconds from the coordinates alone, and one of
  them has already bitten this project once: a fragment built from one frame and scanned
  against a wall from another produced an exchange energy of order 1e-9 hartree and a
  curvature of zero, which reads as a null rather than as a bug.

  So this asserts, for every frame that the campaign will actually use, that the topology
  indices address the atoms the stored geometry says they do, that the fragment is chemically
  intact, and that donor and wall are close enough to exchange at all.

WHAT IS CHECKED, PER FRAME
  1. the coordinate at the stored donor_C index equals the stored donor_C, exactly
  2. likewise for the transferring hydrogen
  3. the donor carbon carries exactly two hydrogens within bonding range
  4. the C-H bond to the transferring hydrogen is physical
  5. the assembled fragment is C5H8 with no atom pair closer than 0.85 A
  6. the nearest donor-to-wall distance is inside exchange range

  A system that fails 6 is reported rather than aborted: an open wall is a physical result,
  not an error, and L754A is expected to sit far out. Failures of 1 to 5 are structural and
  abort the campaign.

Usage: preflight_ensemble.py [n_frames]     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(_REPO)
NF = ROOT / 'results/sapt_bio/native_fragment'
ENS = ROOT / 'results/ensemble_fluctuation'
TAGS = ['WT', 'I553A', 'I552A', 'L754A', 'V750A', 'I538A', 'L546A']
CLAMPS = ['r255', 'r340']
CAP = 1.09
hard, soft = [], []


def build(idx, P):
    dC = idx['donor_C']
    hyd = [h['idx'] for h in idx['hydrogens']]
    on = [i for i in hyd if np.linalg.norm(P[i] - P[dC]) < 1.2]
    atoms = [P[dC]] + [P[i] for i in on]
    atoms += [P[c] for c in idx['carbons'] if c != dC]
    atoms += [P[i] for i in hyd if i not in on]
    for c in idx['caps']:
        v = P[c['beyond']] - P[c['host']]
        atoms.append(P[c['host']] + CAP * v / np.linalg.norm(v))
    return np.array(atoms), len(on)


def main():
    nwant = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f'{"system":>7}{"clamp":>7}{"frames":>8}{"idx ok":>8}{"C5H8":>7}'
          f'{"C-H (A)":>18}{"donor-wall (A)":>20}')
    for tag in TAGS:
        idx = json.loads((NF / f'{tag}_fragment_indices.json').read_text())
        xh = [h['idx'] for h in idx['hydrogens'] if h['transferring']][0]
        for clamp in CLAMPS:
            geo = json.loads((ENS / f'{tag}_{clamp}_frames_geometry.json').read_text())
            X = np.load(ENS / f'{tag}_{clamp}_frames.npy')
            pick = np.linspace(0, geo['n_frames'] - 1, nwant).round().astype(int)
            ch, dw, nidx, nform = [], [], 0, 0
            for fr in pick:
                P = X[fr].astype(float)
                f = geo['frames'][fr]
                a = np.linalg.norm(P[idx['donor_C']] - np.array(f['donor_C']))
                b = np.linalg.norm(P[xh] - np.array(f['xferH']))
                if a < 1e-3 and b < 1e-3:
                    nidx += 1
                else:
                    hard.append(f'{tag}/{clamp}/f{fr}: index mismatch '
                                f'donor_C {a:.3f} A, xferH {b:.3f} A')
                D, non = build(idx, P)
                if non != 2:
                    hard.append(f'{tag}/{clamp}/f{fr}: donor C carries {non} H')
                if len(D) == 13:
                    nform += 1
                else:
                    hard.append(f'{tag}/{clamp}/f{fr}: fragment has {len(D)} atoms')
                dmin_int = min(np.linalg.norm(D[i] - D[j])
                               for i in range(len(D)) for j in range(i + 1, len(D)))
                if dmin_int <= 0.85:
                    hard.append(f'{tag}/{clamp}/f{fr}: internal clash {dmin_int:.3f} A')
                ch.append(float(np.linalg.norm(P[xh] - P[idx['donor_C']])))
                W = np.array([q['position'] for q in f['native_wall']])
                dw.append(float(np.min(np.linalg.norm(D[:, None] - W[None], axis=-1))))
            ch, dw = np.array(ch), np.array(dw)
            if not (0.95 < ch.min() and ch.max() < 1.25):
                hard.append(f'{tag}/{clamp}: C-H out of range {ch.min():.3f}-{ch.max():.3f}')
            far = int((dw >= 6.0).sum())
            if far:
                soft.append(f'{tag}/{clamp}: {far}/{len(dw)} frames beyond exchange range')
            print(f'{tag:>7}{clamp:>7}{len(pick):>8}{nidx:>8}{nform:>7}'
                  f'{ch.min():>9.3f}-{ch.max():<8.3f}{dw.min():>10.2f}-{dw.max():<9.2f}')

    print()
    for s in soft:
        print(f'  note  {s}')
    if hard:
        print(f'\n{len(hard)} STRUCTURAL FAILURES, campaign must not run:')
        for h in hard[:20]:
            print(f'  {h}')
    else:
        print('PRE-FLIGHT CLEAN: every frame addressable, chemically intact, in range')
    sys.exit(len(hard))


if __name__ == '__main__':
    main()
