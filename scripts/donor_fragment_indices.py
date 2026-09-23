#!/usr/bin/env python3
"""Emit the topology indices of the C5H8 donor fragment, once, for reuse across frames.

WHY THIS EXISTS
  Ensemble F-SAPT needs the donor fragment rebuilt at every frame, but the bond graph does
  not change between frames, only the coordinates do. Walking it once and storing indices is
  both cheaper and safer than rewalking per frame, where a distorted frame could in principle
  produce a different distance-based bond assignment.

  It also crosses an environment boundary: parmed lives in the slomd environment and psi4 in
  pauli, so the walk and the quantum chemistry cannot run in the same interpreter. This
  script is the slomd half, and it writes a JSON the pauli half consumes.

WHAT IS EMITTED
  keep      indices of the five carbons, donor carbon first
  hydrogens indices of the hydrogens bonded to those carbons, transferring hydrogen flagged
  caps      (host, beyond) index pairs; the cap hydrogen is placed 1.09 A from host along
            the host-to-beyond direction, recomputed per frame because that direction moves

Usage: donor_fragment_indices.py TAG     (slomd env: parmed)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import parmed
from openmm import app, unit

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
MD, UMB = ROOT / 'md/mcpb', ROOT / 'results/umbrella'
OUT = ROOT / 'results/sapt_bio/native_fragment'


def neighbours(a):
    return [(b.atom2 if b.atom1 is a else b.atom1) for b in a.bonds]


def main():
    tag = sys.argv[1]
    src = (ROOT / 'scripts/extract_native_fragment.py').read_text()
    ns = {}
    exec(src[src.index('ARCH = '):src.index('}\n', src.index('SYSTEMS = {')) + 1], {}, ns)
    info = ns['SYSTEMS'][tag]
    prm = parmed.load_file(str(MD / f'{info["prm"]}.prmtop'))
    dC, xH = prm.atoms[info['donor_C_idx']], prm.atoms[info['xferH_idx']]
    assert xH in neighbours(dC), 'xferH is not bonded to donor_C'

    flank = [o for o in neighbours(dC) if o.element_name == 'C']
    assert len(flank) == 2, f'donor has {len(flank)} carbon neighbours'
    keep, caps = [dC], []
    for f in flank:
        nH = [o for o in neighbours(f) if o.element_name == 'H']
        nC = [o for o in neighbours(f) if o.element_name == 'C']
        assert len(nH) == 1 and len(nC) == 2, f'{f.name} is not sp2 vinyl CH'
        far = [o for o in nC if o is not dC][0]
        assert len([o for o in neighbours(far) if o.element_name == 'H']) == 1
        beyond = [o for o in neighbours(far) if o.element_name == 'C' and o is not f][0]
        keep += [f, far]
        caps.append((far.idx, beyond.idx))

    hyd = [(o.idx, o.idx == xH.idx) for c in keep
           for o in neighbours(c) if o.element_name == 'H']
    nC, nH = len(keep), len(hyd) + len(caps)
    assert (nC, nH) == (5, 8), f'fragment is C{nC}H{nH}, expected C5H8'
    assert sum(1 for _, t in hyd if t) == 1, 'transferring hydrogen not uniquely identified'

    # sanity: the walked indices must reproduce the stored single-snapshot fragment
    pos = np.array([[v.x, v.y, v.z] for v in app.AmberInpcrdFile(
        str(UMB / info['seed'])).positions.value_in_unit(unit.angstrom)])
    ref = json.loads((OUT / f'{tag}_pentadienyl.json').read_text())['donor_fragment']
    refpos = np.array([a['position'] for a in ref if not a['name'].startswith('Hcap')])
    mine = np.array([pos[c.idx] for c in keep] + [pos[i] for i, _ in hyd])
    worst = max(float(np.min(np.linalg.norm(refpos - m, axis=1))) for m in mine)
    assert worst < 1e-6, f'walked fragment differs from the stored one by {worst:.2e} A'

    out = dict(TAG=tag, prm=info['prm'], donor_C=dC.idx, xferH=xH.idx,
               carbons=[c.idx for c in keep],
               hydrogens=[{'idx': i, 'transferring': t} for i, t in hyd],
               caps=[{'host': h, 'beyond': b} for h, b in caps],
               formula=f'C{nC}H{nH}')
    (OUT / f'{tag}_fragment_indices.json').write_text(json.dumps(out, indent=1))
    print(f'{tag}: C{nC}H{nH}  carbons={out["carbons"]}  caps={out["caps"]}')
    print(f'  verified against the stored fragment to {worst:.1e} A')
    print(f'wrote {OUT}/{tag}_fragment_indices.json')
    os._exit(0)


if __name__ == '__main__':
    main()
