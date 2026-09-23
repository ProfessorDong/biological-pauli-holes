#!/usr/bin/env python3
"""Extract the native C5H8 donor fragment at the CLAMPED geometries.

WHY THIS EXISTS
  The coordinate comparison of the paper computes K_sep and K_perp with a methane
  surrogate at the donor carbon. The 2026-09-18 revision audit's first validation item is a
  MATCHED native-donor comparison: the same substrate fragment, the same wall definition and
  the same configurations for both curvatures. The native C5H8 donor already exists, but only
  at the equilibrium near-attack seeds (extract_pentadienyl_donor.py), while the clamped
  geometries carry the acceptor oxygen that defines the transverse plane. Neither set alone
  supports a matched comparison at the tested geometries, so this script builds the native
  donor on the clamped frames.

WHAT IT GUARANTEES
  The donor carbon and transferring hydrogen are asserted against the stored clamped geometry,
  so a wrong restart file cannot silently substitute a different frame. The bond walk is the
  same one extract_pentadienyl_donor.py uses and carries the same chemical assertions: two sp2
  vinyl CH flanks, C5H8 stoichiometry, even electron count, no atom pair closer than 0.85 A,
  and the transferring hydrogen retained.

Usage: extract_native_donor_clamped.py TAG CLAMP      (slomd env: parmed + openmm)
"""
import json
import sys
from pathlib import Path

import numpy as np
import parmed
from openmm import app, unit

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
MD, RG = ROOT / 'md/mcpb', ROOT / 'results/reactive_geometry'
OUT = ROOT / 'results/native_donor_validation'


def neighbours(a):
    return [(b.atom2 if b.atom1 is a else b.atom1) for b in a.bonds]


def main():
    tag = sys.argv[1]
    clamp = sys.argv[2] if len(sys.argv) > 2 else 'r255'
    OUT.mkdir(parents=True, exist_ok=True)

    src = (ROOT / 'scripts/extract_native_fragment.py').read_text()
    ns = {}
    exec(src[src.index('ARCH = '):src.index('}\n', src.index('SYSTEMS = {')) + 1], {}, ns)
    info = ns['SYSTEMS'][tag]

    base = json.loads((RG / f'{tag}_{clamp}_geometry.json').read_text())
    prm = parmed.load_file(str(MD / f'{info["prm"]}.prmtop'))
    rst = app.AmberInpcrdFile(str(RG / f'{tag}_{clamp}_final.rst7'))
    pos = np.array([[v.x, v.y, v.z] for v in rst.positions.value_in_unit(unit.angstrom)])

    dC, xH = prm.atoms[info['donor_C_idx']], prm.atoms[info['xferH_idx']]
    for key, atom in (('donor_C', dC), ('xferH', xH)):
        off = float(np.linalg.norm(np.array(base[key]) - pos[atom.idx]))
        assert off < 1e-6, (
            f'{key} sits {off:.4f} A from {tag}_{clamp}_geometry.json, so this restart is a '
            f'different frame and the donor swap would confound donor with geometry')

    d = float(np.linalg.norm(pos[dC.idx] - pos[xH.idx]))
    assert 1.0 < d < 1.2 and xH in neighbours(dC), f'donor C-H is {d:.3f} A'

    flank = [o for o in neighbours(dC) if o.element_name == 'C']
    assert len(flank) == 2, f'donor has {len(flank)} carbon neighbours'
    keep, caps = [dC], []
    for f in flank:
        nH = [o for o in neighbours(f) if o.element_name == 'H']
        nC = [o for o in neighbours(f) if o.element_name == 'C']
        assert len(nH) == 1 and len(nC) == 2, f'{f.name} is not sp2 vinyl CH'
        far = [o for o in nC if o is not dC][0]
        assert len([o for o in neighbours(far) if o.element_name == 'H']) == 1
        keep += [f, far]
        caps.append((far, [o for o in neighbours(far)
                           if o.element_name == 'C' and o is not f][0]))

    atoms = []
    for c in keep:
        atoms.append((c.element_name, pos[c.idx], c.name))
        atoms += [('H', pos[o.idx], o.name) for o in neighbours(c) if o.element_name == 'H']
    for far, beyond in caps:
        v = pos[beyond.idx] - pos[far.idx]
        atoms.append(('H', pos[far.idx] + v / np.linalg.norm(v) * 1.09, f'Hcap{far.name}'))

    nC = sum(e == 'C' for e, _, _ in atoms)
    nH = sum(e == 'H' for e, _, _ in atoms)
    assert (nC, nH) == (5, 8), f'fragment is C{nC}H{nH}'
    assert (6 * nC + nH) % 2 == 0, 'odd electron count'
    P = np.array([p for _, p, _ in atoms])
    dmin = min(np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(i + 1, len(P)))
    assert dmin > 0.85, f'two atoms {dmin:.3f} A apart'
    assert any(n == xH.name for _, _, n in atoms), 'transferring H was dropped'

    base['donor_fragment'] = [dict(element=e, position=p.tolist(), name=n) for e, p, n in atoms]
    base['donor_fragment_formula'] = f'C{nC}H{nH}'
    base['xferH_name'] = xH.name
    base['restart'] = f'{tag}_{clamp}_final.rst7'
    (OUT / f'{tag}_{clamp}_native_donor.json').write_text(json.dumps(base, indent=2))
    print(f'{tag} {clamp}: C{nC}H{nH} donor, {len(base["native_wall"])} wall atoms, '
          f'C-H {d:.3f} A, dmin {dmin:.3f} A -> {tag}_{clamp}_native_donor.json')


if __name__ == '__main__':
    main()
