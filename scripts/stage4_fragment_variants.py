#!/usr/bin/env python3
"""Stage 4 checks 5 and 6: build the LARGER wall and donor fragments.

The published panel truncates the wall at the side chain (severing CB-CA, capped with H) and the
donor at C5H8 (severing at the two far vinyl carbons, capped with H). The pre-registered stage-4
plan asks whether either truncation is load-bearing. This script builds:

  wall_backbone : side chain + backbone N, CA, C, O of the same residue. The severed peptide
                  bonds are capped with H (N-H 1.01 A, C-H 1.09 A), which leaves a neutral
                  closed-shell H2N-CH(R)-CHO fragment.
  donor_shell2  : C5H8 extended one bond further along the chain in both directions, i.e. the
                  next conjugation shell, capped the same way the original capped C5.

Every fragment is checked for even electron count, no atom pair closer than 0.85 A, and retention
of the transferring hydrogen, exactly as extract_native_donor_clamped.py checks its own.

Usage: stage4_fragment_variants.py TAG [CLAMP]
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, sys
from pathlib import Path
import numpy as np, parmed
from openmm import app, unit

ROOT = Path(_REPO)
RG, MD = ROOT / 'results/reactive_geometry', ROOT / 'md/mcpb'
OUT = ROOT / 'results/native_donor_validation'
Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}


def nbrs(a):
    return [(b.atom2 if b.atom1 is a else b.atom1) for b in a.bonds]


def cap_H(pos, keep_atom, severed_atom, length):
    v = pos[severed_atom.idx] - pos[keep_atom.idx]
    return pos[keep_atom.idx] + v / np.linalg.norm(v) * length


def validate(atoms, label, xferH_name=None):
    ne = sum(Z[e] for e, _, _ in atoms)
    assert ne % 2 == 0, f'{label}: odd electron count {ne}'
    P = np.array([p for _, p, _ in atoms])
    dmin = min(np.linalg.norm(P[i] - P[j])
               for i in range(len(P)) for j in range(i + 1, len(P)))
    assert dmin > 0.85, f'{label}: two atoms {dmin:.3f} A apart'
    if xferH_name:
        assert any(n == xferH_name for _, _, n in atoms), f'{label}: transferring H dropped'
    f = {}
    for e, _, _ in atoms:
        f[e] = f.get(e, 0) + 1
    formula = ''.join(f'{k}{v}' for k, v in sorted(f.items()))
    print(f'  {label:14s} {len(atoms):3d} atoms  {formula:12s} electrons {ne}  dmin {dmin:.3f} A')
    return formula


def main():
    tag = sys.argv[1]
    clamp = sys.argv[2] if len(sys.argv) > 2 else 'r255'
    src = (ROOT / 'scripts/extract_native_fragment.py').read_text()
    ns = {}
    exec(src[src.index('ARCH = '):src.index('}\n', src.index('SYSTEMS = {')) + 1], {}, ns)
    info = ns['SYSTEMS'][tag]

    base = json.loads((OUT / f'{tag}_{clamp}_native_donor.json').read_text())
    prm = parmed.load_file(str(MD / f'{info["prm"]}.prmtop'))
    rst = app.AmberInpcrdFile(str(RG / f'{tag}_{clamp}_final.rst7'))
    pos = np.array([[v.x, v.y, v.z] for v in rst.positions.value_in_unit(unit.angstrom)])

    dC, xH = prm.atoms[info['donor_C_idx']], prm.atoms[info['xferH_idx']]
    off = float(np.linalg.norm(np.array(base['xferH']) - pos[xH.idx]))
    assert off < 1e-6, f'restart is a different frame ({off:.4f} A); would confound geometry'
    print(f'{tag} {clamp}: wall {base["wall_residue"]}, frame verified to {off:.1e} A')

    # ---- check 5: wall side chain + backbone -------------------------------------------
    res = next(r for r in prm.residues
               if r.name == info['wall_res_name'] and r.number == info['wall_res_id'])
    by = {a.name: a for a in res.atoms}
    want = [n for n in ('N', 'H', 'CA', 'HA', 'C', 'O') if n in by]
    sc = [a for a in res.atoms if a.name not in ('N', 'H', 'CA', 'HA', 'C', 'O')]
    wb = [(by[n].element_name, pos[by[n].idx], by[n].name) for n in want]
    wb += [(a.element_name, pos[a.idx], a.name) for a in sc]
    # cap the two severed peptide bonds
    N, C = by['N'], by['C']
    prevC = [o for o in nbrs(N) if o.element_name == 'C' and o not in res.atoms]
    nextN = [o for o in nbrs(C) if o.element_name == 'N' and o not in res.atoms]
    assert len(prevC) == 1 and len(nextN) == 1, 'residue is not mid-chain'
    wb.append(('H', cap_H(pos, N, prevC[0], 1.01), 'HcapN'))
    wb.append(('H', cap_H(pos, C, nextN[0], 1.09), 'HcapC'))
    f_wb = validate(wb, 'wall_backbone')

    # ---- check 6: donor extended one conjugation shell ---------------------------------
    flank = [o for o in nbrs(dC) if o.element_name == 'C']
    keep, caps = [dC], []
    for fl in flank:
        far = [o for o in nbrs(fl) if o.element_name == 'C' and o is not dC][0]
        beyond = [o for o in nbrs(far) if o.element_name == 'C' and o is not fl][0]
        keep += [fl, far, beyond]                     # one shell further than the published C5
        nxt = [o for o in nbrs(beyond) if o.element_name == 'C' and o is not far]
        assert len(nxt) == 1, f'{beyond.name} has {len(nxt)} onward carbons'
        caps.append((beyond, nxt[0]))
    d2 = []
    for c in keep:
        d2.append((c.element_name, pos[c.idx], c.name))
        d2 += [('H', pos[o.idx], o.name) for o in nbrs(c) if o.element_name == 'H']
    for far, beyond in caps:
        d2.append(('H', cap_H(pos, far, beyond, 1.09), f'Hcap{far.name}'))
    f_d2 = validate(d2, 'donor_shell2', xH.name)

    base['wall_backbone'] = [dict(element=e, position=p.tolist(), name=n) for e, p, n in wb]
    base['wall_backbone_formula'] = f_wb
    base['donor_shell2'] = [dict(element=e, position=p.tolist(), name=n) for e, p, n in d2]
    base['donor_shell2_formula'] = f_d2
    p = OUT / f'{tag}_{clamp}_stage4_fragments.json'
    p.write_text(json.dumps(base, indent=2))
    print(f'  -> {p.name}')


if __name__ == '__main__':
    main()
