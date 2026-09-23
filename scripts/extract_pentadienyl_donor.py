#!/usr/bin/env python3
"""Extract the REAL bis-allylic pentadiene donor fragment from the MD snapshot.

WHY
  Every k_exch^bio in this paper uses a constructed methane centred at the donor carbon.
  Li, Soudackov and Hammes-Schiffer (JACS 140, 3068 (2018)) criticize the gas-phase model of
  Champion and co-workers, methane donor against an [OH]^delta- acceptor, as "too small to
  fully describe the interface between the substrate, which has a pi backbone, and the iron
  cofactor". That criticism is about the donor-acceptor coordinate, not the transverse
  donor-to-wall curvature this project measures, so it motivates the present check without
  being tested by it. This replaces the methane by the actual reactive motif taken from the
  same snapshot so that the surrogate's cost can be measured rather than assumed.

WHAT IS EXTRACTED
  The bond graph is walked from the donor carbon, not hardcoded:
      Cb = C=C - C(H)(H*) - C=C  ->  cis,cis-1,4-pentadiene, C5H8, neutral, closed shell
  The two terminal sp2 carbons are capped with hydrogen along the broken C-C bond at
  1.09 A. The transferring hydrogen is retained. Everything else is unchanged.

VERIFICATION PERFORMED HERE
  * the donor carbon is bonded to the transferring hydrogen (1.0-1.2 A)
  * both flanking carbons are sp2 (exactly one H, two C) and doubly bonded onward
  * the assembled fragment is C5H8, an even electron count, hence closed shell
  * no extracted atom lies unphysically close to any other

Usage: extract_pentadienyl_donor.py TAG    (slomd env: parmed + openmm)
"""
import json, sys
from pathlib import Path
import numpy as np
import parmed
from openmm import app, unit

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
MD = ROOT/'md/mcpb'; UMB = ROOT/'results/umbrella'
OUT = ROOT/'results/sapt_bio/native_fragment'
SYS = json.loads((ROOT/'scripts/_sapt_systems.json').read_text()) if (ROOT/'scripts/_sapt_systems.json').exists() else None


def neighbours(a):
    return [(b.atom2 if b.atom1 is a else b.atom1) for b in a.bonds]


def main():
    tag = sys.argv[1]
    # reuse the exact system table the original extraction used
    src = (ROOT/'scripts/extract_native_fragment.py').read_text()
    ns = {}
    # slice from ARCH, not from SYSTEMS: three seeds are built from that archive prefix
    exec(src[src.index('ARCH = '):src.index('}\n', src.index('SYSTEMS = {'))+1], {}, ns)
    info = ns['SYSTEMS'][tag]
    prm = parmed.load_file(str(MD/f'{info["prm"]}.prmtop'))
    seed = app.AmberInpcrdFile(str(UMB/info['seed']))
    pos = np.array([[v.x, v.y, v.z] for v in seed.positions.value_in_unit(unit.angstrom)])
    dC = prm.atoms[info['donor_C_idx']]; xH = prm.atoms[info['xferH_idx']]

    # The donor swap is only a controlled comparison if the pentadienyl fragment is built
    # from the SAME frame as the methane fragment it is compared against. Nothing else in
    # this script would notice a different frame: any equilibrated frame passes every
    # geometric assertion below, because they test chemistry, not identity. So compare the
    # walked reactive atoms against the ones already stored for the methane calculation.
    base = json.loads((OUT/f'{tag}_geometry.json').read_text())
    for key, atom in (('donor_C', dC), ('xferH', xH)):
        off = float(np.linalg.norm(np.array(base[key]) - pos[atom.idx]))
        assert off < 1e-6, (
            f'{key} sits {off:.3f} A from the value in {tag}_geometry.json. The seed '
            f'{info["seed"]} is a different frame, so the donor swap would confound the '
            f'change of donor with a change of geometry.')

    d = np.linalg.norm(pos[dC.idx]-pos[xH.idx])
    assert 1.0 < d < 1.2, f'donor_C is not bonded to xferH ({d:.3f} A)'
    assert xH in neighbours(dC), 'xferH not in donor_C bond list'
    print(f'{tag}: donor C = {dC.name} (idx {dC.idx}), xferH = {xH.name}, C-H = {d:.3f} A')

    # the two flanking carbons must be sp2 vinyl CH
    flank = [o for o in neighbours(dC) if o.element_name == 'C']
    assert len(flank) == 2, f'donor has {len(flank)} carbon neighbours, expected 2'
    keep, caps = [dC], []
    for f in flank:
        nH = [o for o in neighbours(f) if o.element_name == 'H']
        nC = [o for o in neighbours(f) if o.element_name == 'C']
        assert len(nH) == 1 and len(nC) == 2, f'{f.name} is not sp2 vinyl CH'
        far = [o for o in nC if o is not dC][0]          # the doubly bonded partner
        nHf = [o for o in neighbours(far) if o.element_name == 'H']
        nCf = [o for o in neighbours(far) if o.element_name == 'C']
        assert len(nHf) == 1, f'{far.name} is not sp2'
        keep += [f, far]
        beyond = [o for o in nCf if o is not f][0]        # where the chain continues
        caps.append((far, beyond))
        print(f'   flank {f.name} (1 H) = {far.name} (1 H), chain continues to {beyond.name} -> capped')

    atoms = []
    for c in keep:
        atoms.append((c.element_name, pos[c.idx], c.name))
        for o in neighbours(c):
            if o.element_name == 'H':
                atoms.append(('H', pos[o.idx], o.name))
    for far, beyond in caps:
        v = pos[beyond.idx]-pos[far.idx]; v /= np.linalg.norm(v)
        atoms.append(('H', pos[far.idx]+v*1.09, f'Hcap{far.name}'))

    nC = sum(1 for e, _, _ in atoms if e == 'C'); nH = sum(1 for e, _, _ in atoms if e == 'H')
    nelec = 6*nC + nH
    assert (nC, nH) == (5, 8), f'fragment is C{nC}H{nH}, expected C5H8'
    assert nelec % 2 == 0, 'odd electron count: not closed shell'
    P = np.array([p for _, p, _ in atoms])
    dmin = min(np.linalg.norm(P[i]-P[j]) for i in range(len(P)) for j in range(i+1, len(P)))
    assert dmin > 0.85, f'two atoms {dmin:.3f} A apart'
    assert any(n == xH.name for _, _, n in atoms), 'transferring H was dropped'
    print(f'   fragment C{nC}H{nH}, {nelec} electrons (even -> closed shell), '
          f'min interatomic distance {dmin:.3f} A, transferring H retained')

    base['donor_fragment'] = [dict(element=e, position=p.tolist(), name=n) for e, p, n in atoms]
    base['donor_fragment_formula'] = f'C{nC}H{nH}'
    (OUT/f'{tag}_pentadienyl.json').write_text(json.dumps(base, indent=2))
    print(f'wrote {OUT}/{tag}_pentadienyl.json')


if __name__ == '__main__':
    main()
