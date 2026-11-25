#!/usr/bin/env python3
"""Build the model-P topology for one SLO variant (protonation sensitivity analysis).

Model P differs from the production model S only in
  * Glu349, Glu752 -> GLH;  His248 -> HIP;  Lys526, Lys741 -> LYN  (PROPKA3, pH 9)
  * caps on the chain ends that exist only because residues are unresolved in 1F8N:
    ACE before His6, Asn31 and Gly121; NME after Asn18 and Ala116.
See results/protonation_sensitivity/DECLARATION.md.

Input : the model-S replica-1 umbrella window at 2.70 A (final frame), which seeded
        every model-S clamped run, and the model-S topology of the same variant.
Output: results/protonation_sensitivity/build/<TAG>/  (PDBs, tleap input and log,
        model-P prmtop/inpcrd, build_report.json)

Residues are addressed by their zero-based model-S index; crystal number = index + 22
for residues 482-839, and for the residues changed here:
  His6 0, Asn18 12, Asn31 13, Ala116 98, Gly121 99, His248 226, Glu349 327,
  Lys526 504, Lys741 719, Glu752 730.
Usage: psens_build_model_p.py TAG        (slomd env)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import parmed

ROOT = Path(os.environ.get('PAULI_ROOT') or Path(__file__).resolve().parents[1])
MCPB = ROOT / 'md/mcpb'
SUB = ROOT / 'md/substrate'
PRMTOP = {'WT': 'SLO_sub_solv.prmtop'}
TAGS = ('WT', 'I538A', 'I552A', 'I553A', 'L546A', 'L754A', 'V750A')

RENAME = {226: ('HIE', 'HIP'), 327: ('GLU', 'GLH'), 730: ('GLU', 'GLH'),
          504: ('LYS', 'LYN'), 719: ('LYS', 'LYN')}
ACE_BEFORE = (0, 13, 99)      # chain starts: drop H1-H3, add ACE
NME_AFTER = (12, 98)          # chain ends: drop OXT, add NME
EXPECT = {0: 'HIE', 12: 'ASN', 13: 'ASN', 98: 'ALA', 99: 'GLY'}


def unit(v):
    return v / np.linalg.norm(v)


def place(a, b, c, bond, angle_deg, dihedral_deg):
    """Position of d bonded to c, with angle b-c-d and dihedral a-b-c-d (NeRF)."""
    ang, dih = np.radians(angle_deg), np.radians(dihedral_deg)
    bc = unit(c - b)
    n = unit(np.cross(b - a, bc))
    m = np.cross(n, bc)
    d2 = np.array([-bond * np.cos(ang), bond * np.sin(ang) * np.cos(dih),
                   bond * np.sin(ang) * np.sin(dih)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def dihedral(a, b, c, d):
    b0, b1, b2 = a - b, c - b, d - c
    b1 = unit(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def main():
    tag = sys.argv[1]
    assert tag in TAGS, tag
    out = ROOT / 'results/protonation_sensitivity/build' / tag
    out.mkdir(parents=True, exist_ok=True)
    prm = MCPB / PRMTOP.get(tag, f'SLO_{tag}_sub_solv.prmtop')
    seed = ROOT / 'results/umbrella' / tag / 'win_2.70_final.rst7'
    s = parmed.load_file(str(prm), str(seed))
    X = np.array(s.coordinates)
    box = [float(v) for v in s.box[:3]]
    assert all(abs(float(v) - 90.0) < 1e-6 for v in s.box[3:]), 'rectangular box expected'

    res = s.residues
    for i, nm in EXPECT.items():
        assert res[i].name == nm, (tag, i, res[i].name)
    for i, (old, new) in RENAME.items():
        assert res[i].name == old, (tag, i, res[i].name)
    assert res[817].name == 'IE1' and res[818].name == 'FE1' and res[819].name == 'OH1'
    assert res[820].name == 'LIG' and len(res) > 821

    def xyz(r, name):
        return X[[a.idx for a in res[r].atoms if a.name == name][0]]

    protein_heavy = np.array([a.idx for a in s.atoms
                              if a.residue.idx < 820 and a.atomic_number > 1])
    report = dict(tag=tag, seed=str(seed.relative_to(ROOT)), prmtop_S=prm.name, box_A=box,
                  caps={}, removed_atoms=[], renamed={}, waters_removed=0)

    # ---- hydrogen choices on the changed residues ------------------------------------
    acceptors = np.array([a.idx for a in s.atoms if a.atomic_number in (7, 8)
                          and a.residue.name not in ('Na+',)])
    donors_h = np.array([a.idx for a in s.atoms if a.atomic_number == 1
                         and any(b.atomic_number in (7, 8) for b in a.bond_partners)])
    drop = set()
    # Lys -> LYN: remove the hydrogen with the weakest hydrogen-bond partner, name the rest HZ2/HZ3
    lyn_names = {}
    for ri in (504, 719):
        hz = [a for a in res[ri].atoms if a.name in ('HZ1', 'HZ2', 'HZ3')]
        own = {a.idx for a in res[ri].atoms}
        acc = np.array([j for j in acceptors if j not in own])
        dmin = [np.min(np.linalg.norm(X[acc] - X[h.idx], axis=1)) for h in hz]
        gone = hz[int(np.argmax(dmin))]
        drop.add(gone.idx)
        keep = [h for h in hz if h is not gone]
        lyn_names[ri] = {keep[0].idx: 'HZ2', keep[1].idx: 'HZ3'}
        report['removed_atoms'].append(f'{res[ri].name}{ri}:{gone.name}')
        report['renamed'][f'LYN{ri}'] = {gone.name: 'removed', keep[0].name: 'HZ2', keep[1].name: 'HZ3',
                                          'nearest_acceptor_A': [round(float(v), 3) for v in dmin]}
    # Glu -> GLH: the template protonates OE2; put the proton on the oxygen that accepts fewer
    # hydrogen bonds by swapping the oxygen names when needed
    glh_swap = {}
    for ri in (327, 730):
        oe = {a.name: a for a in res[ri].atoms if a.name in ('OE1', 'OE2')}
        own = {a.idx for a in res[ri].atoms}
        hs = np.array([j for j in donors_h if j not in own])
        nacc = {n: int(np.sum(np.linalg.norm(X[hs] - X[a.idx], axis=1) < 2.5)) for n, a in oe.items()}
        swap = nacc['OE2'] > nacc['OE1']
        glh_swap[ri] = swap
        report['renamed'][f'GLH{ri}'] = dict(hbonds_accepted=nacc, protonated='OE1' if swap else 'OE2')
    # chain starts lose H1-H3; chain ends lose OXT
    for ri in ACE_BEFORE:
        for a in res[ri].atoms:
            if a.name in ('H1', 'H2', 'H3'):
                drop.add(a.idx)
    for ri in NME_AFTER:
        for a in res[ri].atoms:
            if a.name == 'OXT':
                drop.add(a.idx)

    # ---- cap heavy atoms ----------------------------------------------------------------
    caps = {}
    for ri in ACE_BEFORE:
        N, CA = xyz(ri, 'N'), xyz(ri, 'CA')
        best = None
        for hn in ('H1', 'H2', 'H3'):
            C = N + 1.335 * unit(xyz(ri, hn) - N)
            CH3 = place(CA, N, C, 1.522, 116.2, 180.0)
            O = place(CA, N, C, 1.229, 122.9, 0.0)
            other = np.array([j for j in protein_heavy if s.atoms[j].residue.idx != ri])
            clear = min(np.min(np.linalg.norm(X[other] - p, axis=1)) for p in (C, CH3, O))
            if best is None or clear > best[0]:
                best = (clear, hn, dict(C=C, O=O, CH3=CH3))
        caps[('ACE', ri)] = best[2]
        report['caps'][f'ACE before {ri}'] = dict(direction_of=best[1], clearance_A=round(float(best[0]), 3))
    for ri in NME_AFTER:
        C, O, CA, OXT = xyz(ri, 'C'), xyz(ri, 'O'), xyz(ri, 'CA'), xyz(ri, 'OXT')
        N = C + 1.335 * unit(OXT - C)
        CH3 = place(CA, C, N, 1.449, 121.9, 180.0)
        other = np.array([j for j in protein_heavy if s.atoms[j].residue.idx != ri])
        clear = min(np.min(np.linalg.norm(X[other] - p, axis=1)) for p in (N, CH3))
        caps[('NME', ri)] = dict(N=N, C=CH3)
        report['caps'][f'NME after {ri}'] = dict(clearance_A=round(float(clear), 3),
                                                  omega_deg=round(float(dihedral(CA, C, N, CH3)), 1))

    # ---- waters that overlap a cap -------------------------------------------------------
    cap_xyz = np.array([p for d in caps.values() for p in d.values()])
    drop_res = set()
    for r in res[821:]:
        if r.name == 'WAT':
            o = X[[a.idx for a in r.atoms if a.name == 'O'][0]]
            if np.min(np.linalg.norm(cap_xyz - o, axis=1)) < 3.0:
                drop_res.add(r.idx)
        else:
            assert r.name == 'Na+', r.name
            ion = X[r.atoms[0].idx]
            assert np.min(np.linalg.norm(cap_xyz - ion, axis=1)) > 3.0, 'ion overlaps a cap'
    report['waters_removed'] = len(drop_res)

    # ---- write PDBs ----------------------------------------------------------------------
    def fmt(serial, name, resn, resi, p, elem):
        nm = name if len(name) == 4 else ' ' + name
        return (f'ATOM  {serial % 100000:5d} {nm:<4s} {resn:>3s}  {resi % 10000:4d}    '
                f'{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          {elem:>2s}\n')

    lines, serial, resi = [], 1, 0
    newnum = {}

    def write_res(r):
        nonlocal serial, resi
        resi += 1
        newnum[r.idx] = resi
        name = RENAME[r.idx][1] if r.idx in RENAME else r.name
        for a in r.atoms:
            if a.idx in drop:
                continue
            an = a.name
            if r.idx in lyn_names and a.idx in lyn_names[r.idx]:
                an = lyn_names[r.idx][a.idx]
            if r.idx in glh_swap and glh_swap[r.idx] and an in ('OE1', 'OE2'):
                an = 'OE2' if an == 'OE1' else 'OE1'
            elem = a.element_name if a.atomic_number > 0 else 'EP'
            if a.residue.name in ('FE1',):
                elem = 'FE'
            lines.append(fmt(serial, an, name, resi, X[a.idx], elem))
            serial += 1

    def write_cap(kind, ri):
        nonlocal serial, resi
        resi += 1
        for an, p in caps[(kind, ri)].items():
            lines.append(fmt(serial, an, kind, resi, p, an[0]))
            serial += 1

    for r in res[:820]:
        if r.idx in ACE_BEFORE:
            write_cap('ACE', r.idx)
        write_res(r)
        if r.idx in NME_AFTER:
            write_cap('NME', r.idx)
            lines.append('TER\n')
        if r.idx in (817, 818):
            lines.append('TER\n')
    lines.append('END\n')
    (out / 'P_solute.pdb').write_text(''.join(lines))

    lig = res[820]
    (out / 'P_lig.pdb').write_text(''.join(
        fmt(k + 1, a.name, 'LIG', 1, X[a.idx], a.element_name) for k, a in enumerate(lig.atoms)) + 'END\n')

    sl, k, ri = [], 1, 0
    for r in res[821:]:
        if r.idx in drop_res:
            continue
        ri += 1
        for a in r.atoms:
            elem = 'NA' if r.name == 'Na+' else ('EP' if a.name == 'EPW' else a.element_name)
            sl.append(fmt(k, a.name, r.name, ri, X[a.idx], elem))
            k += 1
        sl.append('TER\n')
    sl.append('END\n')
    (out / 'P_solvent.pdb').write_text(''.join(sl))

    # ---- tleap: model-S inputs, renumbered bonds, solvent and box from the frame ---------
    def n(old_1based):
        return newnum[old_1based - 1]

    bonds = [(478, 'NE2', 819, 'FE'), (483, 'NE2', 819, 'FE'), (669, 'NE2', 819, 'FE'),
             (818, 'OXT', 819, 'FE'), (819, 'FE', 820, 'O'), (477, 'C', 478, 'N'),
             (478, 'C', 479, 'N'), (482, 'C', 483, 'N'), (483, 'C', 484, 'N'),
             (668, 'C', 669, 'N'), (669, 'C', 670, 'N'), (817, 'C', 818, 'N')]
    leap = [
        'source leaprc.protein.ff19SB', 'source leaprc.water.opc', 'source leaprc.gaff2',
        f'HD1 = loadmol2 {MCPB}/HD1.mol2', f'HD2 = loadmol2 {MCPB}/HD2.mol2',
        f'HD3 = loadmol2 {MCPB}/HD3.mol2', f'IE1 = loadmol2 {MCPB}/IE1.mol2',
        f'FE1 = loadmol2 {MCPB}/FE1.mol2', f'OH1 = loadmol2 {MCPB}/OH1.mol2',
        f'LIG = loadmol2 {SUB}/LIG_final.mol2',
        'loadamberparams frcmod.ionslm_126_opc', f'loadamberparams {MCPB}/SLO_mcpbpy.frcmod',
        f'loadamberparams {SUB}/LIG_final.frcmod',
        f'mol = loadpdb {out}/P_solute.pdb']
    leap += [f'bond mol.{n(a)}.{an} mol.{n(b)}.{bn}' for a, an, b, bn in bonds]
    leap += [f'lig = loadpdb {out}/P_lig.pdb', 'dry = combine {mol lig}',
             f'saveamberparm dry {out}/P_dry.prmtop {out}/P_dry.inpcrd',
             f'solv = loadpdb {out}/P_solvent.pdb', 'complex = combine {mol lig solv}',
             f'set complex box {{{box[0]:.6f} {box[1]:.6f} {box[2]:.6f}}}',
             'charge dry', 'charge complex',
             f'saveamberparm complex {out}/P_solv.prmtop {out}/P_solv.inpcrd', 'quit']
    (out / 'tleap_P.in').write_text('\n'.join(leap) + '\n')
    env = dict(os.environ, AMBERHOME=sys.prefix, PATH=os.path.join(sys.prefix, 'bin') + ':' + os.environ.get('PATH', ''))
    log = subprocess.run([os.path.join(sys.prefix, 'bin', 'tleap'), '-f', str(out / 'tleap_P.in')], cwd=out,
                         capture_output=True, text=True, env=env)
    (out / 'tleap_P.log').write_text(log.stdout + log.stderr)
    report['renumbered_bonds'] = [f'{n(a)}.{an}-{n(b)}.{bn}' for a, an, b, bn in bonds]
    report['tleap_returncode'] = log.returncode
    (out / 'build_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('caps', 'removed_atoms', 'waters_removed',
                                             'tleap_returncode')}, indent=1))


if __name__ == '__main__':
    main()
