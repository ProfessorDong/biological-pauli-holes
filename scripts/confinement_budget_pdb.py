#!/usr/bin/env python3
"""Confinement budget of a transferring hydrogen, evaluated on crystal structures.

Companion to confinement_budget.py, which does the same on the SLO molecular-dynamics
geometries. The weight is the one Eq. (3) of the manuscript assigns to a closed-shell
neighbor at distance d from the transferring hydrogen,

    w_i = exp(-2 d_i / a0),      a0 = 0.5292 A,

and the question is which neighbors carry it.

THE HYDROGEN-PLACEMENT PROBLEM, AND HOW IT IS HANDLED
  A budget built on distances from the transferring hydrogen needs that hydrogen's
  position, and X-ray structures do not resolve it. Two placements are therefore
  evaluated for every system and BOTH are reported:

    'observed'    the hydrogen as modeled in the deposited structure. Available only
                  for neutron structures, and in 4PDJ it is NOT the transferring
                  hydride. The single modeled C4N hydrogen, H41N, refines 0.2 degrees
                  out of the C3N-C4N-C5N plane with a C3N-C4N-C5N angle of 119.6 deg
                  and a ring planar to 0.002 A rms: that is sp2, an oxidized
                  (aromatic) nicotinamide, not the pyramidal sp3 of a hydride donor.
                  The RCSB entry title indeed describes the complex as folate and
                  NADP+, although the coordinate file names the ligands DHF and NDP.
                  It is therefore used only as a SENSITIVITY TEST: a hydrogen roughly
                  an angstrom away from the constructed one, to see whether the budget
                  depends on placement. It is never called the hydride.
    'on-axis'     placed on the donor-acceptor vector at the standard bond length from
                  the donor (1.09 A for C-H, 0.97 A for O-H). This is a construction,
                  not a measurement, and is labeled as such.

  The two differ by roughly an angstrom, which the exponential turns into more than an
  order of magnitude in individual weights. If the qualitative budget survives both,
  it does not depend on the placement. That comparison is the point of running both.

CATEGORIES
  Assignment is by residue, never by distance, so that no cutoff enters:
    donor framework    other atoms of the residue carrying the donor heavy atom
    acceptor group     the residue carrying the acceptor heavy atom
    protein            every amino-acid residue, which is where cavity-lining side
                       chains live
    water              solvent
    other ligand       anything else, e.g. a second cofactor or a bound ion

Usage: confinement_budget_pdb.py     (pauli env; no external dependencies)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
OUT = ROOT / 'results' / 'confinement_budget_crossenzyme.json'

A0 = 0.529177210903
CLOSED_SHELL = {'C', 'N', 'O', 'S', 'P'}
AA = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS',
      'MET','PHE','PRO','SER','THR','TRP','TYR','VAL','MSE','SEC','PYL'}
WATER = {'HOH','DOD','WAT','TIP3','D2O'}
BOND = {'C': 1.09, 'N': 1.01, 'O': 0.97, 'S': 1.34}

SYSTEMS = [
    dict(tag='DHFR', pdb='pauli_data/dhfr/4PDJ_neutron.pdb',
         method='joint X-ray / neutron',
         donor=('NDP', None, 'C4N'), acceptor=('DHF', None, 'C6'),
         observed_H='H41N',
         note='hydride transfer, NADPH C4N to dihydrofolate C6'),
] + [
    dict(tag=f'KSI-{c}', pdb='pauli_data/ksi/2PZV.pdb', method='X-ray', chain=c,
         donor=('IPH', None, 'O1'), acceptor=('ASP', '103', 'OD2'),
         observed_H=None,
         note='oxyanion-hole O-H...O; phenol is a ground-state analog and the '
              'proton position is not determined by X-ray, so only the on-axis '
              'construction is available')
    for c in ('A', 'B', 'C', 'D')
]


def parse_pdb(path):
    """Minimal fixed-column PDB reader. Keeps the highest-occupancy altloc only."""
    atoms = []
    for l in Path(path).read_text().split('\n'):
        if not l.startswith(('ATOM', 'HETATM')):
            continue
        el = l[76:78].strip().upper()
        if not el:                                  # fall back to the atom name
            el = l[12:16].strip()[0]
        atoms.append(dict(
            name=l[12:16].strip(), alt=l[16].strip(), res=l[17:20].strip(),
            chain=l[21], seq=l[22:27].strip(), el=el,
            occ=float(l[54:60]) if l[54:60].strip() else 1.0,
            xyz=np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    best = {}
    for a in atoms:
        key = (a['chain'], a['seq'], a['res'], a['name'])
        if key not in best or a['occ'] > best[key]['occ']:
            best[key] = a
    return list(best.values())


def find(atoms, res, seq, name, chain=None):
    hits = [a for a in atoms if a['res'] == res and a['name'] == name
            and (seq is None or a['seq'] == seq)
            and (chain is None or a['chain'] == chain)]
    if len(hits) != 1:
        raise SystemExit(f'expected 1 atom {res}:{name} (chain {chain}), '
                         f'found {len(hits)}')
    return hits[0]


def budget(atoms, hpos, donor, acceptor):
    """Exponential weight budget about a hydrogen at hpos."""
    dkey = (donor['chain'], donor['seq'], donor['res'])
    akey = (acceptor['chain'], acceptor['seq'], acceptor['res'])
    cat_w, cat_d, per = defaultdict(float), defaultdict(lambda: np.inf), []
    for a in atoms:
        if a['el'] not in CLOSED_SHELL:
            continue
        if a is donor:                              # the donor itself is not a wall
            continue
        d = float(np.linalg.norm(a['xyz'] - hpos))
        if d > 12.0:
            continue
        w = float(np.exp(-2.0 * d / A0))
        key = (a['chain'], a['seq'], a['res'])
        if key == dkey:
            c = 'donor framework'
        elif key == akey:
            c = 'acceptor group'
        elif a['res'] in WATER:
            c = 'water'
        elif a['res'] in AA:
            c = 'protein'
        else:
            c = 'other ligand'
        cat_w[c] += w
        cat_d[c] = min(cat_d[c], d)
        per.append((d, w, c, f"{a['res']}{a['seq']}:{a['name']}"))
    tot = sum(cat_w.values())
    per.sort(key=lambda r: -r[1])
    return dict(
        total=tot,
        categories={c: dict(fraction=cat_w[c] / tot, d_min_A=cat_d[c]) for c in cat_w},
        cumulative={f'{R:.1f}': sum(w for d, w, _, _ in per if d <= R) / tot
                    for R in (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)},
        top=[dict(atom=n, d_A=d, fraction=w / tot, category=c)
             for d, w, c, n in per[:8]],
        # weight beyond 8 A, to show the answer does not depend on any cutoff
        tail_beyond_8A=sum(w for d, w, _, _ in per if d > 8.0) / tot)


def main():
    res = {}
    for S in SYSTEMS:
        atoms = parse_pdb(ROOT / S['pdb'])
        ch = S.get('chain')
        don = find(atoms, *S['donor'], chain=ch)
        acc = find(atoms, *S['acceptor'], chain=ch)
        rDA = float(np.linalg.norm(don['xyz'] - acc['xyz']))
        print(f"\n{S['tag']}  ({S['pdb'].split('/')[-1]}, {S['method']})")
        print(f"  {S['note']}")
        print(f"  donor {don['res']}:{don['name']}  acceptor {acc['res']}:{acc['name']}"
              f"  r_DA = {rDA:.3f} A")

        placements = {}
        if S.get('observed_H'):
            h = find(atoms, S['donor'][0], S['donor'][1], S['observed_H'],
                     chain=ch)
            v1 = h['xyz'] - don['xyz']
            v2 = acc['xyz'] - don['xyz']
            ang = np.degrees(np.arccos(np.dot(v1, v2) /
                                       (np.linalg.norm(v1) * np.linalg.norm(v2))))
            placements['observed'] = h['xyz']
            print(f"  observed H {S['observed_H']}: d(donor) = "
                  f"{np.linalg.norm(v1):.3f} A, {ang:.1f} deg off the D-A axis")
        u = (acc['xyz'] - don['xyz'])
        u = u / np.linalg.norm(u)
        placements['on-axis'] = don['xyz'] + BOND[don['el']] * u
        print(f"  on-axis construction: {BOND[don['el']]:.2f} A from the donor "
              f"along D->A")

        S_out = dict(pdb=S['pdb'], method=S['method'], note=S['note'],
                     r_DA_A=rDA, placements={})
        for pname, hpos in placements.items():
            b = budget(atoms, hpos, don, acc)
            S_out['placements'][pname] = b
            print(f"\n  [{pname}]  total W = {b['total']:.4e}  "
                  f"(weight beyond 8 A: {b['tail_beyond_8A']:.2e} of total)")
            for c, v in sorted(b['categories'].items(),
                               key=lambda kv: -kv[1]['fraction']):
                print(f"     {c:18s} {100*v['fraction']:9.5f} %   "
                      f"nearest {v['d_min_A']:5.2f} A")
            print(f"     cumulative within 2.5 A: "
                  f"{100*b['cumulative']['2.5']:.2f} %,  "
                  f"3.0 A: {100*b['cumulative']['3.0']:.2f} %")
        res[S['tag']] = S_out

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
