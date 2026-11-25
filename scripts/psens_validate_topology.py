#!/usr/bin/env python3
"""Compare a model-P topology with the model-S topology it was built from.

Every atom outside the residues that model P changes must carry the same charge, atom
type, mass and Lennard-Jones parameters, and keep its coordinates from the seed frame.
Every bonded term among unchanged atoms, including the whole MCPB metal site, must be
identical. Differences are allowed only in the declared residues:
  0, 12, 13, 98, 99 (terminal -> internal templates), 226 (HIP), 327 and 730 (GLH),
  504 and 719 (LYN), the five caps, and waters removed for the caps.
Usage: psens_validate_topology.py TAG      (slomd env)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import parmed

ROOT = Path(os.environ.get('PAULI_ROOT') or Path(__file__).resolve().parents[1])
CHANGED = {0, 12, 13, 98, 99, 226, 327, 730, 504, 719}


def main():
    tag = sys.argv[1]
    b = ROOT / 'results/protonation_sensitivity/build' / tag
    rep = json.loads((b / 'build_report.json').read_text())
    S = parmed.load_file(str(ROOT / 'md/mcpb' / rep['prmtop_S']), str(ROOT / rep['seed']))
    P = parmed.load_file(str(b / 'P_solv.prmtop'), str(b / 'P_solv.inpcrd'))
    SX, PX = np.array(S.coordinates), np.array(P.coordinates)   # cached: the property rebuilds the array
    # tleap writes the box with its origin at the corner: a rigid shift of half a box length.
    # Estimate it from the Lys7 CA, apply it to model S, and require it to be uniform below.
    ca_s = [a.idx for a in S.residues[1].atoms if a.name == 'CA'][0]
    ca_p = [a.idx for a in [r for r in P.residues if r.name not in ('ACE', 'NME')][1].atoms if a.name == 'CA'][0]
    shift = PX[ca_p] - SX[ca_s]
    SX = SX + shift
    L = np.array(P.box[:3], float)

    def mi(d):
        """minimum-image displacement: tleap may also wrap molecules into the box"""
        return d - L * np.round(d / L)

    # residue map S -> P: solute in order with the caps skipped, then LIG, then kept solvent
    pres = [r for r in P.residues if r.name not in ('ACE', 'NME')]
    sres = list(S.residues)
    nsol_S = 821
    kept_solvent = [r for r in sres[nsol_S:]]
    # waters removed for caps are absent from P; align solvent by the water oxygen coordinate
    psolv = pres[nsol_S:]
    assert len(sres) - len(psolv) - nsol_S == rep['waters_removed'], 'solvent count mismatch'
    rmap = {i: pres[i] for i in range(nsol_S)}
    j = 0
    for r in kept_solvent:
        a0 = r.atoms[0]
        q = psolv[j]
        if np.linalg.norm(mi(SX[a0.idx] - PX[q.atoms[0].idx])) < 5e-3:   # PDB coordinates carry 3 decimals
            rmap[r.idx] = q
            j += 1
    assert j == len(psolv), (j, len(psolv))

    issues, n_atoms, n_changed = [], 0, 0
    amap = {}
    for si, pr in rmap.items():
        sr = sres[si]
        names = {a.name: a for a in pr.atoms}
        for a in sr.atoms:
            q = names.get(a.name)
            if si in CHANGED:
                n_changed += 1
                if q is not None:
                    amap[a.idx] = q.idx
                continue
            n_atoms += 1
            if q is None:
                issues.append(f'missing {sr.name}{si}:{a.name}')
                continue
            amap[a.idx] = q.idx
            if abs(a.charge - q.charge) > 1e-6 or a.type != q.type or abs(a.mass - q.mass) > 1e-6 \
                    or abs(a.rmin - q.rmin) > 1e-6 or abs(a.epsilon - q.epsilon) > 1e-6:
                issues.append(f'param {sr.name}{si}:{a.name} q {a.charge:.4f}/{q.charge:.4f} '
                              f'type {a.type}/{q.type}')
            if np.linalg.norm(mi(SX[a.idx] - PX[q.idx])) > 5e-3:
                issues.append(f'coord {sr.name}{si}:{a.name}')
        if si not in CHANGED and len(sr.atoms) != len(pr.atoms):
            issues.append(f'atom count {sr.name}{si}: {len(sr.atoms)} vs {len(pr.atoms)}')

    def terms(struct, kind, idxmap=None):
        out = {}
        for t in getattr(struct, kind):
            ats = [getattr(t, f'atom{k}') for k in range(1, 5) if hasattr(t, f'atom{k}')]
            key = tuple(a.idx for a in ats)
            if idxmap is not None:
                if any(k not in idxmap for k in key):
                    continue
                if any(a.residue.idx in CHANGED for a in ats):
                    continue
                key = tuple(idxmap[k] for k in key)
            tp = t.type
            if kind == 'bonds':
                val = (round(tp.k, 4), round(tp.req, 4))
            elif kind == 'angles':
                val = (round(tp.k, 4), round(tp.theteq, 4))
            else:
                val = (round(tp.phi_k, 4), round(tp.per, 4), round(tp.phase, 4))
            out.setdefault(tuple(sorted(key)) if kind == 'bonds' else key, []).append(val)
        return out

    bonded = {}
    for kind in ('bonds', 'angles', 'dihedrals'):
        s_t = terms(S, kind, amap)
        p_t = terms(P, kind)
        miss = 0
        for k, v in s_t.items():
            kk = k if k in p_t else tuple(reversed(k))
            if kk not in p_t or sorted(p_t[kk]) != sorted(v):
                miss += 1
        bonded[kind] = dict(checked=len(s_t), mismatched=miss)
        if miss:
            issues.append(f'{kind}: {miss} of {len(s_t)} terms differ')

    metal = [a for a in P.atoms if a.residue.name in ('FE1', 'OH1', 'HD1', 'HD2', 'HD3', 'IE1')]
    solute_q = sum(a.charge for a in P.atoms if a.residue.name not in ('WAT', 'Na+'))
    out = dict(tag=tag, rigid_shift_A=[round(float(v), 4) for v in shift], unchanged_atoms_checked=n_atoms, atoms_in_changed_residues=n_changed,
               bonded_terms=bonded, metal_site_atoms=len(metal),
               solute_charge=round(solute_q, 4), total_charge=round(sum(a.charge for a in P.atoms), 4),
               n_Na=sum(1 for r in P.residues if r.name == 'Na+'),
               P_residues={i: P.residues[i].name for i in range(0, 6)},
               changed_residue_names=[pres[i].name for i in sorted(CHANGED)],
               issues=issues[:20], n_issues=len(issues), passed=not issues)
    (b / 'topology_validation.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if not issues else 1)


if __name__ == '__main__':
    main()
