#!/usr/bin/env python3
"""Stage 2: locate a transfer geometry on the potential surface, not from the isotope effects.

WHY THIS EXISTS
  Every configuration used so far came from a harmonic restraint on the donor-acceptor distance,
  and the restraint centre was itself obtained by inverting a Franck-Condon model against the same
  kinetic isotope effects the endpoints are then regressed against. That circularity is named in
  the paper. The only way out is to let the electronic structure choose the geometry.

WHAT IT DOES
  A constrained optimization at each of several fixed C-H distances on the 80-atom active-site
  cluster (C25FeH43N7O4, charge +1, multiplicity 6, the high-spin d5 Fe(III) state), at UKS
  B3LYP-D3BJ/def2-SVP. Only r(C-H) is constrained. The donor-acceptor distance, the C-H...O angle
  and the O-H distance are all FREE and relax to whatever the surface prefers, which is exactly
  what the audit asked for: a model that resolves C-H, O-H and C-O rather than imposing them.

  A reactive core of 19 atoms relaxes: the transferring hydrogen, the donor carbon and its two
  flanking vinyl carbons with their hydrogens, the hydroxide oxygen and its hydrogen, the iron,
  its other first-shell oxygens and a first-shell nitrogen. The remaining 61 atoms, including all
  seven valence caps, are frozen in Cartesian space so the cluster cannot reorganize globally.
  This is a frozen-environment relaxed scan and is stated as such; it is not a full optimization.

  The energy maximum along the path is the transfer geometry, and the r_DA it relaxes to is the
  quantity this stage exists to produce. Nothing in it uses a kinetic isotope effect.

Usage: transfer_geometry_scan.py write            build the ORCA inputs
       transfer_geometry_scan.py collect          parse finished jobs
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
SRC = ROOT / 'results/pcet_reactant_v2/wt_win270_pt00.inp'
OUT = ROOT / 'results/transfer_geometry'
XFER_H, DONOR_C, ACC_O, ACC_H, FE = 0, 65, 61, 62, 60
RCH = [1.10, 1.18, 1.26, 1.34, 1.42, 1.50, 1.60]
CORE_CUT = 3.5


def read_cluster():
    lines = SRC.read_text().split('\n')
    i = next(k for k, l in enumerate(lines) if l.startswith('* xyz'))
    at = []
    for l in lines[i + 1:]:
        if l.strip().startswith('*'):
            break
        p = l.split()
        if len(p) == 4:
            at.append([p[0], np.array([float(x) for x in p[1:]])])
    assert len(at) == 80, len(at)
    assert at[DONOR_C][0] == 'C' and at[ACC_O][0] == 'O' and at[FE][0] == 'Fe'
    return at


def core_indices(at):
    P = np.array([a[1] for a in at])
    mid = (P[DONOR_C] + P[ACC_O]) / 2
    sel = {k for k in range(len(at)) if np.linalg.norm(P[k] - mid) < CORE_CUT}
    return sorted(sel | {XFER_H, FE, ACC_O, ACC_H, DONOR_C})


def write():
    at = read_cluster()
    core = core_indices(at)
    frozen = [k for k in range(len(at)) if k not in set(core)]
    OUT.mkdir(parents=True, exist_ok=True)
    u = at[ACC_O][1] - at[DONOR_C][1]
    u = u / np.linalg.norm(u)
    for r in RCH:
        g = [[e, p.copy()] for e, p in at]
        g[XFER_H][1] = at[DONOR_C][1] + u * r          # start on the C->O axis
        tag = f'rch{int(round(r*100)):03d}'
        L = ['! UKS B3LYP def2-SVP D3BJ TightSCF SlowConv Opt',
             '%pal nprocs 4 end',
             '%scf maxiter 300 end',
             '%geom',
             '  MaxIter 80',
             '  Constraints',
             f'    {{ B {DONOR_C} {XFER_H} {r:.4f} C }}']
        L += [f'    {{ C {k} C }}' for k in frozen]
        L += ['  end', 'end', '* xyz 1 6']
        L += [f'{e}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}' for e, p in g]
        L.append('*')
        (OUT / f'{tag}.inp').write_text('\n'.join(L) + '\n')
    (OUT / 'scan_plan.json').write_text(json.dumps(dict(
        source=str(SRC), n_atoms=len(at), charge=1, mult=6,
        method='UKS B3LYP-D3BJ/def2-SVP', xferH=XFER_H, donor_C=DONOR_C,
        acceptor_O=ACC_O, Fe=FE, core_cut_A=CORE_CUT,
        relaxed_atoms=core, n_relaxed=len(core), n_frozen=len(frozen),
        constrained='r(C-H) only; r_DA, C-H...O angle and r(O-H) all free',
        r_CH_grid=RCH), indent=2))
    print(f'wrote {len(RCH)} inputs to {OUT}')
    print(f'  {len(core)} atoms relaxed, {len(frozen)} frozen')
    print(f'  constrained: r(C{DONOR_C}-H{XFER_H}) only')


def collect():
    rows = []
    for r in RCH:
        tag = f'rch{int(round(r*100)):03d}'
        o = OUT / f'{tag}.out'
        if not o.exists():
            continue
        txt = o.read_text(errors='ignore')
        done = 'HURRAY' in txt or 'THE OPTIMIZATION HAS CONVERGED' in txt
        es = [float(l.split()[4]) for l in txt.split('\n')
              if l.startswith('FINAL SINGLE POINT ENERGY')]
        # ORCA writes {tag}.xyz only on convergence; fall back to the last trajectory
        # frame so a slowly converging run can still be read, flagged as unconverged.
        xyz = OUT / f'{tag}.xyz'
        trj = OUT / f'{tag}_trj.xyz'
        geo = None
        src_xyz = xyz if xyz.exists() else (trj if trj.exists() else None)
        if src_xyz is not None:
            ls = src_xyz.read_text().split('\n')
            nat = int(ls[0].split()[0])
            nfr = max(1, len([1 for l in ls if l.strip().isdigit()]))
            off = (nfr - 1) * (nat + 2) + 2 if src_xyz is trj else 2
            P = np.array([[float(x) for x in ls[off + k].split()[1:4]]
                          for k in range(nat)])
            dC, hH, aO = P[DONOR_C], P[XFER_H], P[ACC_O]
            v1, v2 = dC - hH, aO - hH
            geo = dict(
                r_CH=float(np.linalg.norm(hH - dC)),
                r_OH=float(np.linalg.norm(hH - aO)),
                r_DA=float(np.linalg.norm(aO - dC)),
                angle_CHO=float(np.degrees(np.arccos(
                    v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))))
        rows.append(dict(tag=tag, r_CH_target=r, converged=done,
                         E_hartree=es[-1] if es else None, n_steps=len(es), **(geo or {})))
    if not rows:
        print('no finished jobs yet'); return
    ok = [r for r in rows if r['E_hartree'] is not None and 'r_DA' in r]
    if ok:
        E0 = min(r['E_hartree'] for r in ok)
        for r in ok:
            r['rel_kcal'] = (r['E_hartree'] - E0) * 627.5094740631
        print(f'{"target":>7s} {"r_CH":>6s} {"r_OH":>6s} {"r_DA":>6s} {"angle":>7s} '
              f'{"dE(kcal)":>9s} {"steps":>6s} conv')
        for r in sorted(ok, key=lambda x: x['r_CH_target']):
            print(f'{r["r_CH_target"]:7.2f} {r["r_CH"]:6.3f} {r["r_OH"]:6.3f} {r["r_DA"]:6.3f} '
                  f'{r["angle_CHO"]:7.1f} {r["rel_kcal"]:9.2f} {r["n_steps"]:6d} '
                  f'{"yes" if r["converged"] else "NO"}')
        top = max(ok, key=lambda r: r['rel_kcal'])
        print(f'\nbarrier maximum along the scan: {top["rel_kcal"]:.2f} kcal/mol at '
              f'r_CH = {top["r_CH"]:.3f} A, r_DA = {top["r_DA"]:.3f} A, '
              f'angle = {top["angle_CHO"]:.1f} deg')
    (OUT / 'scan_result.json').write_text(json.dumps(rows, indent=2))
    print(f'wrote {OUT}/scan_result.json')


if __name__ == '__main__':
    (write if sys.argv[1] == 'write' else collect)()
