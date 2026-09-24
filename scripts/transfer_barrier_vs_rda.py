#!/usr/bin/env python3
"""Stage 2, corrected: the hydrogen-transfer barrier as a function of donor-acceptor distance.

WHY THE FIRST ATTEMPT WAS WRONG
  Scanning r(C-H) with only that distance constrained looked like the right way to let the surface
  choose a transfer geometry. It is not, because 61 of the 80 cluster atoms were frozen in
  Cartesian space, which pins the substrate backbone and the iron ligand shell and so pins r_DA.
  The scan duly reported that r_DA EXPANDS from 3.443 to 3.666 Angstrom as the hydrogen is pulled
  off, which is the scaffold speaking, not the chemistry. It could say that transfer does not occur
  at the equilibrium geometry; it could not say where transfer becomes competent.

WHAT THIS DOES INSTEAD
  Constraining r_DA is legitimate: the protein scaffold really does restrain it. What must not be
  imposed is its VALUE. So r_DA is constrained across a range and, at each value, two states are
  optimized:

    reactant   r_DA fixed, r(C-H) = 1.10 Angstrom, angle and r(O-H) free
    symmetric  r_DA fixed, r(C-H) = r(O-H) = r_DA/2 + 0.05, the bent transfer-like configuration

  The barrier at that distance is E(symmetric) - E(reactant), and the distance at which it becomes
  thermally accessible is the transfer-competent geometry. Nothing uses a kinetic isotope effect.

WHY THE DIFFERENCE IS THE RIGHT OBSERVABLE
  Forcing r_DA away from its equilibrium value strains the frozen scaffold, and that strain
  contaminates absolute energies. It does not contaminate the barrier: both states at a given r_DA
  carry the same scaffold strain, so it cancels in the difference. Absolute energies are therefore
  NOT comparable across r_DA and are not reported as such; only the per-distance barrier is.

Usage: transfer_barrier_vs_rda.py write | collect      (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(_REPO)
SRC = ROOT / 'results/pcet_reactant_v2/wt_win270_pt00.inp'
OUT = ROOT / 'results/transfer_barrier'
GUESS = ROOT / 'results/pcet_reactant_v2/wt_win270_pt00.gbw'
XFER_H, DONOR_C, ACC_O, ACC_H, FE = 0, 65, 61, 62, 60
RDA = [2.60, 2.75, 2.90, 3.05, 3.20]
CORE_CUT = 4.0


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
    assert len(at) == 80 and at[DONOR_C][0] == 'C' and at[ACC_O][0] == 'O'
    return at


def write():
    at = read_cluster()
    P = np.array([a[1] for a in at])
    mid = (P[DONOR_C] + P[ACC_O]) / 2
    core = sorted({k for k in range(80) if np.linalg.norm(P[k] - mid) < CORE_CUT}
                  | {XFER_H, FE, ACC_O, ACC_H, DONOR_C})
    frozen = [k for k in range(80) if k not in set(core)]
    OUT.mkdir(parents=True, exist_ok=True)
    u = (P[ACC_O] - P[DONOR_C]) / np.linalg.norm(P[ACC_O] - P[DONOR_C])

    for rda in RDA:
        v = rda / 2 + 0.05                       # symmetric, bent: 2v > r_DA guarantees a triangle
        ang = np.degrees(np.arccos(1 - rda ** 2 / (2 * v ** 2)))
        for state, cons, rch in (('react', [f'{{ B {DONOR_C} {XFER_H} 1.1000 C }}'], 1.10),
                                 ('symm', [f'{{ B {DONOR_C} {XFER_H} {v:.4f} C }}',
                                           f'{{ B {ACC_O} {XFER_H} {v:.4f} C }}'], v)):
            g = [[e, p.copy()] for e, p in at]
            # pull the acceptor in along the donor->acceptor axis to the target r_DA
            g[ACC_O][1] = P[DONOR_C] + u * rda
            g[ACC_H][1] = P[ACC_H] + (g[ACC_O][1] - P[ACC_O])
            g[XFER_H][1] = P[DONOR_C] + u * rch
            tag = f'rda{int(round(rda*100)):03d}_{state}'
            L = ['! UKS B3LYP def2-SVP D3BJ TightSCF SlowConv Opt MORead',
                 f'%moinp "{GUESS.name}"',
                 '%pal nprocs 3 end',
                 '%scf maxiter 300 end',
                 '%geom', '  MaxIter 60', '  Constraints',
                 f'    {{ B {DONOR_C} {ACC_O} {rda:.4f} C }}']
            L += ['    ' + c for c in cons]
            L += [f'    {{ C {k} C }}' for k in frozen]
            L += ['  end', 'end', '* xyz 1 6']
            L += [f'{e}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}' for e, p in g]
            L.append('*')
            (OUT / f'{tag}.inp').write_text('\n'.join(L) + '\n')
        print(f'  r_DA {rda:.2f}: symmetric v = {v:.3f} A, implied C-H...O angle {ang:.1f} deg')

    (OUT / 'plan.json').write_text(json.dumps(dict(
        source=str(SRC), charge=1, mult=6, method='UKS B3LYP-D3BJ/def2-SVP',
        r_DA_grid=RDA, core_cut_A=CORE_CUT, n_relaxed=len(core), n_frozen=len(frozen),
        relaxed_atoms=core,
        design='r_DA constrained at each value; barrier = E(symm) - E(react) at the SAME r_DA so '
               'frozen-scaffold strain cancels; absolute energies not comparable across r_DA'),
        indent=2))
    print(f'wrote {2*len(RDA)} inputs to {OUT}; {len(core)} relaxed, {len(frozen)} frozen')


def geom(tag):
    for f in (OUT / f'{tag}.xyz', OUT / f'{tag}_trj.xyz'):
        if not f.exists():
            continue
        ls = f.read_text().split('\n')
        nat = int(ls[0].split()[0])
        nfr = len([1 for l in ls if l.strip().isdigit()])
        off = (nfr - 1) * (nat + 2) + 2 if f.name.endswith('_trj.xyz') else 2
        P = np.array([[float(x) for x in ls[off + k].split()[1:4]] for k in range(nat)])
        v1, v2 = P[DONOR_C] - P[XFER_H], P[ACC_O] - P[XFER_H]
        return dict(r_CH=float(np.linalg.norm(v1)), r_OH=float(np.linalg.norm(v2)),
                    r_DA=float(np.linalg.norm(P[ACC_O] - P[DONOR_C])),
                    angle=float(np.degrees(np.arccos(
                        v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))))
    return None


def collect():
    rows = []
    for rda in RDA:
        rec = dict(r_DA_target=rda)
        for state in ('react', 'symm'):
            tag = f'rda{int(round(rda*100)):03d}_{state}'
            o = OUT / f'{tag}.out'
            if not o.exists():
                continue
            txt = o.read_text(errors='ignore')
            es = [float(l.split()[4]) for l in txt.split('\n')
                  if l.startswith('FINAL SINGLE POINT ENERGY')]
            rec[state] = dict(E=es[-1] if es else None, n=len(es),
                              converged='THE OPTIMIZATION HAS CONVERGED' in txt,
                              **(geom(tag) or {}))
        if 'react' in rec and 'symm' in rec and rec['react'].get('E') and rec['symm'].get('E'):
            rec['barrier_kcal'] = (rec['symm']['E'] - rec['react']['E']) * 627.5094740631
        rows.append(rec)
    done = [r for r in rows if 'barrier_kcal' in r]
    if done:
        print(f'{"r_DA":>6s} {"barrier":>9s} {"r_CH(s)":>8s} {"r_OH(s)":>8s} {"angle(s)":>9s} '
              f'{"cyc R/S":>9s}  conv')
        for r in done:
            s, a = r['symm'], r['react']
            print(f'{r["r_DA_target"]:6.2f} {r["barrier_kcal"]:9.2f} {s.get("r_CH",0):8.3f} '
                  f'{s.get("r_OH",0):8.3f} {s.get("angle",0):9.1f} '
                  f'{a["n"]:4d}/{s["n"]:<4d} '
                  f'{"Y" if a["converged"] else "n"}{"Y" if s["converged"] else "n"}')
        lo = min(done, key=lambda r: r['barrier_kcal'])
        print(f'\nlowest barrier {lo["barrier_kcal"]:.2f} kcal/mol at r_DA = '
              f'{lo["r_DA_target"]:.2f} A')
    else:
        print('no completed pairs yet')
    (OUT / 'barrier_result.json').write_text(json.dumps(rows, indent=2))
    print(f'wrote {OUT}/barrier_result.json')


if __name__ == '__main__':
    (write if sys.argv[1] == 'write' else collect)()
