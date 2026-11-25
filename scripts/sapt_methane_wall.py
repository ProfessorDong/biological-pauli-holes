#!/usr/bin/env python3
"""Methane-wall separation curvature at a near-attack snapshot (surrogate check).

Reimplements the protocol recorded for the lost producer of sapt_bio/sapt_summary.json
(scripts/sapt_campaign_manifest.py): donor = methane constructed at C11 with one C-H pointing at
the transferring hydrogen; wall = methane constructed at the wall atom nearest that hydrogen with
one C-H pointing back at it; the wall methane is translated along the donor-wall axis over
-0.20..+0.20 A in five points; SAPT0/jun-cc-pVDZ, density fitting, frozen core; quadratic fit of
E_exch; k = 2 c2, converted with 0.694770 N/m per kcal/mol/A^2. The methane construction is the
build_methane of sapt_native_fragment.py, unchanged.

Accepted for use only if, with the original hydrogens, it reproduces the seven published values.
Usage: sapt_methane_wall.py TAG OUTDIR      (pauli env; reads native_fragment/<TAG>_geometry.json)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import psi4

ROOT = Path(os.environ.get('PAULI_ROOT') or Path(__file__).resolve().parents[1])
CONV = 0.694770
HA2KCAL = 627.5094740631
DELTAS = [-0.20, -0.10, 0.00, +0.10, +0.20]


def build_methane(center, direction, r=1.09):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    hs = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3.0) * r
    v_from, v_to = hs[0] / r, d
    ax = np.cross(v_from, v_to)
    s, c = np.linalg.norm(ax), np.dot(v_from, v_to)
    if s < 1e-6:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        ax /= s
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * K @ K
    return (R @ hs.T).T + center


def main():
    tag, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    g = json.loads((ROOT / 'results/sapt_bio/native_fragment' / f'{tag}_geometry.json').read_text())
    C, H = np.array(g['donor_C']), np.array(g['xferH'])
    heavy = [(a['name'], np.array(a['position'])) for a in g['native_wall'] if a['element'] != 'H']
    name, W = min(heavy, key=lambda x: np.linalg.norm(x[1] - H))
    r_hw = float(np.linalg.norm(W - H))
    axis = (W - C) / np.linalg.norm(W - C)
    Hd = build_methane(C, H - C)
    Hw = build_methane(W, H - W)
    psi4.set_memory('6 GB')
    psi4.set_num_threads(4)
    psi4.set_options({'basis': 'jun-cc-pvdz', 'scf_type': 'df', 'freeze_core': True})
    E = {}
    for dlt in DELTAS:
        sh = axis * dlt
        L = ['0 1'] + [f'H {h[0]:.6f} {h[1]:.6f} {h[2]:.6f}' for h in Hd] + [f'C {C[0]:.6f} {C[1]:.6f} {C[2]:.6f}']
        L += ['--', '0 1'] + [f'H {h[0] + sh[0]:.6f} {h[1] + sh[1]:.6f} {h[2] + sh[2]:.6f}' for h in Hw]
        L += [f'C {W[0] + sh[0]:.6f} {W[1] + sh[1]:.6f} {W[2] + sh[2]:.6f}', 'units angstrom', 'symmetry c1',
              'no_reorient', 'no_com']
        psi4.core.set_output_file(str(outdir / f'sapt_{tag}_methane_wall_{dlt:+.2f}.out'), False)
        psi4.energy('sapt0', molecule=psi4.geometry('\n'.join(L)))
        E[dlt] = float(psi4.variable('SAPT EXCH ENERGY') * HA2KCAL)
    ds = np.array(DELTAS)
    es = np.array([E[d] for d in DELTAS])
    A = np.column_stack([np.ones_like(ds), ds, ds ** 2])
    p, *_ = np.linalg.lstsq(A, es, rcond=None)
    k_kcal = 2 * p[2]
    out = dict(TAG=tag, wall_atom=name, r_HW_A=r_hw, r_CC_A=float(np.linalg.norm(W - C)),
               k_kcal_A2=float(k_kcal), k_Nm=float(k_kcal * CONV),
               rms_kcal=float(np.sqrt(np.mean((es - A @ p) ** 2))), E_exch_kcal={str(d): E[d] for d in DELTAS},
               xferH=g.get('xferH_label', None))
    (outdir / f'{tag}_methane_wall_result.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('TAG', 'wall_atom', 'r_HW_A', 'r_CC_A', 'k_Nm')}))
    os._exit(0)


if __name__ == '__main__':
    main()
