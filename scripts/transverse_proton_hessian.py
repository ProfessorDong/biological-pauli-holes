#!/usr/bin/env python3
"""The curvature the theory defines: the transverse Hessian of the transferring nucleus.

WHY THIS EXISTS
  The manuscript's k_exch is obtained by translating the whole wall fragment along the
  donor-to-wall axis and fitting a quadratic. That is a fragment-separation curvature. The
  theory it is compared against defines something different: the Hessian of the exchange
  energy with respect to motion of the TRANSFERRING NUCLEUS, projected transverse to the
  reaction coordinate, whose mass-weighted mode carries the zero-point energy and hence any
  isotope effect. Those are different second derivatives of the same surface, and nothing
  guarantees they agree even in sign.

  The distinction is not academic. For a single repulsive centre U(r) = A exp(-beta r), the
  radial curvature is positive while the curvature for a perpendicular displacement at fixed
  longitudinal separation is U'(r)/r, which is NEGATIVE: sliding sideways moves the proton
  away from the wall atom and lowers the energy. A pocket confines transversally only if
  enough wall atoms surround the proton that their contributions sum to a positive Hessian.
  Whether a real active site does that is an empirical question, and this answers it.

WHAT IS COMPUTED
  With every other atom held fixed, the transferring hydrogen is displaced on a grid in the
  plane perpendicular to the donor-acceptor axis, SAPT0 exchange is evaluated at each point,
  and the 2x2 Hessian

      K_perp = P_perp^T (grad_rH grad_rH E_exch) P_perp

  is obtained by least squares on the full quadratic form, so the cross term is fitted rather
  than assumed zero. Eigenvalues and eigenvectors are reported: the eigenvalues are the
  transverse force constants, and a negative one means the exchange interaction is
  transversally ANTI-confining at that geometry.

WHAT IS AND IS NOT CLAIMED
  This is the exchange component alone, not the full restoring Hessian. The covalent C-H bond
  supplies a large positive transverse restoring force that is not included here and that
  dominates the physical vibration. The point of the calculation is not to predict a frequency
  but to establish whether the EXCHANGE contribution, which is the quantity the paper's
  confinement argument is about, confines the proton transversally or not. No zero-point
  energy is assigned to K_perp alone.

Usage: transverse_proton_hessian.py TAG CLAMP [half_width_A] [n_per_axis]   (pauli env)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import psi4

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
RG = ROOT / 'results/reactive_geometry'
OUT = ROOT / 'results/transverse_hessian'
OUT.mkdir(parents=True, exist_ok=True)
Ha2kcal, CONV = 627.5094740631, 0.694770


def build_methane(c, direction, r=1.09):
    """Identical construction to the production scan, so the comparison is controlled."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    H = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3.0) * r
    vf, vt = H[0] / r, d
    ax = np.cross(vf, vt)
    s, cth = np.linalg.norm(ax), float(np.dot(vf, vt))
    if s < 1e-6:
        R = np.eye(3) if cth > 0 else -np.eye(3)
    else:
        ax /= s
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + s * K + (1 - cth) * K @ K
    return (R @ H.T).T + np.asarray(c, float)


def exch(donor_atoms, wall, basis='jun-cc-pVDZ'):
    L = ['0 1'] + [f'{e} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}' for e, p in donor_atoms]
    L += ['--', '0 1']
    L += [f'{a["element"]} ' + ' '.join(f'{v:.8f}' for v in a['position']) for a in wall]
    L.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
    psi4.core.set_output_file(os.devnull, False)
    psi4.core.clean()
    psi4.set_options({'basis': basis, 'scf_type': 'df', 'freeze_core': True})
    psi4.energy('sapt0', molecule=psi4.geometry('\n'.join(L)))
    return float(psi4.variable('SAPT EXCH ENERGY') * Ha2kcal)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'WT'
    clamp = sys.argv[2] if len(sys.argv) > 2 else 'r255'
    half = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    npa = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    psi4.set_memory('8 GB')

    g = json.loads((RG / f'{tag}_{clamp}_geometry.json').read_text())
    dC = np.array(g['donor_C'])
    xH = np.array(g['xferH'])
    aO = np.array(g['acceptor_O'])
    wall = g['native_wall']

    # reaction coordinate, and an orthonormal transverse frame perpendicular to it
    n = (aO - dC) / np.linalg.norm(aO - dC)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = tmp - np.dot(tmp, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    assert abs(np.dot(e1, n)) < 1e-12 and abs(np.dot(e2, n)) < 1e-12, 'frame not orthogonal'

    Hs = build_methane(dC, xH - dC)
    # the surrogate H standing in for the transferring one is the one along the C-H axis
    k = int(np.argmax([np.dot((h - dC) / np.linalg.norm(h - dC), (xH - dC) / np.linalg.norm(xH - dC))
                       for h in Hs]))
    print(f'{tag} {clamp}: donor {g.get("donor_C_label","C")} acceptor {g["acceptor_label"]}, '
          f'r_DA={np.linalg.norm(aO-dC):.3f} A')
    print(f'  grid {npa}x{npa} over +/-{half} A in the plane perpendicular to the reaction axis\n')

    grid = np.linspace(-half, half, npa)
    rows = []
    for a in grid:
        for b in grid:
            H2 = Hs.copy()
            H2[k] = Hs[k] + a * e1 + b * e2
            atoms = [('C', dC)] + [('H', h) for h in H2]
            E = exch(atoms, wall)
            rows.append((a, b, E))
            print(f'    d1={a:+.3f} d2={b:+.3f}  E_exch={E:+9.5f}', flush=True)

    A = np.array([[1, x, y, 0.5 * x * x, x * y, 0.5 * y * y] for x, y, _ in rows], float)
    z = np.array([e for _, _, e in rows], float)
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    resid = float(np.sqrt(np.mean((z - A @ c) ** 2)))
    Kk = np.array([[c[3], c[4]], [c[4], c[5]]])          # kcal/mol/A^2
    K = Kk * CONV                                         # N/m
    w, V = np.linalg.eigh(K)

    ref = json.loads((RG / f'{tag}_{clamp}_native_result.json').read_text())
    print(f'\n  transverse Hessian (N/m):\n    [[{K[0,0]:+8.3f} {K[0,1]:+8.3f}]'
          f'\n     [{K[1,0]:+8.3f} {K[1,1]:+8.3f}]]')
    print(f'  eigenvalues: {w[0]:+.3f}, {w[1]:+.3f} N/m     RMS fit residual {resid:.4f} kcal/mol')
    print(f'  production wall-separation k_exch (K_sep): {ref["k_exch_Nm"]:+.3f} N/m')
    sign = 'CONFINING' if (w > 0).all() else ('ANTI-CONFINING in at least one direction'
                                              if (w < 0).any() else 'mixed')
    print(f'\n  VERDICT: the exchange contribution is {sign} transverse to the reaction axis.')
    if (w < 0).any():
        print('  A negative transverse eigenvalue means sliding the proton sideways LOWERS the')
        print('  exchange energy, so this component cannot by itself raise a transverse ZPE.')

    (OUT / f'{tag}_{clamp}_w{int(round(half*100)):03d}_transverse_hessian.json').write_text(json.dumps(dict(
        tag=tag, clamp=clamp, half_width_A=half, n_per_axis=npa,
        r_DA=float(np.linalg.norm(aO - dC)),
        K_perp_Nm=K.tolist(), eigenvalues_Nm=w.tolist(), eigenvectors=V.tolist(),
        rms_residual_kcal=resid, K_sep_Nm=ref['k_exch_Nm'],
        grid=[[float(a), float(b), float(e)] for a, b, e in rows]), indent=1))
    print(f'\nwrote {OUT}/{tag}_{clamp}_w{int(round(half*100)):03d}_transverse_hessian.json')


if __name__ == '__main__':
    main()
