#!/usr/bin/env python3
"""Stage 3: the TOTAL transverse restoring matrix, on the coordinates of the exchange Hessian.

WHY THIS EXISTS
  The paper reports that the exchange component of the transverse Hessian has a negative
  eigenvalue in most systems, and asserts that this carries no instability because the covalent
  C-H bond supplies a much larger positive restoring term. That term has never been computed. It
  cannot be: every SAPT quantity is an INTERACTION energy between two fragments, and the covalent
  bond is intramolecular to the donor, so it cancels exactly. Only a supermolecular total energy
  contains it.

WHAT IS COMPUTED, on the identical 5x5 transverse grid and the identical fragments
  K_tot(complex)  total-energy Hessian of donor + wall, transferring H displaced alone, every
                  other atom frozen. This is the full constrained transverse restoring matrix.
  K_tot(donor)    the same for the isolated donor fragment. This is the covalent contribution,
                  the term the paper appeals to.
  K_tot(complex) - K_tot(donor) is then the wall's total contribution, which must agree with the
  SAPT total-interaction Hessian K_perp^int already computed on the same grid. That agreement is
  a consistency check the calculation carries with it, not an assumption.

  Two methods are run. HF/jun-cc-pVDZ is matched to SAPT0's own reference determinant and basis,
  so the exchange fraction is formed between quantities of the same theory. B3LYP/jun-cc-pVDZ adds
  correlation and shows whether the conclusion depends on that. No empirical dispersion is applied:
  the dftd3 backend is absent from this environment, and dispersion contributes at the 0.1 N/m
  level against a covalent term two orders of magnitude larger.

WHAT IS STILL NOT CLAIMED
  This is a CONSTRAINED Hessian with every other atom frozen. It is not a relaxed Hessian and not
  a normal-mode analysis; physical frequencies need the mass-weighted total Hessian of the full
  system with its couplings. Nothing here is converted into a frequency or an isotope effect.

Usage: total_restoring_hessian.py TAG [CLAMP] [half_width_A] [n_per_axis]   (pauli env)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import psi4

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
D = ROOT / 'results/native_donor_validation'
Ha2kcal, CONV = 627.5094740631, 0.694770
METHODS = [('hf', 'HF'), ('b3lyp', 'B3LYP')]


def energy(atoms, method, basis='jun-cc-pvdz'):
    L = ['0 1'] + [f'{e} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}' for e, p in atoms]
    L.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
    psi4.core.set_output_file(os.devnull, False)
    psi4.core.clean()
    psi4.set_options({'basis': basis, 'scf_type': 'df'})
    return float(psi4.energy(method, molecule=psi4.geometry('\n'.join(L))) * Ha2kcal)


def quad(rows, col):
    """Full quadratic fit; returns Hessian in N/m, per-eigenvalue standard errors, rms."""
    A = np.array([[1, x, y, 0.5 * x * x, x * y, 0.5 * y * y] for x, y, *_ in rows], float)
    z = np.array([r[col] for r in rows], float)
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    res = A @ c - z
    s2 = float(res @ res) / (len(z) - 6)
    cov = s2 * np.linalg.inv(A.T @ A)
    H = np.array([[c[3], c[4]], [c[4], c[5]]]) * CONV
    w, V = np.linalg.eigh(H)
    se = []
    for k in range(2):
        v = V[:, k]
        gv = np.array([v[0] ** 2, 2 * v[0] * v[1], v[1] ** 2]) * CONV
        se.append(float(np.sqrt(gv @ cov[np.ix_([3, 4, 5], [3, 4, 5])] @ gv)))
    return H, w, np.array(se), float(np.sqrt(s2))


def main():
    tag = sys.argv[1]
    clamp = sys.argv[2] if len(sys.argv) > 2 else 'r255'
    half = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    npa = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    psi4.set_memory('8 GB')
    psi4.set_num_threads(1)

    g = json.loads((D / f'{tag}_{clamp}_native_donor.json').read_text())
    wall = [(a['element'], np.array(a['position'])) for a in g['native_wall']]
    dC, aO = np.array(g['donor_C']), np.array(g['acceptor_O'])
    frag = [(a['element'], np.array(a['position']), a['name']) for a in g['donor_fragment']]
    k = next(i for i, (_, _, n) in enumerate(frag) if n == g['xferH_name'])

    n = (aO - dC) / np.linalg.norm(aO - dC)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = tmp - np.dot(tmp, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    print(f'{tag} {clamp}: supermolecular {len(frag) + len(wall)} atoms, donor alone {len(frag)}',
          flush=True)
    rows = []
    for a in np.linspace(-half, half, npa):
        for b in np.linspace(-half, half, npa):
            don = [(e, (p + a * e1 + b * e2) if i == k else p)
                   for i, (e, p, _) in enumerate(frag)]
            vals = []
            for meth, _ in METHODS:
                vals.append(energy(don + wall, meth))   # complex
                vals.append(energy(don, meth))          # donor alone
            rows.append((a, b, *vals))
            print(f'    d1={a:+.3f} d2={b:+.3f}  ' +
                  '  '.join(f'{lbl}: cplx {vals[2*i]:+14.5f} don {vals[2*i+1]:+14.5f}'
                            for i, (_, lbl) in enumerate(METHODS)), flush=True)

    out = dict(tag=tag, clamp=clamp, half=half, n_per_axis=npa, basis='jun-cc-pVDZ',
               note='constrained transverse Hessian, all atoms but the transferring H frozen')
    for i, (meth, lbl) in enumerate(METHODS):
        Hc, wc, sc, rc = quad(rows, 2 + 2 * i)
        Hd, wd, sd, rd = quad(rows, 3 + 2 * i)
        Hw = Hc - Hd
        ww = np.linalg.eigvalsh(Hw)
        out[lbl] = dict(
            K_tot_complex_eigs=[float(x) for x in wc], K_tot_complex_se=[float(x) for x in sc],
            K_tot_donor_eigs=[float(x) for x in wd], K_tot_donor_se=[float(x) for x in sd],
            K_wall_from_difference_eigs=[float(x) for x in ww],
            spectral_norm_complex=float(max(abs(wc))), rms_complex_kcal=rc, rms_donor_kcal=rd)
        print(f'  {lbl}: K_tot(complex) {wc[0]:+9.3f} , {wc[1]:+9.3f} N/m   '
              f'K_tot(donor) {wd[0]:+9.3f} , {wd[1]:+9.3f}')
        print(f'  {lbl}: wall by difference {ww[0]:+9.4f} , {ww[1]:+9.4f} N/m')

    nd = json.loads((D / f'{tag}_{clamp}_w{int(round(half*100)):03d}_native_transverse.json'
                     ).read_text())
    out['K_perp_exch_eigs'] = nd['K_perp_exch_eigs']
    out['K_perp_int_eigs'] = nd['K_perp_int_eigs']
    for _, lbl in METHODS:
        sn = out[lbl]['spectral_norm_complex']
        out[lbl]['exchange_fraction'] = float(max(abs(np.array(nd['K_perp_exch_eigs']))) / sn)
        out[lbl]['interaction_fraction'] = float(max(abs(np.array(nd['K_perp_int_eigs']))) / sn)
        print(f'  {lbl}: exchange fraction {out[lbl]["exchange_fraction"]*100:.3f}% , '
              f'total-interaction fraction {out[lbl]["interaction_fraction"]*100:.3f}%')
    out['grid'] = [dict(d1=r[0], d2=r[1], **{f'{lbl}_{w}': r[2 + 2*i + j]
                                             for i, (_, lbl) in enumerate(METHODS)
                                             for j, w in enumerate(('complex', 'donor'))})
                   for r in rows]
    (D / f'{tag}_{clamp}_w{int(round(half*100)):03d}_total_restoring.json').write_text(
        json.dumps(out, indent=2))
    os._exit(0)


if __name__ == '__main__':
    main()
