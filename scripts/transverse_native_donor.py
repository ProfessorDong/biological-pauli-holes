#!/usr/bin/env python3
"""Matched native-donor comparison: K_sep and K_perp on the same fragments and configuration.

WHY THIS EXISTS
  The paper's coordinate result is computed with a methane surrogate at the donor carbon. The
  revision audit's first validation item is to repeat it with the native substrate fragment,
  the same wall definition and the same configurations, because "native wall" does not make the
  donor native. This script does exactly that, and it reports three curvatures from one set of
  SAPT0 calculations so that none of them can be compared across a changed input:

    K_sep       fragment-separation curvature: translate the WHOLE wall along the donor-wall
                axis, five points, quadratic fit. This is the paper's k_exch^bio.
    K_perp^exch transverse Hessian of the SAPT0 EXCHANGE energy with respect to displacement of
                the transferring hydrogen alone, projected perpendicular to the donor-acceptor
                axis. This is the paper's K_perp.
    K_perp^int  the same Hessian formed from the SAPT0 TOTAL interaction energy. It costs
                nothing extra, because the same calculation returns both, and it answers a
                question the exchange component alone cannot: whether the full intermolecular
                interaction is transversally confining even where its exchange part is not.

WHAT IT DOES NOT DO
  K_perp^int is still an INTERMOLECULAR quantity. The covalent C-H bond that dominates the
  physical transverse restoring force is intramolecular to the donor fragment and cancels out
  of any interaction energy. The total restoring matrix needs a supermolecular calculation and
  is the separate second item of the validation plan; nothing here should be converted into a
  frequency, a zero-point energy or an isotope effect.

Usage: transverse_native_donor.py TAG [CLAMP] [half_width_A] [n_per_axis]   (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import os
import sys
from pathlib import Path

import numpy as np
import psi4

ROOT = Path(_REPO)
D = ROOT / 'results/native_donor_validation'
Ha2kcal, CONV = 627.5094740631, 0.694770


def sapt(donor, wall, basis='jun-cc-pVDZ'):
    """Return (exchange, total interaction) in kcal/mol for one dimer geometry."""
    L = ['0 1'] + [f'{e} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}' for e, p in donor]
    L += ['--', '0 1']
    L += [f'{a["element"]} ' + ' '.join(f'{v:.8f}' for v in a['position']) for a in wall]
    L.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
    psi4.core.set_output_file(os.devnull, False)
    psi4.core.clean()
    psi4.set_options({'basis': basis, 'scf_type': 'df', 'freeze_core': True})
    psi4.energy('sapt0', molecule=psi4.geometry('\n'.join(L)))
    return (float(psi4.variable('SAPT EXCH ENERGY') * Ha2kcal),
            float(psi4.variable('SAPT TOTAL ENERGY') * Ha2kcal))


def quad2d(rows, col):
    """Least-squares fit of the full quadratic form.

    Returns (Hessian in N/m, rms residual in kcal/mol, gradient in kcal/mol/A). The stage-4
    pre-registration requires the fitted gradient to be reported at every setting alongside the
    eigenvalues, because a large linear term means the expansion centre is off the stationary
    point and the quadratic coefficient is then reading a slope as a curvature.
    """
    A = np.array([[1, x, y, 0.5 * x * x, x * y, 0.5 * y * y] for x, y, _ in rows], float)
    z = np.array([r[col] for r in rows], float)
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    H = np.array([[c[3], c[4]], [c[4], c[5]]]) * CONV
    grad = [float(c[1]), float(c[2])]
    return H, float(np.sqrt(np.mean((A @ c - z) ** 2))), grad


def main():
    tag = sys.argv[1]
    clamp = sys.argv[2] if len(sys.argv) > 2 else 'r255'
    half = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    npa = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    psi4.set_memory('8 GB')
    psi4.set_num_threads(1)

    g = json.loads((D / f'{tag}_{clamp}_native_donor.json').read_text())
    wall = g['native_wall']
    dC, aO = np.array(g['donor_C']), np.array(g['acceptor_O'])
    frag = [(a['element'], np.array(a['position']), a['name']) for a in g['donor_fragment']]
    k = next(i for i, (_, _, n) in enumerate(frag) if n == g['xferH_name'])
    assert np.allclose(frag[k][1], g['xferH'], atol=1e-6), 'transferring H mismatch'

    n = (aO - dC) / np.linalg.norm(aO - dC)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = tmp - np.dot(tmp, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    assert abs(np.dot(e1, n)) < 1e-12 and abs(np.dot(e2, n)) < 1e-12

    print(f'{tag} {clamp}: native {g["donor_fragment_formula"]} donor, wall {g["wall_residue"]} '
          f'({len(wall)} atoms), r_DA={np.linalg.norm(aO - dC):.3f} A', flush=True)

    # --- K_perp on the transferring hydrogen only -----------------------------------
    rows = []
    for a in np.linspace(-half, half, npa):
        for b in np.linspace(-half, half, npa):
            atoms = [(e, (p + a * e1 + b * e2) if i == k else p)
                     for i, (e, p, _) in enumerate(frag)]
            ex, tot = sapt(atoms, wall)
            rows.append((a, b, ex, tot))
            print(f'    perp d1={a:+.3f} d2={b:+.3f}  exch={ex:+9.5f}  int={tot:+9.5f}',
                  flush=True)
    Hx, rx, gx = quad2d([(r[0], r[1], r[2]) for r in rows], 2)
    Ht, rt, gt = quad2d([(r[0], r[1], r[3]) for r in rows], 2)
    wx = np.linalg.eigvalsh(Hx)
    wt = np.linalg.eigvalsh(Ht)

    # --- K_sep by translating the whole wall, same fragments ------------------------
    nearest = min(wall, key=lambda a: np.linalg.norm(np.array(a['position']) - frag[k][1]))
    u = np.array(nearest['position']) - frag[k][1]
    u /= np.linalg.norm(u)
    donor0 = [(e, p) for e, p, _ in frag]
    scan = []
    for d in (-0.20, -0.10, 0.0, 0.10, 0.20):
        moved = [dict(a, position=(np.array(a['position']) + d * u).tolist()) for a in wall]
        ex, _ = sapt(donor0, moved)
        scan.append((d, ex))
        print(f'    sep  delta={d:+.2f}  exch={ex:+9.5f}', flush=True)
    c = np.polyfit([d for d, _ in scan], [e for _, e in scan], 2)
    Ksep = float(2 * c[0] * CONV)

    out = dict(tag=tag, clamp=clamp, half=half, n_per_axis=npa,
               donor='native ' + g['donor_fragment_formula'], wall=g['wall_residue'],
               n_wall_atoms=len(wall), r_DA=float(np.linalg.norm(aO - dC)),
               K_sep_Nm=Ksep,
               K_perp_exch_eigs=[float(x) for x in wx], K_perp_exch_trace=float(wx.sum()),
               K_perp_exch_rms_kcal=rx, K_perp_exch_grad_kcal_per_A=gx,
               K_perp_exch_grad_norm=float(np.hypot(*gx)),
               K_perp_int_eigs=[float(x) for x in wt], K_perp_int_trace=float(wt.sum()),
               K_perp_int_rms_kcal=rt, K_perp_int_grad_kcal_per_A=gt,
               K_perp_int_grad_norm=float(np.hypot(*gt)),
               spectral_norm_exch=float(max(abs(wx))), spectral_norm_int=float(max(abs(wt))),
               ratio_Ksep_over_specnorm_exch=float(Ksep / max(abs(wx))),
               grid=[dict(d1=r[0], d2=r[1], exch=r[2], total=r[3]) for r in rows],
               sep_scan=[dict(delta=d, exch=e) for d, e in scan])
    # the grid density must appear in the name: a 7x7 at the same half-width is a DIFFERENT
    # calculation and silently overwrote the published 5x5 result once.
    suffix = '' if npa == 5 else f'_n{npa}'
    (D / f'{tag}_{clamp}_w{int(round(half * 100)):03d}{suffix}_native_transverse.json'
     ).write_text(
        json.dumps(out, indent=2))
    print(f'  K_sep          = {Ksep:8.3f} N/m')
    print(f'  K_perp^exch    = {wx[0]:+8.4f}, {wx[1]:+8.4f} N/m  (trace {wx.sum():+.4f}, '
          f'rms {rx:.2e} kcal)')
    print(f'  K_perp^int     = {wt[0]:+8.4f}, {wt[1]:+8.4f} N/m  (trace {wt.sum():+.4f}, '
          f'rms {rt:.2e} kcal)')
    print(f'  K_sep / |K_perp^exch|_2 = {Ksep / max(abs(wx)):.1f}')
    os._exit(0)


if __name__ == '__main__':
    main()
