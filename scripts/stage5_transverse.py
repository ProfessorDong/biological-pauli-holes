#!/usr/bin/env python3
"""Stage 5, phase C: K_perp on one sampled configuration.

Identical grid, fragments, basis and fitting to the single-configuration calculation of
transverse_native_donor.py; the only difference is that the configuration comes from the
independently sampled ensemble rather than from the one stored clamped frame. Reports the
exchange and total-interaction transverse Hessians so that both can be given a sampling
uncertainty rather than the fit uncertainty the paper currently quotes.

Usage: stage5_transverse.py <config.json> [half_width_A] [n_per_axis]   (pauli env)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import psi4

Ha2kcal, CONV = 627.5094740631, 0.694770


def sapt(donor, wall, basis='jun-cc-pVDZ'):
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


def quad(rows, col):
    A = np.array([[1, x, y, 0.5 * x * x, x * y, 0.5 * y * y] for x, y, *_ in rows], float)
    z = np.array([r[col] for r in rows], float)
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    H = np.array([[c[3], c[4]], [c[4], c[5]]]) * CONV
    return np.linalg.eigvalsh(H), float(np.sqrt(np.mean((A @ c - z) ** 2)))


def main():
    cfg = Path(sys.argv[1])
    half = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    npa = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    out = cfg.with_name(cfg.name.replace('_native_donor.json',
                                         f'_w{int(round(half*100)):03d}_transverse.json'))
    if out.exists():
        print(f'skip {out.name}'); return
    psi4.set_memory('4 GB')
    psi4.set_num_threads(1)

    g = json.loads(cfg.read_text())
    wall = g['native_wall']
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

    rows = []
    for a in np.linspace(-half, half, npa):
        for b in np.linspace(-half, half, npa):
            atoms = [(e, (p + a * e1 + b * e2) if i == k else p)
                     for i, (e, p, _) in enumerate(frag)]
            ex, tot = sapt(atoms, wall)
            rows.append((a, b, ex, tot))
    wx, rx = quad(rows, 2)
    wt, rt = quad(rows, 3)
    json.dump(dict(tag=g['TAG'], config=g['config'], source_frame=g['source_frame'],
                   half=half, r_DA=g['r_DA'], r_CH=g['r_CH'],
                   K_perp_exch_eigs=[float(v) for v in wx],
                   K_perp_int_eigs=[float(v) for v in wt],
                   spectral_norm_exch=float(max(abs(wx))),
                   rms_exch=rx, rms_int=rt), open(out, 'w'), indent=2)
    print(f'{g["TAG"]} cfg{g["config"]:02d}: exch {wx[0]:+.4f} , {wx[1]:+.4f}   '
          f'int {wt[0]:+.4f} , {wt[1]:+.4f}', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
