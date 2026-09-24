#!/usr/bin/env python3
"""How much of the published k_exch depends on an arbitrary choice nobody made deliberately?

WHY THIS EXISTS
  The methane donor surrogate is built by placing a carbon at the donor position and
  directing ONE of its four C-H bonds along the observed C-H axis. That fixes one degree of
  freedom. It does not fix the azimuth: methane is still free to spin about that axis, and
  the three remaining hydrogens sweep through a full turn as it does. They sit 1.09 A from
  the carbon, which is close enough to the wall to exchange, so k_exch depends on where they
  land.

  Nothing chose that azimuth. It falls out of whichever rotation matrix the construction
  happens to produce, in our implementation the minimal rotation carrying the first
  tetrahedral vertex onto the C-H axis. Every published native-fragment k_exch inherits it.

  If the spread over the azimuth is small, the published series is robust and this is a
  footnote. If it is comparable to the mutation-to-mutation differences the manuscript
  interprets, then part of the panel's variation is an artifact of fragment construction and
  the affected claims have to be weakened. Either way it has to be measured, not assumed.

  Methane has a three-fold axis about any C-H bond, so the azimuth is periodic on 120 deg
  and the scan below covers exactly one period.

  This also bears on the donor-swap result: the unexplained system-to-system variation in
  how faithfully methane reproduces the real bis-allylic unit (0.98x to 1.54x) would be
  explained if the azimuthal spread were of that size.

Usage: methane_azimuth_sensitivity.py [TAG [n_angles]]     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import sys
from pathlib import Path

import numpy as np
import psi4

ROOT = Path(_REPO)
DIR = ROOT / 'results/sapt_bio/native_fragment'
OUT = ROOT / 'results/sapt_bio/donor_fragment'
Ha2kcal, CONV = 627.5094740631, 0.694770
DELTAS = [-0.20, -0.10, 0.00, +0.10, +0.20]


def methane_rotated(center, direction, phi, r=1.09):
    """The same construction as the published scan, then spun by phi about the C-H axis."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    H = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3.0) * r
    vf, vt = H[0] / r, d
    ax = np.cross(vf, vt)
    s = np.linalg.norm(ax)
    c = float(np.dot(vf, vt))
    if s < 1e-6:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        ax /= s
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * K @ K
    P = (R @ H.T).T                                   # published orientation, phi = 0
    K2 = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
    Rz = np.eye(3) + np.sin(phi) * K2 + (1 - np.cos(phi)) * K2 @ K2   # spin about the C-H
    P = (Rz @ P.T).T
    return [('C', np.asarray(center, float))] + [('H', h + np.asarray(center, float))
                                                 for h in P]


def k_of(donor, native, axis, basis):
    E = []
    for delta in DELTAS:
        disp = axis * delta
        L = ['0 1'] + [f'{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}' for e, p in donor]
        L += ['--', '0 1']
        L += [f'{a["element"]} ' + ' '.join(f'{v:.6f}' for v in np.array(a['position']) + disp)
              for a in native]
        L.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
        psi4.core.clean()
        psi4.energy('sapt0', molecule=psi4.geometry('\n'.join(L)))
        E.append(float(psi4.variable('SAPT EXCH ENERGY') * Ha2kcal))
    ds, ys = np.array(DELTAS), np.array(E)
    A = np.column_stack([np.ones_like(ds), ds, ds ** 2])
    p, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return float(2 * p[2] * CONV)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'WT'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    basis = 'jun-cc-pVDZ'
    g = json.loads((DIR / f'{tag}_geometry.json').read_text())
    dC, xH = np.array(g['donor_C']), np.array(g['xferH'])
    native = g['native_wall']
    heavy = [(a['name'], np.array(a['position'])) for a in native if a['element'] != 'H']
    wall = min(heavy, key=lambda t: np.linalg.norm(t[1] - xH))
    axis = (wall[1] - dC) / np.linalg.norm(wall[1] - dC)
    psi4.core.set_output_file(str(OUT / f'psi4_{tag}_azimuth.out'), False)
    psi4.set_options({'basis': basis, 'scf_type': 'df', 'freeze_core': True})

    phis = np.linspace(0.0, 2 * np.pi / 3, n, endpoint=False)   # one 3-fold period
    ks = []
    print(f'{tag}  wall={wall[0]}  one 120 deg period in {n} steps\n')
    print(f'{"phi (deg)":>10}{"k_exch (N/m)":>15}')
    for phi in phis:
        k = k_of(methane_rotated(dC, xH - dC, phi), native, axis, basis)
        ks.append(k)
        print(f'{np.degrees(phi):>10.1f}{k:>15.3f}'
              f'{"   <- the published orientation" if phi == 0 else ""}', flush=True)
    ks = np.array(ks)
    pub = ks[0]
    spread = ks.max() - ks.min()
    print(f'\n  published (phi=0) : {pub:.3f} N/m')
    print(f'  range over azimuth: {ks.min():.3f} to {ks.max():.3f} N/m')
    print(f'  spread            : {spread:.3f} N/m = {spread/pub:.1%} of the published value')
    print(f'  mean +/- sd       : {ks.mean():.3f} +/- {ks.std(ddof=1):.3f} N/m')
    print(f'  published value is {abs(pub-ks.mean())/ks.std(ddof=1):.2f} sd from the '
          f'azimuthal mean')

    (OUT / f'{tag}_azimuth_sensitivity.json').write_text(json.dumps(dict(
        tag=tag, basis=basis, wall_atom=wall[0], n_angles=n,
        phi_deg=np.degrees(phis).tolist(), k_Nm=ks.tolist(),
        published_k_Nm=float(pub), k_min=float(ks.min()), k_max=float(ks.max()),
        spread_Nm=float(spread), spread_fraction=float(spread / pub),
        mean_Nm=float(ks.mean()), sd_Nm=float(ks.std(ddof=1))), indent=1))
    print(f'\nwrote {OUT}/{tag}_azimuth_sensitivity.json')


if __name__ == '__main__':
    main()
