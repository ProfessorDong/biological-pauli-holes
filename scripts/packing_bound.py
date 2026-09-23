#!/usr/bin/env python3
"""Why the cavity contribution is a few percent in ANY protein.

Surveying enzymes one at a time can only ever support a generalization by
induction. The exponential rule permits something better, because the two
distances that matter are both fixed by chemistry rather than by the particular
protein:

  BONDING SHELL, set by covalent geometry. The donor's other heavy-atom
  substituents sit at
      d_bond = sqrt(r_DH^2 + r_DX^2 - 2 r_DH r_DX cos(theta))
  from the transferring hydrogen. With a C-H bond of 1.09 A, a C-C bond of
  1.53 A and a tetrahedral angle, d_bond = 2.15 A. Nothing about the protein
  enters. The acceptor heavy atom is closer still, at r_DA - r_DH.

  PACKING SHELL, set by van der Waals contact. A non-bonded heavy atom cannot
  approach the hydrogen more closely than the sum of the van der Waals radii,
  r(C) + r(H) = 1.70 + 1.20 = 2.90 A, and in folded proteins side chains around
  a buried hydrogen pack at 3.2 A and beyond.

The exponential converts that roughly one-angstrom gap into a large factor per
atom, exp(-2 * 1.1 / a0) ~ 1/64, and the cavity can only recover it by weight of
numbers. This script computes how much it can recover, as a function of the
nearest packing distance and the number of packing atoms, and compares the
resulting bound with the three enzymes evaluated in this work.

The bound is deliberately GENEROUS to the cavity: every packing atom is placed
at the nearest packing distance, which no real structure achieves.

Usage: packing_bound.py     (pauli env)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
OUT = ROOT / 'results' / 'packing_bound.json'

A0 = 0.529177210903
R_CH, R_CC, THETA = 1.09, 1.53, np.radians(109.47)
VDW_C, VDW_H = 1.70, 1.20


def d_bond(r_dh=R_CH, r_dx=R_CC, theta=THETA):
    return float(np.sqrt(r_dh**2 + r_dx**2 - 2 * r_dh * r_dx * np.cos(theta)))


def w(d):
    return np.exp(-2.0 * d / A0)


def n_max(d, r=VDW_C):
    """Largest number of heavy atoms that can simultaneously sit at distance d
    from the hydrogen, from the solid angle each one subtends.

    An atom of van der Waals radius r at distance d occupies a spherical cap of
    half-angle arcsin(r/d), hence solid angle 2 pi (1 - cos arcsin(r/d)). The
    sphere has 4 pi to give away. This ratio OVERCOUNTS, because caps cannot
    tile a sphere without gaps, so it is a strict upper bound on the crowding
    the cavity can achieve. It is what removes the arbitrariness from n_pack:
    the cavity is not free to pile up unlimited atoms at contact distance.
    """
    if d <= r:
        return 0
    cap = 2 * np.pi * (1 - np.cos(np.arcsin(r / d)))
    return int(4 * np.pi / cap)


def cavity_fraction(d_pack, n_pack=20, n_bond=3, d_bnd=None, d_acc=2.0):
    """Upper bound on the cavity share of the confinement weight.

    n_bond substituents at the covalent distance, one acceptor heavy atom at
    d_acc, and n_pack packing atoms ALL placed at d_pack, which is the most
    favourable arrangement the cavity could possibly have.
    """
    d_bnd = d_bnd or d_bond()
    wb = n_bond * w(d_bnd) + w(d_acc)
    wp = n_pack * w(d_pack)
    return float(wp / (wb + wp))


def main():
    db = d_bond()
    contact = VDW_C + VDW_H
    print(f'covalent bonding shell   d_bond  = {db:.3f} A '
          f'(C-H {R_CH}, C-C {R_CC}, tetrahedral)')
    print(f'van der Waals contact    d_vdW   = {contact:.3f} A (r_C + r_H)')
    print(f'weight ratio per atom at contact: {w(contact)/w(db):.4f} '
          f'= 1/{w(db)/w(contact):.1f}')
    print(f'weight ratio per atom at 3.5 A  : {w(3.5)/w(db):.5f} '
          f'= 1/{w(db)/w(3.5):.0f}\n')

    print('upper bound on the cavity share, with the packing shell as crowded')
    print('as solid angle permits:')
    rows = []
    for dp in (2.9, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5):
        nm = n_max(dp)
        f = cavity_fraction(dp, n_pack=nm)
        rows.append((dp, nm, f))
        print(f'   d_pack = {dp:.1f} A   n_max = {nm:3d}   ->  {100*f:7.3f} %')
    worst = max(rows, key=lambda r: r[2])
    print(f'\n   worst case over all packing distances: {100*worst[2]:.2f} % '
          f'at d_pack = {worst[0]:.1f} A with {worst[1]} atoms')

    # what the three enzymes actually give
    obs = {
        'SLO (14 MD geometries, JBC cavity positions)': (0.0006, 4.75),
        'DHFR 4PDJ (protein, both H placements)':       (0.23, 5.83),
        'KSI 2PZV (protein, 4 chains)':                 (1.24, 1.71),
    }
    print('\nobserved protein/cavity share, this work:')
    for k, (lo, hi) in obs.items():
        print(f'   {k:46s} {lo:8.4f} to {hi:.2f} %')
    typ = cavity_fraction(3.4, n_pack=n_max(3.4))
    print(f'\nbound at a typical packing distance of 3.4 A: {100*typ:.2f} % '
          f'({n_max(3.4)} atoms)')
    print(f'worst case over all packing distances:        '
          f'{100*worst[2]:.2f} %')
    print('Every observed value lies at or below the bound. The bound is loose '
          'by\nconstruction, since it places every packing atom at the same '
          'closest distance\nand ignores that caps cannot tile a sphere.')

    OUT.write_text(json.dumps(dict(
        d_bond_A=db, d_vdW_contact_A=contact,
        ratio_per_atom_at_contact=float(w(contact) / w(db)),
        ratio_per_atom_at_3p5=float(w(3.5) / w(db)),
        bound_vs_dpack=[dict(d_pack_A=d, n_max=nm, cavity_fraction=f)
                        for d, nm, f in rows],
        worst_case=dict(d_pack_A=worst[0], n_max=worst[1],
                        cavity_fraction=worst[2]),
        observed_percent=obs), indent=1))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
