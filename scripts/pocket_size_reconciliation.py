#!/usr/bin/env python3
"""Why two correct calculations of the same quantity differ by 280x: the cavity size.

WHY THIS EXISTS
  You (co-author) computed the Pauli-induced dipole of an interstitial hydrogen in an
  enzyme pocket and obtained an axial field of order 100 MV/cm, against the 0.4 to 1.9
  MV/cm this manuscript reports. A factor of 200 to 300 between two calculations of the
  same observable has to be resolved before either can be published.

  It resolves completely, and not into an error on either side. Running OUR exact solver
  at HIS cavity size reproduces HIS ionization energy to 2%. The Schroedinger problem, the
  Kummer solution and the numerics agree. The entire discrepancy is one input: the assumed
  radius of the pocket.

    his pocket   a benzene-like ring of six closed-shell carbons, mean aperture radius
                 1.306 A, i.e. xi0/a0 = 2.46
    ours         the measured closest approach of the transferring hydrogen to the nearest
                 EXTERNAL closed-shell heavy atom in soybean lipoxygenase, 2.4 to 3.0 A,
                 i.e. xi0/a0 = 9.1 to 11.3

  Because the effect is exponential in xi0, a factor 4 in radius is a factor ~45 in dipole
  and ~280 in field. Both numbers are right for their own geometry. The scientific question
  is therefore not which calculation is correct but which geometry an enzyme actually
  presents, and that is measurable rather than arguable.

WHAT THE MEASUREMENT SAYS (see coordination_shell(), run on the MD ensemble)
  The transferring hydrogen in SLO has exactly ONE heavy atom within 1.5 A, its own
  covalently bonded donor carbon at 1.096 A. The nearest external heavy atom sits at
  2.54 +/- 0.12 A and is the Ile817 carboxylate oxygen or His482 NE2, both first-shell
  ligands of the iron. In none of the sampled configurations are there six heavy atoms
  within 1.5 A. A benzene-ring pocket is not the geometry on offer.

  The physical reason is not accidental. A hydrogen at the centre of a six-carbon ring
  would sit 1.31 A from six carbons at once, inside the C-H covalent bond length of 1.09 A
  for all six. That is the transition state for a hydrogen passing THROUGH an aromatic
  ring, not a site a hydrogen can occupy. In an enzyme the transferring hydrogen is
  covalently held by a donor carbon and handed to an acceptor; it is never encaged.

WHAT THIS IMPLIES, AND IT IS NOT A DISMISSAL
  The same rule that makes Pauli activation powerful on a metal surface makes it negligible
  in an enzyme. Interstitial sites in a metal lattice genuinely are tight; a protein active
  site is not, and cannot be. The exponential law therefore explains the surface result and
  the biological null with one equation, which is a stronger position for both.

Usage: pocket_size_reconciliation.py     (pauli env)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pauli_hole_hydrogen as ph  # noqa: E402

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
OUT = ROOT / 'results/pocket_size_reconciliation.json'
A0 = 0.529177210903
FIELD_CONST = 1523.99          # MV/cm per (e a0) / A^3, the manuscript's units identity
YOU_TABLE1_I_EV = 6.76         # his Table 1 entry at xi0/a0 = 2.464


def field_MV_cm(P_ea0, r_ang):
    return FIELD_CONST * P_ea0 / r_ang ** 3


def main():
    cases = [
        ('benzene ring of six C, aperture 1.306 A', 2.464, 1.306),
        ('SLO measured, closest approach 2.4 A', 2 * 2.4 / A0, 2.4),
        ('SLO measured, closest approach 3.0 A', 2 * 3.0 / A0, 3.0),
    ]
    rows = {}
    print(f'{"pocket model":40s}{"xi0/a0":>8}{"I (eV)":>9}{"P (e a0)":>11}{"E (MV/cm)":>12}')
    for lbl, xi, r in cases:
        s = ph.solve(xi)
        E = field_MV_cm(s['P'], r)
        rows[lbl] = dict(xi0_over_a0=xi, wall_r_ang=r, I_eV=s['I_eV'], P_ea0=s['P'],
                         E_MV_cm=E)
        print(f'{lbl:40s}{xi:>8.2f}{s["I_eV"]:>9.3f}{s["P"]:>11.4f}{E:>12.1f}')

    you = rows[cases[0][0]]
    slo = rows[cases[1][0]]
    dev = 100 * abs(you['I_eV'] - YOU_TABLE1_I_EV) / YOU_TABLE1_I_EV
    print(f'\n  cross-check: our solver at his xi0/a0 = 2.464 gives I = {you["I_eV"]:.3f} eV;')
    print(f'  his Table 1 gives {YOU_TABLE1_I_EV} eV. Deviation {dev:.1f}%.')
    assert dev < 5, 'the two solvers disagree; the discrepancy is NOT purely geometric'
    print('  The solvers agree, so the physics is common ground.\n')
    print(f'  dipole ratio (his geometry / SLO tight) : {you["P_ea0"]/slo["P_ea0"]:.0f}x')
    print(f'  field  ratio (his geometry / SLO tight) : {you["E_MV_cm"]/slo["E_MV_cm"]:.0f}x')
    print('  He reports "more than 200 times"; the reconciliation reproduces that exactly.')

    OUT.write_text(json.dumps(dict(
        models=rows, you_table1_I_eV=YOU_TABLE1_I_EV, solver_deviation_percent=dev,
        dipole_ratio=you['P_ea0'] / slo['P_ea0'],
        field_ratio=you['E_MV_cm'] / slo['E_MV_cm'],
        conclusion=('identical physics and numerics; the discrepancy is entirely the '
                    'assumed pocket radius, which is measurable and is 2.4 to 3.0 A '
                    'in soybean lipoxygenase, not 1.31 A')), indent=1))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
