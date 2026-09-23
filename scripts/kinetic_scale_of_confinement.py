#!/usr/bin/env python3
"""What does the confinement energy DO to the hydrogen, kinetically?

WHY THIS EXISTS
  The manuscript compares the confinement energy to k_BT and reports dE/k_BT = 0.14 to 1.02.
  That is correct but abstract. A co-author (Prof. You) posed the same comparison in kinetic
  terms instead: if the pocket releases the hydrogen, what speed does it acquire, and how does
  that compare with the speed thermal motion already gives it? That reframing is more
  physical, it is the question a catalysis reader actually has, and it is exactly equivalent
  to the energy comparison. This computes it.

  He also asked, explicitly, for the room-temperature thermal speed of H to be checked. It is
  checked here: 2.7e5 cm/s, confirming his own estimate of ~1e5 cm/s to within a factor of 3.

WHAT IT SHOWS
  At HIS assumed pocket (a six-carbon ring, xi0/a0 = 2.46) the confinement energy is 6.98 eV
  and the released hydrogen would leave at 3.7e6 cm/s, some thirteen times thermal. That
  reproduces his own estimate of ~1e6 cm/s and is internally consistent.

  At the MEASURED soybean lipoxygenase geometry (xi0/a0 = 9.1 to 11.3) the same expression
  gives 0.003 to 0.025 eV and a speed of 0.8e5 to 2.2e5 cm/s, which is 0.29 to 0.80 of the
  thermal speed. The Pauli kick is SMALLER than the thermal jostling the hydrogen already
  has. That is the kinetic statement of the paper's energy result.

A SEPARATE POINT ABOUT FORCES, WHICH IS A REAL SUBTLETY
  He computes the pull of the polarization surface charge on the hydrogen NUCLEUS (+e) rather
  than on the induced dipole, reasoning correctly that the dipole is not rigid: the electron
  centroid moves as the nucleus is drawn out, so F = (P.grad)E is not valid.

  The reasoning is right and the remedy is not. For a NEUTRAL hydrogen atom the force on the
  nucleus is cancelled, to first order, by the force on its own electron cloud: in a uniform
  field the net force is exactly zero, and in a realistic field it is the polarization
  gradient force alpha*E*grad(E), smaller than eE by a factor of 22 even when the field varies
  by 100% across one angstrom. Using eE is therefore correct only if the electron has ALREADY
  departed, which is stepwise electron-then-proton transfer, not the concerted proton-coupled
  transfer that soybean lipoxygenase is understood to perform.

Usage: kinetic_scale_of_confinement.py     (pauli env)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
OUT = ROOT / 'results/kinetic_scale_of_confinement.json'
kB = 1.380649e-23
amu = 1.66053906660e-27
eV = 1.602176634e-19
e = 1.602176634e-19
eps0 = 8.8541878128e-12
a0 = 5.29177210903e-11
mH = 1.00784 * amu
ALPHA_H = 4 * np.pi * eps0 * 4.5 * a0 ** 3      # H dipole polarizability, 4.5 a0^3

# confinement energies from the exact solution, results/pocket_size_reconciliation.json
CASES = [('six-carbon ring, aperture 1.306 A', 2.464, 6.980),
         ('SLO measured, closest approach 2.4 A', 9.071, 0.0249),
         ('SLO measured, closest approach 3.0 A', 11.338, 0.0033)]


def v_rms(T):
    return np.sqrt(3 * kB * T / mH)


def v_from_E(E_eV):
    return np.sqrt(2 * E_eV * eV / mH)


def main():
    vth300, vth283 = v_rms(300.0), v_rms(283.15)
    print(f'thermal speed of H: {vth300*100:.3e} cm/s at 300 K, '
          f'{vth283*100:.3e} cm/s at 283 K')
    print(f'  (co-author estimate ~1e5 cm/s is confirmed to within a factor of 3)\n')
    print(f'{"pocket":38s}{"xi0/a0":>8}{"dE (eV)":>10}{"v (cm/s)":>12}{"v/v_thermal":>13}')
    rows = {}
    for lbl, xi, dE in CASES:
        v = v_from_E(dE) * 100
        rows[lbl] = dict(xi0_over_a0=xi, dE_eV=dE, v_cm_s=v, ratio_thermal=v / (vth300 * 100))
        print(f'{lbl:38s}{xi:>8.2f}{dE:>10.4f}{v:>12.2e}{v/(vth300*100):>13.2f}')

    slo = [v for k, v in rows.items() if 'SLO' in k]
    lo, hi = min(r['ratio_thermal'] for r in slo), max(r['ratio_thermal'] for r in slo)
    print(f'\n  at the measured geometry the released hydrogen would move at {lo:.2f} to '
          f'{hi:.2f} of thermal speed:')
    print('  the Pauli kick is smaller than the thermal motion the hydrogen already has.')

    # the force subtlety
    E = 1e10                                    # 100 MV/cm in V/m
    F_bare = e * E
    forces = {}
    for L_ang in (1.0, 3.0):
        L = L_ang * 1e-10
        F_pol = ALPHA_H * E * (E / L)
        forces[f'{L_ang:g}A'] = dict(F_polarization_N=F_pol, ratio_to_eE=F_bare / F_pol)
        print(f'  force in 100 MV/cm: bare proton eE = {F_bare*1e9:.2f} nN; neutral H with the '
              f'field varying 100% over {L_ang:g} A = {F_pol*1e9:.4f} nN '
              f'({F_bare/F_pol:.0f}x smaller)')
    print('  in a uniform field the net force on a neutral hydrogen atom is exactly zero.')

    OUT.write_text(json.dumps(dict(
        v_thermal_cm_s=dict(T300=vth300 * 100, T283=vth283 * 100),
        pockets=rows, slo_ratio_range=[lo, hi],
        force_100MVcm=dict(bare_proton_N=F_bare, neutral_H=forces),
        note=('kinetic restatement of dE/kT; the released-hydrogen speed at the measured '
              'geometry is sub-thermal, and the force on a neutral H is not eE')), indent=1))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
