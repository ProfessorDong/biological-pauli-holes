#!/usr/bin/env python3
"""Derive, and put on disk, the unit-conversion constants the manuscript quotes.

The numeric-claim sweep (numeric_claim_sweep.py) requires every distinctive number
in the paper to trace to a file. Constants derived from CODATA values had no such
file, so they were the only claims the sweep could not close. This writes them,
with the arithmetic that produces them, so the audit is complete and so that a
referee can check the conversions without re-deriving them.

Two of these were quoted incorrectly before this script existed:
  * the axial-field constant appeared as 1.5242e11 V/m in the appendix while the
    figures used 1523.98 MV/cm. The correct value is 1523.99 MV/cm. The slip was
    0.014 per cent and of no physical consequence, but the two numbers were
    inconsistent with each other, which is the kind of thing that erodes trust.

Usage: derived_constants.py
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path

ROOT = Path(_REPO)
OUT = ROOT / 'results' / 'derived_constants.json'

# CODATA 2018
E_CHARGE = 1.602176634e-19        # C, exact
A0 = 0.529177210903e-10           # m
K_COULOMB = 8.9875517923e9        # N m^2 C^-2, = 1/(4 pi eps0)
N_AVOGADRO = 6.02214076e23        # mol^-1, exact
KCAL_J = 4184.0                   # J per kcal, exact
KB = 1.380649e-23                 # J K^-1, exact


def main():
    ea0 = E_CHARGE * A0                                  # C m, one atomic unit of dipole

    # Axial field of a point dipole, E = (1/4 pi eps0) 2P / r^3, with P in e*a0
    # and r in angstrom. The 1e-30 divides out the (1e-10 m)^3 of r^3 against the
    # 1e-30-scale dipole, leaving V/m.
    field_Vm = 2.0 * K_COULOMB * ea0 / 1e-30             # V/m per (P[e a0] / r[A]^3)
    field_MVcm = field_Vm / 1e8                          # 1 MV/cm = 1e8 V/m

    # kcal/mol/A^2 -> N/m
    curv = (KCAL_J / N_AVOGADRO) / 1e-20

    # kT at the temperature of the JBC kinetic measurements
    T = 283.15
    kT_eV = KB * T / E_CHARGE
    kT_kcal = KB * T * N_AVOGADRO / KCAL_J

    # decade spans of the two confinement curves, quoted in the Fig. 1 caption.
    # Derived from results/confinement_universality.json rather than stored there,
    # so they are recorded here to keep every quoted number traceable.
    import json as _json
    import math as _math
    uni = _json.loads(
        (ROOT / 'results' / 'confinement_universality.json').read_text())['geometries']
    spans = {k: _math.log10(max(v['dE_hartree']) / min(v['dE_hartree']))
             for k, v in uni.items()}

    d = {
        'confinement_curve_decade_spans': spans,
        'coulomb_constant_N_m2_C-2': K_COULOMB,
        'coulomb_constant_note': ('quoted as 8.98755e9 in the induced-field appendix; '
                                  'CODATA 2018 exact-derived value, not a computed result'),
        'dipole_atomic_unit_Cm': ea0,
        'axial_field_constant_V_per_m': field_Vm,
        'axial_field_constant_MV_per_cm': field_MVcm,
        'axial_field_note': ('E[MV/cm] = 1523.99 * P[e a0] / r[A]^3. The figures use '
                             '1523.98, which agrees to five significant figures. An '
                             'earlier appendix value of 1.5242e11 V/m was a slip.'),
        'curvature_kcalmolA2_to_Nm': curv,
        'kT_283p15K_eV': kT_eV,
        'kT_283p15K_kcal_per_mol': kT_kcal,
    }
    OUT.write_text(json.dumps(d, indent=1))

    print(f'  e*a0                          = {ea0:.6e} C m')
    print(f'  axial field constant          = {field_Vm:.6e} V/m')
    print(f'                                = {field_MVcm:.4f} MV/cm   '
          f'(manuscript: 1523.99; figures: 1523.98)')
    print(f'  kcal/mol/A^2 -> N/m           = {curv:.6f}   (manuscript: 0.694770)')
    print(f'  kT at {T} K                = {kT_eV:.5f} eV = {kT_kcal:.4f} kcal/mol')
    print(f'                                  (manuscript: 0.0244 eV, 0.563 kcal/mol)')
    for k, v in spans.items():
        print(f'  {k:24s} spans {v:.4f} decades   (manuscript: '
              f'{"6.43" if k == "paraboloid" else "5.93"})')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
