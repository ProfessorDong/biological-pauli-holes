#!/usr/bin/env python3
"""Rows of the exact-solution table: cavity size, ionization energy I, induced dipole P and the axial
dipole field E at r = 2.55 A, for two solid-surface sites, the two ends of the lipoxygenase window, and
the cavity that the field measured in ketosteroid isomerase would require.

Solver: the 50-digit mpmath solution of scripts/xi0_convention_correction.py, imported unchanged. The
scipy solver of scripts/pauli_hole_hydrogen.py, which produced the superseded surface rows through
scripts/pauli_hole_survey.py, is run alongside every row as a cross-check.

Inputs
  Pd(110) channel entrance. You, Cheng, Cheng and Huang (Phys. Rev. B 43, 7293, 1991) treat a hydrogen at the
    rhombic entrance of each (110) channel as sitting in a hemisphere of radius
    r = 0.5 [0.5 (3.87 - 1.72) + 0.5 x 2.74] = 1.2225 A, which they print as 1.22 A. That radius is xi0.
    The row is evaluated at the printed 1.22 A; the unrounded 1.2225 A is reported alongside.
    The superseded row used xi0 = 1.37 A. In You, Liu and Gao (Chem. Phys. Impact 4, 100058, 2022) 1.37 A is the height
    difference between neighboring Pd layers of the (110) surface, half the 2.74 A Pd-Pd distance, not a
    hole radius. It is reproduced here only to document the superseded values.
  NaCl, surface Na+ site replaced by hydrogen. You, Ye and Du (Phys. Rev. B 41, 8180, 1990) take
    zeta0 = 3.5 a0. The superseded row was computed at 1.85 A = 3.496 a0; it is reproduced as a check.
  Lipoxygenase window. The per-system mean contact distances d, printed as 2.42 and 2.95 A
    (results/pauli_hole_survey/slo_xi0_full.json holds 2.4232 and 2.9530 A); xi0 = 2d. Evaluated at the
    printed values; the unrounded extremes are reported alongside.
  Required. The dipole whose axial field at 2.55 A equals the 144 MV/cm measured by vibrational Stark
    spectroscopy in ketosteroid isomerase (Fried et al., Science 346, 1510, 2014), and the cavity xi0
    that produces it, found by bisection in nu1.
Field: E = 1523.99 P / r^3 MV/cm with P in e a0 and r in A (CODATA 2018 constants).
Writes results/audit_2026_11/surface_rows.json.
Usage: audit_surface_rows.py   (pauli env; about two minutes)
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get('PAULI_ROOT') or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / 'scripts'))

from mpmath import mp, mpf, exp as mexp, log as mlog          # noqa: E402
import xi0_convention_correction as xc                          # noqa: E402  (sets mp.dps = 50)
import pauli_hole_hydrogen as ph                                 # noqa: E402

FIELD_C = mpf('1523.99')       # MV/cm per (e a0 / A^3)
R_FIELD = mpf('2.55')          # A
KSI = mpf('144')               # MV/cm
RY = xc.RY_eV
A0 = xc.A0


def row_at(xi0_over_a0):
    """Exact quantities at a given xi0/a0, with the scipy cross-check."""
    nu1, nu, u0 = xc.solve(xi0_over_a0)
    P = xc.dipole(nu1, nu, u0)
    I = RY / nu ** 2
    E = FIELD_C * P / R_FIELD ** 3
    chk = ph.solve(float(xi0_over_a0))
    return dict(xi0_over_a0=float(xi0_over_a0), xi0_A=float(mpf(xi0_over_a0) * A0),
                d_A=float(mpf(xi0_over_a0) * A0 / 2), nu1=float(nu1), I_eV=float(I),
                dE_eV=float(RY - I), P_ea0=float(P), E_MVcm=float(E), pct_of_144=float(100 * E / KSI),
                scipy_check=dict(I_eV=chk['I_eV'], P_ea0=chk['P']))


def required_row():
    """Bisection in t = ln(nu1): P(nu1) rises as the cavity tightens; match P to the KSI field."""
    P_req = KSI * R_FIELD ** 3 / FIELD_C

    def P_of(t):
        nu1 = mexp(t)
        u0 = xc.first_zero(-nu1, mpf(1))
        return xc.dipole(nu1, 1 + nu1, u0), u0 * (1 + nu1), nu1

    a, b = mlog(mpf('0.8')), mlog(mpf('1.0'))      # P = 1.363 and 1.581 there (validation table)
    Pa = P_of(a)[0] - P_req
    if Pa * (P_of(b)[0] - P_req) > 0:
        raise RuntimeError('no bracket for the required dipole')
    for _ in range(60):
        m = (a + b) / 2
        Pm = P_of(m)[0] - P_req
        if Pa * Pm <= 0:
            b = m
        else:
            a, Pa = m, Pm
    P, x, nu1 = P_of((a + b) / 2)
    out = row_at(x)
    out.update(P_required_ea0=float(P_req), nu1_bisection=float(nu1))
    return out


def main():
    a0 = float(A0)
    res = dict(method='xi0_convention_correction.solve/dipole (mpmath, 50 digits); scipy cross-check from '
                      'pauli_hole_hydrogen.solve',
               field='E = 1523.99 P / 2.55^3 MV/cm; percentages of 144 MV/cm', rows={}, superseded_checks={})
    print('superseded rows (checks)')
    for lab, xi_A in (('Pd(110), xi0 = 1.37 A (layer height, not a hole radius)', '1.37'),
                      ('NaCl, xi0 = 1.85 A = 3.496 a0', '1.85')):
        r = row_at(mpf(xi_A) / A0)
        res['superseded_checks'][lab] = r
        print(f"  {lab:58s} I {r['I_eV']:.2f}  P {r['P_ea0']:.4f}  E {r['E_MVcm']:.1f} ({r['pct_of_144']:.1f}%)")

    print('table rows')
    inputs = [
        ('Pd(110) channel entrance, xi0 = 1.22 A (You et al. 1991, printed)', mpf('1.22') / A0),
        ('Pd(110) channel entrance, xi0 = 1.2225 A (formula, unrounded)', mpf('1.2225') / A0),
        ('NaCl surface Na+ site, xi0 = 3.50 a0 (You and Ye 1990)', mpf('3.5')),
        ('SLO smallest mean contact, d = 2.42 A (printed)', 2 * mpf('2.42') / A0),
        ('SLO largest mean contact, d = 2.95 A (printed)', 2 * mpf('2.95') / A0),
        ('SLO smallest mean contact, d = 2.4232 A (unrounded)', 2 * mpf('2.4232') / A0),
        ('SLO largest mean contact, d = 2.9530 A (unrounded)', 2 * mpf('2.9530') / A0),
    ]
    for lab, x in inputs:
        r = row_at(x)
        res['rows'][lab] = r
        print(f"  {lab:66s} xi0/a0 {r['xi0_over_a0']:.4f}  xi0 {r['xi0_A']:.4f} A  d {r['d_A']:.4f}  "
              f"I {r['I_eV']:.4f}  P {r['P_ea0']:.5f}  E {r['E_MVcm']:.3f} ({r['pct_of_144']:.2f}%)  "
              f"[scipy I {r['scipy_check']['I_eV']:.4f} P {r['scipy_check']['P_ea0']:.5f}]")
    r = required_row()
    res['rows']['Required for the measured 144 MV/cm at 2.55 A'] = r
    print(f"  required: P {r['P_required_ea0']:.5f}  xi0/a0 {r['xi0_over_a0']:.5f}  xi0 {r['xi0_A']:.4f} A  "
          f"d {r['d_A']:.4f} A  I {r['I_eV']:.4f}  P {r['P_ea0']:.5f}  E {r['E_MVcm']:.2f}")
    dest = ROOT / 'results' / 'audit_2026_11' / 'surface_rows.json'
    dest.write_text(json.dumps(res, indent=1) + '\n')
    print(f'wrote {dest.relative_to(ROOT)}  (a0 = {a0} A)')


if __name__ == '__main__':
    main()
