#!/usr/bin/env python3
"""The confinement energy at the biological wall distances, in BOTH solved geometries.

WHY THIS EXISTS
  The manuscript reported the permitted confinement energy as 0.14 to 1.02 k_BT and
  presented it as a ceiling. That number is the PARABOLOID alone. The paper solves a
  second geometry, a hydrogen atom centered in a hard sphere, and at the same wall
  distances the sphere gives roughly eight times as much. A ceiling quoted from one of
  two solved geometries is not a ceiling, and a referee reproduced the discrepancy
  independently. This computes both, exactly, from the same solver that produced
  confinement_universality.json, so the manuscript can report the range honestly.

WHAT IS AND IS NOT GEOMETRY-DEPENDENT HERE
  ENERGY: both geometries raise the ground state, and by different amounts. Both are
  reported.

  INDUCED DIPOLE AND FIELD: these exist for the PARABOLOID ONLY. A hydrogen atom at the
  center of a hard sphere stays centrosymmetric, so <z> = 0 and there is no induced
  dipole to speak of. The field numbers in the manuscript are therefore paraboloidal by
  construction, not a choice between geometries, and this script does not recompute them.

WHY THE TWO DIFFER
  The exponential rate is common to both, 2.011 and 2.002 in units of 1/a_0, which is the
  paper's universality claim and is unaffected. What differs is the algebraic prefactor:
  the paraboloid carries (d/a_0)^1.17 and the sphere (d/a_0)^2.17, one full power of d
  apart. Over the biological range that power is worth about an order of magnitude. The
  universality claim is about the EXPONENT; it never licensed quoting one geometry's
  PREFACTOR as a bound.

Usage: kbt_ceiling_both_geometries.py     (pauli env)
"""
import json
import sys
from pathlib import Path

from mpmath import mp, mpf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confinement_universality import (  # noqa: E402
    A0_ANG, d_of_eps_paraboloid, d_of_eps_sphere, dE, invert,
)

mp.dps = 60

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
OUT = ROOT / 'results' / 'kbt_ceiling_both_geometries.json'

HARTREE_EV = 27.211386245988
EV_KCAL = 23.060547830619
KT_283_EV = 0.0244                 # k_B T at 283.15 K, the JBC kinetic temperature
D_ANG = ['2.4', '3.0']             # H to nearest external closed-shell heavy atom, SLO


def main():
    kT = mpf(KT_283_EV)
    print(f'k_BT at 283.15 K = {KT_283_EV} eV = {KT_283_EV*EV_KCAL:.3f} kcal/mol\n')
    print(f'{"d (A)":>7s}{"d/a0":>8s}{"geometry":>13s}{"dE (Eh)":>13s}'
          f'{"dE (eV)":>11s}{"kcal/mol":>10s}{"dE/kT":>8s}')

    res = {}
    for d_str in D_ANG:
        d_ang = mpf(d_str)
        d_a0 = d_ang / mpf(A0_ANG)
        row = {}
        for name, wall in (('paraboloid', d_of_eps_paraboloid),
                           ('sphere', d_of_eps_sphere)):
            eps = invert(wall, d_a0)
            e_h = dE(1 + eps)
            e_ev = e_h * HARTREE_EV
            row[name] = dict(dE_hartree=float(e_h), dE_eV=float(e_ev),
                             dE_kcal=float(e_ev * EV_KCAL),
                             dE_over_kT=float(e_ev / kT))
            print(f'{d_str:>7s}{float(d_a0):>8.3f}{name:>13s}{float(e_h):>13.4e}'
                  f'{float(e_ev):>11.5f}{float(e_ev*EV_KCAL):>10.4f}'
                  f'{float(e_ev/kT):>8.2f}')
        row['sphere_over_paraboloid'] = (row['sphere']['dE_over_kT']
                                         / row['paraboloid']['dE_over_kT'])
        res[d_str] = row
        print(f'{"":>7s}{"":>8s}{"ratio":>13s}{row["sphere_over_paraboloid"]:>13.2f}\n')

    lo_p = res['3.0']['paraboloid']['dE_over_kT']
    hi_p = res['2.4']['paraboloid']['dE_over_kT']
    lo_s = res['3.0']['sphere']['dE_over_kT']
    hi_s = res['2.4']['sphere']['dE_over_kT']
    print(f'  paraboloid over the biological range: {lo_p:.2f} to {hi_p:.2f} kT')
    print(f'  sphere     over the biological range: {lo_s:.2f} to {hi_s:.2f} kT')
    print(f'  combined envelope of the two solved geometries: '
          f'{min(lo_p, lo_s):.2f} to {max(hi_p, hi_s):.2f} kT')
    print('\n  The exponential rate is common to both (2.011 vs 2.002 per a_0).')
    print('  The prefactor is not: (d/a0)^1.17 vs (d/a0)^2.17, one power of d apart.')
    print('  A dipole and a field exist for the paraboloid only; a hydrogen atom')
    print('  centered in a sphere stays centrosymmetric.')

    # The manuscript also quotes the near/far RATIO, because that, and not the
    # absolute energy, is what the mutation argument rests on. The prefactor does
    # not cancel exactly, but it survives only as a small factor against the
    # exponential, which is why the relative conclusion is geometry-robust.
    import math
    fits = {'paraboloid': (0.3550, 1.1719, 2.0112), 'sphere': (0.8876, 2.1741, 2.0017)}

    def dE_fit(g, d):
        C, pw, r = fits[g]
        x = d / A0_ANG
        return C * x ** pw * math.exp(-r * x)

    ratios = {}
    print('\n  near/far ratio, the quantity the mutation argument uses:')
    for dn, df in ((2.1, 3.5), (2.1, 4.5)):
        rp = dE_fit('paraboloid', dn) / dE_fit('paraboloid', df)
        rs = dE_fit('sphere', dn) / dE_fit('sphere', df)
        expo = math.exp(-2 * (dn - df) / A0_ANG)
        ratios[f'{dn}_vs_{df}'] = dict(paraboloid=rp, sphere=rs,
                                       geometry_factor=max(rp, rs) / min(rp, rs),
                                       shared_exponential=expo)
        print(f'    {dn} A vs {df} A: paraboloid {rp:.1f}, sphere {rs:.1f}, '
              f'geometry changes it {max(rp,rs)/min(rp,rs):.2f}x, '
              f'shared exponential {expo:.0f}x')

    OUT.write_text(json.dumps(dict(
        near_far_ratios=ratios,
        kT_283K_eV=KT_283_EV, d_angstrom=D_ANG, by_distance=res,
        paraboloid_kT_range=[lo_p, hi_p], sphere_kT_range=[lo_s, hi_s],
        combined_envelope_kT=[min(lo_p, lo_s), max(hi_p, hi_s)],
        note=('energy is geometry-dependent and both solved geometries are reported; '
              'the induced dipole and its field are paraboloidal only, because a '
              'hydrogen atom centered in a hard sphere has no induced dipole')), indent=1))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
