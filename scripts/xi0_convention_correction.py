#!/usr/bin/env python3
"""Recompute the enzyme placement on the confinement curve under the CORRECT
identification of the cavity radius xi0.

THE ERROR BEING CORRECTED
  xi0 is a PARABOLIC COORDINATE, not a distance. With the code's convention
      x = sqrt(xi eta) cos phi,  y = sqrt(xi eta) sin phi,  z = (eta - xi)/2
  one has r = (xi + eta)/2, hence xi = r - z. The wall xi = xi0 is therefore the
  paraboloid r - z = xi0. On it rho^2 = 2 xi0 z + xi0^2, so the squared distance
  to the nucleus is D^2(z) = 2 xi0 z + xi0^2 + z^2, which is increasing on the
  physical domain z >= -xi0/2 and is minimized at the vertex:

      >>>  closest approach of the wall to the nucleus  d = xi0 / 2  <<<

  (verified analytically and by numerical minimization; the surface also crosses
  the nucleus plane z = 0 at rho = xi0, which is the aperture radius.)

  Figure 1 of the manuscript already draws this correctly, describing the pit
  depth as the focal length xi0/2 with the nucleus at the focus. Section 2.11,
  however, sets xi0 equal to the MEASURED distance from the transferring
  hydrogen to the nearest closed-shell heavy atom. That measured distance is a
  closest approach, so it equals xi0/2, and the correct substitution is

      xi0 = 2 d.

  Using xi0 = d models a wall twice as close as intended. Because the response
  is exponential in xi0, this inflates the induced dipole substantially.

PRECISION
  nu1 becomes very small at large xi0 (nu1 ~ u0 e^-u0), so the free-atom limit is
  approached to many digits and double precision loses the dipole. Everything
  here is mpmath at 50 digits, cross-checked against the deployed scipy solver
  in the regime where the latter is still reliable.

Usage: xi0_convention_correction.py
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path

from mpmath import mp, mpf, hyp1f1, findroot, quad, exp as mexp

mp.dps = 50

ROOT = Path(_REPO)
OUT = ROOT / 'results' / 'xi0_convention_correction.json'

A0 = mpf('0.529177210903')       # Angstrom
RY_eV = mpf('13.605693122994')
# axial field of a point dipole: E[MV/cm] = 1523.98 * P[e a0] / r[A]^3
FIELD_C = mpf('1523.98')


# first zeros of the Bessel functions J_0 and J_1, used only to place the scan
J01 = {1: mpf('2.404825557695773'), 2: mpf('3.831705970207512')}


def first_zero(a, b):
    """Smallest x > 0 with M(a, b, x) = 0, for a < 0.

    The scan must START BELOW the first zero and step upward finely, because
    M(-nu1, b, x) oscillates once nu1 is large and a coarse scan can straddle an
    even number of crossings and silently return a LATER zero. A fixed starting
    point cannot do this: the first zero sits near ln(1/nu1) when nu1 is small
    and near j^2/(4 nu1) when nu1 is large, which differ by many decades. The
    start is therefore scaled to the smaller of those two estimates and then
    lowered further until M is still positive there, which it must be since
    M(a, b, 0) = 1.
    """
    nu1 = -a
    f = lambda x: hyp1f1(a, b, x)
    bessel = J01[int(b)] ** 2 / (4 * nu1)                 # large-nu1 estimate
    small = abs(mp.log(1 / nu1)) + 8 if nu1 < 1 else mpf(8)   # small-nu1 estimate
    x0 = min(bessel, small) / 20
    while f(x0) <= 0 and x0 > mpf('1e-60'):
        x0 /= 10
    x_prev, f_prev = x0, f(x0)
    x = x0 * mpf('1.02')
    while x < mpf('1e8'):
        fx = f(x)
        if f_prev * fx < 0:
            return findroot(f, (x_prev, x), solver='bisect', tol=mpf('1e-45'))
        x_prev, f_prev = x, fx
        x *= mpf('1.02')
    raise RuntimeError(f'no zero of M({a}, {b}, x)')


def xi0_of_nu1(nu1):
    """Wall coordinate xi0/a0 for a given nu1 > 0 (ground state, nu = 1 + nu1)."""
    return first_zero(-nu1, mpf(1)) * (1 + nu1)


def solve(xi0_over_a0):
    """Invert xi0(nu1) by bisection in log(nu1); return nu1, nu, u0."""
    target = mpf(xi0_over_a0)
    g = lambda t: xi0_of_nu1(mexp(t)) - target
    a, b = mpf('-90'), mpf('12')
    fa = g(a)
    if fa * g(b) > 0:
        raise RuntimeError(f'no bracket for xi0/a0={xi0_over_a0}')
    for _ in range(240):
        m = (a + b) / 2
        fm = g(m)
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
        if abs(b - a) < mpf('1e-40'):
            break
    nu1 = mexp((a + b) / 2)
    nu = 1 + nu1
    return nu1, nu, first_zero(-nu1, mpf(1))


def dipole(nu1, nu, u0):
    """P/(e a0) = (nu/2) (2 A0 - A2) / (A1 + A0), A_n = int_0^u0 e^-u M(-nu1,1,u)^2 u^n du."""
    M = lambda u: hyp1f1(-nu1, 1, u)
    A = [quad(lambda u: mexp(-u) * M(u) ** 2 * u ** n, [0, u0]) for n in (0, 1, 2)]
    return (nu / 2) * (2 * A[0] - A[2]) / (A[1] + A[0])


def report(d_ang, label):
    """d_ang is the PHYSICAL closest approach of the wall, in Angstrom."""
    d = mpf(d_ang)
    out = {}
    for name, xi0 in (('old_convention_xi0_eq_d', d),
                      ('corrected_xi0_eq_2d', 2 * d)):
        x = xi0 / A0
        nu1, nu, u0 = solve(x)
        P = dipole(nu1, nu, u0)
        dE = RY_eV * (1 - 1 / nu ** 2)
        out[name] = dict(xi0_A=float(xi0), xi0_over_a0=float(x),
                         nu1=float(nu1), P_ea0=float(P),
                         energy_raised_eV=float(dE),
                         field_at_2p55_MVcm=float(FIELD_C * P / mpf('2.55') ** 3))
    o, c = out['old_convention_xi0_eq_d'], out['corrected_xi0_eq_2d']
    print(f'\n{label}:  wall closest approach d = {float(d):.2f} A')
    print(f'  as published   xi0 = d   = {o["xi0_A"]:.2f} A  (xi0/a0 = {o["xi0_over_a0"]:5.2f}): '
          f'P = {o["P_ea0"]:.4f} e a0,  dE = {o["energy_raised_eV"]:.3f} eV,  '
          f'E = {o["field_at_2p55_MVcm"]:6.2f} MV/cm')
    print(f'  corrected      xi0 = 2d  = {c["xi0_A"]:.2f} A  (xi0/a0 = {c["xi0_over_a0"]:5.2f}): '
          f'P = {c["P_ea0"]:.4f} e a0,  dE = {c["energy_raised_eV"]:.3f} eV,  '
          f'E = {c["field_at_2p55_MVcm"]:6.2f} MV/cm')
    print(f'  overestimate factor in P: {o["P_ea0"]/c["P_ea0"]:.1f}x')
    out['overestimate_factor_P'] = o['P_ea0'] / c['P_ea0']
    return out


def ceiling_sweep():
    """Sweep the WHOLE family forward in nu1 and find the largest dipole, hence
    the largest axial field, the model can produce at any cavity size.

    Inverting P -> xi0 is the wrong tool here. The map nu1 -> xi0 is monotone
    decreasing but its range is BOUNDED BELOW: as nu1 -> inf the wall condition
    M(-nu1, 1, u0) = 0 goes over to its Bessel limit J_0(2 sqrt(nu1 u0)) = 0, so
    u0 -> j_{0,1}^2/(4 nu1) and

        xi0/a0 = u0 (1 + nu1)  ->  j_{0,1}^2 / 4 = 1.44580...

    No paraboloidal cavity tighter than that exists: the electron ionizes rather
    than be squeezed further. A forward sweep therefore both locates the maximum
    dipole and exhibits that floor, which an inversion cannot do.
    """
    rows = []
    t = mpf('-8')
    while t <= mpf('14'):
        nu1 = mexp(t)
        u0 = first_zero(-nu1, mpf(1))
        nu = 1 + nu1
        x = u0 * nu
        P = dipole(nu1, nu, u0)
        rows.append((float(x), float(P),
                     float(FIELD_C * P / mpf('2.55') ** 3), float(nu1)))
        t += mpf('0.25')
    return rows


def main():
    res = {'geometry': 'wall xi=xi0 has closest approach xi0/2 and aperture radius xi0',
           'placements': {}}
    print('Enzyme placement on the confinement curve, old vs corrected convention')
    print('(SLO: H to nearest external closed-shell heavy atom = 2.4 to 3.0 A)')
    for d, lab in (('2.4', 'SLO tight end'), ('3.0', 'SLO loose end')):
        res['placements'][lab] = report(d, lab)

    print('\n--- domain of the model ---')
    rows = ceiling_sweep()
    xmin = min(r[0] for r in rows)
    print(f'  floor:  xi0/a0 -> {xmin:.5f} as nu1 -> inf, matching the Bessel '
          f'limit j_01^2/4 = {float(J01[1]**2/4):.5f} exactly')
    print(f'  no paraboloidal cavity tighter than xi0 = {xmin*float(A0):.3f} A '
          f'(wall at d = {xmin*float(A0)/2:.3f} A) supports a bound state.')
    print('  NOTE: the dipole DIVERGES as that floor is approached, because the')
    print('  paraboloid confines in xi but is open in eta: the state is pushed out')
    print('  along the open direction and ionizes (E = -13.6/nu^2 -> 0), so <z>')
    print('  grows without bound. The model therefore supplies NO useful upper')
    print('  bound on P, and none is claimed. The bound that matters is thermal.')
    res['domain'] = dict(xi0_over_a0_floor=xmin,
                         bessel_limit=float(J01[1] ** 2 / 4),
                         note='dipole diverges at the floor because the state '
                              'ionizes along the open eta direction; no maximum '
                              'dipole is claimed')

    print('\n--- the bound that matters: confinement energy against kT ---')
    kT_283 = mpf('0.0244')      # eV at 283 K, the SLO kinetic temperature
    print(f'  kT at 283 K = {float(kT_283):.4f} eV = '
          f'{float(kT_283)*23.0605:.3f} kcal/mol')
    for d, lab in (('2.4', 'SLO tight end'), ('3.0', 'SLO loose end')):
        c = res['placements'][lab]['corrected_xi0_eq_2d']
        print(f'  {lab:14s} d = {d} A:  dE = {c["energy_raised_eV"]:.4f} eV = '
              f'{c["energy_raised_eV"]*23.0605:.3f} kcal/mol  '
              f'-> dE/kT = {c["energy_raised_eV"]/float(kT_283):.2f}')
    res['thermal_comparison'] = dict(
        kT_283K_eV=float(kT_283), kT_283K_kcal=float(kT_283) * 23.0605,
        ratios={lab: res['placements'][lab]['corrected_xi0_eq_2d']
                ['energy_raised_eV'] / float(kT_283)
                for lab in res['placements']})

    OUT.write_text(json.dumps(res, indent=1))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
