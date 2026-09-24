#!/usr/bin/env python3
"""Is the exponential suppression of confinement a property of the PARABOLOID,
or a universal property of a bound state meeting a wall?

This is the keystone test for the paper's central claim. If the exponential rate
is geometry-dependent, there is no theorem and no selection rule. If the rate is
the same across geometrically unrelated cavities, then the decay constant is set
by the bound state itself and the polynomial prefactor is all that geometry
controls. That second case is a theorem, and it is what we test here.

TWO EXACTLY SOLVABLE, GEOMETRICALLY UNRELATED CONFINEMENTS
--------------------------------------------------------
Atomic units throughout (a0 = 1, energies in hartree, E_free = -1/2).
Both problems reduce to a zero of a Kummer function M(a, b, x), but with
DIFFERENT b and different scaling, so they are genuinely different problems.

(A) PARABOLOIDAL WALL (You & Ye, Phys. Rev. B 41, 8180 (1990))
      parabolic coordinates  xi = r - z,  eta = r + z
      wall: the paraboloid of revolution xi = xi0, psi = 0 on it
      ground state: nu = 1 + nu1, wall condition  M(-nu1, 1, u0) = 0,
      u0 = xi0/nu.
    GEOMETRY. On the axis rho = 0 the surface xi = xi0 satisfies r - z = xi0
    with r = |z|, which for z < 0 gives -2z = xi0, i.e. a vertex at z = -xi0/2.
      >>> the wall's CLOSEST APPROACH to the nucleus is d = xi0/2 <<<
    This is the point of the whole script: xi0 is a parabolic coordinate, not a
    distance, and the two differ by exactly a factor of two.

(B) SPHERICAL BOX, nucleus at the center (Michels 1937; Sommerfeld & Welker 1938)
      wall: the sphere r = R, psi = 0 on it
      l = 0 radial function u(r) = r e^{-r/nu} M(1 - nu, 2, 2r/nu)
      wall condition  M(1 - nu, 2, 2R/nu) = 0
    Here the closest approach is simply d = R.

In both, E = -1/(2 nu^2), so the confinement energy shift is
      dE(d) = 1/2 - 1/(2 nu^2)  > 0.

WHAT IS FITTED
  ln dE(d) against d over a range of d where dE spans many decades. A pure
  exponential exp(-q d) with polynomial prefactor d^p gives
      ln dE = const + p ln d - q d,
  so we fit that three-parameter form and report q for each geometry.

PRECISION. dE falls to ~1e-12 hartree by d ~ 9 a0, so double precision cannot
resolve it against E ~ 0.5. Everything is done in mpmath at 60 digits.

Usage: confinement_universality.py
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path

import numpy as np
from mpmath import mp, mpf, hyp1f1, findroot, log as mlog

mp.dps = 60

ROOT = Path(_REPO)
OUT = ROOT / 'results' / 'confinement_universality.json'

A0_ANG = 0.529177210903


# -------------------------------------------------------- Kummer first zero
def first_zero(a, b):
    """Smallest x > 0 with M(a, b, x) = 0, for a < 0. Bracket by walking a grid
    that is geometric in x, since the zero moves to large x as a -> 0-."""
    f = lambda x: hyp1f1(a, b, x)
    x_prev, f_prev = mpf('1e-6'), f(mpf('1e-6'))
    x = mpf('1e-3')
    while x < mpf('1e6'):
        fx = f(x)
        if f_prev * fx < 0:
            return findroot(f, (x_prev, x), solver='bisect', tol=mpf('1e-50'))
        x_prev, f_prev = x, fx
        x *= mpf('1.05')
    raise RuntimeError(f'no zero found for M({a}, {b}, x)')


def invert(wall_of_nu, target_d, lo=mpf('1e-32'), hi=mpf('2')):
    """Given the monotone DECREASING map eps = nu - 1 -> wall distance, find the
    eps reproducing target_d. Confinement RAISES the energy, so nu > 1 always.

    Bisection is done on log(eps), because eps spans ~30 decades over the range
    of wall distances of interest and a linear bisection would waste all its
    iterations at the top of the interval. mpmath's own bisect tests |f|, which
    cannot be driven below the precision of d itself, so the loop is manual and
    terminates on the RELATIVE width of the bracket.
    """
    from mpmath import log as _log, exp as _exp
    g = lambda eps: wall_of_nu(eps) - target_d
    a, b = _log(lo), _log(hi)
    fa = g(_exp(a))
    if fa * g(_exp(b)) > 0:
        raise RuntimeError(f'target d={target_d} outside bracket')
    for _ in range(200):
        m = (a + b) / 2
        fm = g(_exp(m))
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
        if abs(b - a) < mpf('1e-40'):
            break
    return _exp((a + b) / 2)


# ------------------------------------------------------------------ paraboloid
def d_of_eps_paraboloid(eps):
    """Wall closest approach d = xi0/2 for ground-state nu = 1 + eps.

    Wall condition M(-eps, 1, u0) = 0 with u0 = xi0/nu, so xi0 = u0 * (1 + eps).
    """
    nu = 1 + eps
    u0 = first_zero(-eps, mpf(1))
    return u0 * nu / 2


# --------------------------------------------------------------- spherical box
def d_of_eps_sphere(eps):
    """Wall radius R for a hydrogen atom centered in a hard sphere, nu = 1 + eps.

    l = 0 radial function u(r) = r e^{-r/nu} M(1 - nu, 2, 2r/nu); the node
    condition u(R) = 0 gives M(-eps, 2, 2R/nu) = 0, so R = x0 * nu / 2.
    """
    nu = 1 + eps
    x0 = first_zero(-eps, mpf(2))
    return x0 * nu / 2


def dE(nu):
    """Confinement energy shift in hartree, positive."""
    return mpf('0.5') - 1 / (2 * nu ** 2)


def fit_exponential(d, y):
    """Least squares for ln y = c + p ln d - q d. Returns (c, p, q)."""
    d = np.asarray(d, float)
    y = np.asarray(y, float)
    A = np.column_stack([np.ones_like(d), np.log(d), -d])
    coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    return coef


def main():
    # d is the PHYSICAL closest approach of the wall to the nucleus, in a0.
    ds = [mpf(x) / 2 for x in range(8, 25)]          # 4.0 .. 12.0 a0

    rows = {'paraboloid': [], 'sphere': []}
    for d in ds:
        eps_p = invert(d_of_eps_paraboloid, d)
        eps_s = invert(d_of_eps_sphere, d)
        rows['paraboloid'].append((float(d), float(dE(1 + eps_p)), float(eps_p)))
        rows['sphere'].append((float(d), float(dE(1 + eps_s)), float(eps_s)))

    print('  d/a0    dE paraboloid      dE sphere        ratio')
    for (d, ep, _), (_, es, _) in zip(rows['paraboloid'], rows['sphere']):
        print(f'  {d:5.2f}   {ep:.6e}   {es:.6e}   {ep/es:9.3f}')

    out = {'note': 'd is the closest approach of the wall to the nucleus, in a0',
           'geometries': {}}
    print()
    for name in ('paraboloid', 'sphere'):
        d = [r[0] for r in rows[name]]
        y = [r[1] for r in rows[name]]
        c, p, q = fit_exponential(d, y)
        # decay constant with the polynomial held at the fitted power
        out['geometries'][name] = dict(
            c=float(c), poly_power=float(p), decay_rate_per_a0=float(q),
            d_a0=d, dE_hartree=y)
        print(f'  {name:11s}  dE ~ {np.exp(c):.4g} * (d/a0)^{p:.3f} '
              f'* exp(-{q:.4f} d/a0)')

    qp = out['geometries']['paraboloid']['decay_rate_per_a0']
    qs = out['geometries']['sphere']['decay_rate_per_a0']
    print(f'\n  decay rates: paraboloid {qp:.4f} /a0, sphere {qs:.4f} /a0, '
          f'difference {abs(qp-qs):.4f}')
    print(f'  reference value 2/a0 = 2.0000  (|psi|^2 decay of the 1s state)')

    out['summary'] = dict(q_paraboloid=qp, q_sphere=qs,
                          q_reference_2_over_a0=2.0,
                          max_deviation_from_2=max(abs(qp - 2), abs(qs - 2)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
