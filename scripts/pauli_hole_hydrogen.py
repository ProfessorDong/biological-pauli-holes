"""Hydrogen atom confined in a paraboloidal Pauli hole: reproduce and extend You & Ye.

Solves the problem posed in You, Liu & Gao, Chem. Phys. Impact 4 (2022) 100058,
whose Table 1 is quoted from You & Ye, Phys. Rev. B 41, 8180 (1990).

MODEL
  Free-hydrogen Hamiltonian, parabolic coordinates (xi, eta, phi):
      x = sqrt(xi*eta) cos(phi),  y = sqrt(xi*eta) sin(phi),  z = (eta - xi)/2
  The Pauli wall is the paraboloid of revolution xi = xi0, impenetrable to the
  electron, giving the boundary condition  psi(xi0, eta, phi) = 0.

  Separated solution (their Eq. 6), with rho1 = xi/(2 a0 nu), rho2 = eta/(2 a0 nu):
      psi ~ exp(-rho1-rho2) * M(-nu1, |m|+1, 2 rho1) * M(-nu2, |m|+1, 2 rho2) * exp(i m phi)
  Natural boundary condition at eta -> inf forces nu2 = 0, 1, 2, ...
  Quantum-number relation (their Eq. 10):  nu1 + nu2 + |m| + 1 = nu
  Energy (their Eq. 9):  E = -e^2 / (2 a0 nu^2)  =>  I = |E| = 13.6057 eV / nu^2

  GROUND STATE: nu2 = m = 0, so nu = 1 + nu1, and the wall condition is
      M(-nu1, 1, u0) = 0   with   u0 = 2 rho1|_wall = xi0 / (a0 nu).
  Given xi0/a0 this is one scalar equation for nu1, solved here by bisection.

DIPOLE
  P = e <z>, with dv = (xi+eta)/4 dxi deta dphi and xi integrated only to xi0.
  Substituting u = xi/(a0 nu), w = eta/(a0 nu) and doing the w integrals analytically
  (int w^k e^-w dw = k!) gives the closed form used below:

      P/(e a0) = (nu/2) * (2 A0 - A2) / (A1 + A0),
      A_n = int_0^{u0} exp(-u) M(-nu1,1,u)^2 u^n du.

  Free-atom limit nu1 -> 0, u0 -> inf: A0=1, A1=1, A2=2  =>  P = 0, as required.

LARGE-xi0 ASYMPTOTICS (derived in the docstring of asymptotic_P below)
      P/(e a0)  ~  (xi0/a0)^3 * exp(-xi0/a0) / 4
  i.e. EXPONENTIAL decay, not a power law.

Usage: pauli_hole_hydrogen.py
"""
import numpy as np
from scipy.special import hyp1f1
from scipy.optimize import brentq
from scipy.integrate import quad

RY_eV = 13.605693122994    # e^2 / (2 a0)
A0_ANG = 0.529177210903    # Bohr radius in Angstrom


def M(nu1, u):
    return hyp1f1(-nu1, 1.0, u)


def first_zero_u0(nu1, u_max=200.0, n_scan=200000):
    """First positive zero of M(-nu1, 1, u), located by a fine forward scan.

    A scan is used rather than an expanding bracket because hyp1f1 overflows to
    inf/nan at large argument, and because for large nu1 the function oscillates
    (Laguerre-like) so an expanding bracket can skip past the FIRST zero.
    """
    if nu1 <= 0:
        return np.inf
    # log-spaced: first zero ranges over many decades (~1/nu1 for large nu1,
    # ~ln(1/nu1) for small nu1), so a linear grid aliases at one end or the other
    grid = np.geomspace(1e-8, u_max, n_scan)
    vals = M(nu1, grid)
    ok = np.isfinite(vals)
    cand = np.where(ok & (vals <= 0.0))[0]
    cand = cand[cand > 0]
    for i in cand:                       # first index with a genuine + -> - crossing
        if ok[i-1] and vals[i-1] > 0.0:
            # NOTE: brentq passes the root variable FIRST, so M(nu1, u) must be
            # wrapped rather than passed with args=(nu1,), which would evaluate
            # M(u, nu1) with the arguments transposed.
            return brentq(lambda u: M(nu1, u), grid[i-1], grid[i],
                          xtol=1e-14, rtol=8.9e-16)
    return np.inf


def solve_nu1(xi0_over_a0):
    """Self-consistency: first zero of M(-nu1,1,.) must equal xi0/(a0*(1+nu1))."""
    def resid(nu1):
        return first_zero_u0(nu1) * (1.0 + nu1) - xi0_over_a0
    # resid increases as nu1 -> 0 (u0 -> inf) and decreases for large nu1
    lo, hi = 1e-7, 1e-7
    while resid(hi) > 0:
        hi *= 1.6
        if hi > 1e4:
            raise RuntimeError('no bracket')
    while resid(lo) < 0:
        lo /= 1.6
        if lo < 1e-14:
            raise RuntimeError('no bracket')
    return brentq(resid, lo, hi, xtol=1e-14, rtol=1e-14)


def dipole(nu1, u0):
    nu = 1.0 + nu1
    def An(n):
        f = lambda u: np.exp(-u) * M(nu1, u)**2 * u**n
        val, _ = quad(f, 0.0, u0, limit=400)
        return val
    A0_, A1_, A2_ = An(0), An(1), An(2)
    return 0.5 * nu * (2.0*A0_ - A2_) / (A1_ + A0_)


def solve(xi0_over_a0):
    nu1 = solve_nu1(xi0_over_a0)
    u0 = first_zero_u0(nu1)
    nu = 1.0 + nu1
    return dict(xi0=xi0_over_a0, nu1=nu1, nu=nu, u0=u0,
                I_eV=RY_eV/nu**2, P=dipole(nu1, u0))


def asymptotic_P(x):
    """Leading large-xi0 behaviour.

    For nu1 -> 0, M(-nu1,1,u) = 1 - nu1*h(u) + O(nu1^2) with
    h(u) = sum_{n>=1} u^n/(n n!) ~ e^u/u.  The wall condition M=0 gives
    nu1 ~ u0 e^{-u0}, so nu1 vanishes EXPONENTIALLY in u0 (hence in xi0).
    Expanding the A_n to first order in nu1 leaves 2A0 - A2 ~ u0^3 e^{-u0}
    and A1 + A0 -> 2, and nu -> 1, so P/(e a0) ~ u0^3 e^{-u0}/4 with u0 -> xi0/a0.
    """
    return x**3 * np.exp(-x) / 4.0


PUBLISHED = [  # You & Ye Table 1 as reproduced in Chem. Phys. Impact 4 (2022) 100058
    (5.32, 12.57, 0.238), (3.13, 9.44, 0.607), (2.53, 6.94, 0.893),
    (2.26, 5.31, 1.14),  (2.11, 4.20, 1.36),  (2.00, 3.40, 1.58),
    (1.84, 2.18, 2.11),  (1.76, 1.51, 2.63),  (1.66, 0.85, 3.64),
    (1.61, 0.54, 4.65),
]

if __name__ == '__main__':
    print('=' * 78)
    print('VALIDATION against the published Table 1')
    print('=' * 78)
    print(f"{'xi0/a0':>7} | {'I pub':>7}{'I calc':>8}{'dev':>8} | "
          f"{'P pub':>7}{'P calc':>8}{'dev':>8}")
    print('-' * 78)
    dI, dP = [], []
    for x, Ip, Pp in PUBLISHED:
        r = solve(x)
        eI = 100*(r['I_eV']-Ip)/Ip
        eP = 100*(r['P']-Pp)/Pp
        dI.append(abs(eI)); dP.append(abs(eP))
        print(f"{x:>7.2f} | {Ip:>7.2f}{r['I_eV']:>8.2f}{eI:>7.1f}% | "
              f"{Pp:>7.3f}{r['P']:>8.3f}{eP:>7.1f}%")
    print(f"\nmean |deviation|:  I {np.mean(dI):.1f}%   P {np.mean(dP):.1f}%")

    print('\n' + '=' * 78)
    print('EXTENSION to the enzyme range (what our SLO active site needs)')
    print('=' * 78)
    print(f"{'xi0/a0':>7}{'xi0 (A)':>9}{'I (eV)':>9}{'raise':>8}"
          f"{'P (e a0)':>10}{'asympt':>9}{'E@2.55A':>11}{'% of 150':>10}")
    print('-' * 78)
    k = 8.9875517873681764e9; ea0 = 8.4783e-30
    for x in [5.32, 6.0, 6.41, 6.79, 7.0, 7.49, 8.0, 8.39, 8.68, 8.99, 9.14, 10.0]:
        r = solve(x)
        E = k*2*r['P']*ea0/(2.55e-10)**3 / 1e8
        print(f"{x:>7.2f}{x*A0_ANG:>9.2f}{r['I_eV']:>9.3f}{RY_eV-r['I_eV']:>8.3f}"
              f"{r['P']:>10.4f}{asymptotic_P(x):>9.4f}{E:>8.1f} MV/cm{100*E/150:>9.1f}%")
