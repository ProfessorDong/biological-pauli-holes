"""Independent 50-digit check of the paraboloidal dipole asymptote, Eq. (expbound):

    P/(e a0)  ~  (xi0/a0)^3 exp(-xi0/a0) / 4.

Ports the double-precision solver's conventions (pauli_hole_hydrogen.py) to mpmath without
importing it: nu1 solves u0(nu1)*(1+nu1) = xi0/a0 with u0 the first zero of M(-nu1,1,u);
P/(e a0) = (nu/2)(2A0 - A2)/(A1 + A0), A_n = int_0^u0 e^-u M(-nu1,1,u)^2 u^n du.

Finding (2026-10-29): the ratio exact/asymptote is 0.973 at xi0/a0 = 10, crosses unity near 13,
peaks at 1.019 near 25 and is still 1.015 at 50. The leading law is right; its subleading
corrections decay slowly. No reported biological number uses the asymptote; those come from the
exact solver. Writes results/asymptote_check_50dps.json.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import json
from mpmath import mp, mpf, hyp1f1, quad, findroot, exp, log
mp.dps = 50
def M(nu1, u): return hyp1f1(-nu1, 1, u)
def first_zero(nu1, start):
    u, step = mpf(start), mpf('0.05')
    f0 = M(nu1, u)
    while True:
        u2 = u + step; f1 = M(nu1, u2)
        if f0 * f1 < 0: break
        u, f0 = u2, f1
        if u > 400: raise RuntimeError('no zero')
    return findroot(lambda t: M(nu1, t), (u, u2), solver='bisect', tol=mpf(10)**-45)
def solve(x):
    x = mpf(x); guess = x * exp(-x)
    resid = lambda n: first_zero(n, max(mpf(1), x - 8)) * (1 + n) - x
    nu1 = findroot(resid, (guess / 5, guess * 5), solver='bisect', tol=mpf(10)**-45)
    u0 = first_zero(nu1, max(mpf(1), x - 8)); nu = 1 + nu1
    A = [quad(lambda u: exp(-u) * M(nu1, u)**2 * u**n, [0, u0]) for n in range(3)]
    return nu / 2 * (2 * A[0] - A[2]) / (A[1] + A[0])
rows = []
for x in (8, 10, 12, 15, 20, 25, 30, 40, 50):
    P = solve(x); asym = mpf(x)**3 * exp(-mpf(x)) / 4
    rows.append({"xi0_over_a0": x, "P_over_ea0": mp.nstr(P, 20), "asymptote": mp.nstr(asym, 20), "ratio": float(P / asym)})
out = {"equation": "P/(e a0) ~ (xi0/a0)^3 exp(-xi0/a0)/4", "dps": mp.dps, "rows": rows,
       "ratio_span": [min(r["ratio"] for r in rows), max(r["ratio"] for r in rows)]}
with open(_os.path.join(_REPO, 'results', 'asymptote_check_50dps.json'), 'w') as f:
    json.dump(out, f, indent=2)
for r in rows: print(f"  xi0/a0 {r['xi0_over_a0']:3d}  ratio {r['ratio']:.6f}")
