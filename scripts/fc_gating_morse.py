"""Franck-Condon gating with MORSE wells (supersedes the harmonic treatment).

WHY THIS SUPERSEDES THE HARMONIC CALCULATION
Soudackov and Hammes-Schiffer (J. Chem. Phys. 143, 194101 (2015)) warn explicitly
that harmonic treatments overestimate the vibronic coupling at short proton
donor-acceptor distances, because a harmonic well has no repulsive wall to
prevent unphysically close donor-acceptor contact, and that "the use of more
realistic potentials ... such as Morse-like potentials with repulsive walls
preventing the proton donor and acceptor from unphysical close contact" removes
that artefact. Our first gating estimate used harmonic wells at
k = 500 N/m and inferred a wild-type reactive distance of 2.55 A, which is far
inside the 3.22 A van der Waals C...O contact. That number is therefore suspect
in exactly the way this reference describes.

Meyer and Klinman (Chem. Phys. 319, 283 (2005)) saw the same effect empirically:
fitting the SLO isotope data with harmonic hydrogenic wavefunctions gave a
transfer distance of 0.75 A, while Morse wavefunctions gave 1.02 A. The
more realistic potential yields a LONGER, less compressed transfer distance.

METHOD
Both wells are Morse potentials with the parameters used in the established SLO
PCET treatment (Hatcher, Soudackov & Hammes-Schiffer, JACS 2007):
    D_CH = 77 kcal/mol, beta_CH = 2.068 A^-1, R0_CH = 1.09 A
    D_OH = 82 kcal/mol, beta_OH = 2.442 A^-1, R0_OH = 0.96 A
chosen there to reproduce the experimental 2900 and 3500 cm^-1 stretches. The
1D nuclear Schrodinger equation is solved numerically on each well for H and D
(the same validated solver used for the QM proton scans), the ground-state
overlap is integrated numerically, and KIE_FC = |S_H|^2/|S_D|^2 is swept over
r_DA. No expansion of the coupling is made, which is the approach Soudackov and
Hammes-Schiffer recommend over linear or quadratic expansion.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from proton_schrodinger import solve_1d

# Hatcher/Soudackov/Hammes-Schiffer SLO Morse parameters
D_CH, B_CH, R0_CH = 77.0, 2.068, 1.09      # kcal/mol, A^-1, A
D_OH, B_OH, R0_OH = 82.0, 2.442, 0.96
M_H, M_D = 1.008, 2.014
HBAR = 1.054571817e-34; AMU = 1.66053906660e-27; JKCAL = 4184.0/6.02214076e23


def morse(q, D, beta, q0):
    x = np.exp(-beta * (q - q0))
    return D * (x*x - 2.0*x)


def ground_state(q, V, m):
    eps, psi = solve_1d(q, V, m, K=1)
    p = psi[0]
    p = p / np.sqrt(np.trapezoid(p*p, q))
    return float(eps[0]), p


def fc_kie(r_DA, npts=4000):
    """Franck-Condon KIE from numerically solved Morse ground states."""
    # q is measured FROM THE DONOR, so the product O-H bond length is (r_DA - q).
    # CORRECTION 2026-09-17: the product Morse must be evaluated in its own bond
    # coordinate. Writing morse(q, D_OH, B_OH, r_DA - R0_OH) puts the minimum in the
    # right place but mirrors the well: the steep repulsive wall then faces decreasing
    # q (H moving AWAY from the acceptor) and the flat dissociative limb faces
    # increasing q (H driven INTO the acceptor), which is backwards. A harmonic well is
    # reflection-symmetric, so the harmonic regression tests could not see this.
    qA = r_DA - R0_OH
    lo, hi = min(R0_CH, qA) - 0.9, max(R0_CH, qA) + 0.9
    q = np.linspace(lo, hi, npts)
    VR = morse(q, D_CH, B_CH, R0_CH)
    VP = morse(r_DA - q, D_OH, B_OH, R0_OH)
    out = {}
    for m, tag in ((M_H, 'H'), (M_D, 'D')):
        _, pR = ground_state(q, VR, m)
        _, pP = ground_state(q, VP, m)
        out[tag] = abs(float(np.trapezoid(pR*pP, q)))
    return out['H'], out['D'], (out['H']/out['D'])**2 if out['D'] > 0 else np.nan


def harmonic_kie(r_DA, k_Nm=500.0):
    """The superseded harmonic result, for explicit comparison."""
    def alpha(m):
        return math.sqrt(k_Nm*m*AMU)/HBAR*1e-20
    qD, qA = R0_CH, r_DA - 0.98
    S = {}
    for m, tag in ((M_H, 'H'), (M_D, 'D')):
        a = alpha(m)
        S[tag] = math.sqrt(2*math.sqrt(a*a)/(2*a))*math.exp(-a*a*(qA-qD)**2/(2*(2*a)))
    return (S['H']/S['D'])**2


import sys as _sys
if __name__ == '__main__' and '--test' not in _sys.argv:
    KIE = {'WT':66, 'I839A':62, 'L754A':106, 'V750A':62,
           'I538A':100, 'L546A':131, 'I553A':148, 'I552A':66}

    grid = np.linspace(2.40, 4.20, 91)
    rows = []
    for r in grid:
        sh, sd, kie = fc_kie(float(r))
        rows.append((float(r), sh, sd, float(kie)))
    R = np.array([x[0] for x in rows]); K = np.array([x[3] for x in rows])

    good = np.isfinite(K) & (K > 1)
    R, K = R[good], K[good]
    print(f'{"r_DA":>7} {"transfer dq":>12} {"KIE_FC(Morse)":>15} {"KIE_FC(harm)":>13}')
    for r, k in list(zip(R, K))[::10]:
        print(f'{r:>7.2f} {r-R0_CH-R0_OH:>12.3f} {k:>15.3g} {harmonic_kie(r):>13.3g}')

    from scipy.interpolate import interp1d
    inv = interp1d(np.log(K), R, bounds_error=False)
    inv_h = interp1d(np.log([harmonic_kie(r) for r in R]), R, bounds_error=False)

    print(f'\n{"variant":<8}{"KIE":>5}{"r_DA Morse":>12}{"dq Morse":>10}'
          f'{"r_DA harm":>11}{"dq harm":>9}')
    res = {}
    for v, k in KIE.items():
        rm = float(inv(math.log(k))); rh = float(inv_h(math.log(k)))
        res[v] = dict(KIE=k, r_DA_morse=rm, dq_morse=rm-R0_CH-R0_OH,
                      r_DA_harm=rh, dq_harm=rh-R0_CH-0.98)
        print(f'{v:<8}{k:>5}{rm:>12.3f}{rm-R0_CH-R0_OH:>10.3f}'
              f'{rh:>11.3f}{rh-R0_CH-0.98:>9.3f}')

    wt = res['WT']
    print(f'\n  WT transfer distance dq:')
    print(f'    this work, Morse      : {wt["dq_morse"]:.3f} A')
    print(f'    this work, harmonic   : {wt["dq_harm"]:.3f} A  (superseded)')
    print(f'    Meyer-Klinman, Morse  : 1.02 A   (fitted to KIE temperature data)')
    print(f'    Meyer-Klinman, harm.  : 0.75 A')
    print(f'    van der Waals contact : 1.17 A   (at r_DA = 3.22 A)')

    out = Path(_REPO + '/results/fc_gating_morse.json')
    out.write_text(json.dumps(dict(
        morse_params=dict(D_CH=D_CH, beta_CH=B_CH, R0_CH=R0_CH,
                          D_OH=D_OH, beta_OH=B_OH, R0_OH=R0_OH),
        per_variant=res,
        curve=dict(r_DA=R.tolist(), KIE_FC=K.tolist())), indent=2))
    print(f'\nwrote {out}')


# ---------------------------------------------------------------------------
# Regression tests for the Franck-Condon isotope ordering.
#
# WHY THESE EXIST: the manuscript printed this expression with the mass ordering
# reversed, in the main text as (alpha_H - alpha_D) and in the appendix as
# (sqrt(m_H) - sqrt(m_D)), both of which give KIE < 1 for every nonzero
# displacement; the appendix form also dropped a factor of 1/2. The CODE was
# always right, so no reported number changed, but nothing in the repository
# would have caught the disagreement between the code and the typeset equation.
# These would have.
#
# The zero-displacement and closed-form tests use harmonic_kie, whose two wells
# share a curvature. They do NOT hold for the Morse branch, whose C-H and O-H
# wells differ in depth and width, so its overlap is not unity even at dq = 0.
#
# Run: python scripts/fc_gating_morse.py --test
# ---------------------------------------------------------------------------
R_COINCIDENT = 1.09 + 0.98          # dq = 0


def _analytic_kie(r_DA, k=500.0, m_H=1.008, m_D=2.014):
    """exp[(gamma_D - gamma_H) dq^2 / 2] with gamma_X = sqrt(k m_X)/hbar."""
    import math
    HBAR, AMU = 1.054571817e-34, 1.66053906660e-27
    dq = (r_DA - 1.09 - 0.98) * 1e-10
    def g(m):
        return math.sqrt(k * m * AMU) / HBAR
    return math.exp((g(m_D) - g(m_H)) * dq * dq / 2)


def test_product_well_is_oriented_toward_the_acceptor():
    """The product O-H Morse must be steep where H is driven INTO the acceptor.

    This is the check that was missing when the product well was mirrored. It cannot be
    written with a harmonic well, which is symmetric about its minimum; it needs the
    Morse asymmetry. With q measured from the donor and the acceptor at r_DA, moving the
    hydrogen 0.25 A PAST the product minimum toward the acceptor must cost much more than
    moving it 0.25 A back toward the donor.
    """
    r_DA = 2.70
    qmin = r_DA - R0_OH                       # product minimum, in donor-based q
    toward_acceptor = morse(r_DA - (qmin + 0.25), D_OH, B_OH, R0_OH)
    toward_donor = morse(r_DA - (qmin - 0.25), D_OH, B_OH, R0_OH)
    assert toward_acceptor > toward_donor + 5.0, (
        f'product well is mirrored: pushing H into the acceptor costs '
        f'{toward_acceptor:.2f} kcal/mol but pulling it away costs {toward_donor:.2f}')


def test_zero_displacement_gives_unity():
    assert abs(harmonic_kie(R_COINCIDENT) - 1.0) < 1e-9


def test_heavier_isotope_gives_kie_above_one():
    for r in (2.2, 2.5, 2.8, 3.2, 3.8):
        assert harmonic_kie(r) > 1.0, (r, harmonic_kie(r))
        assert fc_kie(r)[2] > 1.0, (r, fc_kie(r)[2])


def test_reciprocal_symmetry():
    """KIE_{D/H} is the reciprocal of KIE_{H/D}; fc_kie returns (S_H, S_D, ratio)."""
    for r in (2.5, 3.0):
        sh, sd, kie = fc_kie(r)
        assert abs((sd / sh) ** 2 - 1.0 / kie) < 1e-9 / kie, r


def test_matches_closed_form_harmonic():
    for r in (2.5, 2.6, 3.0):
        assert abs(_analytic_kie(r) / harmonic_kie(r) - 1.0) < 1e-6, r


def _run_tests():
    import sys as _s
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    bad = 0
    for f in fns:
        try:
            f(); print(f'  ok    {f.__name__}')
        except AssertionError as e:
            print(f'  FAIL  {f.__name__}: {e}'); bad += 1
    print(f'{len(fns)-bad}/{len(fns)} passed')
    _s.exit(bad)


if __name__ == '__main__' and '--test' in _sys.argv:
    _run_tests()
