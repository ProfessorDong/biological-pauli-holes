#!/usr/bin/env python3
"""Benchmark our exchange pipeline against the exact asymptotic exchange coupling of H2.

WHY THIS EXISTS
  The manuscript's locality claim rests on exchange being exponentially short-ranged. We
  validated the CONFINEMENT side against Laughlin et al.'s published spherical-box energies.
  This validates the EXCHANGE side against the one case where the asymptotics are known
  exactly: Herring and Flicker, Phys. Rev. 134, A362 (1964), Eq. (19),

      J   = -0.821 R^(5/2) e^(-2R),      2J = -1.641 R^(5/2) e^(-2R)

  with 2J = E(singlet) - E(triplet), energies in hartree and R in bohr. Their point is that
  the Heitler-London form is wrong even asymptotically, by a coefficient 1.47 against the
  correct 0.821, because it mistreats the mutual avoidance of the exchanging electrons.

WHY IT MATTERS TO THIS PAPER
  It tests two things a referee will ask about. First, that a full-CI treatment reproduces a
  known exact exchange asymptote, so the exponential form we rely on is not an artifact of
  our own fitting. Second, and more usefully, it shows how slowly the leading asymptotic is
  approached: the remainder in Eq. (19) is one power of R down, so the ratio to the leading
  term drifts toward unity only gradually. That is the same lesson the Laughlin comparison
  taught for the sphere, and it is why we report local effective exponents rather than
  asymptotic limits.

METHOD
  Full CI is exact within the basis for a two-electron system, so the only error is basis
  incompleteness, which we probe by repeating in a larger basis. The singlet and triplet are
  computed at the same geometry in the same basis, so basis-set superposition error largely
  cancels in the difference.

Usage: herring_flicker_benchmark.py [--basis aug-cc-pVQZ] [--rmax 11]
"""
import json, math, sys
from pathlib import Path
import psi4

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
OUT = ROOT / 'results' / 'herring_flicker'
OUT.mkdir(parents=True, exist_ok=True)
HF_COEFF, HF_POWER, HF_RATE = 1.641, 2.5, 2.0     # Herring & Flicker Eq. (19), for |2J|


def two_J(R, basis):
    """E(singlet) - E(triplet) at separation R (bohr), by full CI."""
    out = {}
    for mult, ref in ((1, 'rhf'), (3, 'rohf')):
        psi4.core.clean()
        psi4.geometry(f"0 {mult}\nH 0 0 0\nH 0 0 {R}\nunits bohr\n"
                      f"symmetry c1\nno_reorient\nno_com\n")
        psi4.set_options({'basis': basis, 'reference': ref,
                          'e_convergence': 1e-12, 'd_convergence': 1e-11,
                          'scf_type': 'pk', 'freeze_core': 'false'})
        out[mult] = psi4.energy('fci')
    return out[1] - out[3], out[1], out[3]


def main():
    basis = 'aug-cc-pVQZ'
    rmax = 11
    if '--basis' in sys.argv: basis = sys.argv[sys.argv.index('--basis') + 1]
    if '--rmax' in sys.argv:  rmax = int(sys.argv[sys.argv.index('--rmax') + 1])
    psi4.set_memory('8 GB'); psi4.core.be_quiet()
    rows = {}
    print(f'FCI/{basis}   2J = E(S) - E(T),  hartree,  R in bohr\n')
    print(f'{"R":>5}{"E(S)":>20}{"E(T)":>20}{"2J":>14}{"HF Eq.(19)":>14}{"ratio":>8}')
    for R in range(6, rmax + 1):
        tj, es, et = two_J(float(R), basis)
        pred = -HF_COEFF * R ** HF_POWER * math.exp(-HF_RATE * R)
        rows[R] = dict(E_singlet=es, E_triplet=et, twoJ=tj,
                       herring_flicker=pred, ratio=tj / pred)
        print(f'{R:>5}{es:>20.12f}{et:>20.12f}{tj:>14.4e}{pred:>14.4e}{tj/pred:>8.4f}')
    (OUT / f'benchmark_{basis}.json').write_text(json.dumps(
        dict(basis=basis, method='FCI',
             reference='Herring & Flicker, Phys. Rev. 134, A362 (1964), Eq. (19)',
             asymptote='2J = -1.641 R^(5/2) exp(-2R)', rows=rows), indent=1))
    print(f'\nwrote {OUT}/benchmark_{basis}.json')


if __name__ == '__main__':
    main()
