#!/usr/bin/env python3
"""Where does our exchange curvature sit among the published stiffnesses for this enzyme?

WHY THIS EXISTS
  The manuscript reports k_exch^bio in N/m and compares it only against a water dimer we
  computed ourselves. That leaves the number floating: a referee cannot tell whether
  8 N/m is large or small for soybean lipoxygenase. Two independent groups have published
  quantities for THIS enzyme that carry the same units once converted, and the manuscript
  currently cites neither.

  SOUDACKOV AND HAMMES-SCHIFFER, Faraday Discuss. 195, 171 (2016), Table 1, fit the
  experimental KIE magnitudes and temperature dependences of WT and the I553 series to an
  analytical PCET rate expression, extracting a donor-acceptor equilibrium distance R0 and
  a donor-acceptor mode frequency Omega for an assumed effective mass M. A harmonic mode of
  mass M and frequency Omega has force constant k = M (2 pi c Omega)^2. That is a stiffness
  for the SAME enzyme obtained from EXPERIMENT rather than from a fragment calculation.

  SALNA, BENABBAS AND CHAMPION, J. Phys. Chem. A 121, 2199 (2017), report that forces of
  order 1 nN are needed to compress the donor-acceptor equilibrium from the isolated
  CH...[OH]- value of 3.04 A to 2.8 A. Force over displacement is a secant stiffness.

WHAT THIS DOES AND DOES NOT ESTABLISH
  The published quantities are LONGITUDINAL: they stiffen or compress the donor-acceptor
  axis. Ours is TRANSVERSE, the curvature of the wall contact perpendicular to that axis.
  They are different second derivatives of the same surface and the comparison is a
  calibration of scale, not an identity. Reporting them on one axis tells the reader what
  order of magnitude the enzyme works at; it does not license equating them.

Usage: stiffness_ladder.py     (pauli env)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
OUT = ROOT / 'results/sapt_bio/stiffness_ladder.json'

AMU = 1.66053906660e-27      # kg
C_CM = 2.99792458e10         # cm/s

# Soudackov and Hammes-Schiffer, Faraday Discuss. 195, 171 (2016), Table 1.
# Transcribed from the paper. R0 in angstrom, Omega in cm^-1, for two assumed masses.
FD2016 = {
    'WT':    {'M100': (2.77, 132.8), 'M10': (2.88, 368.2)},
    'I553V': {'M100': (2.89, 105.5), 'M10': (3.00, 316.3)},
    'I553A': {'M100': (2.97, 96.3),  'M10': (3.08, 295.1)},
    'I553L': {'M100': (3.00, 92.8),  'M10': (3.10, 266.4)},
    'I553G': {'M100': (3.24, 82.0),  'M10': (3.35, 257.5)},
}
# Salna, Benabbas and Champion, J. Phys. Chem. A 121, 2199 (2017).
CHAMPION_FORCE_N = 1e-9          # "forces on the order of ~1 nN"
CHAMPION_R0_A, CHAMPION_RC_A = 3.04, 2.80


def k_of(M_amu, omega_cm):
    return M_amu * AMU * (2 * np.pi * C_CM * omega_cm) ** 2


def main():
    print('Donor-acceptor mode force constant from the experimental KIE fits')
    print('  Soudackov and Hammes-Schiffer, Faraday Discuss. 195, 171 (2016), Table 1\n')
    print(f'{"variant":>8}{"R0 (A)":>9}{"Om (cm-1)":>11}{"k (N/m)":>10}   |'
          f'{"R0 (A)":>9}{"Om (cm-1)":>11}{"k (N/m)":>10}')
    print(f'{"":>8}{"M = 100 amu":^30}   |{"M = 10 amu":^30}')
    rows = {}
    for v, d in FD2016.items():
        r1, o1 = d['M100']
        r2, o2 = d['M10']
        k1, k2 = k_of(100, o1), k_of(10, o2)
        rows[v] = dict(R0_M100=r1, omega_M100=o1, k_M100=k1,
                       R0_M10=r2, omega_M10=o2, k_M10=k2)
        print(f'{v:>8}{r1:>9.2f}{o1:>11.1f}{k1:>10.1f}   |{r2:>9.2f}{o2:>11.1f}{k2:>10.1f}')

    kc = CHAMPION_FORCE_N / ((CHAMPION_R0_A - CHAMPION_RC_A) * 1e-10)
    print(f'\nSecant stiffness implied by the Champion compressive force')
    print(f'  ~{CHAMPION_FORCE_N*1e9:.0f} nN to go {CHAMPION_R0_A} -> {CHAMPION_RC_A} A'
          f'  =>  {kc:.0f} N/m  (order of magnitude only, the force is quoted as ~1 nN)')

    # our own numbers, read from disk rather than retyped
    ds = json.loads((ROOT / 'results/sapt_bio/donor_fragment/'
                            'donor_surrogate_summary.json').read_text())
    tags, km, kp = ds['tags'], ds['k_methane_Nm'], ds['k_pentadienyl_Nm']
    print(f'\nOur transverse exchange curvature, native wall, this work')
    print(f'  methane donor      : {min(km):.3f} to {max(km):.3f} N/m across seven variants')
    print(f'  real C5H8 donor    : {min(kp):.3f} to {max(kp):.3f} N/m across seven variants')
    wt = tags.index('WT')
    print(f'  wild type          : {km[wt]:.2f} (methane) / {kp[wt]:.2f} (C5H8) N/m')

    kFD = [rows[v]['k_M100'] for v in rows]
    print(f'\nTHE LADDER, all in N/m, longitudinal unless marked')
    print(f'  {min(kFD):.0f} to {max(kFD):.0f}   donor-acceptor mode, fitted to experimental '
          f'KIEs (Faraday Discuss. 2016)')
    print(f'  {kc:.0f}          donor-acceptor compression, fitted force (J. Phys. Chem. A 2017)')
    print(f'  40.5        water dimer, computed here, longitudinal')
    print(f'  {kp[wt]:.1f}        OUR wild-type wall contact, TRANSVERSE, real donor')
    print(f'\n  The transverse wall contact is roughly {rows["WT"]["k_M100"]/kp[wt]:.0f}x softer '
          f'than the wild-type donor-acceptor mode that experiment implies.')
    print('  Different second derivatives of the same surface, so this places our quantity '
          'on a scale;\n  it does not equate the two.')

    # The mutation that both datasets share, and it moves the two the OPPOSITE way.
    ours = {t: k for t, k in zip(tags, kp)}
    print(f'\nI553A is the one variant present in both datasets:')
    print(f'  fitted D-A mode  : WT {rows["WT"]["k_M100"]:.1f} -> I553A '
          f'{rows["I553A"]["k_M100"]:.1f} N/m   (softens {rows["WT"]["k_M100"]/rows["I553A"]["k_M100"]:.1f}x)')
    print(f'  our wall contact : WT {ours["WT"]:.1f} -> I553A {ours["I553A"]:.1f} N/m   '
          f'(stiffens {ours["I553A"]/ours["WT"]:.1f}x)')
    print('  These move in OPPOSITE directions. They are different coordinates, so this is')
    print('  not a contradiction, but it is the sharpest available check on whether the wall')
    print('  contact tracks the mode that actually carries the isotope effect. It does not.')

    OUT.write_text(json.dumps(dict(
        faraday_discuss_2016=rows, champion_secant_k_Nm=kc,
        champion_force_N=CHAMPION_FORCE_N,
        champion_R0_A=CHAMPION_R0_A, champion_Rc_A=CHAMPION_RC_A,
        ours_methane_Nm=dict(zip(tags, km)), ours_pentadienyl_Nm=dict(zip(tags, kp)),
        water_dimer_Nm=40.50,
        note=('published stiffnesses are longitudinal along the donor-acceptor axis; '
              'ours is transverse to it, so the ladder calibrates scale and does not '
              'equate the quantities')), indent=1))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
