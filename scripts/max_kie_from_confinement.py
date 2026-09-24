#!/usr/bin/env python3
"""How much of the kinetic isotope effect can confinement possibly move?

The bound of the manuscript is an energy, and enzymology is argued in kinetic
isotope effects. This converts one to the other.

THE MECHANISM, AND WHY IT IS THE ONLY ONE AVAILABLE
  The confinement energy is electronic: it is a shift of the Born-Oppenheimer
  surface, which depends on nuclear charge and not on nuclear mass, so by itself
  it produces no isotope effect at all. Confinement can reach the isotope effect
  only by changing the shape of the surface on which the nucleus moves, that is,
  by adding curvature to the transverse potential of the transferring hydrogen
  and so shifting its zero-point energy. That is the channel evaluated here.

THE CALCULATION
  Transverse motion of the transferring hydrogen sees a total curvature

      k_total = k_cov + k_exch,

  the covalent contribution from its own bond to the donor plus the exchange
  contribution from the wall. Only the second is what cavity mutation changes,
  and the SAPT campaign of the manuscript measures it directly: k_exch spans
  0.004 to 16.4 N/m across the seven JBC variants, a factor of 4000.

  For a harmonic mode, ZPE = (hbar/2) sqrt(k/m), so

      dZPE(H-D) = (hbar/2) sqrt(k_total/m_H) (1 - 1/sqrt(2)),

  and the isotope effect a variant can gain over another from this channel is

      KIE ratio = exp[ (dZPE_1 - dZPE_2) / (k_B T) ].

  This is deliberately GENEROUS in three ways: it assumes the transverse mode is
  fully present in the reactant and fully absent at the transition state, so
  nothing cancels; it uses the full 4000-fold measured span of k_exch; and the
  two-mode variant counts both transverse directions.

THE POINT
  k_exch is a perturbation on a much larger covalent curvature, and ZPE goes as
  sqrt(k). Adding 16 N/m to a covalent 100 N/m changes the frequency by 8 per
  cent, not by a factor. The 4000-fold engineered span in the wall descriptor
  therefore buys very little isotope effect, which is why a descriptor built on
  the wall can vary over three orders of magnitude while the isotope effect it
  is supposed to control does not follow.

Usage: max_kie_from_confinement.py     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path

import numpy as np

ROOT = Path(_REPO)
OUT = ROOT / 'results' / 'max_kie_from_confinement.json'

HBAR = 1.054571817e-34
AMU = 1.66053907e-27
KB = 1.380649e-23
J_KCAL = 1.0 / 4184.0 * 6.02214076e23      # J -> kcal/mol
T = 283.15
KT_KCAL = KB * T * J_KCAL

M_H, M_D = 1.008 * AMU, 2.014 * AMU
K_EXCH_MIN, K_EXCH_MAX = 0.004, 16.381     # N/m, native-fragment SAPT, this work
KIE_LO, KIE_HI = 62.0, 148.0               # observed ladder, Hu 2019 JBC panel


def dzpe_HD(k, n_modes=1):
    """Zero-point energy difference H minus D for n transverse modes, kcal/mol."""
    z = 0.5 * HBAR * (np.sqrt(k / M_H) - np.sqrt(k / M_D))
    return n_modes * z * J_KCAL


def main():
    print(f'kT at {T:.1f} K = {KT_KCAL:.4f} kcal/mol')
    print(f'observed KIE ladder {KIE_LO:.0f} to {KIE_HI:.0f}  ->  spread in '
          f'ddG = {KT_KCAL*np.log(KIE_HI/KIE_LO):.3f} kcal/mol '
          f'(factor {KIE_HI/KIE_LO:.2f})\n')

    rows = []
    print('  k_cov   modes    dZPE(min k_exch)  dZPE(max k_exch)   gain    max KIE ratio')
    for k_cov in (0.0, 50.0, 100.0, 200.0, 500.0):
        for nm in (1, 2):
            lo = dzpe_HD(k_cov + K_EXCH_MIN, nm)
            hi = dzpe_HD(k_cov + K_EXCH_MAX, nm)
            gain = hi - lo
            ratio = np.exp(gain / KT_KCAL)
            rows.append(dict(k_cov_Nm=k_cov, n_modes=nm, dzpe_lo=lo, dzpe_hi=hi,
                             gain_kcal=gain, max_kie_ratio=ratio))
            print(f'  {k_cov:5.0f}     {nm}      {lo:8.4f}          {hi:8.4f}    '
                  f'{gain:7.4f}      {ratio:6.3f}')

    ref = [r for r in rows if r['k_cov_Nm'] == 100.0 and r['n_modes'] == 2][0]
    obs = KIE_HI / KIE_LO
    print(f"\n  representative case, covalent 100 N/m and both transverse modes:")
    print(f"     confinement can move the KIE by at most a factor "
          f"{ref['max_kie_ratio']:.3f}")
    print(f"     the panel actually spans a factor {obs:.2f}")
    print(f"     confinement can account for "
          f"{100*np.log(ref['max_kie_ratio'])/np.log(obs):.1f} % of the observed "
          f"spread in ln KIE")
    print(f"\n  the isolated-mode limit, k_cov = 0, is reported for completeness only:")
    iso = [r for r in rows if r['k_cov_Nm'] == 0.0 and r['n_modes'] == 1][0]
    print(f"     it gives {iso['max_kie_ratio']:.2f}, but a transferring hydrogen is")
    print(f"     covalently bound and its transverse curvature is never zero, so this")
    print(f"     row is an upper limit of no physical relevance.")

    OUT.write_text(json.dumps(dict(
        T_K=T, kT_kcal=KT_KCAL, k_exch_range_Nm=[K_EXCH_MIN, K_EXCH_MAX],
        observed_KIE=[KIE_LO, KIE_HI], observed_factor=obs,
        rows=rows, representative=ref,
        fraction_of_observed_lnKIE_spread=float(
            np.log(ref['max_kie_ratio']) / np.log(obs))), indent=1))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
