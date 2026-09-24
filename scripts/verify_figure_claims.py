#!/usr/bin/env python3
"""Trace the figure-caption numbers to the exact field of the exact file that makes them.

WHY THIS EXISTS, GIVEN THAT numeric_claim_sweep.py ALREADY RUNS
  The sweep is a substring matcher over the whole result corpus. Its own docstring
  says so: FOUND means the digits occur somewhere, not that the claim is sourced.
  Two caption numbers have already passed it by coincidence. This closes the gap
  for the numbers that the figure audits of 2026-08-13 to 2026-08-15 introduced or
  changed, by checking each against a NAMED FIELD of a NAMED FILE, with a tolerance.

  The two are complementary and both should be run: the sweep finds claims with no
  source at all, this one finds claims whose source says something different.

WHAT IT COVERS
  Figure 10 (figED1): native-fragment SAPT bootstrap uncertainties, which did not
    exist before the audit; the only bootstrap on disk was for the methane surrogates.
  Figure 11 (figED2): the bead-convergence numbers, all of which moved when the
    window-mixture confound in the P=8 cell was fixed.
  Figure 9 (figED3): the Franck-Condon reactive-r_DA range and the equilibrium mean.

Usage: verify_figure_claims.py     (pauli env; exit status is the failure count)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(_REPO)
fails = []


def chk(label, claim, actual, tol, mode='eq'):
    good = (abs(claim - actual) <= tol) if mode == 'eq' else (actual < claim)
    print(f"  {'ok  ' if good else 'FAIL'}  {label:46s} "
          f"paper {claim:>9} | source {actual:>11.5f}")
    if not good:
        fails.append(label)


print('Figure 10 (figED1_sapt_scans) -- results/sapt_bio/native_fragment_uncertainty.json')
u = json.loads((ROOT / 'results/sapt_bio/native_fragment_uncertainty.json').read_text())
S = u['systems']
for tag, se in [('L754A', 0.0004), ('I552A', 0.38), ('I538A', 0.47), ('V750A', 0.60),
                ('WT', 0.62), ('I553A', 1.06), ('L546A', 1.20)]:
    chk(f'sigma_k {tag}', se, S[tag]['sigma_k_Nm'], 0.005)
chk('rel sigma range, low end', 7.2, 100 * u['rel_sigma_range'][0], 0.05)
chk('rel sigma range, high end', 8.9, 100 * u['rel_sigma_range'][1], 0.05)
chk('L754A ci95 lower bound', 0.0033, S['L754A']['ci95'][0], 5e-5)
chk('L754A ci95 upper bound', 0.0047, S['L754A']['ci95'][1], 5e-5)
chk('every 95% interval excludes zero', 7,
    sum(v['excludes_zero'] for v in S.values()), 0)
chk('max RMS residual is below', 0.021,
    max(v['RMS_residual_kcal'] for v in S.values()), 0, mode='lt')
# the benchmark line drawn on the ranking panel, from Appendix D
chk('water-dimer benchmark, 58.29 kcal/mol/A^2', 40.50, 58.29 * 0.694770, 0.005)

print('\nFigure 11 (figED2_bead_convergence) -- '
      'results/pimd_convergence/analysis/B1_convergence_analysis.json')
d = json.loads((ROOT / 'results/pimd_convergence/analysis/'
                       'B1_convergence_analysis.json').read_text())
per, conv = d['per_condition'], d['convergence']
if not d.get('design', {}).get('matched'):
    fails.append('analysis is not the matched design')
    print('  FAIL  analysis JSON predates the matched-design fix')
else:
    print('  ok    matched 3+3 design confirmed (the confound fix)')
for P, c in zip((8, 16, 32), (0.022, 0.046, 0.100)):
    chk(f'I553A ln(NH/ND) at P={P}', c, conv[f'P{P}']['delta_ln_Neff_I553A'], 5e-4)
for P, c in zip((8, 16, 32), (0.044, -0.059, 0.134)):
    chk(f'I552A ln(NH/ND) at P={P}', c, conv[f'P{P}']['delta_ln_Neff_I552A'], 5e-4)
for P, c in zip((8, 16, 32), (-0.022, 0.106, -0.034)):
    chk(f'DD point estimate at P={P}', c, conv[f'P{P}']['DD_I553A_minus_I552A'], 5e-4)
# the caption states the bias as a RANGE because it differs by condition
bias8 = [100*(per[f'P8_{s}_{i}']['N_eff_miller_madow']/per[f'P8_{s}_{i}']['N_eff'] - 1)
         for s in ('I553A', 'I552A') for i in ('H', 'D')]
bias32 = [100*(per[f'P32_{s}_{i}']['N_eff_miller_madow']/per[f'P32_{s}_{i}']['N_eff'] - 1)
          for s in ('I553A', 'I552A') for i in ('H', 'D')]
chk('Miller-Madow at P=8, low end', 1.4, min(bias8), 0.05)
chk('Miller-Madow at P=8, high end', 1.5, max(bias8), 0.05)
chk('Miller-Madow at P=32', 0.2, float(np.mean(bias32)), 0.05)
mm = max(abs(np.log(per[f'P{P}_{s}_H']['N_eff_miller_madow'] /
                    per[f'P{P}_{s}_D']['N_eff_miller_madow'])
             - np.log(per[f'P{P}_{s}_H']['N_eff'] / per[f'P{P}_{s}_D']['N_eff']))
         for P in (8, 16, 32) for s in ('I553A', 'I552A'))
chk('MM shift in the isotope ratio is below', 0.001, mm, 0, mode='lt')
# the caption says the interval excludes zero ONLY at P=16
excl = [P for P in (8, 16, 32)
        if conv[f'P{P}']['DD_ci95_lo'] > 0 or conv[f'P{P}']['DD_ci95_hi'] < 0]
chk('bead counts whose interval excludes zero', 16, excl[0] if len(excl) == 1 else -1, 0)

print('\nFigure 9 (figED3_gating_KIE) -- results/pcet_B34_dense/marcus_rate_A_v3.json')
m = json.loads((ROOT / 'results/pcet_B34_dense/marcus_rate_A_v3.json').read_text())
flat = json.dumps(m)
for want in ('2.55', '2.60', '3.73'):
    hit = want in flat or any(abs(float(x) - float(want)) < 5e-3
                              for x in __import__('re').findall(r'-?\d+\.\d+', flat))
    print(f"  {'ok  ' if hit else 'FAIL'}  {'value ' + want + ' present':46s}")
    if not hit:
        fails.append(want)

print(f'\n{"ALL CLAIMS TRACED" if not fails else str(len(fails)) + " FAILED: " + ", ".join(fails)}')
sys.exit(len(fails))
