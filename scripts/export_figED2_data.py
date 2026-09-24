#!/usr/bin/env python3
"""Export Figure 11's three panels to pgfplots tables from the convergence analysis.

Every value comes from results/pimd_convergence/analysis/B1_convergence_analysis.json, the
same file verify_figures.py checks the manuscript against.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path

ROOT = Path(_REPO)
d = json.loads((ROOT / 'results/pimd_convergence/analysis/B1_convergence_analysis.json').read_text())
pc, cv = d['per_condition'], d['convergence']
Ps = (8, 16, 32)

with (ROOT / 'figures/figED2_occupancy.dat').open('w') as f:
    f.write('# Figure 11a: per-condition occupancy N~ = exp H[rho] against bead count.\n')
    f.write('P\tI553A_H\tI553A_D\tI552A_H\tI552A_D\n')
    for P in Ps:
        f.write(f'{P}\t' + '\t'.join(f"{pc[f'P{P}_{s}_{i}']['N_eff']:.4f}"
                                     for s in ('I553A','I552A') for i in ('H','D')) + '\n')

with (ROOT / 'figures/figED2_shift.dat').open('w') as f:
    f.write('# Figure 11b: per-system isotope shift ln(N~_H/N~_D).\n')
    f.write('P\tI553A\tI552A\n')
    for P in Ps:
        k = cv[f'P{P}']
        f.write(f"{P}\t{k['delta_ln_Neff_I553A']:.6f}\t{k['delta_ln_Neff_I552A']:.6f}\n")

with (ROOT / 'figures/figED2_contrast.dat').open('w') as f:
    f.write('# Figure 11c: between-mutant contrast with block-bootstrap 95% intervals\n')
    f.write('# (30-frame blocks, 3000 resamples) and P(sign>0).\n')
    f.write('P\tDD\telo\tehi\tpsign\n')
    for P in Ps:
        k = cv[f'P{P}']
        f.write(f"{P}\t{k['DD_I553A_minus_I552A']:.6f}\t"
                f"{k['DD_I553A_minus_I552A']-k['DD_ci95_lo']:.6f}\t"
                f"{k['DD_ci95_hi']-k['DD_I553A_minus_I552A']:.6f}\t{k['P_sign_positive']:.2f}\n")

print('wrote figED2_occupancy.dat, figED2_shift.dat, figED2_contrast.dat')
for P in Ps:
    k = cv[f'P{P}']
    print(f"  P={P:2d}  DD={k['DD_I553A_minus_I552A']:+.5f}  "
          f"ci95 [{k['DD_ci95_lo']:+.4f},{k['DD_ci95_hi']:+.4f}]  P(sign)={k['P_sign_positive']:.2f}")
