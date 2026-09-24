#!/usr/bin/env python3
"""Export Figure 1's data from the solver output to a pgfplots table.

The TikZ figure reads this file rather than carrying numbers in its source, so the artwork cannot
drift from the solver. Regenerate after any change to confinement_universality.py.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path
import numpy as np

ROOT = Path(_REPO)
K = 627.5094740631                      # hartree -> kcal/mol
g = json.loads((ROOT / 'results/confinement_universality.json').read_text())['geometries']
dp = np.array(g['paraboloid']['d_a0'])
ep = np.array(g['paraboloid']['dE_hartree'])
es = np.array(g['sphere']['dE_hartree'])
assert np.array_equal(dp, np.array(g['sphere']['d_a0'])), 'the two geometries use different grids'
out = ROOT / 'figures/fig1_selection_rule.dat'
with out.open('w') as f:
    f.write('# Figure 1 data. Generated from results/confinement_universality.json by\n')
    f.write('# scripts/export_fig1_data.py. Do not edit by hand.\n')
    f.write('# dE computed at 60-digit precision by scripts/confinement_universality.py.\n')
    f.write('# d: closest approach of wall to nucleus (a0). E: confinement shift (kcal/mol).\n')
    f.write('# ratio: sphere/paraboloid.\n')
    f.write('d\tEpar\tEsph\tratio\n')
    for i in range(len(dp)):
        f.write(f'{dp[i]:.4f}\t{ep[i]*K:.10e}\t{es[i]*K:.10e}\t{es[i]/ep[i]:.6f}\n')
print(f'wrote {out.name}, {len(dp)} rows')
