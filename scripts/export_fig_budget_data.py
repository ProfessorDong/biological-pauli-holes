#!/usr/bin/env python3
"""Export Figure 4's data to pgfplots tables so the TikZ carries no numbers.

Reproduces exactly what render_fig_budget.py computes:
  panel a  0.25 A shells over the WT reactive geometry, atom count and the share of the
           total range weight each shell carries.
  panel b  the cumulative share inside a shell of given radius, recomputed on a fine grid
           from the per-atom distances so the true STEP shape is visible. Interpolating the
           six tabulated radii would draw a smooth ramp and misrepresent it.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path
import numpy as np

ROOT = Path(_REPO)
FIG = ROOT / 'figures'
d = json.loads((ROOT / 'results/confinement_budget.json').read_text())

# ---- panel a -------------------------------------------------------------------
wt = d['WT_r255']
dd = np.array(wt['per_atom']['d_A'])
ww = np.array(wt['per_atom']['weight'])
bins = np.arange(1.75, 6.51, 0.25)
mid = 0.5 * (bins[:-1] + bins[1:])
count, _ = np.histogram(dd, bins=bins)
wsum, _ = np.histogram(dd, bins=bins, weights=ww / wt['total_weight'])
with (FIG / 'fig_budget_a.dat').open('w') as f:
    f.write('# Figure 4a. From results/confinement_budget.json via export_fig_budget_data.py.\n')
    f.write('# WT reactive geometry, 0.25 A shells. share is of the total range weight.\n')
    f.write('mid\tcount\tshare\n')
    for m, c, s in zip(mid, count, wsum):
        f.write(f'{m:.4f}\t{c:d}\t{s if s > 0 else float("nan"):.6e}\n')

# ---- panel b -------------------------------------------------------------------
radii = np.arange(2.0, 5.001, 0.02)
geoms = sorted([k for k in d if isinstance(d[k], dict) and 'categories' in d[k]])
cols = {}
for g in geoms:
    D = np.array(d[g]['per_atom']['d_A'])
    W = np.array(d[g]['per_atom']['weight'])
    cols[g] = [100.0 * W[D <= R].sum() / d[g]['total_weight'] for R in radii]
with (FIG / 'fig_budget_b.dat').open('w') as f:
    f.write('# Figure 4b. Cumulative share of the range weight inside a shell of radius r,\n')
    f.write('# recomputed per atom so the step structure survives. 14 clamped geometries.\n')
    f.write('r\t' + '\t'.join(geoms) + '\n')
    for i, R in enumerate(radii):
        f.write(f'{R:.3f}\t' + '\t'.join(f'{cols[g][i]:.4f}' for g in geoms) + '\n')

cav = [d[g]['categories']['JBC cavity side chain']['d_min_A'] for g in geoms]
print(f'wrote fig_budget_a.dat ({len(mid)} shells) and fig_budget_b.dat '
      f'({len(radii)} radii x {len(geoms)} geometries)')
print(f'mutated-position nearest approach: {min(cav):.2f} to {max(cav):.2f} A')
