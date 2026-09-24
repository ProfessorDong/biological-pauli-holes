#!/usr/bin/env python3
"""Export Figure 6 panel b to a pgfplots table, with the symlog transform applied.

matplotlib drew panel b on symlog(linthresh=0.05, linscale=0.55). pgfplots has no symlog, so the
transform is applied here and the axis is drawn linear with ticks relabelled to the originals.
Values are read from the same per-system files the manuscript quotes.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path
import numpy as np

ROOT = Path(_REPO)
T = ROOT / 'results/transverse_hessian'
LT, LS = 0.05, 0.55
A = LS / (1.0 - 0.1)
KIE = {'WT': 66, 'V750A': 62, 'I552A': 66, 'I538A': 100,
       'L754A': 106, 'L546A': 131, 'I553A': 148}


def sym(y):
    y = float(y); s = np.sign(y)
    return s * A * abs(y) / LT if abs(y) <= LT else s * (A + np.log10(abs(y) / LT))


rows = []
for f in sorted(T.glob('*_r255_w015_transverse_hessian.json')):
    g = json.loads(f.read_text())
    e = sorted(g['eigenvalues_Nm'])
    rows.append((g['tag'], g['K_sep_Nm'], e[1], e[0]))
rows.sort(key=lambda r: r[1])                      # ordered by K_sep, as the figure is

out = ROOT / 'figures/fig_coordinate.dat'
with out.open('w') as fh:
    fh.write('# Figure 6b. From results/transverse_hessian/*_r255_w015_transverse_hessian.json\n')
    fh.write('# via scripts/export_fig_coordinate_data.py. sym* are symlog-transformed\n')
    fh.write('# (linthresh 0.05, linscale 0.55); raw values follow for checking.\n')
    fh.write('i\tsystem\tkie\tsym_ksep\tsym_hi\tsym_lo\tksep\thi\tlo\n')
    for i, (t, ks, hi, lo) in enumerate(rows):
        fh.write(f'{i}\t{t}\t{KIE[t]}\t{sym(ks):.6f}\t{sym(hi):.6f}\t{sym(lo):.6f}\t'
                 f'{ks:.4f}\t{hi:.5f}\t{lo:.5f}\n')

print(f'wrote {out.name}, {len(rows)} systems ordered by K_sep')
print('tick positions for {-0.2,0,0.2,1,10}: ' +
      ', '.join(f'{v}->{sym(v):.4f}' for v in (-0.2, 0, 0.2, 1, 10)))
ks = [r[1] for r in rows]; ev = [v for r in rows for v in r[2:]]
print(f'K_sep spans {min(ks):.3f} to {max(ks):.3f} N/m   (caption: 0.506 to 13.800)')
print(f'K_perp eigenvalues {min(ev):+.3f} to {max(ev):+.3f} N/m   (caption: -0.222 to +0.333)')
print(f'systems with a negative eigenvalue: {sum(1 for r in rows if r[3] < 0)} of {len(rows)}')
