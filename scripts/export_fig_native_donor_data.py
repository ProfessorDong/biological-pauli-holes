#!/usr/bin/env python3
"""Export Figure 5's data to pgfplots tables, including the symlog transform for panel b.

Reuses render_fig_native_donor.fit so the eigenvalues and their standard errors are the same
numbers the superseded matplotlib figure drew, not a reimplementation.

Panel b was drawn on a matplotlib symlog axis (linthresh 0.05, linscale 0.5). pgfplots has no
symlog, so the transform is applied here and the axis is drawn linear with ticks placed at the
transformed positions and labelled with the original values:
    |y| <= t :  y' = sign(y) * a * |y|/t
    |y| >  t :  y' = sign(y) * (a + log10(|y|/t)),   a = linscale/(1 - 1/10)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(_REPO)
sys.path.insert(0, str(ROOT / 'scripts'))
from render_fig_native_donor import fit, SYS, D, T

LT, LS = 0.05, 0.5
A = LS / (1.0 - 0.1)


def sym(y):
    y = float(y)
    s = np.sign(y)
    return s * A * abs(y) / LT if abs(y) <= LT else s * (A + np.log10(abs(y) / LT))


rows = []
for t in SYS:
    d = json.loads((D / f'{t}_r255_w015_native_transverse.json').read_text())
    g = [(r['d1'], r['d2'], r['exch'], r['total']) for r in d['grid']]
    ne, ns = fit(g, 2)
    ie, isd = fit(g, 3)
    m = json.loads((T / f'{t}_r255_w015_transverse_hessian.json').read_text())
    mg = [(r[0], r[1], r[2]) for r in (m.get('grid') or m.get('rows'))]
    me, ms = fit(mg, 2)
    rows.append((t, me[0], ms[0], ne[0], ns[0], ne[0], ne[1], ie[0], ie[1]))

with (ROOT / 'figures/fig_native_donor.dat').open('w') as f:
    f.write('# Figure 5. From scripts/export_fig_native_donor_data.py, using the same fit()\n')
    f.write('# as the superseded matplotlib figure. e = eigenvalue (N/m), s = standard error.\n')
    f.write('# sym* columns are the matplotlib symlog transform (linthresh 0.05, linscale 0.5).\n')
    f.write('i\tsystem\tmet_e\tmet_s\tnat_e\tnat_s\tsym_x1\tsym_x2\tsym_i1\tsym_i2\n')
    for i, (t, me, ms, ne, ns, x1, x2, i1, i2) in enumerate(rows):
        f.write(f'{i}\t{t}\t{me:.6f}\t{ms:.6f}\t{ne:.6f}\t{ns:.6f}\t'
                f'{sym(x1):.6f}\t{sym(x2):.6f}\t{sym(i1):.6f}\t{sym(i2):.6f}\n')

# connector segments for panel b: two rows per system, blank line between, so that
# \addplot table draws seven separate vertical segments rather than one polyline.
for tag, c1, c2, off in (('exch', 6, 7, -0.16), ('int', 8, 9, +0.16)):
    with (ROOT / f'figures/fig_native_donor_seg_{tag}.dat').open('w') as f:
        f.write(f'# panel b connectors, {tag}. x y, blank line separates segments.\n')
        f.write('x\ty\n')
        for i, r in enumerate(rows):
            y1, y2 = sym(r[c1 - 1]) if False else None, None
        for i, (t, me, ms, ne, ns, x1, x2, i1, i2) in enumerate(rows):
            a, b = (sym(x1), sym(x2)) if tag == 'exch' else (sym(i1), sym(i2))
            f.write(f'{i + off:.3f}\t{a:.6f}\n{i + off:.3f}\t{b:.6f}\n\n')

print('wrote fig_native_donor.dat and the two connector tables')
print('symlog tick positions for {-0.4,-0.2,0,0.2,1,2}:')
print('  ' + ', '.join(f'{v}->{sym(v):.4f}' for v in (-0.4, -0.2, 0, 0.2, 1, 2)))
neg = sum(1 for r in rows if r[3] < 0)
print(f'negative smaller eigenvalue, native donor: {neg} of {len(rows)}')
print('sigma of the smaller eigenvalue (native):')
for t, me, ms, ne, ns, *_ in rows:
    print(f'  {t:6s} {ne:+.5f} +/- {ns:.5f}  = {abs(ne)/ns:5.1f} sigma')
