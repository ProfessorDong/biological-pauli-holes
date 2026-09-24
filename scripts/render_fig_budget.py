#!/usr/bin/env python3
"""Figure: the exponential confinement budget of the transferring hydrogen.

Panel a  how many closed-shell heavy atoms sit at each distance from the
         transferring hydrogen in the wild-type reactive geometry, against how
         much of the confinement weight that shell actually carries. Steric bulk
         grows with shell volume while weight collapses exponentially.
Panel b  cumulative share of the total weight as a function of shell radius, all
         fourteen geometries overlaid, with the range of nearest approaches of
         the six mutated cavity positions marked.

The point of the figure is that the exponential decides membership on its own.
No boundary is drawn between "wall" and "reaction partner": every closed-shell
atom is included and weighted, and the cavity side chains that the JBC panel
mutates fall where the exponential has already reduced them to nothing much.

Honest ranges, recomputed over all fourteen geometries on 2026-09-23:
  within 2.5 A: 82.4 to 94.0 % of the weight;  within 3.0 A: 95.1 to 98.9 %
  the six mutated positions together: 0.0006 % to 4.534 %, median 0.070 %
The maximum, 4.53 %, occurs for L754A_r255, where a cavity position comes to
2.71 A. It is quoted rather than buried: the claim is that
the cavity contribution is small, not that it is always negligible.

STALE VALUES CORRECTED HERE (2026-09-23 audit). This docstring still carried the
PRE-REPAIR numbers after the V750A side-chain mask was fixed: "within 2.5 A: 85.1",
"within 3.0 A: 95.1 to 98.3" and "0.0006 % to 4.8 %, median 0.125 %". The caption
and the plotted data had been updated; the generator's own documentation had not.

Usage: render_fig_budget.py     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(_REPO)
SRC = ROOT / 'results' / 'confinement_budget.json'
OUT = ROOT / 'figures' / 'fig_budget.pdf'

A0 = 0.529177210903
ANG = 'Å'

CATS = [
    ('substrate framework',   '#2b6cb0', 'o'),
    ('acceptor hydroxide O',  '#c05621', 's'),
    ('cofactor first shell',  '#276749', '^'),
    ('other protein',         '#718096', 'v'),
    ('JBC cavity side chain', '#9b2c2c', 'D'),
    ('water',                 '#63b3ed', '.'),
]

plt.rcParams.update({
    'font.size': 7.0, 'axes.labelsize': 7.0, 'axes.titlesize': 7.2,
    'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4, 'legend.fontsize': 5.8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
})

data = json.loads(SRC.read_text())
fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.15, 2.25))

# ---------------------------------------------------------------- panel a
wt = data['WT_r255']
pa = wt['per_atom']
d = np.array(pa['d_A'])
w = np.array(pa['weight'])
cat = np.array(pa['category'])
tot = wt['total_weight']

# Plotting w against d would be a tautology: w is DEFINED as exp(-2d/a0), so
# every atom lies on that curve by construction and the panel would test
# nothing. The informative contrast is between how many neighbors sit at each
# distance and how much of the confinement they actually carry. Steric bulk
# grows with the shell volume; weight collapses exponentially. They cross.
bins = np.arange(1.75, 6.51, 0.25)
mid = 0.5 * (bins[:-1] + bins[1:])
count, _ = np.histogram(d, bins=bins)
wsum, _ = np.histogram(d, bins=bins, weights=w / tot)

axA.bar(mid, count, width=0.22, color='#a0aec0', edgecolor='#4a5568',
        linewidth=0.35, zorder=2, label='closed-shell atoms in bin')
axA.set_xlim(1.8, 6.5)
axA.set_ylim(0, max(count) * 1.32)
axA.set_xlabel(f'distance from transferring H, $d$ ({ANG})', labelpad=1.5)
axA.set_ylabel('number of atoms in bin', labelpad=1.5)
axA.set_title('a  bulk sits far, weight sits near', loc='left', pad=3)
axA.tick_params(pad=1.5)
axA.spines['top'].set_visible(False)

axA2 = axA.twinx()
axA2.plot(mid[wsum > 0], wsum[wsum > 0], 'o-', color='#9b2c2c', lw=1.0, ms=2.6,
          mfc='#9b2c2c', mec='black', mew=0.3, zorder=3,
          label='share of confinement weight')
axA2.set_yscale('log')
axA2.set_ylim(1e-9, 3)
axA2.set_ylabel('share of confinement weight', color='#9b2c2c', labelpad=1,
                fontsize=6.4)
axA2.tick_params(axis='y', colors='#9b2c2c', pad=1.5)
axA2.spines['top'].set_visible(False)
axA2.spines['right'].set_color('#9b2c2c')

h1, l1 = axA.get_legend_handles_labels()
h2, l2 = axA2.get_legend_handles_labels()
axA.legend(h1 + h2, l1 + l2, loc='upper center', frameon=True, framealpha=0.85,
           borderpad=0.3, handletextpad=0.4, labelspacing=0.25)

# ---------------------------------------------------------------- panel b
# The stored cumulative is tabulated at only six radii, and the curve rises from
# 0.0 % at 2.0 A to 89.4 % at 2.5 A. Joining six points draws that as a smooth
# ramp, which misrepresents the shape: the true cumulative is a STEP function
# with jumps at the individual atom distances. Recomputed here on a fine grid
# from the per-atom data so the steps are visible. This strengthens the panel,
# because discrete jumps at bonding distances make the point better than a ramp.
radii = np.arange(2.0, 5.001, 0.02)
cav_d = []
for k, v in data.items():
    dd = np.array(v['per_atom']['d_A'])
    ww = np.array(v['per_atom']['weight'])
    tot_k = v['total_weight']
    cum = [100.0 * ww[dd <= R].sum() / tot_k for R in radii]
    axB.plot(radii, cum, '-', color='#4a5568', lw=0.6, alpha=0.55, zorder=2)
    c = v['categories'].get('JBC cavity side chain')
    if c:
        cav_d.append(c['d_min_A'])
axB.axhline(95, color='#9b2c2c', lw=0.6, ls=':', zorder=1)
axB.text(2.05, 95.8, '95 %', fontsize=5.8, color='#9b2c2c')

lo, hi = min(cav_d), max(cav_d)
axB.axvspan(lo, hi, color='#9b2c2c', alpha=0.13, zorder=0)
axB.text((lo + hi) / 2, 22,
         'nearest approach of the\nsix mutated positions\n'
         f'({lo:.2f} to {hi:.2f} {ANG})',
         fontsize=5.8, ha='center', va='center', color='#9b2c2c',
         linespacing=1.3)

axB.set_xlim(2.0, 5.0)
axB.set_ylim(0, 104)
axB.set_xlabel(f'shell radius ({ANG})', labelpad=1.5)
axB.set_ylabel('cumulative weight inside shell (%)', labelpad=1.5)
axB.set_title('b  all fourteen geometries', loc='left', pad=3)
axB.tick_params(pad=1.5)
for s in ('top', 'right'):
    axB.spines[s].set_visible(False)

fig.subplots_adjust(left=0.088, right=0.945, top=0.885, bottom=0.175, wspace=0.62)
fig.savefig(OUT, format='pdf')
print(f'wrote {OUT}')
print(f'  cavity nearest approach spans {lo:.2f} to {hi:.2f} {ANG}')
