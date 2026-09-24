#!/usr/bin/env python3
"""Figure 1 -- The exponential selection rule.

a  Two confinements that share a closest approach d but nothing else: the
   paraboloid xi = xi0 of the surface treatment, open along its axis, and a
   closed sphere of radius R. Both are drawn with d = 1.
b  Confinement energy against d for each, over 6.43 decades for the paraboloid and 5.93
   for the sphere; an earlier draft said eight, which was wrong). The two run
   parallel. The thermal energy at the temperature of the SLO measurements and
   the range of wall distances the enzyme actually provides are marked.
c  The ratio of the two, which grows LINEARLY in d. This is the assumption-free
   form of the claim: equal exponential rates with polynomial prefactors
   differing by one power give a linear ratio, whereas any mismatch in the rates
   would show up here as exponential drift.

All values are read from results/confinement_universality.json, produced by
confinement_universality.py at 60-digit precision. Nothing is re-derived here.

Usage: render_fig1_selection_rule.py     (pauli env)
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
SRC = ROOT / 'results' / 'confinement_universality.json'
OUT = ROOT / 'figures' / 'fig1_selection_rule.pdf'

A0 = 0.529177210903          # Angstrom
HA_KCAL = 627.509474         # hartree -> kcal/mol
KT_283 = 0.563               # kcal/mol at 283 K, the JBC kinetic temperature
SLO_D = (2.4, 3.0)           # Angstrom, H to nearest closed-shell heavy atom

CPAR, CSPH = '#2b6cb0', '#9b2c2c'
ANG = 'Å'

plt.rcParams.update({
    'font.size': 7.0, 'axes.labelsize': 7.0, 'axes.titlesize': 7.2,
    'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4, 'legend.fontsize': 6.0,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
})

g = json.loads(SRC.read_text())['geometries']
d = np.array(g['paraboloid']['d_a0'])
ep = np.array(g['paraboloid']['dE_hartree']) * HA_KCAL
es = np.array(g['sphere']['dE_hartree']) * HA_KCAL

fig = plt.figure(figsize=(5.15, 2.05))
gs = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.35, 0.85],
                      left=0.058, right=0.988, top=0.885, bottom=0.20,
                      wspace=0.46)
axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

# ---------------------------------------------------------------- panel a
# paraboloid xi = xi0 is r - z = xi0, i.e. z = (rho^2 - xi0^2)/(2 xi0);
# its vertex sits at z = -xi0/2, so xi0 = 2 puts the wall 1 unit from the
# nucleus. The sphere is drawn with radius 1. Same d, unrelated shapes.
xi0, XL, XR = 2.0, -1.75, 1.75
rho = np.linspace(-1.75, 1.75, 400)
th = np.linspace(0, 2 * np.pi, 400)

# left: the open paraboloid, vertex one unit below its focus
axA.plot(XL + rho, (rho ** 2 - xi0 ** 2) / (2 * xi0), '-', color=CPAR, lw=1.3,
         zorder=3)
axA.plot([XL], [0], 'o', ms=3.2, color='black', zorder=5)
axA.annotate('', xy=(XL, -1), xytext=(XL, 0), zorder=4,
             arrowprops=dict(arrowstyle='<->', lw=0.7, color='black'))
axA.text(XL + 0.11, -0.5, '$d$', fontsize=7.0, va='center')
axA.text(XL, -1.62, 'paraboloid', color=CPAR, fontsize=6.2, ha='center')
axA.text(XL, -2.02, 'open', color=CPAR, fontsize=5.6, ha='center', alpha=0.85)

# right: the closed sphere, radius one about its center
axA.plot(XR + np.cos(th), np.sin(th), '-', color=CSPH, lw=1.3, zorder=3)
axA.plot([XR], [0], 'o', ms=3.2, color='black', zorder=5)
axA.annotate('', xy=(XR, -1), xytext=(XR, 0), zorder=4,
             arrowprops=dict(arrowstyle='<->', lw=0.7, color='black'))
axA.text(XR + 0.11, -0.5, '$d$', fontsize=7.0, va='center')
axA.text(XR, -1.62, 'sphere', color=CSPH, fontsize=6.2, ha='center')
axA.text(XR, -2.02, 'closed', color=CSPH, fontsize=5.6, ha='center', alpha=0.85)

axA.set_xlim(-3.6, 3.1)
axA.set_ylim(-2.35, 1.5)
axA.set_aspect('equal')
axA.set_anchor('N')
axA.set_title('a  same $d$, unrelated shapes', loc='left', pad=3)
axA.set_xticks([])
axA.set_yticks([])
for s in axA.spines.values():
    s.set_visible(False)

# ---------------------------------------------------------------- panel b
axB.plot(d, ep, 'o-', color=CPAR, lw=1.1, ms=2.4, mew=0, label='paraboloid')
axB.plot(d, es, 's-', color=CSPH, lw=1.1, ms=2.4, mew=0, label='sphere')
axB.set_yscale('log')
axB.set_xlim(3.6, 12.4)
axB.set_ylim(2e-7, 60)

axB.axhline(KT_283, color='#2f855a', lw=0.8, ls='--', zorder=1)
axB.text(10.4, KT_283 / 3.4, '$k_BT$, 283 K', fontsize=5.8, color='#2f855a',
         ha='center', va='center')
lo, hi = SLO_D[0] / A0, SLO_D[1] / A0
axB.axvspan(lo, hi, color='#744210', alpha=0.15, zorder=0)
axB.annotate(f'wall distances the\nenzyme provides\n({SLO_D[0]} to {SLO_D[1]} {ANG})',
             xy=(hi, 3e-6), xytext=(7.1, 2.2e-6),
             fontsize=5.8, ha='left', va='center', color='#744210',
             linespacing=1.25,
             arrowprops=dict(arrowstyle='-', lw=0.5, color='#744210',
                             shrinkA=1, shrinkB=1))

axB.set_xlabel('wall distance $d$ ($a_0$)', labelpad=1.5)
axB.set_ylabel('confinement energy (kcal mol$^{-1}$)', labelpad=1.5,
               fontsize=6.6)
axB.set_title('b  parallel decay, 6.4 and 5.9 decades', loc='left', pad=3)
axB.legend(loc='upper right', frameon=True, framealpha=0.88, borderpad=0.3,
           handletextpad=0.5, labelspacing=0.25, handlelength=1.5)
axB.tick_params(pad=1.5)
for s in ('top', 'right'):
    axB.spines[s].set_visible(False)

# ---------------------------------------------------------------- panel c
ratio = es / ep
axC.plot(d, ratio, 'o', color='#4a5568', ms=2.6, mew=0, zorder=3)
m, b = np.polyfit(d, ratio, 1)
axC.plot(d, m * d + b, '-', color='#4a5568', lw=0.8, zorder=2)
axC.set_xlim(3.6, 12.4)
axC.set_xticks([4, 6, 8, 10, 12])
axC.set_ylim(0, 26)
axC.set_xlabel('wall distance $d$ ($a_0$)', labelpad=1.5)
axC.set_ylabel('sphere / paraboloid', labelpad=1.5)
axC.set_title('c  ratio is linear in $d$', loc='left', pad=3)
axC.text(0.05, 0.93, f'slope {m:.2f} per $a_0$', transform=axC.transAxes,
         fontsize=5.8, va='top')
axC.tick_params(pad=1.5)
for s in ('top', 'right'):
    axC.spines[s].set_visible(False)

fig.savefig(OUT, format='pdf')
print(f'wrote {OUT}')
print(f'  ratio grows linearly, slope {m:.3f} per a0, intercept {b:.3f}')
print(f'  paraboloid rate {g["paraboloid"]["decay_rate_per_a0"]:.4f}, '
      f'sphere rate {g["sphere"]["decay_rate_per_a0"]:.4f} per a0')
