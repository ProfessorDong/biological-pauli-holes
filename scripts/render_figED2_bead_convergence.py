#!/usr/bin/env python3
"""Extended Data bead-convergence figure (figED2, Fig. 11 in the PRX Life manuscript):
bead-count convergence of the equilibrium isotope-specific delocalization observable
on I553A/I552A x {H, D} x P in {8, 16, 32}.

  (a) Per-condition occupancy Ntilde_i = exp H[rho_i] against P
  (b) Per-system isotope shift ln(Ntilde_H / Ntilde_D) against P
  (c) Between-mutant contrast DD against P with block-bootstrap 95% intervals

AUDIT CORRECTIONS (2026-08-15)

 (i)   THE ABSCISSA WAS CONFOUNDED. Fixed upstream in pimd_convergence_analysis.py,
       not here: the P=8 cell pooled 6 replicas at r335 and 3 at r360 while P=16 and
       P=32 pooled 3 and 3, so the x axis varied the window mixture as well as the
       bead count. All three cells now use the matched 3+3 design. This changed the
       numbers: DD at P=8 moved from +0.0097 to -0.0214, a sign flip.

 (ii)  A FALSE CLAIM IN THE CAPTION FELL WITH IT. The caption said both systems drift
       "monotonically up" in Ntilde_H with P. That was never true for I552A
       (248.5 -> 237.5 -> 272.4), and under the matched design it is not true for
       I553A either (241.0 -> 226.6 -> 251.1). Neither series is monotonic and the
       caption no longer says otherwise.

 (iii) THE ERROR BARS WERE INVISIBLE. errorbar(..., linewidth=0) also zeroes
       elinewidth, because elinewidth defaults to linewidth. The 95% intervals of
       panel (c) rendered as detached caps with no connecting whisker, which reads
       as four stray tick marks rather than as an interval. elinewidth is now set
       explicitly and the connector is suppressed with linestyle='none'.

 (iv)  TYPE WAS INVERTED AND OVERSET. Tick labels were 9 pt while the axis labels
       they annotate were 6 pt. Panel (c) carried the full definition of DD as its
       y label at 9 pt, which overran the whole figure height, covered panel (b) and
       its "32" tick, and collided with the suptitle. The definition now lives in
       the caption where there is room for it, and type is consistent.

 (v)   PRINT SCALE. bbox_inches='tight' grew the canvas to 5.698 in, which was then
       included at \\textwidth = 5.15 in, a silent 0.90 shrink. Fixed margins and no
       tight bbox, so it prints at its design size.

Output: figures/figED2_bead_convergence.pdf
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
JSON_PATH = ROOT / 'results/pimd_convergence/analysis/B1_convergence_analysis.json'
OUT_PDF = ROOT / 'figures/figED2_bead_convergence.pdf'

data = json.loads(JSON_PATH.read_text())
per, conv = data['per_condition'], data['convergence']
assert data.get('design', {}).get('matched'), \
    'analysis JSON predates the matched-design fix; re-run pimd_convergence_analysis.py'

P_vals = [8, 16, 32]
systems = ['I553A', 'I552A']
isos = ['H', 'D']
sys_colour = {'I553A': '#2c7fb8', 'I552A': '#c9812f'}
iso_marker = {'H': 'o', 'D': 's'}

plt.rcParams.update({
    'font.size': 7.0, 'axes.labelsize': 7.0, 'axes.titlesize': 7.0,
    'xtick.labelsize': 6.0, 'ytick.labelsize': 6.0, 'legend.fontsize': 5.6,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
})

fig, axes = plt.subplots(1, 3, figsize=(5.15, 1.95))

# ------------------------------------------------------------------ (a)
ax = axes[0]
for s in systems:
    for i in isos:
        ys = [per[f'P{P}_{s}_{i}']['N_eff'] for P in P_vals]
        ax.plot(P_vals, ys, marker=iso_marker[i], color=sys_colour[s],
                markersize=3.4, markeredgecolor='k', markeredgewidth=0.4,
                linewidth=1.0, alpha=0.9, label=f'{s}·{i}')
ax.set_ylabel(r'$\tilde N_i = \exp H[\rho_i]$', labelpad=1.5)
ax.set_title('(a) per-condition occupancy', loc='left', pad=3.0)
ax.legend(loc='upper center', ncol=2, frameon=False, handlelength=1.2,
          columnspacing=0.8, handletextpad=0.4, borderpad=0.1)
ax.set_ylim(195, 300)

# ------------------------------------------------------------------ (b)
ax = axes[1]
for s in systems:
    ys = [conv[f'P{P}'][f'delta_ln_Neff_{s}'] for P in P_vals]
    ax.plot(P_vals, ys, marker='o', color=sys_colour[s], markersize=3.4,
            markeredgecolor='k', markeredgewidth=0.4, linewidth=1.2, label=s)
ax.axhline(0, color='k', ls=':', lw=0.6, alpha=0.6)
ax.set_ylabel(r'$\ln(\tilde N_H / \tilde N_D)$', labelpad=1.5)
ax.set_title('(b) isotope shift', loc='left', pad=3.0)
ax.legend(loc='lower right', frameon=False, handlelength=1.2,
          handletextpad=0.4, borderpad=0.1)

# ------------------------------------------------------------------ (c)
ax = axes[2]
xs = np.array(P_vals, dtype=float)
means = np.array([conv[f'P{P}']['DD_bootstrap_mean'] for P in P_vals])
ci_lo = np.array([conv[f'P{P}']['DD_ci95_lo'] for P in P_vals])
ci_hi = np.array([conv[f'P{P}']['DD_ci95_hi'] for P in P_vals])
p_sign = [conv[f'P{P}']['P_sign_positive'] for P in P_vals]
# elinewidth explicitly: it defaults to linewidth, and linewidth must be 0 here to
# suppress the line joining the three points, which would imply an interpolation
# between bead counts that does not exist.
ax.errorbar(xs, means, yerr=[means - ci_lo, ci_hi - means],
            fmt='o', color='#4d4d4d', markersize=3.8, markeredgecolor='k',
            markeredgewidth=0.4, ecolor='#7d7d7d', elinewidth=0.9, capsize=2.0,
            capthick=0.7, linestyle='none')
ax.axhline(0, color='k', ls=':', lw=0.6, alpha=0.6)
for x, y, hi, p in zip(xs, means, ci_hi, p_sign):
    ax.annotate(f'{p:.2f}', xy=(x, hi), xytext=(0, 3), textcoords='offset points',
                fontsize=5.4, ha='center', va='bottom', color='#4d4d4d')
ax.set_ylabel(r'$\Delta\Delta$', labelpad=1.5)
ax.set_title('(c) mutant contrast', loc='left', pad=3.0)
ax.set_ylim(-0.16, 0.27)
ax.text(0.5, 0.965, r'$P(\mathrm{sign}>0)$ above each interval', fontsize=5.2,
        transform=ax.transAxes, ha='center', va='top', color='#4d4d4d')

for ax in axes:
    ax.set_xlabel('bead count $P$', labelpad=1.5)
    ax.set_xscale('log', base=2)
    ax.set_xticks(P_vals)
    ax.set_xticklabels([str(P) for P in P_vals])
    ax.set_xlim(6.6, 38)
    ax.tick_params(pad=1.5)
    ax.grid(True, alpha=0.22, linewidth=0.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.20, wspace=0.40)
fig.savefig(OUT_PDF, format='pdf')
plt.close(fig)
print(f'wrote {OUT_PDF}')
for P in P_vals:
    c = conv[f'P{P}']
    print(f"  P={P:<3d} I553A {c['delta_ln_Neff_I553A']:+.4f}  "
          f"I552A {c['delta_ln_Neff_I552A']:+.4f}  "
          f"DD {c['DD_bootstrap_mean']:+.4f} "
          f"[{c['DD_ci95_lo']:+.4f}, {c['DD_ci95_hi']:+.4f}]  "
          f"P(sign)={c['P_sign_positive']:.2f}")
