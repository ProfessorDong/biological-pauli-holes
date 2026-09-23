#!/usr/bin/env python3
"""Main figure: the two curvatures, side by side, on the same seven variants.

The audit's figure recommendation was that the coordinate result deserves a main figure before
another schematic of the proposed mechanism. This is that figure. Panel a shows what each
calculation displaces. Panel b puts the fragment-separation curvature and both eigenvalues of the
transverse proton Hessian on the same systems, which is the whole argument in one view: different
magnitude, different sign, different order.

Usage: render_fig_coordinate.py     (pauli env)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
OUT = ROOT / 'figures/fig_coordinate.pdf'
KIE = {'WT': 66, 'V750A': 62, 'I552A': 66, 'I538A': 100, 'L754A': 106, 'L546A': 131, 'I553A': 148}

plt.rcParams.update({'font.size': 8.4, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'font.family': 'sans-serif', 'pdf.fonttype': 42})


def main():
    d = json.loads((ROOT / 'results/transverse_hessian/transverse_vs_separation.json').read_text())
    rec = {r['tag']: r for r in d['records']
           if r['clamp'] == 'r255' and abs(r['half'] - 0.15) < 1e-9}
    tags = sorted(rec, key=lambda t: rec[t]['Ksep'])
    ksep = np.array([rec[t]['Ksep'] for t in tags])
    w1 = np.array([rec[t]['w1'] for t in tags])
    w2 = np.array([rec[t]['w2'] for t in tags])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 2.75),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.45], wspace=0.34))

    # ---- panel a: what each calculation moves -----------------------------------------
    axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.axis('off')
    axa.text(0.0, 0.95, 'a', fontweight='bold', fontsize=10, transform=axa.transAxes)
    # donor, hydrogen, acceptor, wall
    axa.plot([0.16], [0.60], 'o', ms=9, color='0.25')
    axa.plot([0.42], [0.60], 'o', ms=5.5, color='#1f77b4')
    axa.plot([0.68], [0.60], 'o', ms=9, color='0.25')
    axa.plot([0.42], [0.27], 'o', ms=10, color='#c2410c')
    axa.text(0.16, 0.70, 'donor C', ha='center', fontsize=7.4)
    axa.text(0.42, 0.51, 'H', ha='center', fontsize=7.4, color='#1f77b4')
    axa.text(0.68, 0.70, 'acceptor O', ha='center', fontsize=7.4)
    axa.text(0.42, 0.17, 'closed-shell wall', ha='center', fontsize=7.4, color='#c2410c')
    axa.plot([0.16, 0.68], [0.60, 0.60], '-', lw=0.8, color='0.55')
    # K_sep: the wall translates along the donor-wall axis
    axa.annotate('', xy=(0.62, 0.40), xytext=(0.62, 0.27),
                 arrowprops=dict(arrowstyle='<->', lw=1.1, color='#c2410c'))
    axa.text(0.67, 0.32, r'$K_{\rm sep}$: wall moves', fontsize=7.4, color='#c2410c',
             va='center')
    # K_perp: the hydrogen moves transverse to the reaction axis
    axa.annotate('', xy=(0.42, 0.93), xytext=(0.42, 0.68),
                 arrowprops=dict(arrowstyle='<->', lw=1.1, color='#1f77b4'))
    axa.text(0.47, 0.83, r'$K_\perp$: H moves', fontsize=7.4, color='#1f77b4')
    axa.text(0.0, 0.055, 'coordinate schematic, drawn collinear for clarity;',
             fontsize=7.0, style='italic', color='0.35')
    axa.text(0.0, -0.02, r'the sampled C$-$H$\cdots$O angles are 88$\degree$ to 118$\degree$',
             fontsize=7.0, style='italic', color='0.35')

    # ---- panel b: the two quantities on the same systems ------------------------------
    x = np.arange(len(tags))
    axb.axhline(0, lw=0.7, color='0.6')
    lo = np.minimum(w1, w2)
    hi = np.maximum(w1, w2)
    axb.bar(x - 0.26, ksep, width=0.25, color='#c2410c', label=r'$K_{\rm sep}$')
    axb.bar(x + 0.01, hi, width=0.25, color='#1f77b4',
            label=r'$K_\perp$ larger eigenvalue')
    axb.bar(x + 0.27, lo, width=0.25, color='#93c5fd', edgecolor='#1f77b4',
            linewidth=0.5, label=r'$K_\perp$ smaller eigenvalue')
    axb.set_yscale('symlog', linthresh=0.05, linscale=0.55)
    axb.set_xticks(x)
    axb.set_xticklabels([f'{t}\n{KIE[t]}' for t in tags], fontsize=7.0)
    axb.set_ylabel(r'curvature (N m$^{-1}$)', fontsize=8.0)
    axb.set_yticks([-0.2, 0, 0.2, 1, 10])
    axb.set_yticklabels(['$-0.2$', '0', '0.2', '1', '10'])
    axb.text(-0.085, 1.04, 'b', fontweight='bold', fontsize=10, transform=axb.transAxes)
    axb.legend(frameon=False, fontsize=6.8, loc='upper left', handlelength=1.1,
               labelspacing=0.25, borderpad=0.15)
    axb.text(0.5, -0.30, 'variant, and its kinetic isotope effect', ha='center',
             fontsize=8.0, transform=axb.transAxes)
    axb.tick_params(length=2.4, pad=1.6)
    for s in ('top', 'right'):
        axb.spines[s].set_visible(False)

    fig.savefig(OUT, format='pdf', bbox_inches='tight')
    neg = int(sum(1 for t in tags if min(rec[t]['w1'], rec[t]['w2']) < 0))
    print(f'wrote {OUT}')
    print(f'  K_sep {ksep.min():.3f} to {ksep.max():.3f} N/m')
    print(f'  K_perp eigenvalues {min(w1.min(), w2.min()):.3f} to {max(w1.max(), w2.max()):.3f} N/m')
    print(f'  {neg} of {len(tags)} systems have a negative eigenvalue')


if __name__ == '__main__':
    main()
