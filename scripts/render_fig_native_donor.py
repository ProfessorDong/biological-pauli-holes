#!/usr/bin/env python3
"""Figure: the matched native-donor validation of the coordinate result.

Panel a asks whether the sign of the transverse exchange Hessian is a property of the methane
surrogate. It is not: the smaller eigenvalue keeps its sign in every system that resolves, and the
one system that does not resolve with the native donor did not resolve with methane either.
Panel b asks whether isolating the exchange term manufactured the negative eigenvalue. It did not:
the total intermolecular interaction is negative definite in six of seven systems.

Error bars are 2 sigma, propagated from the least-squares fit covariance to each eigenvalue.

Usage: render_fig_native_donor.py     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(_REPO)
D = ROOT / 'results/native_donor_validation'
T = ROOT / 'results/transverse_hessian'
OUT = ROOT / 'figures/fig_native_donor.pdf'
SYS = ['L754A', 'I552A', 'I538A', 'L546A', 'I553A', 'V750A', 'WT']
CONV = 0.694770

plt.rcParams.update({'font.size': 8.4, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'font.family': 'sans-serif', 'pdf.fonttype': 42})


def fit(grid, key):
    A = np.array([[1, r[0], r[1], 0.5 * r[0] ** 2, r[0] * r[1], 0.5 * r[1] ** 2]
                  for r in grid], float)
    z = np.array([r[key] for r in grid], float)
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    res = A @ c - z
    cov = float(res @ res) / (len(z) - 6) * np.linalg.inv(A.T @ A)
    H = np.array([[c[3], c[4]], [c[4], c[5]]]) * CONV
    w, V = np.linalg.eigh(H)
    se = []
    for k in range(2):
        v = V[:, k]
        g = np.array([v[0] ** 2, 2 * v[0] * v[1], v[1] ** 2]) * CONV
        se.append(float(np.sqrt(g @ cov[np.ix_([3, 4, 5], [3, 4, 5])] @ g)))
    return w, np.array(se)


def main():
    nat_e, nat_s, int_e, int_s, met_e, met_s = {}, {}, {}, {}, {}, {}
    for t in SYS:
        d = json.loads((D / f'{t}_r255_w015_native_transverse.json').read_text())
        g = [(r['d1'], r['d2'], r['exch'], r['total']) for r in d['grid']]
        nat_e[t], nat_s[t] = fit(g, 2)
        int_e[t], int_s[t] = fit(g, 3)
        m = json.loads((T / f'{t}_r255_w015_transverse_hessian.json').read_text())
        mg = [(r[0], r[1], r[2]) for r in (m.get('grid') or m.get('rows'))]
        met_e[t], met_s[t] = fit(mg, 2)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 2.85),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.30))
    x = np.arange(len(SYS))

    # ---- panel a: smaller eigenvalue, methane vs native ---------------------------
    axa.axhline(0, lw=0.7, color='0.55')
    axa.errorbar(x - 0.16, [met_e[t][0] for t in SYS], yerr=[2 * met_s[t][0] for t in SYS],
                 fmt='o', ms=4.2, lw=0, elinewidth=1.0, capsize=2.2,
                 color='#9ca3af', ecolor='#9ca3af', label='methane donor')
    axa.errorbar(x + 0.16, [nat_e[t][0] for t in SYS], yerr=[2 * nat_s[t][0] for t in SYS],
                 fmt='o', ms=4.2, lw=0, elinewidth=1.0, capsize=2.2,
                 color='#1f77b4', ecolor='#1f77b4', label=r'native $\mathrm{C_5H_8}$ donor')
    iv = SYS.index('V750A')
    axa.annotate('unresolved\nin both', xy=(iv + 0.16, nat_e['V750A'][0]), xytext=(iv - 1.15, 0.185),
                 fontsize=6.6, color='0.35', ha='center',
                 arrowprops=dict(arrowstyle='-', lw=0.5, color='0.55'))
    axa.set_xticks(x); axa.set_xticklabels(SYS, fontsize=6.9, rotation=35, ha='right')
    axa.set_ylabel(r'smaller eigenvalue of $K_\perp^{\rm exch}$  (N m$^{-1}$)', fontsize=7.6)
    axa.legend(frameon=False, fontsize=6.9, loc='upper left', handletextpad=0.4)
    axa.text(-0.20, 1.04, 'a', fontweight='bold', fontsize=10, transform=axa.transAxes)
    axa.tick_params(length=2.4, pad=1.6)
    for s in ('top', 'right'): axa.spines[s].set_visible(False)

    # ---- panel b: exchange vs total interaction, native donor ---------------------
    axb.axhline(0, lw=0.7, color='0.55')
    for i, t in enumerate(SYS):
        axb.plot([i - 0.17, i - 0.17], nat_e[t], '-', lw=1.0, color='#1f77b4', zorder=1)
        axb.plot([i + 0.17, i + 0.17], int_e[t], '-', lw=1.0, color='#c2410c', zorder=1)
    axb.plot(x - 0.17, [nat_e[t][0] for t in SYS], 'o', ms=3.8, color='#1f77b4',
             label=r'$K_\perp^{\rm exch}$')
    axb.plot(x - 0.17, [nat_e[t][1] for t in SYS], 'o', ms=3.8, color='#1f77b4', mfc='white')
    axb.plot(x + 0.17, [int_e[t][0] for t in SYS], 's', ms=3.8, color='#c2410c',
             label=r'$K_\perp^{\rm int}$ (total)')
    axb.plot(x + 0.17, [int_e[t][1] for t in SYS], 's', ms=3.8, color='#c2410c', mfc='white')
    axb.set_yscale('symlog', linthresh=0.05, linscale=0.5)
    axb.set_yticks([-0.4, -0.2, 0, 0.2, 1, 2])
    axb.set_yticklabels(['$-0.4$', '$-0.2$', '0', '0.2', '1', '2'])
    axb.set_xticks(x); axb.set_xticklabels(SYS, fontsize=6.9, rotation=35, ha='right')
    axb.set_ylabel(r'both eigenvalues, native donor  (N m$^{-1}$)', fontsize=7.6)
    axb.legend(frameon=False, fontsize=6.9, loc='upper center', ncol=2,
               handletextpad=0.4, columnspacing=1.1, bbox_to_anchor=(0.52, 1.10))
    axb.text(-0.20, 1.04, 'b', fontweight='bold', fontsize=10, transform=axb.transAxes)
    axb.tick_params(length=2.4, pad=1.6)
    for s in ('top', 'right'): axb.spines[s].set_visible(False)

    fig.savefig(OUT, format='pdf', bbox_inches='tight')
    nneg = sum(nat_e[t][0] + 2 * nat_s[t][0] < 0 for t in SYS)
    nint = sum(int_e[t][1] < 0 for t in SYS)
    print(f'wrote {OUT}')
    print(f'  native exch: {nneg} of 7 negative at 2 sigma')
    print(f'  native int : {nint} of 7 negative definite')


if __name__ == '__main__':
    main()
