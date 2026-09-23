#!/usr/bin/env python3
"""Extended Data scan figure (figED1, Fig. 10 in the PRX Life manuscript) --
SAPT0/jun-cc-pVDZ five-point transverse exchange scans and quadratic fits for the
seven JBC-2019 SLO systems.

AUDIT CORRECTIONS TO THE SUPERSEDED VERSION (sapt_scans_ED_figure.py)

 (i)   WRONG SERIES PLOTTED. The old script parsed /tmp/sapt_<TAG>_<d>.out, which
       are the METHANE-SURROGATE scans (WT k = 15.577 N/m, range 0.063-23.419).
       Every k_exch^bio value quoted in the manuscript and in Table S5 comes from
       the NATIVE-FRAGMENT campaign (WT k = 7.885 N/m, range 0.004-16.381). The
       figure therefore disagreed with the text it illustrated. This script reads
       the native-fragment results committed under results/sapt_bio/native_fragment/.

 (ii)  NOT REPRODUCIBLE. Reading from /tmp meant the figure could not be rebuilt:
       those scratch files are gone. All inputs are now in-repo JSON.

 (iii) PRINT SCALE. The old figure was 10.5 in wide and was included at \textwidth
       (5.15 in), a scale factor of 0.49, which reduced its 8 pt type to 3.9 pt on
       the page. Designed here at 5.15 in so it prints at its design size.

 (iv)  RESIDUAL PANELS DROPPED, and the reason stated. The quadratic fits are
       essentially exact: RMS residual is 0.000-0.021 kcal/mol against exchange
       energies of order 1 kcal/mol. A separate residual row plotted noise at
       high magnification and invited over-reading. The RMS is annotated per panel
       instead, which is the quantity a reader needs.

 (vi)  UNCERTAINTY, FOR THE RIGHT SERIES. The caption promised a bootstrap
       standard error and a water-dimer benchmark line, neither of which this
       figure drew, and the only bootstrap on disk was for the METHANE series
       (sapt_uncertainty.json). Native-fragment uncertainties are now computed by
       sapt_native_uncertainty.py and are read here. They are fit precision only,
       and sigma_k/k is 7.2-8.9 per cent for every system, so they are drawn as
       error bars but are NOT offered as a significance test; the caption says so.

 (v)   L754A FLAGGED. Its nearest wall is ASN672, not the LEU732 that faces the
       transferring H in the other six systems, and its curvature is ~0. That is a
       change of wall identity, not a small change of wall distance, and the panel
       now says so rather than presenting it as one point among seven.

Output: figures/figED1_sapt_scans.pdf
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
SRC = ROOT / 'results/sapt_bio/native_fragment'
OUT = ROOT / 'figures/figED1_sapt_scans.pdf'
UNC = ROOT / 'results/sapt_bio/native_fragment_uncertainty.json'

# Water-dimer methodological benchmark, longitudinal (Appendix: proof of principle).
# Every biological curvature falls below it, which is the point of showing it.
K_WATER_DIMER = 40.50

ANG = 'Å'
CONV = 0.694770          # kcal/mol/A^2 -> N/m
DELTAS = ['-0.2', '-0.1', '0.0', '0.1', '0.2']

# KIE at 10 C, Hu, Sharma & Klinman, J. Biol. Chem. 294, 18069 (2019).
KIE = {'WT': 66, 'V750A': 62, 'I552A': 66, 'I538A': 100,
       'L754A': 106, 'L546A': 131, 'I553A': 148}

# panel order: ascending curvature, so the trend is readable left to right
ORDER = ['L754A', 'I552A', 'I538A', 'V750A', 'WT', 'I553A', 'L546A']

plt.rcParams.update({
    'font.size': 7.0, 'axes.labelsize': 7.0, 'axes.titlesize': 7.0,
    'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2, 'legend.fontsize': 6.2,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
})

# ---------------------------------------------------------------- load
data = {}
for tag in ORDER:
    d = json.loads((SRC / f'{tag}_native_result.json').read_text())
    x = np.array([float(k) for k in DELTAS])
    y = np.array([d['deltas'][k]['exch'] for k in DELTAS])
    data[tag] = dict(x=x, y=y, k_Nm=d['k_exch_Nm'], r_HW=d['r_HW_A'],
                     wall=d['wall_residue'], rms=d['RMS_residual_kcal'])

unc = json.loads(UNC.read_text())['systems']
for tag, D in data.items():
    D['sigma_k'] = unc[tag]['sigma_k_Nm']
    assert np.isclose(unc[tag]['k_exch_Nm'], D['k_Nm'], rtol=1e-9), tag

# refit here so the drawn curve is the fit, not a redrawn claim about it
for tag, D in data.items():
    A = np.column_stack([np.ones_like(D['x']), D['x'], D['x'] ** 2])
    p, *_ = np.linalg.lstsq(A, D['y'], rcond=None)
    D['coef'] = p
    k_check = 2.0 * p[2] * CONV
    if not np.isclose(k_check, D['k_Nm'], rtol=1e-6, atol=1e-9):
        raise SystemExit(f'{tag}: refit k={k_check:.6f} != stored {D["k_Nm"]:.6f}')
    D['rms_check'] = float(np.sqrt(np.mean((D['y'] - A @ p) ** 2)))

print('refit reproduces every stored k_exch to 1e-6 relative')

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 4, figsize=(5.15, 2.85))

xs = np.linspace(-0.22, 0.22, 200)
for ax, tag in zip(axes.ravel()[:7], ORDER):
    D = data[tag]
    ax.plot(xs, np.polyval(D['coef'][::-1], xs), '-', color='#2b6cb0', lw=1.0,
            zorder=2)
    ax.plot(D['x'], D['y'], 'o', ms=2.8, mfc='white', mec='#1a365d', mew=0.7,
            zorder=3)
    special = (tag == 'L754A')
    ax.set_title(f'{tag}  KIE {KIE[tag]}', fontsize=6.6, pad=2.0,
                 color='#9b2c2c' if special else 'black')
    # the scan falls monotonically from upper left to lower right, so the upper
    # right corner is the only region guaranteed clear of the curve
    # Units are stated in the caption, not on the panel: at 1.2 in wide, adding
    # "N m^-1" to the k line overruns the axes and collides with the tick labels.
    # L754A needs four decimals or its standard error prints as 0.000.
    nd = 4 if D['k_Nm'] < 0.1 else 2
    kfmt = f'{D["k_Nm"]:.{nd}f} $\\pm$ {D["sigma_k"]:.{nd}f}'
    ax.text(0.97, 0.96,
            f'$k$ = {kfmt}\n'
            f'$r_{{HW}}$ = {D["r_HW"]:.2f} {ANG}\n'
            f'rms {D["rms"]:.3f}',
            transform=ax.transAxes, fontsize=5.5, va='top', ha='right',
            linespacing=1.25)
    ax.tick_params(pad=1.5)
    ax.yaxis.get_offset_text().set_fontsize(5.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

# L754A is a different wall residue; say so on the panel, in the lower left,
# which the descending scan leaves clear
axes.ravel()[0].text(0.03, 0.015, 'wall is ASN672,\nnot LEU732',
                     transform=axes.ravel()[0].transAxes, fontsize=5.4,
                     va='bottom', ha='left', color='#9b2c2c', linespacing=1.25)

# ---------------------------------------------------------------- summary panel
# A scatter of k against r_HW was unreadable: four of the seven systems lie
# within 0.12 A of each other and their labels collided. A sorted horizontal bar
# chart separates every system and shows the full four-decade range honestly.
ax = axes.ravel()[7]
bars = sorted(ORDER, key=lambda t: data[t]['k_Nm'])
ypos = np.arange(len(bars))
ax.barh(ypos, [max(data[t]['k_Nm'], 2.5e-3) for t in bars], height=0.66,
        color=['#9b2c2c' if t == 'L754A' else '#2b6cb0' for t in bars],
        edgecolor='black', linewidth=0.4,
        xerr=[data[t]['sigma_k'] for t in bars],
        error_kw=dict(ecolor='black', elinewidth=0.5, capsize=1.2, capthick=0.5))
ax.axvline(K_WATER_DIMER, ls='--', color='#4a5568', lw=0.7, zorder=4)
# below the shortest bar, the only region of this panel the bars leave clear
ax.text(K_WATER_DIMER * 0.8, -0.62, f'water dimer {K_WATER_DIMER:.1f}',
        fontsize=4.8, ha='right', va='bottom', color='#4a5568')
ax.set_yticks(ypos)
ax.set_yticklabels(bars, fontsize=5.6)
ax.set_xscale('log')
ax.set_xlim(2.5e-3, 90)
ax.set_xticks([1e-2, 1e-1, 1, 10])
ax.set_ylim(-0.7, len(bars) - 0.3)
ax.set_title('curvature by system', fontsize=6.6, pad=2.0)
ax.set_xlabel('$k_{\\mathrm{exch}}^{\\mathrm{bio}}$ (N m$^{-1}$)', fontsize=6.4,
              labelpad=1.5)
ax.tick_params(pad=1.5)
ax.tick_params(axis='y', length=0)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)

# shared axis labels for the seven scan panels
fig.supxlabel(f'wall displacement $\\delta$ ({ANG})', fontsize=7.0, y=0.015)
fig.supylabel('SAPT0 exchange energy (kcal mol$^{-1}$)', fontsize=7.0, x=0.008)

fig.subplots_adjust(left=0.095, right=0.985, top=0.925, bottom=0.135,
                    wspace=0.42, hspace=0.55)
fig.savefig(OUT, format='pdf')
print(f'wrote {OUT}')

for tag in ORDER:
    D = data[tag]
    print(f'  {tag:6s} k={D["k_Nm"]:7.3f} N/m  r_HW={D["r_HW"]:.3f}  '
          f'wall={D["wall"]:7s} rms={D["rms"]:.4f}')
