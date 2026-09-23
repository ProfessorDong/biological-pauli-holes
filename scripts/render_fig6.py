#!/usr/bin/env python3
"""Render Fig 6: isotope-specific quantum reactive-plane density from the full 36-trajectory
RPMD campaign for I553A and I552A x {H, D}. 2x2 grid of 24x24 histograms of the
transferring-particle bead positions projected onto (r_HO, theta_CHO), with N_eff and
ln(N_eff,H/D) annotated per panel/system.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

DATA = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/pimd_prod')
OUT  = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/sn-article-template/fig6_pimd_density.pdf')


def load_projected(system, iso):
    """All bead-frame samples pooled across the 9 replicas (6 at r=3.35, 3 at r=3.60)."""
    r_HO_all, theta_all = [], []
    for r0_tag in ['r335', 'r360']:
        for rep in [1, 2, 3, 4, 5, 6]:
            p = DATA/f'{system}_{iso}_{r0_tag}_rep{rep}_pimd.npz'
            if not p.exists(): continue
            d = np.load(p)
            xH = np.asarray(d['xferH_positions'])
            dpos = np.asarray(d['donor_pos'])[:, None, :]
            apos = np.asarray(d['acceptor_pos'])[:, None, :]
            v_HO = apos - xH; v_HC = dpos - xH
            r_HO = np.linalg.norm(v_HO, axis=-1)
            r_HC = np.linalg.norm(v_HC, axis=-1)
            cos_t = np.einsum('fbi,fbi->fb', v_HO, v_HC) / (r_HO*r_HC + 1e-12)
            theta = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
            r_HO_all.append(r_HO.flatten()); theta_all.append(theta.flatten())
    return np.concatenate(r_HO_all), np.concatenate(theta_all)


data = {(s, i): load_projected(s, i) for s in ['I553A', 'I552A'] for i in ['H', 'D']}

# Global range for common histogram (1-99 pct on pooled data)
r_all = np.concatenate([r for r, _ in data.values()])
t_all = np.concatenate([t for _, t in data.values()])
r_range = (float(np.percentile(r_all, 1)), float(np.percentile(r_all, 99)))
t_range = (float(np.percentile(t_all, 1)), float(np.percentile(t_all, 99)))
NB = 24

# Compute histograms and entropies
def hist_and_entropy(r, t):
    H, xe, ye = np.histogram2d(r, t, bins=NB, range=[r_range, t_range])
    p = H / (H.sum() + 1e-30)
    nz = p > 0
    return H, xe, ye, -np.sum(p[nz] * np.log(p[nz]))

hists = {k: hist_and_entropy(*v) for k, v in data.items()}
S     = {k: v[3] for k, v in hists.items()}
N_eff = {k: np.exp(S[k]) for k in S}

# Common colour scale for all four panels (probability per bin)
max_prob = max((h/ (h.sum()+1e-30)).max() for h, *_ in hists.values())

fig, axes = plt.subplots(2, 2, figsize=(5.15, 4.5),
                         sharex=True, sharey=True,
                         gridspec_kw=dict(wspace=0.08, hspace=0.12,
                                          left=0.11, right=0.86, top=0.90, bottom=0.10))

# Ordering: rows=system, cols=isotope
sys_order = ['I553A', 'I552A']
iso_order = ['H', 'D']

im = None
for i, s in enumerate(sys_order):
    for j, iso in enumerate(iso_order):
        H, xe, ye, S_val = hists[(s, iso)]
        p = H / (H.sum() + 1e-30)
        ax = axes[i, j]
        im = ax.pcolormesh(xe, ye, p.T,
                           cmap='viridis',
                           norm=Normalize(vmin=0, vmax=max_prob),
                           shading='auto', rasterized=True)
        ax.set_xlim(r_range); ax.set_ylim(t_range)
        # panel label
        ax.text(0.04, 0.95, f'{s} · {iso}',
                transform=ax.transAxes, fontsize=8, weight='bold',
                va='top', ha='left', color='w',
                bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'))
        # N_eff annotation
        ax.text(0.96, 0.06, f'$\\tilde{{N}}={N_eff[(s,iso)]:.1f}$',
                transform=ax.transAxes, fontsize=7.2, ha='right', va='bottom',
                color='w',
                bbox=dict(facecolor='black', alpha=0.5, pad=2, edgecolor='none'))
        if i == 1:
            ax.set_xlabel('$r_{HO}$ (Å)', fontsize=8)
        if j == 0:
            ax.set_ylabel('$\\theta_{CHO}$ (deg)', fontsize=8)
        ax.tick_params(labelsize=8)

# per-row (per-system) delta annotation between H and D columns
for i, s in enumerate(sys_order):
    dSHD = S[(s, 'H')] - S[(s, 'D')]
    # Ntilde, NOT N_eff. The caption states explicitly that this is not the
    # rate-weighted N_eff of the rate decomposition, so the artwork must not
    # carry an "eff" subscript. An earlier version did. The label also used to
    # sit outside the axes at x = 1.02, where the colorbar covered it.
    axes[i, 1].annotate(r'$\ln(\tilde N_H/\tilde N_D)=' + f'{dSHD:+.3f}$',
                        xy=(0.96, 0.90), xycoords='axes fraction',
                        fontsize=7.2, ha='right', va='center', color='white',
                        bbox=dict(boxstyle='round,pad=0.22', fc='#00000088',
                                  ec='none'))

# Common colorbar
cbar_ax = fig.add_axes([0.885, 0.10, 0.02, 0.80])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('probability per bin', fontsize=7.2)
cb.ax.tick_params(labelsize=8)

fig.suptitle(r'Isotope-specific bead density on the reactive plane '
             r'($r_{\mathrm{HO}}, \theta_{\mathrm{CHO}}$)  (RPMD, $P=8$ beads)',
             fontsize=8, y=0.97)

fig.savefig(OUT, format='pdf', bbox_inches='tight', dpi=200)
plt.close(fig)

# Report
DD = (S[('I553A','H')]-S[('I553A','D')]) - (S[('I552A','H')]-S[('I552A','D')])
print(f'wrote {OUT}')
print(f'N_eff  I553A:  H={N_eff[("I553A","H")]:.1f}   D={N_eff[("I553A","D")]:.1f}')
print(f'N_eff  I552A:  H={N_eff[("I552A","H")]:.1f}   D={N_eff[("I552A","D")]:.1f}')
print(f'DeltaDelta ln(N_eff,H/D) I553A - I552A = {DD:+.4f}')
