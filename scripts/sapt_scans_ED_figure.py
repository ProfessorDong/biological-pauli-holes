#!/usr/bin/env python3
"""A1 + A2: parse the seven SAPT scans, refit quadratics with bootstrap uncertainty
on k_exch^bio, render an Extended Data figure showing per-system exchange energy
scans + residuals + summary (k_exch vs r_HW on log axes)."""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import re, math, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TMP = Path('/tmp')
SYSTEMS = ['WT', 'V750A', 'I552A', 'I538A', 'L754A', 'L546A', 'I553A']
DELTAS = [-0.20, -0.10, 0.00, +0.10, +0.20]
KIE = {'WT':66,'I553A':148,'L754A':106,'V750A':62,'I538A':100,'L546A':131,'I552A':66}
r_HW = {'WT':3.394,'L754A':4.812,'I553A':3.384,'I552A':3.723,'V750A':3.500,'I538A':3.434,'L546A':4.387}
CONV = 0.694770   # kcal/mol/A^2 -> N/m


def parse_exch(tag, delta):
    """Return the exchange energy in kcal/mol from Psi4 SAPT0 output."""
    sign = '+' if delta >= 0 else '-'
    fname = TMP / f'sapt_{tag}_{sign}{abs(delta):.2f}.out'
    if not fname.exists():
        return None
    txt = fname.read_text()
    m = re.search(r'Exchange\s+[-\d.]+\s+\[mEh\]\s+([-\d.]+)\s+\[kcal/mol\]', txt)
    if m is None:
        return None
    return float(m.group(1))


# Parse everything
scans = {}
for tag in SYSTEMS:
    xs, ys = [], []
    for d in DELTAS:
        e = parse_exch(tag, d)
        if e is None:
            print(f'  missing {tag} delta={d:+.2f}')
            continue
        xs.append(d); ys.append(e)
    scans[tag] = (np.array(xs), np.array(ys))
    print(f'  {tag}: {len(xs)} points')


def fit_quadratic(x, y):
    """Fit E = c0 + c1*d + c2*d^2 by least squares. Return coeffs, residuals, SE(c2)."""
    A = np.column_stack([np.ones_like(x), x, x**2])
    p, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ p
    n, k = len(x), 3
    if n <= k:
        return p, resid, float('nan')
    sigma2 = float(np.sum(resid**2) / (n - k))
    cov = sigma2 * np.linalg.inv(A.T @ A)
    se_c2 = math.sqrt(cov[2, 2])
    return p, resid, se_c2


def bootstrap_curvature(x, y, n_boot=5000, rng=None):
    """Parametric residual bootstrap on the 5-point scan; returns k_exch_Nm samples."""
    if rng is None: rng = np.random.default_rng(20260803)
    A = np.column_stack([np.ones_like(x), x, x**2])
    p, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ p
    sigma = float(np.std(resid, ddof=3)) if len(x) > 3 else float(np.std(resid))
    out = np.empty(n_boot)
    for i in range(n_boot):
        y_star = A @ p + rng.normal(0.0, sigma, size=len(x))
        p_star, *_ = np.linalg.lstsq(A, y_star, rcond=None)
        out[i] = 2 * p_star[2] * CONV     # kcal/mol/A^2 -> N/m
    return out


# Fit each system, collect uncertainties
results = {}
for tag in SYSTEMS:
    x, y = scans[tag]
    p, resid, se_c2 = fit_quadratic(x, y)
    boot = bootstrap_curvature(x, y)
    k_Nm = 2 * p[2] * CONV
    results[tag] = dict(
        coeffs_kcal=p.tolist(),
        residuals_kcal=resid.tolist(),
        k_exch_Nm=float(k_Nm),
        k_exch_Nm_bootstrap_mean=float(boot.mean()),
        k_exch_Nm_bootstrap_sem=float(boot.std()),
        k_exch_Nm_bootstrap_ci95=[float(np.percentile(boot, 2.5)),
                                  float(np.percentile(boot, 97.5))],
        r_HW=r_HW[tag],
        KIE=KIE[tag],
    )
    print(f'  {tag}: k = {k_Nm:.3f} ± {boot.std():.3f} N/m  '
          f'95%CI [{np.percentile(boot,2.5):.3f}, {np.percentile(boot,97.5):.3f}]  '
          f'RMS_resid = {math.sqrt(np.mean(resid**2)):.4f} kcal/mol')

# JSON dump for the paper
OUT_JSON = Path(_REPO + '/results/sapt_bio/sapt_uncertainty.json')
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f'wrote {OUT_JSON}')


# ============================== FIGURE ==============================
# 2x4 grid: 7 per-system scans + 1 summary (k vs r_HW).
fig = plt.figure(figsize=(10.5, 5.0))
gs = fig.add_gridspec(2, 4, wspace=0.35, hspace=0.55, left=0.06, right=0.985, top=0.93, bottom=0.10)

# Colour palette by increasing r_HW (sequential blue)
r_HW_sorted = sorted(r_HW.items(), key=lambda kv: kv[1])
colour = {tag: plt.cm.viridis(0.15 + 0.7*i/6) for i, (tag, _) in enumerate(r_HW_sorted)}

for i, tag in enumerate(SYSTEMS):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    x, y = scans[tag]
    p = results[tag]['coeffs_kcal']
    xs_smooth = np.linspace(x.min(), x.max(), 100)
    ys_smooth = p[0] + p[1]*xs_smooth + p[2]*xs_smooth**2
    ax.plot(xs_smooth, ys_smooth, '-', color=colour[tag], linewidth=1.5, alpha=0.9)
    ax.plot(x, y, 'o', color=colour[tag], markersize=5, markeredgecolor='k', markeredgewidth=0.5)
    ax.set_title(f'{tag}  (KIE={KIE[tag]}, $r_{{HW}}$={r_HW[tag]:.2f}\\,\\AA)', fontsize=8)
    ax.set_xlabel(r'$\delta$ (\AA)', fontsize=8)
    ax.set_ylabel(r'$E_{\mathrm{exch}}$ (kcal/mol)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    # Residual annotation
    rms = math.sqrt(np.mean(np.asarray(results[tag]['residuals_kcal'])**2))
    ax.text(0.03, 0.97, f'$k$={results[tag]["k_exch_Nm"]:.2f}\\,N/m\n$\\pm${results[tag]["k_exch_Nm_bootstrap_sem"]:.2f}\nRMS={rms:.3g}',
            transform=ax.transAxes, fontsize=6.5, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.85, pad=1.5, edgecolor='0.7'))

# Summary panel: k_exch vs r_HW
ax = fig.add_subplot(gs[1, 3])
for tag in SYSTEMS:
    ax.errorbar(r_HW[tag], results[tag]['k_exch_Nm'],
                yerr=results[tag]['k_exch_Nm_bootstrap_sem'],
                fmt='o', color=colour[tag], markersize=6,
                markeredgecolor='k', markeredgewidth=0.5, capsize=2)
    ax.annotate(tag, xy=(r_HW[tag], results[tag]['k_exch_Nm']),
                xytext=(4, 4), textcoords='offset points', fontsize=6.5)
ax.set_yscale('log')
ax.set_xlabel(r'$r_{HW}$ (\AA)', fontsize=8)
ax.set_ylabel(r'$k_{\mathrm{exch}}^{\mathrm{bio}}$ (N/m, log)', fontsize=8)
ax.set_title('Summary: $k$ vs nearest wall distance', fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(True, alpha=0.25, linewidth=0.4, which='both')
# Reference: H..H..H model (~40 N/m)
ax.axhline(40.5, ls='--', color='0.5', lw=0.8, alpha=0.6)
ax.text(0.98, 40.5, r'H$\cdots$H$\cdots$H model', ha='right', va='bottom',
        transform=ax.get_yaxis_transform(), fontsize=6.5, color='0.4')

fig.suptitle(r'SAPT0/jun-cc-pVDZ transverse exchange scans across seven JBC-2019 SLO systems  '
             r'($E_{\mathrm{exch}}$ vs wall displacement $\delta$)', fontsize=9, y=0.99)

OUT_PDF = Path(_REPO + '/sn-article-template/figED1_sapt_scans.pdf')
fig.savefig(OUT_PDF, format='pdf', bbox_inches='tight', dpi=200)
plt.close(fig)
print(f'wrote {OUT_PDF}')
