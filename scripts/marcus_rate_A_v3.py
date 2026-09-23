"""A (v3): Franck-Condon-informed Marcus KIE analysis for WT SLO.

The naive equilibrium-r_DA Marcus rate is astronomically small because the reactant
C-H and product O-H equilibrium wells are separated by Δq ~ 1.3-1.9 Å at typical
r_DA ~ 3.4-4.0 Å, giving vibronic overlaps ~10^{-30}. The observed WT rate ~300 s^{-1}
therefore cannot be produced at equilibrium geometry: it MUST be produced at rare,
gated configurations with much smaller r_DA.

This is exactly the rare-configuration-selection prediction of the framework's
k = f * q_2 decomposition. We quantify it here by asking: at what r_DA does the
Franck-Condon-only Marcus KIE match experimental KIE = 66?

Model: harmonic donor and acceptor wells (k = 500 N/m, typical C-H and O-H stretch),
at positions q_D = r_CH = 1.09 A from the donor C and q_A = r_DA - r_OH = r_DA - 0.98
from the donor C. Vary r_DA from 2.5 to 4.0 A.
"""
import math, json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HBAR = 1.054571817e-34
AMU_KG = 1.66053906660e-27
ANGSTROM_M = 1e-10
J_PER_KCAL = 4184.0 / 6.02214076e23

R_CH = 1.09
R_OH = 0.98
K_CH_Nm = 500.0     # typical C-H stretch
K_OH_Nm = 500.0     # typical O-H stretch


def alpha_from_curvature(k_Nm, m_amu):
    m_kg = m_amu * AMU_KG
    return math.sqrt(k_Nm * m_kg) / HBAR * ANGSTROM_M**2


def harmonic_gaussian_overlap(alpha_D, alpha_A, q_D, q_A):
    pre = math.sqrt(2*math.sqrt(alpha_D*alpha_A) / (alpha_D + alpha_A))
    arg = -alpha_D * alpha_A * (q_A - q_D)**2 / (2*(alpha_D + alpha_A))
    return pre * math.exp(arg)


def FC_KIE(r_DA):
    """Franck-Condon-only KIE at donor-acceptor distance r_DA."""
    q_D = R_CH
    q_A = r_DA - R_OH
    S = {}
    for m, name in [(1.008,'H'), (2.014,'D'), (3.016,'T')]:
        aD = alpha_from_curvature(K_CH_Nm, m)
        aA = alpha_from_curvature(K_OH_Nm, m)
        S[name] = harmonic_gaussian_overlap(aD, aA, q_D, q_A)
    return dict(r_DA=r_DA, Delta_q=q_A-q_D,
                S_H=S['H'], S_D=S['D'], S_T=S['T'],
                KIE_HD=(S['H']/S['D'])**2, KIE_HT=(S['H']/S['T'])**2)


if __name__ == '__main__':
    # Sweep r_DA from 2.4 to 4.0 A
    r_DA_grid = np.linspace(2.4, 4.0, 41)
    results = [FC_KIE(r) for r in r_DA_grid]

    # Print key values
    print(f'Franck-Condon-only KIE across a range of r_DA (harmonic C-H and O-H wells, k=500 N/m)')
    print(f'{"r_DA(Å)":>8} {"Δq(Å)":>7} {"|S_H|²":>12} {"|S_D|²":>12} {"KIE_HD":>12}')
    for r in results[::4]:
        print(f'{r["r_DA"]:>8.2f} {r["Delta_q"]:>7.3f} {r["S_H"]**2:>12.3e} {r["S_D"]**2:>12.3e} '
              f'{r["KIE_HD"]:>12.2e}')

    # Find r_DA for which KIE_HD = 66 (experimental WT)
    kies = np.array([r['KIE_HD'] for r in results])
    r_arr = np.array([r['r_DA'] for r in results])
    # Log-interpolate
    from scipy.interpolate import interp1d
    log_kie_interp = interp1d(np.log(kies), r_arr)
    r_reactive_66 = float(log_kie_interp(np.log(66)))
    r_reactive_148 = float(log_kie_interp(np.log(148))) if np.log(148) < np.log(kies).max() else None
    r_reactive_100 = float(log_kie_interp(np.log(100)))
    print(f'\nEffective reactive r_DA that reproduces exp KIE=66 (WT):   {r_reactive_66:.3f} Å')
    print(f'Effective reactive r_DA that reproduces exp KIE=100 (I538A): {r_reactive_100:.3f} Å')
    if r_reactive_148: print(f'Effective reactive r_DA that reproduces exp KIE=148 (I553A): {r_reactive_148:.3f} Å')

    # Compare with the r_DA of the five B3.4-dense snapshots. These come from BIASED
    # umbrella windows and are NOT an equilibrium distribution; the variable was once
    # named r_DA_equilibrium and that word propagated into the figure legend.
    B34_DIR = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/pcet_B34_dense')
    r_DA_equilibrium = []
    for w in ['270','320','345','370','395']:
        with open(B34_DIR/f'wt_win{w}_scan.json') as f:
            d = json.load(f)
        r_DA_equilibrium.append(d['r_DA'])
    r_DA_equilibrium = np.array(r_DA_equilibrium)
    print(f'\nSampled r_DA of the 5 stratified WT snapshots (not an equilibrium distribution): '
          f'mean {r_DA_equilibrium.mean():.3f} ± {r_DA_equilibrium.std():.3f} Å')
    print(f'Effective reactive r_DA / snapshot mean = {r_reactive_66:.3f} / {r_DA_equilibrium.mean():.3f} '
          f'= {r_reactive_66/r_DA_equilibrium.mean():.3f}')

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.8))
    ax.semilogy(r_arr, kies, 'k-', linewidth=1.6, label='Franck-Condon KIE (k=500 N/m wells)')
    ax.axhspan(62, 148, color='#d7301f', alpha=0.16, lw=0,
               label='observed JBC-2019 KIEs, 62 to 148')
    ax.axhline(62, ls=':', color='#d7301f', alpha=0.8, lw=0.9)
    ax.axhline(148, ls=':', color='#d7301f', alpha=0.8, lw=0.9)
    ax.text(3.95, 205, 'observed KIEs\n62 to 148 (V750A to I553A)', color='#d7301f',
            fontsize=7, ha='right', va='bottom')
    # Mark the equilibrium r_DA range
    ax.axvspan(r_DA_equilibrium.min(), r_DA_equilibrium.max(),
               alpha=0.12, color='gray',
               label='equilibrium $r_{\\mathrm{DA}}$ sampled (B3.4)')
    for rr, lab in [(r_reactive_66, f'{r_reactive_66:.2f}'),
                    (r_reactive_148, f'{r_reactive_148:.2f}')]:
        if rr is None:
            continue
        ax.axvline(rr, ls='--', color='#2c7fb8', alpha=0.75, lw=1.0)
    if r_reactive_148 is not None:
        ax.annotate('', xy=(r_reactive_66, 3e6), xytext=(r_reactive_148, 3e6),
                    arrowprops=dict(arrowstyle='<->', color='#2c7fb8', lw=0.9))
        ax.text(0.5*(r_reactive_66+r_reactive_148), 6e6,
                f'reactive $r_{{\\mathrm{{DA}}}}$\n{r_reactive_66:.2f} to '
                f'{r_reactive_148:.2f} \u00c5',
                fontsize=7, color='#2c7fb8', ha='center', va='bottom')
    ax.set_xlabel('$r_{\\mathrm{DA}}$ at reactive configuration (\u00c5)', fontsize=10)
    ax.set_ylabel('Franck-Condon KIE   $|S_{H}|^2/|S_{D}|^2$', fontsize=10)
    ax.set_title(r'Effective reactive $r_{\mathrm{DA}}$ predicted from Franck-Condon KIE', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=7.5, loc='upper right')
    ax.set_xlim(2.4, 4.0)
    ax.set_ylim(1, 1e12)

    OUT_PDF = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/sn-article-template/figED3_gating_KIE.pdf')
    fig.savefig(OUT_PDF, format='pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'\nwrote {OUT_PDF}')

    # Save results
    with open(B34_DIR/'marcus_rate_A_v3.json','w') as f:
        json.dump({'r_DA_grid': r_arr.tolist(),
                   'KIE_HD_grid': kies.tolist(),
                   'K_CH_Nm': K_CH_Nm, 'K_OH_Nm': K_OH_Nm,
                   'R_CH': R_CH, 'R_OH': R_OH,
                   'r_reactive_KIE66': r_reactive_66,
                   'r_reactive_KIE100': r_reactive_100,
                   'r_reactive_KIE148': r_reactive_148,
                   'equilibrium_r_DA_mean': float(r_DA_equilibrium.mean()),
                   'equilibrium_r_DA_std':  float(r_DA_equilibrium.std())}, f, indent=2)
    print(f'wrote {B34_DIR/"marcus_rate_A_v3.json"}')
