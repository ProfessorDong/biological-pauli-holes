"""A: First-principles-informed Marcus PCET rate for WT SLO.

Combines the B3.4-dense per-snapshot QM ΔZPE and well curvatures with literature-
benchmarked Marcus parameters (V_el, λ, ΔG from the Hammes-Schiffer SLO PCET series)
to assemble a semi-empirical Fermi Golden Rule rate for isotope H, D.

Approximations:
  1. Both the reactant (donor) and product (acceptor) wells are treated as harmonic,
     with curvatures taken from the cubic-spline second derivative of the QM PES at
     each local minimum.
  2. The vibronic overlap S_{0,0} between reactant and product proton ground states
     is computed from Gaussian overlap of the two isotope-specific harmonic
     eigenstates, separated by Δq = q_acceptor - q_donor.
  3. |V_el|^2 and (λ, ΔG) are taken from the SLO literature; only the vibronic
     overlap and the reactant-well ZPE are first-principles-derived here. The result
     is therefore a "first-principles-informed" rate estimate, not a fully self-
     contained calculation.

Rate expression (Fermi Golden Rule, 0-0 dominant approximation):
    k_i = (2π/ℏ) |V_el|^2 |S_{0,0}^{(i)}|^2 (4π λ k_B T)^{-1/2}
          × exp[-(ΔG + λ + ε_ν^{(i)} - ε_μ^{(i)})^2 / (4 λ k_B T)]
For 0-0 (μ=ν=0), ε_ν - ε_μ = ZPE_A - ZPE_D on the two wells (isotope-dependent).
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, math
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline

DIR = Path(_REPO + '/results/pcet_B34_dense')
Ha2kcal = 627.5094740631
kT_10C = 1.987e-3 * (283.15)                 # kcal/mol
kT_40C = 1.987e-3 * (313.15)

# Physical constants
HBAR = 1.054571817e-34                        # J s
J_PER_KCAL = 4184.0 / 6.02214076e23
AMU_KG = 1.66053906660e-27
ANGSTROM_M = 1e-10

# ---- literature-benchmarked SLO PCET parameters ----
# V_el: Hammes-Schiffer & coworkers estimate V_el ~ 0.02 eV = 0.46 kcal/mol for SLO PCET.
# Cite: e.g. Hammes-Schiffer JPCB 2011; Hu Klinman JBC 2019; Meyer Klinman Biochem 2005
V_EL_kcal = 0.46
# λ: reorganization energy for SLO PCET literature range 15-25 kcal/mol; use 20
LAMBDA_kcal = 20.0
# ΔG: nearly ergoneutral, ~1-2 kcal/mol
DELTA_G_kcal = 1.0


def harmonic_gaussian_overlap(alpha_D, alpha_A, q_D, q_A):
    """Overlap of two harmonic ground-state Gaussians on the proton coordinate q.
    ψ_D(q) = (α_D/π)^(1/4) exp(-α_D (q-q_D)^2 / 2)
    ψ_A(q) = (α_A/π)^(1/4) exp(-α_A (q-q_A)^2 / 2)
    with α_i = m_i ω_i / ℏ in inverse-Å^2 (so overlaps are dimensionless).
    Analytic 1D integral:
       S = (2*sqrt(α_D α_A)/(α_D+α_A))^(1/2) * exp[-α_D α_A (q_A-q_D)^2 / (2(α_D+α_A))]
    """
    pre = math.sqrt(2*math.sqrt(alpha_D*alpha_A) / (alpha_D + alpha_A))
    arg = -alpha_D * alpha_A * (q_A - q_D)**2 / (2*(alpha_D + alpha_A))
    return pre * math.exp(arg)


def alpha_from_curvature(k_kcal_A2, m_amu):
    """Return the Gaussian width parameter α = m ω / ℏ in units of Å^{-2}.
    ω = sqrt(k/m); with k in N/m and m in kg gives ω in s^{-1}.
    α [1/m^2] = m*ω/ℏ = sqrt(k m)/ℏ
    Convert to 1/Å^2 by multiplying by (1e-10)^2.
    """
    k_Nm = k_kcal_A2 * 0.694770
    m_kg = m_amu * AMU_KG
    if k_Nm <= 0: return 1.0     # bogus fallback for negative curvature
    alpha_m2 = math.sqrt(k_Nm * m_kg) / HBAR       # 1/m^2
    alpha_A2 = alpha_m2 * ANGSTROM_M**2             # 1/Å^2
    return alpha_A2


def zpe_kcal(k_kcal_A2, m_amu):
    """Harmonic ZPE in kcal/mol from curvature (kcal/mol/Å^2) and mass (amu)."""
    if k_kcal_A2 <= 0: return 0.0
    k_Nm = k_kcal_A2 * 0.694770
    m_kg = m_amu * AMU_KG
    omega = math.sqrt(k_Nm / m_kg)
    return 0.5 * HBAR * omega / J_PER_KCAL


def marcus_rate_00(V_el_kcal, S_00, dE_kcal, lam_kcal, T_K):
    """Fermi Golden Rule 0-0 rate contribution in s^-1.
    dE = ΔG + ZPE_A(product) - ZPE_D(reactant).
    """
    kT = 1.987e-3 * T_K                       # kcal/mol
    V_el_J = V_el_kcal * J_PER_KCAL
    # (2π/ℏ) |V_el|^2 has units J/s per J^2 = 1/(J s) -- need to convert to s^-1
    # rate = (2π/ℏ) |V_el|^2 |S|^2 (4π λ kT)^{-1/2} exp[...]
    # units: J/(ℏ · J) * unitless * 1/sqrt(J) * unitless = 1/s (after conversion)
    # Simplest: convert V_el to J, λ, kT, dE all to J
    lam_J = lam_kcal * J_PER_KCAL
    kT_J  = kT * J_PER_KCAL
    dE_J  = dE_kcal * J_PER_KCAL
    prefac = (2 * math.pi / HBAR) * V_el_J**2 * (S_00**2)
    density = 1.0 / math.sqrt(4 * math.pi * lam_J * kT_J)
    arg = -(dE_J + lam_J)**2 / (4 * lam_J * kT_J)
    return prefac * density * math.exp(arg)


def analyse_snapshot(json_path, V_el=V_EL_kcal, lam=LAMBDA_kcal, dG=DELTA_G_kcal):
    with open(json_path) as f: d = json.load(f)
    pts = [pt for pt in d['points'] if pt['E_hartree'] is not None]
    q = np.array([pt['q_along_A'] for pt in pts])
    E = np.array([pt['E_hartree'] for pt in pts])
    E_kcal = (E - E.min()) * Ha2kcal
    cs = CubicSpline(q, E_kcal)
    q_fine = np.linspace(q[0], q[-1], 1001)
    V_fine = cs(q_fine)
    # locate donor + acceptor wells
    i1 = int(np.argmin(V_fine[:len(q_fine)//2]))
    i2 = len(q_fine)//2 + int(np.argmin(V_fine[len(q_fine)//2:]))
    q_D = q_fine[i1]; V_D = V_fine[i1]
    q_A = q_fine[i2]; V_A = V_fine[i2]
    # local curvatures
    k_D = float(cs(q_D, 2))                    # kcal/mol/Å^2
    k_A = float(cs(q_A, 2))
    # per-isotope quantities
    result = {'win':json_path.stem, 'r_DA':float(d['r_DA']),
              'q_donor':float(q_D), 'q_acc':float(q_A),
              'V_donor_kcal':float(V_D), 'V_acc_kcal':float(V_A),
              'k_donor_kcal_A2':float(k_D), 'k_acc_kcal_A2':float(k_A),
              'dG_apparent_kcal':float(V_A - V_D)}
    for m_amu, name in [(1.008,'H'), (2.014,'D'), (3.016,'T')]:
        aD = alpha_from_curvature(k_D, m_amu)
        aA = alpha_from_curvature(k_A, m_amu)
        S00 = harmonic_gaussian_overlap(aD, aA, q_D, q_A)
        zpeD = zpe_kcal(k_D, m_amu)
        zpeA = zpe_kcal(k_A, m_amu)
        # dE for 0-0 = ΔG(literature) + (ZPE_A - ZPE_D)
        dE_00 = dG + (zpeA - zpeD)
        k_10 = marcus_rate_00(V_el, S00, dE_00, lam, 283.15)
        k_40 = marcus_rate_00(V_el, S00, dE_00, lam, 313.15)
        result[f'{name}_S00'] = float(S00)
        result[f'{name}_ZPE_donor_kcal'] = float(zpeD)
        result[f'{name}_ZPE_acc_kcal']   = float(zpeA)
        result[f'{name}_dE_00_kcal']     = float(dE_00)
        result[f'{name}_k_10C_s^-1']     = float(k_10)
        result[f'{name}_k_40C_s^-1']     = float(k_40)
    # KIE = k_H / k_D
    result['KIE_10C_S00only'] = float(result['H_S00']**2 / result['D_S00']**2)
    result['KIE_10C_full']    = result['H_k_10C_s^-1'] / result['D_k_10C_s^-1']
    result['KIE_40C_full']    = result['H_k_40C_s^-1'] / result['D_k_40C_s^-1']
    return result


if __name__ == '__main__':
    print(f'Literature Marcus parameters: V_el={V_EL_kcal:.2f} kcal/mol '
          f'(={V_EL_kcal/23.06:.3f} eV), λ={LAMBDA_kcal:.1f} kcal/mol, ΔG={DELTA_G_kcal:.1f} kcal/mol')
    print()
    results = []
    for w in ['270','320','345','370','395']:
        p = DIR/f'wt_win{w}_scan.json'
        if not p.exists(): continue
        r = analyse_snapshot(p)
        results.append(r)
        print(f'{r["win"]:>16}: q_D={r["q_donor"]:.3f} q_A={r["q_acc"]:.3f} '
              f'S00(H)={r["H_S00"]:.2e} S00(D)={r["D_S00"]:.2e}  '
              f'k_H(10C)={r["H_k_10C_s^-1"]:.2e}  k_D(10C)={r["D_k_10C_s^-1"]:.2e}  '
              f'KIE(10C)={r["KIE_10C_full"]:.2f}')

    # Aggregate
    kH_10 = np.array([r['H_k_10C_s^-1'] for r in results])
    kD_10 = np.array([r['D_k_10C_s^-1'] for r in results])
    kH_40 = np.array([r['H_k_40C_s^-1'] for r in results])
    kD_40 = np.array([r['D_k_40C_s^-1'] for r in results])
    print(f'\n=== Ensemble means across 5 stratified WT snapshots ===')
    print(f'  <k_H>(10°C) = {kH_10.mean():.2e} s^-1  (geom mean {np.exp(np.log(kH_10).mean()):.2e})')
    print(f'  <k_D>(10°C) = {kD_10.mean():.2e} s^-1  (geom mean {np.exp(np.log(kD_10).mean()):.2e})')
    print(f'  KIE(10°C)   = {(kH_10/kD_10).mean():.2f} ± {(kH_10/kD_10).std():.2f}')
    print(f'  KIE(40°C)   = {(kH_40/kD_40).mean():.2f} ± {(kH_40/kD_40).std():.2f}')
    print(f'\n  vs. experimental WT k_H(10°C) ~= 300 s^-1, KIE(10°C) = 66, KIE(40°C) = 52')

    with open(DIR/'marcus_rate_A.json','w') as f:
        json.dump({'per_snapshot':results,
                   'literature_V_el_kcal':V_EL_kcal, 'literature_lambda_kcal':LAMBDA_kcal,
                   'literature_dG_kcal':DELTA_G_kcal,
                   'mean_k_H_10C':float(kH_10.mean()), 'mean_k_D_10C':float(kD_10.mean()),
                   'mean_KIE_10C':float((kH_10/kD_10).mean()),
                   'std_KIE_10C':float((kH_10/kD_10).std()),
                   'mean_KIE_40C':float((kH_40/kD_40).mean()),
                   'std_KIE_40C':float((kH_40/kD_40).std())}, f, indent=2)
    print(f'\nwrote {DIR/"marcus_rate_A.json"}')
