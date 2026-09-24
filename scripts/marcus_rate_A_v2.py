"""A (v2): First-principles-informed Marcus PCET rate for WT SLO — corrected model.

The v1 attempt located the "acceptor well" on the single-Born-Oppenheimer UKS surface
at unphysically short r_HO (~0.68 A), because that surface doesn't allow the electronic
state to relax from Fe(III) reactant to Fe(II) product; the "well" it finds there is
an artificial spline minimum, not the true PCET product state.

Corrected model: use LITERATURE-BENCHMARKED product-well geometry (Fe(II)-OH_2 with
r_OH = 0.98 Å) and a Fe-O-H stretch curvature representative of a hydroxide/water O-H
bond (~500 N/m = 720 kcal/mol/Å^2). Keep the reactant-well ZPE, curvature, and position
from B3.4-dense QM PES. Compute Franck-Condon overlap between:
   ψ_D^{i}(q): Gaussian centered at q_donor, curvature k_donor (QM), mass m_i
   ψ_A^{i}(q): Gaussian centered at q_acc_lit = r_DA - 0.98, curvature k_acc_lit, mass m_i
Assemble Fermi Golden Rule rate with literature V_el, λ, ΔG.

The result is a "first-principles reactant + literature product" hybrid rate. Fully
first-principles product-side quantities require CDFT (Task B).
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

# Physical constants
HBAR = 1.054571817e-34
J_PER_KCAL = 4184.0 / 6.02214076e23
AMU_KG = 1.66053906660e-27
ANGSTROM_M = 1e-10

# ---- literature-benchmarked SLO PCET parameters ----
V_EL_kcal = 0.46          # Hammes-Schiffer JPC-B 2011 etc: V_el ~ 0.02 eV for SLO
LAMBDA_kcal = 20.0        # SLO reorganisation energy, PCET literature
DELTA_G_kcal = 1.0        # nearly ergoneutral
R_OH_PRODUCT = 0.98       # Fe-OH_2 O-H bond length, standard
K_OH_LIT_Nm = 500.0       # ~500 N/m O-H stretch (~3000 cm^-1)


def harmonic_gaussian_overlap(alpha_D, alpha_A, q_D, q_A):
    pre = math.sqrt(2*math.sqrt(alpha_D*alpha_A) / (alpha_D + alpha_A))
    arg = -alpha_D * alpha_A * (q_A - q_D)**2 / (2*(alpha_D + alpha_A))
    return pre * math.exp(arg)


def alpha_from_curvature(k_Nm, m_amu):
    m_kg = m_amu * AMU_KG
    if k_Nm <= 0: return 1.0
    alpha_m2 = math.sqrt(k_Nm * m_kg) / HBAR
    return alpha_m2 * ANGSTROM_M**2       # per Å^2


def zpe_kcal(k_Nm, m_amu):
    if k_Nm <= 0: return 0.0
    m_kg = m_amu * AMU_KG
    omega = math.sqrt(k_Nm / m_kg)
    return 0.5 * HBAR * omega / J_PER_KCAL


def marcus_rate_00(V_el_kcal, S_00, dE_kcal, lam_kcal, T_K):
    lam_J = lam_kcal * J_PER_KCAL
    kT_J  = (1.987e-3 * T_K) * J_PER_KCAL
    dE_J  = dE_kcal * J_PER_KCAL
    V_el_J = V_el_kcal * J_PER_KCAL
    prefac = (2 * math.pi / HBAR) * V_el_J**2 * (S_00**2)
    density = 1.0 / math.sqrt(4 * math.pi * lam_J * kT_J)
    arg = -(dE_J + lam_J)**2 / (4 * lam_J * kT_J)
    return prefac * density * math.exp(arg)


def analyse_snapshot(json_path):
    with open(json_path) as f: d = json.load(f)
    pts = [pt for pt in d['points'] if pt['E_hartree'] is not None]
    q = np.array([pt['q_along_A'] for pt in pts])
    E = np.array([pt['E_hartree'] for pt in pts])
    E_kcal = (E - E.min()) * Ha2kcal
    cs = CubicSpline(q, E_kcal)
    q_fine = np.linspace(q[0], q[-1], 1001)
    V_fine = cs(q_fine)
    # Locate the QM reactant (donor) well only (α < 0.5 half)
    i1 = int(np.argmin(V_fine[:len(q_fine)//2]))
    q_D = q_fine[i1]
    k_D_kcal = float(cs(q_D, 2))              # kcal/mol/Å^2
    k_D_Nm = k_D_kcal * 0.694770
    if k_D_Nm <= 0: return None
    r_DA = d['r_DA']
    # Product well: LITERATURE geometry — H at r_HO = 0.98 Å from O = q_A = r_DA - 0.98 from donor C
    q_A = r_DA - R_OH_PRODUCT
    Delta_q = q_A - q_D                        # proton transfer distance
    result = {'win':json_path.stem, 'r_DA':float(r_DA),
              'q_donor':float(q_D), 'q_acc_literature':float(q_A),
              'Delta_q_A':float(Delta_q),
              'k_donor_Nm':float(k_D_Nm),
              'k_acc_literature_Nm':K_OH_LIT_Nm}
    for m_amu, name in [(1.008,'H'), (2.014,'D'), (3.016,'T')]:
        aD = alpha_from_curvature(k_D_Nm, m_amu)
        aA = alpha_from_curvature(K_OH_LIT_Nm, m_amu)
        S00 = harmonic_gaussian_overlap(aD, aA, q_D, q_A)
        zpeD = zpe_kcal(k_D_Nm, m_amu)
        zpeA = zpe_kcal(K_OH_LIT_Nm, m_amu)
        dE_00 = DELTA_G_kcal + (zpeA - zpeD)
        k_10 = marcus_rate_00(V_EL_kcal, S00, dE_00, LAMBDA_kcal, 283.15)
        k_40 = marcus_rate_00(V_EL_kcal, S00, dE_00, LAMBDA_kcal, 313.15)
        result[f'{name}_S00'] = float(S00)
        result[f'{name}_ZPE_donor_kcal'] = float(zpeD)
        result[f'{name}_ZPE_acc_kcal']   = float(zpeA)
        result[f'{name}_dE_00_kcal']     = float(dE_00)
        result[f'{name}_k_10C_s^-1']     = float(k_10)
        result[f'{name}_k_40C_s^-1']     = float(k_40)
    result['KIE_10C_S00_only'] = float(result['H_S00']**2 / result['D_S00']**2)
    result['KIE_10C_full'] = result['H_k_10C_s^-1'] / result['D_k_10C_s^-1']
    result['KIE_40C_full'] = result['H_k_40C_s^-1'] / result['D_k_40C_s^-1']
    return result


if __name__ == '__main__':
    print(f'Literature-benchmarked SLO Marcus parameters:')
    print(f'  V_el = {V_EL_kcal:.2f} kcal/mol ({V_EL_kcal/23.06:.3f} eV)')
    print(f'  λ = {LAMBDA_kcal:.1f} kcal/mol')
    print(f'  ΔG = {DELTA_G_kcal:.1f} kcal/mol')
    print(f'  r_OH(product) = {R_OH_PRODUCT} Å')
    print(f'  k_OH(product) = {K_OH_LIT_Nm} N/m')
    print()
    print(f'{"win":<16} {"r_DA":>6} {"q_D":>6} {"q_A_lit":>8} {"Δq":>6} '
          f'{"k_D(N/m)":>9} {"S00(H)":>10} {"S00(D)":>10} '
          f'{"k_H(10C)":>10} {"KIE(10C)":>9} {"KIE(40C)":>9}')
    results = []
    for w in ['270','320','345','370','395']:
        p = DIR/f'wt_win{w}_scan.json'
        if not p.exists(): continue
        r = analyse_snapshot(p)
        if r is None: continue
        results.append(r)
        print(f'{r["win"]:<16} {r["r_DA"]:>6.3f} {r["q_donor"]:>6.3f} {r["q_acc_literature"]:>8.3f} '
              f'{r["Delta_q_A"]:>6.3f} {r["k_donor_Nm"]:>9.1f} '
              f'{r["H_S00"]:>10.3e} {r["D_S00"]:>10.3e} '
              f'{r["H_k_10C_s^-1"]:>10.2e} {r["KIE_10C_full"]:>9.2f} {r["KIE_40C_full"]:>9.2f}')

    kH_10 = np.array([r['H_k_10C_s^-1'] for r in results])
    kD_10 = np.array([r['D_k_10C_s^-1'] for r in results])
    kH_40 = np.array([r['H_k_40C_s^-1'] for r in results])
    kD_40 = np.array([r['D_k_40C_s^-1'] for r in results])
    kie_10 = kH_10 / kD_10; kie_40 = kH_40 / kD_40
    print(f'\n=== Ensemble across 5 stratified WT snapshots ===')
    print(f'  <k_H>(10°C) = {kH_10.mean():.2e} s^-1   geom mean = {np.exp(np.log(kH_10).mean()):.2e}')
    print(f'  <k_D>(10°C) = {kD_10.mean():.2e} s^-1   geom mean = {np.exp(np.log(kD_10).mean()):.2e}')
    print(f'  KIE(10°C)   = {kie_10.mean():.1f} ± {kie_10.std():.1f}   geom mean = {np.exp(np.log(kie_10).mean()):.1f}')
    print(f'  KIE(40°C)   = {kie_40.mean():.1f} ± {kie_40.std():.1f}   geom mean = {np.exp(np.log(kie_40).mean()):.1f}')
    print(f'\n  vs. experimental WT k_H(10°C) ~= 300 s^-1, KIE(10°C) = 66, KIE(40°C) = 52')
    print(f'  first-principles / experimental k_H ratio: {kH_10.mean()/300:.2e}')

    with open(DIR/'marcus_rate_A.json','w') as f:
        json.dump({'per_snapshot':results,
                   'literature_V_el_kcal':V_EL_kcal, 'literature_lambda_kcal':LAMBDA_kcal,
                   'literature_dG_kcal':DELTA_G_kcal,
                   'literature_r_OH_product_A':R_OH_PRODUCT,
                   'literature_k_OH_product_Nm':K_OH_LIT_Nm,
                   'mean_k_H_10C_s^-1':float(kH_10.mean()),
                   'mean_k_D_10C_s^-1':float(kD_10.mean()),
                   'geom_mean_KIE_10C':float(np.exp(np.log(kie_10).mean())),
                   'geom_mean_KIE_40C':float(np.exp(np.log(kie_40).mean())),
                   'std_log_KIE_10C':float(np.log(kie_10).std()),
                   }, f, indent=2)
    print(f'\nwrote {DIR/"marcus_rate_A.json"}')
