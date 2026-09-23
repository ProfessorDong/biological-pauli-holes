#!/usr/bin/env python3
"""Regenerate the SAPT k_exch table with the CORRECT unit-conversion factor.

Prior code used 1 kcal/mol/A^2 = 69.4784 N/m, which is 100x too large.
Correct: 1 kcal/mol/A^2 = 0.694770 N/m.

For each of the seven JBC systems, prints and JSON-dumps:
  - r_HW: nearest heavy-atom wall distance (A)
  - k_kcal: fitted transverse curvature (kcal/mol/A^2)  [as originally computed]
  - k_Nm_corrected: same in N/m using 0.694770 factor
  - E_bio_kcal: exchange energy at biological geometry
  - ZPE_H, ZPE_D: harmonic zero-point energy per transverse mode (kcal/mol)
  - dZPE_H_D: single-mode isotope shift (kcal/mol)
  - KIE at 10 C
"""
import json, math
from pathlib import Path

N_A = 6.02214076e23
J_per_kcal = 4184.0
kcal_per_particle = J_per_kcal / N_A     # J
A2_in_m2 = 1e-20
correct_factor = kcal_per_particle / A2_in_m2   # N/m per kcal/mol/A^2
assert abs(correct_factor - 0.694770) < 1e-5, correct_factor

hbar = 1.054571817e-34
amu  = 1.66053906660e-27

def zpe_kcal(k_Nm, m_amu):
    if k_Nm <= 0: return float('nan')
    w = math.sqrt(k_Nm / (m_amu * amu))
    return (hbar * w / 2.0) / kcal_per_particle

RAW = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/sapt_bio/sapt_summary.json')
OUT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/sapt_bio/sapt_summary_corrected.json')

with open(RAW) as f:
    raw = json.load(f)

# Model reference from Supplementary Note S4 proof-of-principle
MODEL_k_kcal = 58.30
MODEL_k_Nm   = MODEL_k_kcal * correct_factor

print(f'Conversion factor: 1 kcal/mol/A^2 = {correct_factor:.6f} N/m')
print(f'Model H..H..H reference (SI S4): k_kcal={MODEL_k_kcal}, k_Nm={MODEL_k_Nm:.2f}')
print()
print(f'{"sys":<7} {"KIE":>4} {"r_HW":>6} {"k_kcal":>9} {"k_Nm(old)":>10} {"k_Nm(new)":>10} '
      f'{"E_bio":>7} {"ZPE_H":>7} {"ZPE_D":>7} {"dZPE":>7} {"k/model":>8}')

corrected = {}
for sys_name in ['WT','V750A','I552A','I538A','L754A','L546A','I553A']:
    d = raw[sys_name]
    k_kcal = d['k_exch_kcal_A2']
    k_Nm_old = d['k_exch_Nm']
    k_Nm_new = k_kcal * correct_factor
    zH = zpe_kcal(k_Nm_new, 1.008)
    zD = zpe_kcal(k_Nm_new, 2.014)
    dz = zH - zD
    ratio = k_Nm_new / MODEL_k_Nm
    print(f'{sys_name:<7} {d["KIE"]:>4} {d["r_HW"]:>6.3f} {k_kcal:>9.3f} '
          f'{k_Nm_old:>10.1f} {k_Nm_new:>10.3f} {d["E_exch_bio_kcal"]:>7.3f} '
          f'{zH:>7.4f} {zD:>7.4f} {dz:>7.4f} {ratio:>8.3f}')
    corrected[sys_name] = dict(
        r_HW_A=d['r_HW'],
        k_kcal_A2=k_kcal,
        k_Nm=k_Nm_new,
        E_exch_bio_kcal=d['E_exch_bio_kcal'],
        ZPE_H_kcal=zH, ZPE_D_kcal=zD, dZPE_HD_kcal=dz,
        k_over_model=ratio,
        KIE_10C=d['KIE'],
    )

# Aggregate stats
import statistics
k_vals = [corrected[s]['k_Nm'] for s in corrected]
dz_vals = [corrected[s]['dZPE_HD_kcal'] for s in corrected]
print()
print(f'Across the 7 systems:')
print(f'  k_exch^bio N/m:   min={min(k_vals):.2f}  median={statistics.median(k_vals):.2f}  '
      f'max={max(k_vals):.2f}   mean={statistics.mean(k_vals):.2f}')
print(f'  k_exch/model:     min={min(v["k_over_model"] for v in corrected.values()):.3f}  '
      f'max={max(v["k_over_model"] for v in corrected.values()):.3f}')
print(f'  dZPE(H-D) kcal/mol per mode:   min={min(dz_vals):.4f}  max={max(dz_vals):.4f}')

# WT vs the JBC panel spread (biological Pauli quantity does vary with mutation)
print()
print('WT vs mutants:')
wt_k = corrected["WT"]["k_Nm"]
for s in ['V750A','I552A','I538A','L754A','L546A','I553A']:
    print(f'  {s}: k_exch = {corrected[s]["k_Nm"]:.3f} N/m  (WT-normalised = {corrected[s]["k_Nm"]/wt_k:.3f})')

with open(OUT, 'w') as f:
    json.dump(dict(
        conversion_factor_kcalA2_to_Nm=correct_factor,
        conversion_factor_INCORRECT_previously_used=69.4784,
        model_HHH_reference_Nm=MODEL_k_Nm,
        per_system_corrected=corrected,
    ), f, indent=2)
print()
print(f'wrote {OUT}')
