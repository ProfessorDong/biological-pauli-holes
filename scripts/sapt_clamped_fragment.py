"""SAPT0/jun-cc-pVDZ k_exch on a CLAMPED geometry (reactive-geometry test).

Derived mechanically from sapt_native_fragment.py so the descriptor is bit-for-bit
the main-text 2.8 quantity; only the input geometry and output naming differ.
Run in the pauli env (psi4)."""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, json, numpy as np, psi4
from pathlib import Path

TAG, CLAMP = sys.argv[1], sys.argv[2]
DIR = Path(_REPO + '/results/reactive_geometry')
with open(DIR/f'{TAG}_{CLAMP}_geometry.json') as f:
    geom = json.load(f)

donor_C_pos = np.array(geom['donor_C'])
xferH_pos   = np.array(geom['xferH'])
native      = [(a['element'], np.array(a['position']), a['name']) for a in geom['native_wall']]

# donor methane at donor_C, one C-H directed at the transferring hydrogen
def build_methane(center, direction, r=1.09):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    Hs_local = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)
    Hs_local = Hs_local / np.sqrt(3.0) * r
    v_from = Hs_local[0]/r; v_to = d
    ax = np.cross(v_from, v_to); s = np.linalg.norm(ax); c = np.dot(v_from, v_to)
    if s < 1e-6: R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        ax /= s
        K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
        R = np.eye(3) + s*K + (1-c)*K@K
    return (R @ Hs_local.T).T + center

Hs_donor = build_methane(donor_C_pos, xferH_pos - donor_C_pos)

# Find nearest wall atom to xferH; scan axis is donor -> nearest-wall
min_d, nearest_wall_pos, nearest_name = np.inf, None, None
for e, p, n in native:
    if e == 'H': continue
    d = np.linalg.norm(p - xferH_pos)
    if d < min_d: min_d = d; nearest_wall_pos = p; nearest_name = n
axis = (nearest_wall_pos - donor_C_pos) / np.linalg.norm(nearest_wall_pos - donor_C_pos)
print(f'nearest wall atom: {nearest_name}  r_HW={min_d:.3f} A')

Ha2kcal = 627.5094740631
DELTAS = [-0.20, -0.10, 0.00, +0.10, +0.20]

psi4.set_memory('6 GB'); psi4.set_num_threads(4)
psi4.set_options({'basis':'jun-cc-pvdz','scf_type':'df','freeze_core':True})

results = {}
for delta in DELTAS:
    disp = axis * delta
    lines = ['0 1']
    for h in Hs_donor:
        lines.append(f'H {h[0]:.6f} {h[1]:.6f} {h[2]:.6f}')
    lines.append(f'C {donor_C_pos[0]:.6f} {donor_C_pos[1]:.6f} {donor_C_pos[2]:.6f}')
    lines.append('--')
    lines.append('0 1')
    for e, p, n in native:
        pd = p + disp
        lines.append(f'{e} {pd[0]:.6f} {pd[1]:.6f} {pd[2]:.6f}')
    lines.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
    outfile = DIR/f'sapt_{TAG}_{CLAMP}_{delta:+.2f}.out'
    psi4.core.set_output_file(str(outfile), False)
    try:
        mol = psi4.geometry('\n'.join(lines))
        psi4.energy('sapt0', molecule=mol)
    except Exception as ex:
        print(f'  delta={delta:+.2f}: FAIL {ex}'); continue
    v = psi4.variable
    E = {
        'exch': float(v('SAPT EXCH ENERGY')*Ha2kcal),
        'elst': float(v('SAPT ELST ENERGY')*Ha2kcal),
        'ind' : float(v('SAPT IND ENERGY')*Ha2kcal),
        'disp': float(v('SAPT DISP ENERGY')*Ha2kcal),
        'tot' : float(v('SAPT0 TOTAL ENERGY')*Ha2kcal),
    }
    results[str(delta)] = E
    print(f'  delta={delta:+.2f}: E_exch={E["exch"]:+7.3f}  E_tot={E["tot"]:+7.3f}')

CONV = 0.694770  # kcal/mol/A^2 -> N/m
if len(results) >= 3:
    ds = np.array(sorted(float(k) for k in results))
    Es = np.array([results[str(d)]['exch'] for d in ds])
    A = np.column_stack([np.ones_like(ds), ds, ds**2])
    p, *_ = np.linalg.lstsq(A, Es, rcond=None)
    resid = Es - A@p
    k_kcal = 2*p[2]; k_Nm = k_kcal * CONV
    RMS = float(np.sqrt(np.mean(resid**2)))
    print(f'\n{TAG}: native-fragment k_exch = {k_kcal:.3f} kcal/mol/A^2 = {k_Nm:.3f} N/m  (RMS residual = {RMS:.4f} kcal/mol)')
    out = dict(TAG=TAG, clamp=CLAMP, wall_residue=geom['wall_residue'],
               nearest_wall_atom=nearest_name, r_HW_A=float(min_d),
               k_exch_kcal_A2=float(k_kcal), k_exch_Nm=float(k_Nm),
               RMS_residual_kcal=RMS, deltas=results)
    with open(DIR/f'{TAG}_{CLAMP}_native_result.json','w') as f:
        json.dump(out, f, indent=2)
    print(f'wrote {DIR/f"{TAG}_{CLAMP}_native_result.json"}')
