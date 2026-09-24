#!/usr/bin/env python3
"""Does jun-cc-pVDZ underestimate the SAPT exchange curvature?

WHY THIS TEST EXISTS
  The Herring-Flicker benchmark (scripts/herring_flicker_benchmark.py) showed that the
  exchange coupling of two hydrogen atoms at 5-6 angstrom separation is acutely sensitive to
  DIFFUSE basis functions: at R = 11 bohr the computed |2J| rose from 0.960 to 0.986 of the
  exact asymptote when aug-cc-pVQZ was replaced by d-aug-cc-pVQZ. Exchange lives in the
  wavefunction tail, and truncating the tail truncates the exchange.

  Every k_exch^bio in this paper was computed with jun-cc-pVDZ, which is the aug-cc-pVDZ
  basis with the diffuse shell REMOVED FROM HYDROGEN and the highest diffuse angular momentum
  removed from heavy atoms. If the benchmark lesson transfers, our exchange curvatures are
  systematic UNDERESTIMATES, and the size of the bias is what this script measures.

WHAT IS HELD FIXED
  Identical geometry, identical fragments, identical five displacements, identical quadratic
  fit. Only the orbital basis changes. Differences are therefore attributable to the basis.

Usage: sapt_basis_sensitivity.py TAG [basis1 basis2 ...]
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json, sys
from pathlib import Path
import numpy as np
import psi4

ROOT = Path(_REPO)
DIR = ROOT / 'results/sapt_bio/native_fragment'
OUT = ROOT / 'results/sapt_bio/basis_sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
Ha2kcal = 627.5094740631
CONV = 0.694770
DELTAS = [-0.20, -0.10, 0.00, +0.10, +0.20]


def build_methane(center, direction, r=1.09):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    H = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)/np.sqrt(3.0)*r
    v_from, v_to = H[0]/r, d
    ax = np.cross(v_from, v_to); s = np.linalg.norm(ax); c = float(np.dot(v_from, v_to))
    if s < 1e-6:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        ax /= s
        K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
        R = np.eye(3) + s*K + (1-c)*K@K
    return (R @ H.T).T + center


def scan(geom, basis):
    dC = np.array(geom['donor_C']); xH = np.array(geom['xferH'])
    native = [(a['element'], np.array(a['position'])) for a in geom['native_wall']]
    Hs = build_methane(dC, xH - dC)
    heavy = [(e,p) for e,p in native if e != 'H']
    wall = min(heavy, key=lambda t: np.linalg.norm(t[1]-xH))[1]
    axis = (wall - dC)/np.linalg.norm(wall - dC)
    psi4.set_options({'basis': basis, 'scf_type': 'df', 'freeze_core': True})
    E = {}
    for delta in DELTAS:
        disp = axis*delta
        L = ['0 1'] + [f'H {h[0]:.6f} {h[1]:.6f} {h[2]:.6f}' for h in Hs]
        L += [f'C {dC[0]:.6f} {dC[1]:.6f} {dC[2]:.6f}', '--', '0 1']
        L += [f'{e} {(p+disp)[0]:.6f} {(p+disp)[1]:.6f} {(p+disp)[2]:.6f}' for e,p in native]
        L.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
        psi4.core.set_output_file(str(OUT/f'psi4_{basis}_{delta:+.2f}.out'), False)
        psi4.core.clean()
        psi4.energy('sapt0', molecule=psi4.geometry('\n'.join(L)))
        E[delta] = float(psi4.variable('SAPT EXCH ENERGY')*Ha2kcal)
        print(f'    delta={delta:+.2f}  E_exch={E[delta]:+8.4f} kcal/mol', flush=True)
    ds = np.array(DELTAS); ys = np.array([E[d] for d in DELTAS])
    A = np.column_stack([np.ones_like(ds), ds, ds**2])
    p, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return dict(exch=E, k_kcal=float(2*p[2]), k_Nm=float(2*p[2]*CONV),
                rms=float(np.sqrt(np.mean((ys-A@p)**2))))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'WT'
    bases = sys.argv[2:] or ['jun-cc-pVDZ', 'aug-cc-pVDZ']
    geom = json.loads((DIR/f'{tag}_geometry.json').read_text())
    psi4.set_memory('12 GB'); psi4.set_num_threads(8)
    res = {}
    for b in bases:
        print(f'\n{tag}  basis = {b}', flush=True)
        try:
            res[b] = scan(geom, b)
            print(f'  -> k_exch = {res[b]["k_Nm"]:.3f} N/m  (RMS {res[b]["rms"]:.4f})', flush=True)
        except Exception as ex:
            print(f'  FAILED: {str(ex)[:90]}', flush=True)
    if 'jun-cc-pVDZ' in res:
        base = res['jun-cc-pVDZ']['k_Nm']
        print(f'\n{"basis":>16}{"k_exch (N/m)":>15}{"vs jun-cc-pVDZ":>17}')
        for b, v in res.items():
            print(f'{b:>16}{v["k_Nm"]:>15.3f}{v["k_Nm"]/base:>16.3f}x')
    (OUT/f'{tag}_basis_sensitivity.json').write_text(json.dumps(
        dict(tag=tag, geometry=str(DIR/f'{tag}_geometry.json'), results=res), indent=1))
    print(f'\nwrote {OUT}/{tag}_basis_sensitivity.json')


if __name__ == '__main__':
    main()
