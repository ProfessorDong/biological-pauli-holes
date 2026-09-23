#!/usr/bin/env python3
"""Does the methane donor surrogate bias k_exch? One variable changed: the donor fragment.

WHAT PROMPTED THIS, STATED PRECISELY
  Li, Soudackov and Hammes-Schiffer (JACS 140, 3068 (2018)) criticize the gas-phase model
  of Champion and co-workers, methane as proton donor against [OH]^delta- as acceptor, as
  "too small to fully describe the interface between the substrate, which has a pi backbone,
  and the iron cofactor," and show that with a large enough model the energy along the
  donor-acceptor distance R_CO is soft rather than needing a protein compressive force.

  Their claim is about the DONOR-ACCEPTOR coordinate against the iron-bound acceptor. Our
  k_exch is a TRANSVERSE curvature of the donor against a protein side-chain wall, a
  different coordinate and a different partner, and our surrogate is a different
  construction from Champion's. So this calculation is NOT a test of their claim and cannot
  confirm or refute it. What their criticism does establish is that a one-carbon stand-in
  for a conjugated substrate is a real hazard worth measuring, and that is what this does:
  a robustness check of our own observable against our own surrogate.

HELD IDENTICAL to the published native-fragment calculation: the snapshot, the wall
fragment, the scan axis (donor carbon -> nearest wall heavy atom), the five displacements,
the basis, the SAPT order, and the quadratic fit. ONLY the donor changes, from a constructed
methane centred on the donor carbon to the real C5H8 bis-allylic unit taken from the same
snapshot by scripts/extract_pentadienyl_donor.py.

Usage: sapt_pentadienyl_vs_methane.py TAG [basis]
"""
import json, sys
from pathlib import Path
import numpy as np
import psi4

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
DIR = ROOT/'results/sapt_bio/native_fragment'
OUT = ROOT/'results/sapt_bio/donor_fragment'
OUT.mkdir(parents=True, exist_ok=True)
Ha2kcal, CONV = 627.5094740631, 0.694770
DELTAS = [-0.20, -0.10, 0.00, +0.10, +0.20]


def methane(center, direction, r=1.09):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    H = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]], float)/np.sqrt(3.0)*r
    vf, vt = H[0]/r, d
    ax = np.cross(vf, vt); s = np.linalg.norm(ax); c = float(np.dot(vf, vt))
    if s < 1e-6:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        ax /= s
        K = np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
        R = np.eye(3) + s*K + (1-c)*K@K
    out = [('C', np.asarray(center, float))]
    out += [('H', h) for h in (R @ H.T).T + np.asarray(center, float)]
    return out


def scan(donor, native, dC, axis, basis, label):
    psi4.set_options({'basis': basis, 'scf_type': 'df', 'freeze_core': True})
    E = {}
    for delta in DELTAS:
        disp = axis*delta
        L = ['0 1'] + [f'{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}' for e, p in donor]
        L += ['--', '0 1']
        L += [f'{a["element"]} ' + ' '.join(f'{v:.6f}' for v in np.array(a['position'])+disp)
              for a in native]
        L.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
        psi4.core.set_output_file(str(OUT/f'psi4_{label}_{delta:+.2f}.out'), False)
        psi4.core.clean()
        psi4.energy('sapt0', molecule=psi4.geometry('\n'.join(L)))
        E[delta] = float(psi4.variable('SAPT EXCH ENERGY')*Ha2kcal)
        print(f'    delta={delta:+.2f}  E_exch={E[delta]:+8.4f}', flush=True)
    ds = np.array(DELTAS); ys = np.array([E[d] for d in DELTAS])
    A = np.column_stack([np.ones_like(ds), ds, ds**2])
    p, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return dict(exch=E, k_kcal=float(2*p[2]), k_Nm=float(2*p[2]*CONV),
                rms=float(np.sqrt(np.mean((ys-A@p)**2))), n_atoms=len(donor))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'WT'
    basis = sys.argv[2] if len(sys.argv) > 2 else 'jun-cc-pVDZ'
    g = json.loads((DIR/f'{tag}_pentadienyl.json').read_text())
    dC = np.array(g['donor_C']); xH = np.array(g['xferH'])
    native = g['native_wall']
    heavy = [(a['name'], np.array(a['position'])) for a in native if a['element'] != 'H']
    wall = min(heavy, key=lambda t: np.linalg.norm(t[1]-xH))
    axis = (wall[1]-dC)/np.linalg.norm(wall[1]-dC)
    print(f'{tag}  basis={basis}  wall={wall[0]}  '
          f'r_HW={np.linalg.norm(wall[1]-xH):.3f} A\n')
    donors = {
        'methane':     methane(dC, xH-dC),
        'pentadienyl': [(a['element'], np.array(a['position'])) for a in g['donor_fragment']],
    }
    # A donor and a wall taken from different frames raise nothing anywhere else in this
    # pipeline: SAPT returns an exchange energy of order 1e-9 Eh, the quadratic fit reports
    # k_exch = 0 without complaint, and the result looks like a null rather than a bug.
    # Exchange is a contact property, so demand contact.
    W = np.array([a['position'] for a in native], float)
    for name, D in donors.items():
        P = np.array([p for _, p in D], float)
        dmin = float(np.min(np.linalg.norm(P[:, None, :] - W[None, :, :], axis=-1)))
        assert dmin < 6.0, (f'nearest {name}-to-wall distance is {dmin:.2f} A. The donor '
                            f'and the wall are not in the same frame.')

    res = {}
    for name, D in donors.items():
        print(f'  donor = {name}  ({len(D)} atoms)')
        res[name] = scan(D, native, dC, axis, basis, f'{tag}_{name}')
        print(f'   -> k_exch = {res[name]["k_Nm"]:.3f} N/m  (RMS {res[name]["rms"]:.4f})\n')
    m, p = res['methane']['k_Nm'], res['pentadienyl']['k_Nm']
    print(f'{"donor":>14}{"atoms":>7}{"k_exch (N/m)":>15}{"vs methane":>13}')
    print(f'{"methane":>14}{res["methane"]["n_atoms"]:>7}{m:>15.3f}{1.0:>12.3f}x')
    print(f'{"pentadienyl":>14}{res["pentadienyl"]["n_atoms"]:>7}{p:>15.3f}{p/m:>12.3f}x')
    (OUT/f'{tag}_donor_comparison.json').write_text(json.dumps(
        dict(tag=tag, basis=basis, wall_atom=wall[0],
             formula=g['donor_fragment_formula'], results=res,
             ratio_pentadienyl_over_methane=p/m), indent=1))
    print(f'\nwrote {OUT}/{tag}_donor_comparison.json')


if __name__ == '__main__':
    main()
