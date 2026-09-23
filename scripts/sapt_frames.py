"""SAPT0 k_exch for every frame of a clamped run (ensemble-fluctuation test).

Step 2 of the bridge (pauli env: psi4). Loops over the frame geometries written by
extract_frames_geometry.py, running the SAME 5-point SAPT0/jun-cc-pVDZ scan and
quadratic curvature fit as the audited main-text 2.8 descriptor, once per frame.
One psi4 session serves all frames of a system+clamp, avoiding 40 process starts.

Output: {TAG}_{CLAMP}_kexch_frames.json holding the per-frame k_exch series.

Usage: sapt_frames.py <TAG> <clamp>
"""
import sys, json, numpy as np, psi4
from pathlib import Path

TAG, CLAMP = sys.argv[1], sys.argv[2]
EF = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/ensemble_fluctuation')
geom = json.loads((EF / f'{TAG}_{CLAMP}_frames_geometry.json').read_text())

Ha2kcal = 627.5094740631
CONV = 0.694770                      # kcal/mol/A^2 -> N/m
DELTAS = [-0.20, -0.10, 0.00, +0.10, +0.20]

psi4.set_memory('4 GB'); psi4.set_num_threads(4)
psi4.set_options({'basis': 'jun-cc-pvdz', 'scf_type': 'df', 'freeze_core': True})
psi4.core.be_quiet()


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


results, failed = [], 0
for fr in geom['frames']:
    dC = np.array(fr['donor_C']); xH = np.array(fr['xferH'])
    native = [(a['element'], np.array(a['position'])) for a in fr['native_wall']]
    Hs = build_methane(dC, xH - dC)
    # scan axis: donor -> nearest heavy wall atom to the transferring hydrogen
    best, npos = np.inf, None
    for e, p in native:
        if e == 'H':
            continue
        dd = np.linalg.norm(p - xH)
        if dd < best:
            best, npos = dd, p
    axis = (npos - dC) / np.linalg.norm(npos - dC)

    Es, ok = [], True
    for delta in DELTAS:
        disp = axis * delta
        lines = ['0 1']
        for h in Hs:
            lines.append(f'H {h[0]:.6f} {h[1]:.6f} {h[2]:.6f}')
        lines.append(f'C {dC[0]:.6f} {dC[1]:.6f} {dC[2]:.6f}')
        lines.append('--'); lines.append('0 1')
        for e, p in native:
            q = p + disp
            lines.append(f'{e} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}')
        lines.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
        try:
            mol = psi4.geometry('\n'.join(lines))
            psi4.energy('sapt0', molecule=mol)
            Es.append(float(psi4.variable('SAPT EXCH ENERGY') * Ha2kcal))
        except Exception:
            ok = False
            break
        finally:
            psi4.core.clean()
    if not ok or len(Es) < 3:
        failed += 1
        continue
    ds = np.array(DELTAS); Ev = np.array(Es)
    A = np.column_stack([np.ones_like(ds), ds, ds**2])
    p_, *_ = np.linalg.lstsq(A, Ev, rcond=None)
    rms = float(np.sqrt(np.mean((Ev - A @ p_)**2)))
    results.append(dict(frame=fr['frame'], k_exch_Nm=float(2*p_[2]*CONV),
                        rms_residual_kcal=rms))
    if len(results) % 10 == 0:
        print(f'  {TAG} {CLAMP}: {len(results)}/{geom["n_frames"]} frames', flush=True)

k = np.array([r['k_exch_Nm'] for r in results])
out = EF / f'{TAG}_{CLAMP}_kexch_frames.json'
out.write_text(json.dumps(dict(TAG=TAG, clamp=CLAMP, n_ok=len(results), n_failed=failed,
                               k_exch_Nm=k.tolist(), per_frame=results), indent=1))
print(f'{TAG} {CLAMP}: n={len(k)} failed={failed}  '
      f'<k_exch>={k.mean():.4g}  sd={k.std(ddof=1):.4g} N/m  -> {out.name}')
