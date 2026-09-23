#!/usr/bin/env python3
"""Stage 4 checks 3, 4, 5, 6: basis, SAPT order, and fragment size on K_perp.

One variable changes per run; the geometry, the displacement axes, the grid and the fit are
identical to transverse_native_donor.py, whose quad2d this imports rather than reimplements.
Grid points are farmed to a process pool because aug-cc-pVTZ and SAPT2+ cost 15 and 10 minutes
per point, which is 7.3 and 5.0 hours if run serially.

Reporting follows the pre-registration: both eigenvalues AND the fitted gradient at every
setting, the rms residual labelled as a fit residual and never as a sampling uncertainty.

Usage: stage4_sensitivity.py LABEL TAG METHOD BASIS DONOR WALL [half] [npa] [nproc] [mem]
  DONOR: native | shell2        WALL: sidechain | backbone
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
D = ROOT / 'results/native_donor_validation'
OUT = D / 'stage4'
Ha2kcal, CONV = 627.5094740631, 0.694770
_cfg = {}


def init(method, basis, mem):
    _cfg['method'], _cfg['basis'], _cfg['mem'] = method, basis, mem


def one_point(job):
    """One SAPT call. Returns (d1, d2, exch, total) in kcal/mol."""
    import psi4
    a, b, donor, wall = job
    L = ['0 1'] + [f'{e} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}' for e, p in donor]
    L += ['--', '0 1'] + [f'{e} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}' for e, p in wall]
    L.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
    psi4.set_memory(_cfg['mem'])
    psi4.set_num_threads(1)
    psi4.core.set_output_file(os.devnull, False)
    psi4.core.clean()
    psi4.set_options({'basis': _cfg['basis'], 'scf_type': 'df', 'freeze_core': True})
    psi4.energy(_cfg['method'], molecule=psi4.geometry('\n'.join(L)))
    return (a, b, float(psi4.variable('SAPT EXCH ENERGY') * Ha2kcal),
            float(psi4.variable('SAPT TOTAL ENERGY') * Ha2kcal))


def main():
    sys.path.insert(0, str(ROOT / 'scripts'))
    from transverse_native_donor import quad2d          # identical fit, not a reimplementation

    label, tag, method, basis, dsel, wsel = sys.argv[1:7]
    half = float(sys.argv[7]) if len(sys.argv) > 7 else 0.15
    npa = int(sys.argv[8]) if len(sys.argv) > 8 else 5
    nproc = int(sys.argv[9]) if len(sys.argv) > 9 else 4
    mem = sys.argv[10] if len(sys.argv) > 10 else '12 GB'
    OUT.mkdir(parents=True, exist_ok=True)

    src = f'{tag}_r255_stage4_fragments.json' if (dsel == 'shell2' or wsel == 'backbone') \
        else f'{tag}_r255_native_donor.json'
    g = json.loads((D / src).read_text())
    dkey = 'donor_shell2' if dsel == 'shell2' else 'donor_fragment'
    wkey = 'wall_backbone' if wsel == 'backbone' else 'native_wall'
    frag = [(a['element'], np.array(a['position']), a['name']) for a in g[dkey]]
    wall = [(a['element'], np.array(a['position'])) for a in g[wkey]]

    k = next(i for i, (_, _, n) in enumerate(frag) if n == g['xferH_name'])
    assert np.allclose(frag[k][1], g['xferH'], atol=1e-6), 'transferring H mismatch'
    dC, aO = np.array(g['donor_C']), np.array(g['acceptor_O'])

    # displacement axes: identical construction to the published panel
    n = (aO - dC) / np.linalg.norm(aO - dC)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = tmp - np.dot(tmp, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    assert abs(np.dot(e1, n)) < 1e-12 and abs(np.dot(e2, n)) < 1e-12

    jobs = []
    for a in np.linspace(-half, half, npa):
        for b in np.linspace(-half, half, npa):
            moved = [(e, (p + a * e1 + b * e2) if i == k else p)
                     for i, (e, p, _) in enumerate(frag)]
            jobs.append((float(a), float(b), moved, wall))

    print(f'{label}: {tag} {method}/{basis} donor={dsel}({len(frag)}) wall={wsel}({len(wall)}) '
          f'half={half} {npa}x{npa}={len(jobs)} points on {nproc} workers', flush=True)
    rows = []
    with ProcessPoolExecutor(nproc, initializer=init, initargs=(method, basis, mem)) as ex:
        for i, r in enumerate(ex.map(one_point, jobs), 1):
            rows.append(r)
            print(f'    [{i:3d}/{len(jobs)}] d1={r[0]:+.3f} d2={r[1]:+.3f} '
                  f'exch={r[2]:+10.5f} int={r[3]:+10.5f}', flush=True)

    Hx, rx, gx = quad2d([(r[0], r[1], r[2]) for r in rows], 2)
    Ht, rt, gt = quad2d([(r[0], r[1], r[3]) for r in rows], 2)
    wx, wt = np.linalg.eigvalsh(Hx), np.linalg.eigvalsh(Ht)
    res = dict(label=label, tag=tag, method=method, basis=basis, donor=dsel, wall=wsel,
               n_donor_atoms=len(frag), n_wall_atoms=len(wall), half=half, n_per_axis=npa,
               K_perp_exch_eigs=[float(v) for v in wx],
               K_perp_exch_grad_kcal_per_A=gx, K_perp_exch_grad_norm=float(np.hypot(*gx)),
               K_perp_exch_rms_kcal=rx,
               K_perp_int_eigs=[float(v) for v in wt],
               K_perp_int_grad_kcal_per_A=gt, K_perp_int_grad_norm=float(np.hypot(*gt)),
               K_perp_int_rms_kcal=rt,
               grid=[dict(d1=r[0], d2=r[1], exch=r[2], total=r[3]) for r in rows])
    (OUT / f'{label}.json').write_text(json.dumps(res, indent=2))
    print(f'  exch eigs {wx[0]:+.5f} {wx[1]:+.5f} | grad {np.hypot(*gx):.4f} kcal/mol/A | '
          f'fit rms {rx:.2e} kcal/mol  -> stage4/{label}.json', flush=True)


if __name__ == '__main__':
    main()
