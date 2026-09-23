#!/usr/bin/env python3
"""One grid point of a stage-4 sensitivity run, as its own OS process.

Psi4 is not reliable inside a forked worker pool here, so parallelism is one process per point,
the pattern the 101-configuration stage-5 campaign used successfully. A point whose output file
already exists is skipped, so a driver can be re-run to resume.

Usage: stage4_point.py LABEL TAG METHOD BASIS DONOR WALL HALF NPA I J MEM
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
D = ROOT / 'results/native_donor_validation'
Ha2kcal = 627.5094740631


def axes(dC, aO):
    n = (aO - dC) / np.linalg.norm(aO - dC)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = tmp - np.dot(tmp, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    assert abs(np.dot(e1, n)) < 1e-12 and abs(np.dot(e2, n)) < 1e-12
    return e1, e2


def main():
    label, tag, method, basis, dsel, wsel = sys.argv[1:7]
    half, npa, i, j = float(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])
    mem = sys.argv[11] if len(sys.argv) > 11 else '12 GB'
    out = D / 'stage4' / 'points' / f'{label}_{i:02d}_{j:02d}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f'skip {out.name}'); return

    src = f'{tag}_r255_stage4_fragments.json' if (dsel == 'shell2' or wsel == 'backbone') \
        else f'{tag}_r255_native_donor.json'
    g = json.loads((D / src).read_text())
    frag = [(a['element'], np.array(a['position']), a['name'])
            for a in g['donor_shell2' if dsel == 'shell2' else 'donor_fragment']]
    wall = [(a['element'], np.array(a['position']))
            for a in g['wall_backbone' if wsel == 'backbone' else 'native_wall']]
    k = next(t for t, (_, _, n) in enumerate(frag) if n == g['xferH_name'])
    assert np.allclose(frag[k][1], g['xferH'], atol=1e-6), 'transferring H mismatch'
    e1, e2 = axes(np.array(g['donor_C']), np.array(g['acceptor_O']))

    grid = np.linspace(-half, half, npa)
    a, b = float(grid[i]), float(grid[j])
    donor = [(e, (p + a * e1 + b * e2) if t == k else p)
             for t, (e, p, _) in enumerate(frag)]

    import psi4
    L = ['0 1'] + [f'{e} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}' for e, p in donor]
    L += ['--', '0 1'] + [f'{e} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}' for e, p in wall]
    L.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
    psi4.set_memory(mem)
    psi4.set_num_threads(1)
    psi4.core.set_output_file(os.devnull, False)
    psi4.set_options({'basis': basis, 'scf_type': 'df', 'freeze_core': True})
    psi4.energy(method, molecule=psi4.geometry('\n'.join(L)))
    rec = dict(label=label, i=i, j=j, d1=a, d2=b,
               exch=float(psi4.variable('SAPT EXCH ENERGY') * Ha2kcal),
               total=float(psi4.variable('SAPT TOTAL ENERGY') * Ha2kcal),
               method=method, basis=basis, donor=dsel, wall=wsel,
               n_donor=len(frag), n_wall=len(wall))
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, out)                      # atomic, survives a mid-write kill
    print(f'{label} [{i},{j}] d1={a:+.3f} d2={b:+.3f} exch={rec["exch"]:+10.5f}')


if __name__ == '__main__':
    main()
