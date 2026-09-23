#!/usr/bin/env python3
"""Prove every SAPT fragment coordinate came from the MD frame its metadata names.

WHY THIS EXISTS
  Task C6 re-ran rep-1 umbrella sampling for WT, I553A, L754A and DM on the Blackwell GPU
  and wrote the new trajectories over the old paths. The native-fragment SAPT geometries
  for WT, I553A and L754A had been extracted on 2026-08-06 from the trajectories that the
  re-run replaced, so from 2026-08-10 the `seed` field in those three files named a file
  whose contents no longer matched them. The published k_exch values were never wrong: the
  frames survive in results/umbrella/_RTX4060_rep1_archive_2026-08-10/, and the seed fields
  and the SYSTEMS table now point there. But nothing in the pipeline would have caught the
  mismatch, and one thing did silently go wrong because of it: the pentadienyl donor swap
  built its fragment from the replacement frame while comparing it against a wall from the
  original, putting the two 11 A apart.

  So this exists to make the coordinate provenance a checked property rather than an
  assumption. It is cheap, it has no arguments, and it should be run after anything that
  touches results/umbrella or the native-fragment extraction.

WHAT IT CHECKS, per system, for both the methane and the pentadienyl fragment file
  1. the `seed` recorded in the JSON is the one the SYSTEMS table specifies
  2. that seed file exists
  3. every real atom of the fragment coincides with an atom of that frame, exactly
  4. every link-atom cap sits at a C-H bond length from the heavy atom it caps
  5. donor and wall are close enough to exchange, which a frame mismatch would break

  Caps are excluded from check 3 by construction: they are built by the extractor along a
  bond vector and are not present in the MD frame. That is the one legitimate reason for a
  fragment coordinate to be absent from the trajectory, and it is checked separately.

Usage: verify_sapt_geometry_provenance.py [--fix-seed-field]     (slomd env: openmm)
       exit status is the number of failed checks
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from openmm import app, unit

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
DIR = ROOT / 'results/sapt_bio/native_fragment'
UMB = ROOT / 'results/umbrella'
CAP_BOND = 1.09          # the C-H distance the extractor places link atoms at
CONTACT = 6.0            # A; beyond this the exchange energy is numerically zero

fails = []


def systems():
    """Read the one authoritative tag -> seed table, from the extraction script itself."""
    src = (ROOT / 'scripts/extract_native_fragment.py').read_text()
    ns = {}
    exec(src[src.index('ARCH = '):src.index('}\n', src.index('SYSTEMS = {')) + 1], {}, ns)
    return ns['SYSTEMS']


def note(ok, label, detail=''):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label:52s} {detail}")
    if not ok:
        fails.append(label)


def frame(seed):
    return np.array([[v.x, v.y, v.z] for v in
                     app.AmberInpcrdFile(str(UMB / seed)).positions
                     .value_in_unit(unit.angstrom)])


def check_file(path, seed, pos, kind):
    g = json.loads(path.read_text())
    note(g['seed'] == seed, f'{path.name}: seed field', f"{g['seed']}")

    groups = [('wall', g['native_wall'])]
    if 'donor_fragment' in g:
        groups.append(('donor', g['donor_fragment']))
    named = [('donor_C', np.array(g['donor_C'])), ('xferH', np.array(g['xferH']))]

    worst, wn, caps = 0.0, '', []
    for _, atoms in groups:
        for a in atoms:
            p = np.array(a['position'], float)
            if a['name'].startswith('Hcap'):
                caps.append((a['name'], p, atoms))
                continue
            d = float(np.min(np.linalg.norm(pos - p, axis=1)))
            if d > worst:
                worst, wn = d, a['name']
    for name, p in named:
        d = float(np.min(np.linalg.norm(pos - p, axis=1)))
        if d > worst:
            worst, wn = d, name
    note(worst < 1e-6, f'{path.name}: every real atom is in the frame',
         f'worst {worst:.1e} A ({wn})')

    capbad = []
    for name, p, atoms in caps:
        host = name[4:]
        hp = [np.array(a['position'], float) for a in atoms if a['name'] == host]
        if not hp:
            capbad.append(f'{name} has no host {host}')
            continue
        d = float(np.linalg.norm(p - hp[0]))
        if abs(d - CAP_BOND) > 1e-3:
            capbad.append(f'{name} at {d:.4f} A')
    note(not capbad, f'{path.name}: {len(caps)} link cap(s) at {CAP_BOND} A',
         '; '.join(capbad))

    W = np.array([a['position'] for a in g['native_wall']], float)
    D = (np.array([a['position'] for a in g['donor_fragment']], float)
         if 'donor_fragment' in g else np.array([g['donor_C'], g['xferH']], float))
    dmin = float(np.min(np.linalg.norm(D[:, None, :] - W[None, :, :], axis=-1)))
    note(dmin < CONTACT, f'{path.name}: {kind} donor is in contact with the wall',
         f'nearest {dmin:.2f} A')


def main():
    fix = '--fix-seed-field' in sys.argv
    SYS = systems()
    for tag, info in SYS.items():
        seed = info['seed']
        print(f'\n{tag}   seed = {seed}')
        if not (UMB / seed).exists():
            note(False, f'{tag}: seed file exists', str(UMB / seed))
            continue
        pos = frame(seed)
        for suffix, kind in (('_geometry.json', 'methane'),
                             ('_pentadienyl.json', 'pentadienyl')):
            path = DIR / f'{tag}{suffix}'
            if not path.exists():
                print(f'  --    {path.name} absent, skipped')
                continue
            if fix:
                g = json.loads(path.read_text())
                if g['seed'] != seed:
                    g['seed'] = seed
                    path.write_text(json.dumps(g, indent=2))
                    print(f'  fixed seed field in {path.name}')
            check_file(path, seed, pos, kind)

    print(f'\n{"ALL PROVENANCE VERIFIED" if not fails else f"{len(fails)} FAILED"}')
    for f in fails:
        print(f'   {f}')
    os._exit(len(fails))


if __name__ == '__main__':
    main()
