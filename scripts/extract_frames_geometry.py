"""Extract wall + donor fragments for EVERY saved frame of a clamped run.

Step 1 of the ensemble-fluctuation bridge (slomd env: parmed). Writes one JSON
holding all frames' geometries, which the psi4-env worker then loops over. Uses
exactly the residue table, keep-lists and 1.09 A CB cap of the audited main-text
2.8 extractor, so only the frame index differs between this and the published
single-configuration descriptor.

Usage: extract_frames_geometry.py <TAG> <clamp>
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, json, numpy as np, parmed
from pathlib import Path

TAG, CLAMP = sys.argv[1], sys.argv[2]
ROOT = Path(_REPO)
MD = ROOT / 'md/mcpb'
EF = ROOT / 'results/ensemble_fluctuation'

SYSTEMS = {
    'WT':    dict(prm='SLO_sub_solv',       xferH=13031, donor=13002, wall='LEU', wid=732),
    'I553A': dict(prm='SLO_I553A_sub_solv', xferH=13022, donor=12993, wall='LEU', wid=732),
    'I552A': dict(prm='SLO_I552A_sub_solv', xferH=13022, donor=12993, wall='LEU', wid=732),
    'L754A': dict(prm='SLO_L754A_sub_solv', xferH=13023, donor=12993, wall='ASN', wid=672),
    'V750A': dict(prm='SLO_V750A_sub_solv', xferH=13025, donor=12996, wall='LEU', wid=732),
    'I538A': dict(prm='SLO_I538A_sub_solv', xferH=13022, donor=12993, wall='LEU', wid=732),
    'L546A': dict(prm='SLO_L546A_sub_solv', xferH=13022, donor=12993, wall='LEU', wid=732),
}
info = SYSTEMS[TAG]
fn = EF / f'{TAG}_{CLAMP}_frames.npy'
if not fn.exists():
    print(f'ERROR: {fn} not found'); sys.exit(1)
frames = np.load(fn)                       # (nframes, natoms, 3) in angstrom

prm = parmed.load_file(str(MD / f'{info["prm"]}.prmtop'))
wall = None
for r in prm.residues:
    if r.name == info['wall'] and r.number == info['wid']:
        wall = r; break
if wall is None:
    print(f'ERROR: {info["wall"]}{info["wid"]} not found'); sys.exit(1)

keep = (['CB','HB2','HB3','CG','HG','CD1','HD11','HD12','HD13','CD2','HD21','HD22','HD23']
        if info['wall'] == 'LEU' else
        ['CB','HB2','HB3','CG','OD1','ND2','HD21','HD22'])
keep_idx = [(a.idx, a.name, 'H' if a.name.startswith('H') else a.name[0])
            for a in wall.atoms if a.name in keep]
cb_idx = next(a.idx for a in wall.atoms if a.name == 'CB')
ca_idx = next(a.idx for a in wall.atoms if a.name == 'CA')

out = []
for k in range(frames.shape[0]):
    pos = frames[k].astype(float)
    native = [dict(element=e, position=pos[i].tolist(), name=n) for i, n, e in keep_idx]
    d = pos[ca_idx] - pos[cb_idx]
    d /= np.linalg.norm(d)
    native.append(dict(element='H', position=(pos[cb_idx] + d*1.09).tolist(), name='HcapCB'))
    out.append(dict(frame=k,
                    donor_C=pos[info['donor']].tolist(),
                    xferH=pos[info['xferH']].tolist(),
                    native_wall=native))

f = EF / f'{TAG}_{CLAMP}_frames_geometry.json'
f.write_text(json.dumps(dict(TAG=TAG, clamp=CLAMP,
                             wall_residue=f'{info["wall"]}{info["wid"]}',
                             n_frames=len(out), frames=out), indent=1))
print(f'{TAG} {CLAMP}: {len(out)} frames x {len(native)} wall atoms -> {f.name}')
