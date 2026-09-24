"""Step 1 of B2 native-fragment SAPT: extract per-atom coordinates for the donor methane
(centred at substrate C11) and the wall native-fragment (Leu isobutane or Asn acetamide)
from the topology + snapshot. Writes a JSON that the psi4-env SAPT scan reads.

Run in slomd env (which has parmed and openmm)."""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, json, numpy as np, parmed
from openmm import app, unit
from pathlib import Path

TAG = sys.argv[1]
MD = Path(_REPO + '/md/mcpb')
UMB = Path(_REPO + '/results/umbrella')

# WT, I553A and L754A were extracted from the rep-1 trajectories that task C6 later
# re-ran in place on the Blackwell GPU. The frames these fragments came from therefore
# live in the C6 archive, not at results/umbrella/<system>/, and the seeds below point
# there so that re-running this script reproduces the published geometries bit for bit.
# verify_sapt_geometry_provenance.py checks every atom of all seven against these paths.
ARCH = '_RTX4060_rep1_archive_2026-08-10/'

SYSTEMS = {
    'WT':    dict(prm='SLO_sub_solv',       xferH_idx=13031, seed=ARCH+'WT/win_2.95_final.rst7',
                  donor_C_idx=13002, wall_res_name='LEU', wall_res_id=732),
    'I553A': dict(prm='SLO_I553A_sub_solv', xferH_idx=13022, seed=ARCH+'I553A/win_2.70_final.rst7',
                  donor_C_idx=12993, wall_res_name='LEU', wall_res_id=732),
    'I552A': dict(prm='SLO_I552A_sub_solv', xferH_idx=13022, seed='I552A_rep3/win_2.70_final.rst7',
                  donor_C_idx=12993, wall_res_name='LEU', wall_res_id=732),
    'L754A': dict(prm='SLO_L754A_sub_solv', xferH_idx=13023, seed=ARCH+'L754A/win_2.95_final.rst7',
                  donor_C_idx=12993, wall_res_name='ASN', wall_res_id=672),
    'V750A': dict(prm='SLO_V750A_sub_solv', xferH_idx=13025, seed='V750A/win_2.95_final.rst7',
                  donor_C_idx=12996, wall_res_name='LEU', wall_res_id=732),
    'I538A': dict(prm='SLO_I538A_sub_solv', xferH_idx=13022, seed='I538A_rep3/win_2.70_final.rst7',
                  donor_C_idx=12993, wall_res_name='LEU', wall_res_id=732),
    'L546A': dict(prm='SLO_L546A_sub_solv', xferH_idx=13022, seed='L546A_rep3/win_2.70_final.rst7',
                  donor_C_idx=12993, wall_res_name='LEU', wall_res_id=732),
}
info = SYSTEMS[TAG]

prm = parmed.load_file(str(MD/f'{info["prm"]}.prmtop'))
seed = app.AmberInpcrdFile(str(UMB/info['seed']))
pos = np.array([[v.x, v.y, v.z] for v in seed.positions.value_in_unit(unit.angstrom)])

wall_atoms = None
for r in prm.residues:
    if r.name == info['wall_res_name'] and r.number == info['wall_res_id']:
        wall_atoms = r.atoms; break
if wall_atoms is None:
    print(f'ERROR: residue {info["wall_res_name"]}{info["wall_res_id"]} not found'); sys.exit(1)

if info['wall_res_name'] == 'LEU':
    keep = ['CB','HB2','HB3','CG','HG','CD1','HD11','HD12','HD13','CD2','HD21','HD22','HD23']
else:  # ASN
    keep = ['CB','HB2','HB3','CG','OD1','ND2','HD21','HD22']

wall_by_name = {a.name: (a.idx, pos[a.idx]) for a in wall_atoms}
native = []
for name in keep:
    if name not in wall_by_name:
        print(f'  warn: {name} missing'); continue
    _, p = wall_by_name[name]
    elem = 'H' if name.startswith('H') else name[0]
    native.append(dict(element=elem, position=p.tolist(), name=name))

# Cap H at CB (replacing CA connection)
CB_pos = wall_by_name['CB'][1]
CA_pos = None
for a in wall_atoms:
    if a.name == 'CA': CA_pos = pos[a.idx]; break
if CA_pos is not None:
    d = CB_pos - CA_pos; d /= np.linalg.norm(d)
    cap = CB_pos - d * 1.09
    native.append(dict(element='H', position=cap.tolist(), name='HcapCB'))

donor_C_pos = pos[info['donor_C_idx']].tolist()
xferH_pos = pos[info['xferH_idx']].tolist()

out = dict(
    TAG=TAG, wall_residue=f'{info["wall_res_name"]}{info["wall_res_id"]}',
    donor_C=donor_C_pos, xferH=xferH_pos,
    native_wall=native,
    seed=info['seed'],
)
OUT = Path(_REPO + '/results/sapt_bio/native_fragment')
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT/f'{TAG}_geometry.json','w') as f:
    json.dump(out, f, indent=2)
print(f'wrote {OUT/f"{TAG}_geometry.json"}')
print(f'donor C at {donor_C_pos}')
print(f'xfer H at {xferH_pos}')
print(f'native wall has {len(native)} atoms')
