"""SAPT bridge: extract native wall + donor fragments from a CLAMPED trajectory.

Feeds the reactive-geometry campaign into the SAME audited fragment machinery used
for main-text 2.8 (extract_native_fragment.py), so the descriptor is identical and
ONLY the donor-acceptor geometry differs. Leu walls give capped isobutane, Asn walls
capped acetamide; the cap is placed along the CB-CA vector at 1.09 A exactly as in
the published pipeline.

Usage: extract_clamped_fragment.py <TAG> <clamp>     clamp in {r255, r340}
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import sys, json, numpy as np, parmed
from openmm import app, unit
from pathlib import Path

TAG, CLAMP = sys.argv[1], sys.argv[2]
ROOT = Path(_REPO)
MD = ROOT / 'md/mcpb'
RG = ROOT / 'results/reactive_geometry'

# identical system table to extract_native_fragment.py (audited), except the
# structure is taken from the clamped run rather than an umbrella window
SYSTEMS = {
    'WT':    dict(prm='SLO_sub_solv',       xferH_idx=13031, donor_C_idx=13002,
                  wall_res_name='LEU', wall_res_id=732),
    'I553A': dict(prm='SLO_I553A_sub_solv', xferH_idx=13022, donor_C_idx=12993,
                  wall_res_name='LEU', wall_res_id=732),
    'I552A': dict(prm='SLO_I552A_sub_solv', xferH_idx=13022, donor_C_idx=12993,
                  wall_res_name='LEU', wall_res_id=732),
    'L754A': dict(prm='SLO_L754A_sub_solv', xferH_idx=13023, donor_C_idx=12993,
                  wall_res_name='ASN', wall_res_id=672),
    'V750A': dict(prm='SLO_V750A_sub_solv', xferH_idx=13025, donor_C_idx=12996,
                  wall_res_name='LEU', wall_res_id=732),
    'I538A': dict(prm='SLO_I538A_sub_solv', xferH_idx=13022, donor_C_idx=12993,
                  wall_res_name='LEU', wall_res_id=732),
    'L546A': dict(prm='SLO_L546A_sub_solv', xferH_idx=13022, donor_C_idx=12993,
                  wall_res_name='LEU', wall_res_id=732),
}
info = SYSTEMS[TAG]
rst = RG / f'{TAG}_{CLAMP}_final.rst7'
if not rst.exists():
    print(f'ERROR: {rst} not found'); sys.exit(1)

prm = parmed.load_file(str(MD / f'{info["prm"]}.prmtop'))
seed = app.AmberInpcrdFile(str(rst))
pos = np.array([[v.x, v.y, v.z] for v in seed.positions.value_in_unit(unit.angstrom)])

wall_atoms = None
for r in prm.residues:
    if r.name == info['wall_res_name'] and r.number == info['wall_res_id']:
        wall_atoms = r.atoms; break
if wall_atoms is None:
    print(f'ERROR: {info["wall_res_name"]}{info["wall_res_id"]} not found'); sys.exit(1)

if info['wall_res_name'] == 'LEU':
    keep = ['CB','HB2','HB3','CG','HG','CD1','HD11','HD12','HD13',
            'CD2','HD21','HD22','HD23']
else:
    keep = ['CB','HB2','HB3','CG','OD1','ND2','HD21','HD22']

native, CB_pos, CA_pos = [], None, None
for a in wall_atoms:
    if a.name == 'CB': CB_pos = pos[a.idx]
    if a.name == 'CA': CA_pos = pos[a.idx]
    if a.name in keep:
        el = 'H' if a.name.startswith('H') else a.name[0]
        native.append(dict(element=el, position=pos[a.idx].tolist(), name=a.name))

# H cap closing the severed CB-CA bond, along the original bond vector at 1.09 A
d = (CA_pos - CB_pos) / np.linalg.norm(CA_pos - CB_pos)
cap = CB_pos + d * 1.09
native.append(dict(element='H', position=cap.tolist(), name='HcapCB'))

# CORRECTION 2026-09-17: the field formerly called r_DA_clamped was the donor-carbon to
# transferring-hydrogen distance, i.e. the constrained C-H bond at about 1.096 A, not the
# donor-acceptor distance the name implied. It is renamed r_CH, and the real donor-acceptor
# distance is added, resolved from the topology (the acceptor oxygen is the OH1 oxygen bonded
# to the iron) rather than assumed.
_fe = [a for a in prm.atoms if a.mass > 50]
assert len(_fe) == 1, f'{TAG}: expected one metal, found {len(_fe)}'
_acc = [(b.atom2 if b.atom1 is _fe[0] else b.atom1) for b in _fe[0].bonds]
_acc = [a for a in _acc if a.residue.name == 'OH1']
assert len(_acc) == 1, f'{TAG}: {len(_acc)} OH1 oxygens bonded to Fe, expected 1'
_acc_idx = _acc[0].idx
_dC, _xH = info['donor_C_idx'], info['xferH_idx']
assert 1.0 < float(np.linalg.norm(pos[_dC] - pos[_xH])) < 1.2, \
    f'{TAG}: donor C and transferring H are not bonded'

out = dict(TAG=TAG, clamp=CLAMP,
           wall_residue=f'{info["wall_res_name"]}{info["wall_res_id"]}',
           donor_C=pos[_dC].tolist(),
           xferH=pos[_xH].tolist(),
           acceptor_O=pos[_acc_idx].tolist(),
           acceptor_idx=int(_acc_idx),
           acceptor_label=f'{_acc[0].residue.name}{_acc[0].residue.number}:{_acc[0].name}',
           native_wall=native,
           r_CH=float(np.linalg.norm(pos[_dC] - pos[_xH])),
           r_DA_clamped=float(np.linalg.norm(pos[_dC] - pos[_acc_idx])))
f = RG / f'{TAG}_{CLAMP}_geometry.json'
f.write_text(json.dumps(out, indent=2))
print(f'{TAG} {CLAMP}: {len(native)} wall atoms (incl. cap), wrote {f.name}')
