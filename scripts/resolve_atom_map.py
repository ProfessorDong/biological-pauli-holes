#!/usr/bin/env python3
"""Resolve the reactive atoms from topology, and refuse to proceed if they are wrong.

WHY THIS EXISTS
  The reactive-geometry and ensemble-fluctuation campaigns passed hardcoded zero-based atom
  indices for every variant. Six of the seven share the same indices; V750A is offset by
  three because its topology differs. The campaign scripts used the common value for V750A
  as well, so that system was restrained between LIG:C11 and the Ile817 carboxylate oxygen
  instead of between the bis-allylic donor C14 and the iron-bound hydroxide. Those indices
  are valid atoms, so nothing raised, nothing crashed, and the saved colvar sat neatly on the
  requested distance while the intended coordinate stayed near 5.7 A. The error is invisible
  in every output the campaign produced.

  The remedy is not a corrected constant. It is to stop writing indices down at all: resolve
  them from the topology every time, assert the chemistry, and print what was resolved.

WHAT IS ASSERTED, PER VARIANT
  1. the donor is a LIG carbon named C14 carrying exactly two hydrogens
  2. the transferring hydrogen is covalently bonded to that carbon
  3. the acceptor oxygen belongs to residue OH1 and is covalently bonded to the iron
  4. donor and acceptor are distinct residues

  Any failure raises rather than returning a usable map, because a campaign that runs on a
  silently wrong pair is worse than one that does not run.

NOTE ON ELEMENTS
  parmed reports MCPB metal-site atoms with element_name 'Y'. Identify the iron by mass and
  the acceptor by residue name and bonding, never by element_name.

Usage: resolve_atom_map.py [--json results/atom_map.json]    (slomd env: parmed)
"""
import json
import os
import sys
from pathlib import Path

import parmed

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
PRM = {'WT': 'SLO_sub_solv', 'I553A': 'SLO_I553A_sub_solv', 'I552A': 'SLO_I552A_sub_solv',
       'L754A': 'SLO_L754A_sub_solv', 'V750A': 'SLO_V750A_sub_solv',
       'I538A': 'SLO_I538A_sub_solv', 'L546A': 'SLO_L546A_sub_solv',
       'DM': 'SLO_DM_sub_solv'}


def neighbours(a):
    return [(b.atom2 if b.atom1 is a else b.atom1) for b in a.bonds]


def resolve(tag, prm_name):
    p = parmed.load_file(str(ROOT / f'md/mcpb/{prm_name}.prmtop'))
    fe = [a for a in p.atoms if a.mass > 50]
    assert len(fe) == 1, f'{tag}: expected one heavy metal, found {len(fe)}'
    fe = fe[0]
    acc = [a for a in neighbours(fe) if a.residue.name == 'OH1']
    assert len(acc) == 1, f'{tag}: {len(acc)} OH1 oxygens bonded to Fe, expected 1'
    acc = acc[0]
    assert acc.mass > 15 and acc.mass < 17, f'{tag}: acceptor mass {acc.mass} is not oxygen'
    dC = [a for a in p.atoms if a.residue.name == 'LIG' and a.name == 'C14']
    assert len(dC) == 1, f'{tag}: {len(dC)} LIG:C14 atoms, expected 1'
    dC = dC[0]
    hs = [h for h in neighbours(dC) if h.element_name == 'H']
    assert len(hs) == 2, f'{tag}: donor C14 carries {len(hs)} hydrogens, expected 2'
    assert dC.residue is not acc.residue, f'{tag}: donor and acceptor share a residue'
    return dict(
        prmtop=prm_name, n_atoms=len(p.atoms),
        donor_C=dC.idx, donor_C_label=f'{dC.residue.name}{dC.residue.number}:{dC.name}',
        acceptor_O=acc.idx,
        acceptor_label=f'{acc.residue.name}{acc.residue.number}:{acc.name}',
        Fe=fe.idx, H_on_donor=[h.idx for h in hs], H_labels=[h.name for h in hs])


def main():
    out, bad = {}, []
    print(f'{"variant":8s}{"donor":>8s}  {"label":16s}{"acceptor":>9s}  {"label":14s}{"H on donor":>18s}')
    for tag, prm in PRM.items():
        try:
            m = resolve(tag, prm)
            out[tag] = m
            print(f'{tag:8s}{m["donor_C"]:>8d}  {m["donor_C_label"]:16s}'
                  f'{m["acceptor_O"]:>9d}  {m["acceptor_label"]:14s}'
                  f'{str(m["H_on_donor"]):>18s}')
        except AssertionError as e:
            bad.append(str(e))
            print(f'{tag:8s}  FAILED: {e}')
    if bad:
        print('\nATOM MAP INVALID; do not run any campaign from it')
        os._exit(1)
    # the offset that caused the original error, made explicit
    offsets = {t: m['donor_C'] for t, m in out.items()}
    odd = [t for t, v in offsets.items() if v != offsets['I553A']]
    print(f'\n  variants NOT sharing I553A\'s donor index ({offsets["I553A"]}): '
          f'{", ".join(f"{t}={offsets[t]}" for t in odd)}')
    print('  this is why hardcoding one index across variants is unsafe')
    dest = ROOT / 'results/atom_map.json'
    dest.write_text(json.dumps(out, indent=1))
    print(f'\nwrote {dest}')
    os._exit(0)


if __name__ == '__main__':
    main()
