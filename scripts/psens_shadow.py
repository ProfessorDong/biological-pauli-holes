#!/usr/bin/env python3
"""Build a shadow repository so the unchanged analysis scripts run on another model's data.

Every downstream script derives its paths from PAULI_ROOT. A shadow root reproduces the
directory layout those scripts expect, with the inputs linked in, so a script run with
PAULI_ROOT=<shadow> reads that model's data and writes only inside the shadow.

  shadow_S  model-S inputs, unmodified scripts. Used once to show that the mechanism
            reproduces the published model-S files exactly.
  shadow_P  model-P inputs. Five scripts carry hard-coded model-S atom indices and wall
            residue numbers; their shadow copies get the model-P values, each mapped by
            residue and atom name and checked, and every replacement is counted.
Usage: psens_shadow.py S|P      (slomd env)
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

import parmed

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / 'results/protonation_sensitivity'
TAGS = ('WT', 'I538A', 'I552A', 'I553A', 'L546A', 'L754A', 'V750A')
PRM = {t: ('SLO_sub_solv' if t == 'WT' else f'SLO_{t}_sub_solv') for t in TAGS}
PATCHED = ('extract_native_fragment.py', 'extract_clamped_fragment.py', 'extract_frames_geometry.py',
           'confinement_budget.py', 'slo_xi0_full.py')
CAPS_BEFORE = {0: 1, 12: 1, 13: 3, 98: 3, 99: 5}   # caps preceding a model-S residue index


def p_residue(si):
    """model-S residue index -> model-P residue index (five caps are inserted)."""
    if si <= 12:
        return si + 1
    if si <= 98:
        return si + 3
    return si + 5


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def atom_map(tag):
    """model-S atom index -> model-P atom index for the LIG and wall atoms the scripts name."""
    S = parmed.load_file(str(ROOT / 'md/mcpb' / f'{PRM[tag]}.prmtop'))
    P = parmed.load_file(str(PS / 'build' / tag / 'P_solv.prmtop'))
    m = {}
    for a in S.residues[820].atoms:                     # LIG
        q = [b for b in P.residues[p_residue(820)].atoms if b.name == a.name]
        assert len(q) == 1 and P.residues[p_residue(820)].name == 'LIG'
        m[a.idx] = q[0].idx
    for si in (672, 732):
        assert S.residues[si].name == P.residues[p_residue(si)].name
    return m, S, P


def main():
    kind = sys.argv[1]
    assert kind in ('S', 'P')
    sh = PS / f'shadow_{kind}'
    if sh.exists():
        shutil.rmtree(sh)
    (sh / 'scripts').mkdir(parents=True)
    report = dict(kind=kind, links=0, patches={})

    for t in TAGS:
        src = ROOT / 'md/mcpb' / f'{PRM[t]}.prmtop' if kind == 'S' else PS / 'build' / t / 'P_solv.prmtop'
        link(src, sh / 'md/mcpb' / f'{PRM[t]}.prmtop')
        rg = ROOT / 'results/reactive_geometry' if kind == 'S' else PS / 'reactive_geometry'
        ef = ROOT / 'results/ensemble_fluctuation' if kind == 'S' else PS / 'ensemble_fluctuation'
        for c in ('r255', 'r340'):
            for suf in ('_final.rst7', '_colvar.dat'):
                f = rg / f'{t}_{c}{suf}'
                if f.exists() or kind == 'P':      # model-P outputs may not exist yet
                    link(f, sh / 'results/reactive_geometry' / f.name)
            for suf in ('_frames.npy', '_colvar.dat'):
                f = ef / f'{t}_{c}{suf}'
                if f.exists() or kind == 'P':
                    link(f, sh / 'results/ensemble_fluctuation' / f.name)
        f5 = ROOT / 'results/stage5_configs' / f'{t}_frames.npy' if kind == 'S' else PS / 'stage5_md' / f'{t}_r255_frames.npy'
        if f5.exists() or kind == 'P':
            link(f5, sh / 'results/stage5_configs' / f'{t}_frames.npy')
        # the single-snapshot F-SAPT decomposition: read only for two comparison fields
        dec = ROOT / 'results/sapt_bio/donor_fragment' / f'{t}_fsapt_decomposition.json'
        (sh / 'results/sapt_bio/donor_fragment').mkdir(parents=True, exist_ok=True)
        shutil.copy(dec, sh / 'results/sapt_bio/donor_fragment' / dec.name)

    # fragment indices of the native C5H8 donor
    for t in TAGS:
        fi = json.loads((ROOT / 'results/sapt_bio/native_fragment' / f'{t}_fragment_indices.json').read_text())
        if kind == 'P':
            m, S, P = atom_map(t)
            fi['prm'] = PRM[t]
            fi['donor_C'], fi['xferH'] = m[fi['donor_C']], m[fi['xferH']]
            fi['carbons'] = [m[i] for i in fi['carbons']]
            fi['hydrogens'] = [dict(h, idx=m[h['idx']]) for h in fi['hydrogens']]
            fi['caps'] = [dict(host=m[c['host']], beyond=m[c['beyond']]) for c in fi['caps']]
            fi['_model'] = 'P (indices mapped from model S by residue and atom name)'
            for i in fi['carbons'] + [fi['donor_C']]:
                assert P.atoms[i].residue.name == 'LIG' and P.atoms[i].atomic_number == 6
        out = sh / 'results/sapt_bio/native_fragment' / f'{t}_fragment_indices.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fi, indent=1))

    # scripts: links, except patched copies for model P
    maps = {t: atom_map(t)[0] for t in TAGS} if kind == 'P' else None
    for f in sorted((ROOT / 'scripts').iterdir()):
        if f.suffix not in ('.py', '.sh'):
            continue
        dst = sh / 'scripts' / f.name
        if kind == 'S' or f.name not in PATCHED:
            link(f, dst)
            report['links'] += 1
            continue
        txt, n = f.read_text(), 0
        lines = txt.split('\n')
        tag = None
        for i, line in enumerate(lines):
            mt = re.match(r"\s*'(\w+)'\s*:\s*dict\(", line)
            if mt:
                tag = mt.group(1) if mt.group(1) in TAGS else None
            elif re.match(r"\s*}", line):
                tag = None                        # end of the SYSTEMS table
            if tag is None:
                continue

            def sub_atom(mo, tag=tag):
                return f"{mo.group(1)}={maps[tag][int(mo.group(2))]}"

            new, k1 = re.subn(r"\b(xferH_idx|donor_C_idx|xferH|donor)=(\d{5})\b", sub_atom, line)
            new, k2 = re.subn(r"\b(wall_res_id|wid)=(\d{3})\b",
                              lambda mo: f"{mo.group(1)}={p_residue(int(mo.group(2)))}", new)
            lines[i], n = new, n + k1 + k2
        expect = {'confinement_budget.py': 14, 'slo_xi0_full.py': 14}.get(f.name, 21)
        assert n == expect, f'{f.name}: {n} replacements, expected {expect}'
        dst.write_text('\n'.join(lines))
        report['patches'][f.name] = n
    (sh / 'shadow_report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
