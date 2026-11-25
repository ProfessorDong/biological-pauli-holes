#!/usr/bin/env python3
"""Build the shadow roots for the L754A pro-S correction (both models).

  shadow_SpS  model S: inputs (topologies, frames, restarts, umbrella seeds) are linked read-only;
              the published per-variant outputs of the six other variants are COPIED, never linked,
              so no script can write through a link into results/; L754A outputs are left out and
              regenerated; the five index-bearing scripts get L754A H25 (13022) in place of H26 (13023).
  shadow_PpS  model P: a copy of shadow_P with the L754A outputs removed and the translated scripts
              changed from 13046 (H26) to 13045 (H25).
See results/protonation_sensitivity/L754A_PRO_S_DECLARATION.md.
Usage: psens_proS_setup.py      (slomd env)
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import parmed

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / 'results/protonation_sensitivity'
PATCHED = ('extract_native_fragment.py', 'extract_clamped_fragment.py', 'extract_frames_geometry.py',
           'confinement_budget.py', 'slo_xi0_full.py')
OUTDIRS = ('reactive_geometry', 'ensemble_fluctuation', 'sapt_bio/native_fragment', 'sapt_bio/donor_fragment',
           'transverse_hessian', 'native_donor_validation', 'stage5_configs', 'sapt_bio')


def patch_l754a(text, old, new, name):
    """Change the L754A hydrogen in the SYSTEMS entry only; the count must be exactly one."""
    lines = text.split('\n')
    n = 0
    for i, line in enumerate(lines):
        if re.match(r"\s*'L754A'\s*:\s*dict\(", line):
            new_line, k = re.subn(r"\b(xferH_idx|xferH)=" + str(old) + r"\b", lambda m: f"{m.group(1)}={new}", line)
            lines[i] = new_line
            n += k
    assert n == 1, f'{name}: {n} L754A hydrogen replacements, expected 1'
    return '\n'.join(lines)


def is_l754a(p):
    return p.name.startswith('L754A') or p.name.startswith('s4_wall_bb_L754A')


def build_S():
    sh = PS / 'shadow_SpS'
    if sh.exists():
        shutil.rmtree(sh)
    (sh / 'scripts').mkdir(parents=True)
    for f in sorted((ROOT / 'scripts').iterdir()):
        if f.suffix not in ('.py', '.sh'):
            continue
        dst = sh / 'scripts' / f.name
        if f.name in PATCHED:
            dst.write_text(patch_l754a(f.read_text(), 13023, 13022, f.name))
        else:
            os.symlink(f, dst)
    # read-only inputs
    (sh / 'md').mkdir()
    os.symlink(ROOT / 'md/mcpb', sh / 'md/mcpb')
    (sh / 'results').mkdir()
    os.symlink(ROOT / 'results/umbrella', sh / 'results/umbrella')
    for d in ('reactive_geometry', 'ensemble_fluctuation', 'stage5_configs'):
        (sh / 'results' / d).mkdir(parents=True, exist_ok=True)
        for f in (ROOT / 'results' / d).iterdir():
            if f.suffix in ('.rst7', '.npy', '.dat') and f.is_file():
                os.symlink(f, sh / 'results' / d / f.name)
    # published outputs of the six other variants: copied
    n = 0
    for d in OUTDIRS:
        src = ROOT / 'results' / d
        for f in src.rglob('*.json'):
            rel = f.relative_to(ROOT / 'results')
            if d == 'sapt_bio' and len(rel.parts) > 2:
                continue                      # sub-folders handled by their own entries
            if is_l754a(f) or any(part.startswith('L754A') for part in rel.parts):
                continue
            dst = sh / 'results' / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(f, dst)
                n += 1
    return sh, n


def build_P():
    src, sh = PS / 'shadow_P', PS / 'shadow_PpS'
    if sh.exists():
        shutil.rmtree(sh)
    subprocess.run(['cp', '-a', str(src), str(sh)], check=True)
    removed = 0
    for f in list((sh / 'results').rglob('*')):
        if (f.is_file() or f.is_symlink()) and f.suffix == '.json' and is_l754a(f):
            f.unlink()
            removed += 1
    if (sh / 'results/stage5_configs/L754A').exists():
        shutil.rmtree(sh / 'results/stage5_configs/L754A')
    for name in PATCHED:
        p = sh / 'scripts' / name
        p.write_text(patch_l754a(p.read_text(), 13046, 13045, name))
    return sh, removed


def main():
    S = parmed.load_file(str(ROOT / 'md/mcpb/SLO_L754A_sub_solv.prmtop'))
    P = parmed.load_file(str(PS / 'build/L754A/P_solv.prmtop'))
    assert (S.atoms[13022].name, S.atoms[13023].name) == ('H25', 'H26')
    assert (P.atoms[13045].name, P.atoms[13046].name) == ('H25', 'H26')
    shS, n = build_S()
    shP, r = build_P()
    rep = dict(shadow_SpS=str(shS), copied_outputs=n, shadow_PpS=str(shP), removed_L754A_outputs=r,
               H25_S=13022, H25_P=13045)
    (PS / 'proS').mkdir(exist_ok=True)
    (PS / 'proS/setup_report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
