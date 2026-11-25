#!/usr/bin/env python3
"""Collate the per-variant results and run the panel-level analyses in a shadow root.

  psens_panel.py S   link the published model-S intermediates into shadow_S, rerun the two
                     pre-registered analyses there and compare with the published results
  psens_panel.py P   collate the model-P per-variant results exactly as the model-S files
                     are laid out, then run every panel-level script in shadow_P
Collation follows the published files: kexch_<clamp>.json is {TAG: k}; a per-frame list
keeps the finite values in frame order and omits frames that failed.
"""
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / 'results/protonation_sensitivity'
SLOMD = '/home/liang/anaconda3/envs/slomd/bin/python'
PAULI = '/home/liang/anaconda3/envs/pauli/bin/python'
TAGS = ('I538A', 'I552A', 'I553A', 'L546A', 'L754A', 'V750A', 'WT')
PANEL = [(PAULI, 'analyze_reactive_geometry.py'), (PAULI, 'analyze_ensemble_fluctuation.py'),
         (PAULI, 'exponential_law_panel.py'), (PAULI, 'exponential_law_prefactor.py'),
         (PAULI, 'transverse_vs_separation.py'),
         (PAULI, 'native_donor_panel.py'), (PAULI, 'native_donor_eigen_uncertainty.py'),
         (PAULI, 'total_restoring_panel.py'), (PAULI, 'duarte_projection_check.py'),
         (PAULI, 'stage5_ensemble_panel.py'), (SLOMD, 'confinement_budget.py'), (SLOMD, 'slo_xi0_full.py')]


def run(sh, py, script):
    p = subprocess.run([py, str(sh / 'scripts' / script)], cwd=str(sh), capture_output=True, text=True,
                       env=dict(os.environ, PAULI_ROOT=str(sh), OMP_NUM_THREADS='4'))
    (sh / 'panel_logs').mkdir(exist_ok=True)
    (sh / 'panel_logs' / f'{script}.log').write_text(p.stdout + p.stderr)
    print(f'{script:36s} rc={p.returncode}', flush=True)
    return p.returncode


def collate(sh):
    rg, ef = sh / 'results/reactive_geometry', sh / 'results/ensemble_fluctuation'
    for c in ('r255', 'r340'):
        k = {}
        for t in TAGS:
            f = rg / f'{t}_{c}_native_result.json'
            if f.exists():
                k[t] = json.loads(f.read_text())['k_exch_Nm']
        (rg / f'kexch_{c}.json').write_text(json.dumps(k, indent=1))
        frames = {}
        for t in TAGS:
            vals = []
            for i in range(40):
                f = ef / f'{t}_{c}_f{i:03d}.json'
                if f.exists():
                    v = json.loads(f.read_text()).get('k_exch_Nm')
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        vals.append(v)
            frames[t] = vals
            (ef / f'{t}_{c}_kexch_frames.json').write_text(json.dumps(
                dict(TAG=t, clamp=c, n_ok=len(vals), k_exch_Nm=vals), indent=1))
        (ef / f'kexch_frames_{c}.json').write_text(json.dumps(frames, indent=1))
        print(c, 'static k for', sorted(k), '| frames per variant', {t: len(v) for t, v in frames.items()})


def collate_replace_l754a(sh):
    """Model S with H25 for L754A: keep the published per-variant lists, replace only L754A."""
    rg, ef = sh / 'results/reactive_geometry', sh / 'results/ensemble_fluctuation'
    for c in ('r255', 'r340'):
        k = json.loads((rg / f'kexch_{c}.json').read_text())
        k['L754A'] = json.loads((rg / f'L754A_{c}_native_result.json').read_text())['k_exch_Nm']
        (rg / f'kexch_{c}.json').write_text(json.dumps(k, indent=1))
        fr = json.loads((ef / f'kexch_frames_{c}.json').read_text())
        vals = []
        for i in range(40):
            f = ef / f'L754A_{c}_f{i:03d}.json'
            if f.exists():
                v = json.loads(f.read_text()).get('k_exch_Nm')
                if isinstance(v, (int, float)) and math.isfinite(v):
                    vals.append(v)
        fr['L754A'] = vals
        (ef / f'kexch_frames_{c}.json').write_text(json.dumps(fr, indent=1))
        (ef / f'L754A_{c}_kexch_frames.json').write_text(json.dumps(dict(TAG='L754A', clamp=c, n_ok=len(vals), k_exch_Nm=vals), indent=1))
        print(c, 'L754A static k', round(k['L754A'], 4), '| L754A frames', len(vals))


def main():
    kind = sys.argv[1]
    sh = PS / f'shadow_{kind}'
    if kind == 'SpS':
        collate_replace_l754a(sh)
        for py, script in PANEL + [(PAULI, 'sapt_native_uncertainty.py')]:
            run(sh, py, script)
        return
    if kind == 'PpS':
        collate(sh)
        for py, script in PANEL:
            run(sh, py, script)
        return
    if kind == 'S':
        for c in ('r255', 'r340'):
            for src, dst in ((ROOT / 'results/reactive_geometry' / f'kexch_{c}.json', 'results/reactive_geometry'),
                             (ROOT / 'results/ensemble_fluctuation' / f'kexch_frames_{c}.json', 'results/ensemble_fluctuation')):
                shutil.copy(src, sh / dst / src.name)
        for script, rel in (('analyze_reactive_geometry.py', 'results/reactive_geometry/analysis_result.json'),
                            ('analyze_ensemble_fluctuation.py', 'results/ensemble_fluctuation/analysis_result.json')):
            run(sh, PAULI, script)
            a, b = json.loads((sh / rel).read_text()), json.loads((ROOT / rel).read_text())
            print(f'  {rel}: identical to published = {a == b}')
        return
    collate(sh)
    for py, script in PANEL:
        run(sh, py, script)


if __name__ == '__main__':
    main()
