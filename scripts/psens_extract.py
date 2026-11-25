#!/usr/bin/env python3
"""Run the model-S extraction scripts inside a shadow root (protonation sensitivity).

  psens_extract.py S   run every extraction step on model-S data in shadow_S and compare
                       each output with the published model-S file; any difference fails
  psens_extract.py P [TAG ...]   run the same steps on model-P data in shadow_P
Steps per variant: extract_clamped_fragment (r255, r340), extract_frames_geometry (r255,
r340), extract_native_donor_clamped (r255), stage5_extract (16); then, once for the panel,
confinement_budget and slo_xi0_full.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PS = ROOT / 'results/protonation_sensitivity'
SLOMD = '/home/liang/anaconda3/envs/slomd/bin/python'
TAGS = ('WT', 'I538A', 'I552A', 'I553A', 'L546A', 'L754A', 'V750A')


def same(a, b, path='', tol=1e-9):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f'{path}: keys {sorted(set(a) ^ set(b))}']
        return [d for k in a for d in same(a[k], b[k], f'{path}.{k}', tol)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} vs {len(b)}']
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f'{path}[{i}]', tol)]
    if isinstance(a, float) or isinstance(b, float):
        try:
            ok = abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
        except (TypeError, ValueError):
            ok = a == b
        return [] if ok else [f'{path}: {a} vs {b}']
    return [] if a == b else [f'{path}: {a!r} vs {b!r}']


def run(sh, script, *args):
    env = dict(os.environ, PAULI_ROOT=str(sh))
    p = subprocess.run([SLOMD, str(sh / 'scripts' / script), *args], env=env,
                       capture_output=True, text=True, cwd=str(sh))
    return p.returncode, (p.stdout + p.stderr)[-600:]


def steps(tags):
    for t in tags:
        for c in ('r255', 'r340'):
            yield ('extract_clamped_fragment.py', (t, c), f'results/reactive_geometry/{t}_{c}_geometry.json')
            yield ('extract_frames_geometry.py', (t, c), f'results/ensemble_fluctuation/{t}_{c}_frames_geometry.json')
        yield ('extract_native_donor_clamped.py', (t, 'r255'), f'results/native_donor_validation/{t}_r255_native_donor.json')
        yield ('stage5_extract.py', (t, '16'), f'results/stage5_configs/{t}/selection.json')
    yield ('confinement_budget.py', (), 'results/confinement_budget.json')
    yield ('slo_xi0_full.py', (), 'results/pauli_hole_survey/slo_xi0_full.json')


def main():
    kind = sys.argv[1]
    tags = tuple(sys.argv[2:]) or TAGS
    sh = PS / f'shadow_{kind}'
    report = []
    for script, args, rel in steps(tags):
        rc, tail = run(sh, script, *args)
        rec = dict(script=script, args=list(args), output=rel, rc=rc)
        out = sh / rel
        if rc != 0 or not out.exists():
            rec['error'] = tail
        elif kind == 'S':
            diffs = same(json.loads(out.read_text()), json.loads((ROOT / rel).read_text()))
            if script == 'stage5_extract.py':           # the configurations themselves, too
                for f in sorted(out.parent.glob(f'{args[0]}_cfg*_native_donor.json')):
                    diffs += same(json.loads(f.read_text()),
                                  json.loads((ROOT / 'results/stage5_configs' / args[0] / f.name).read_text()),
                                  f.name)
            rec['identical_to_published'] = not diffs
            rec['diffs'] = diffs[:5]
        report.append(rec)
        flag = rec.get('identical_to_published', 'n/a' if 'error' not in rec else 'ERROR')
        print(f'{script:34s} {" ".join(args):12s} rc={rc}  match={flag}', flush=True)
    (sh / f'extract_report_{"_".join(tags) if len(tags) < 7 else "all"}.json').write_text(json.dumps(report, indent=1))
    bad = [r for r in report if r['rc'] != 0 or 'error' in r or r.get('identical_to_published') is False]
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
