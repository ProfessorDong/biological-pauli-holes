#!/usr/bin/env python3
"""Achieved donor-acceptor distance of every clamped run, defined exactly as the two
pre-registered analyses define their covariate: the mean of the whole colvar series of the run
(analyze_reactive_geometry.load_geometry for the static runs, analyze_ensemble_fluctuation.load
for the ensemble runs). The two sets are separate runs from the same seed structure with
independent random seeds.

Also gives the Pearson correlation of each set of seven run means with ln KIE (10 C series of
Hu et al. 2019, the KIE table of both analysis scripts).

Writes results/audit_2026_11/achieved_rda.json.
Usage: audit_achieved_rda.py"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('PAULI_ROOT') or Path(__file__).resolve().parents[1])
KIE = {'WT': 66, 'L754A': 106, 'V750A': 62, 'I538A': 100, 'L546A': 131, 'I553A': 148, 'I552A': 66}
SYSTEMS = ['WT', 'V750A', 'I552A', 'I538A', 'L754A', 'L546A', 'I553A']
RUNS = {'static': ROOT / 'results' / 'reactive_geometry',
        'ensemble': ROOT / 'results' / 'ensemble_fluctuation'}


def run_means(directory, clamp):
    return {s: float(np.loadtxt(directory / f'{s}_{clamp}_colvar.dat').mean()) for s in SYSTEMS}


def main():
    y = np.log(np.array([KIE[s] for s in SYSTEMS], float))
    out = dict(definition='mean of the whole colvar series of each clamped run (r_DA in A), as in the '
                          'pre-registered analysis scripts', kie='Hu et al. 2019, 10 C', sets={})
    for run, d in RUNS.items():
        for clamp in ('r255', 'r340'):
            m = run_means(d, clamp)
            x = np.array([m[s] for s in SYSTEMS])
            out['sets'][f'{run}_{clamp}'] = dict(
                means=m, min=float(x.min()), max=float(x.max()), spread=float(x.max() - x.min()),
                pearson_with_lnKIE=float(np.corrcoef(x, y)[0, 1]))
    dest = ROOT / 'results' / 'audit_2026_11' / 'achieved_rda.json'
    dest.write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out['sets'].items():
        print(f"  {k:<15} {v['min']:.3f} to {v['max']:.3f} A   R(ln KIE) = {v['pearson_with_lnKIE']:+.3f}")
    print(f'  wrote {dest.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
