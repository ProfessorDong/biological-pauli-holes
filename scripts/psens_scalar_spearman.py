#!/usr/bin/env python3
"""Spearman rank correlations of K_sep with scalar summaries of the methane-series K_perp.

Recomputes Table XII (tab:scalarsummary) from the transverse-Hessian files, which until
now had no producer script for four of its five rows. Exact two-sided p values enumerate
all 7! = 5040 orderings; |rho| ties are counted with a 1e-12 tolerance.
  psens_scalar_spearman.py S   reads results/transverse_hessian (published model S)
  psens_scalar_spearman.py P   reads shadow_P/results/transverse_hessian (model P)
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TAGS = ('I538A', 'I552A', 'I553A', 'L546A', 'L754A', 'V750A', 'WT')


def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def exact_p(a, b):
    rho = spearman(a, b)
    rb = ranks(b)
    ra = ranks(a)
    n_ge, n = 0, 0
    for perm in itertools.permutations(range(len(a))):
        r = float(np.corrcoef(ra, rb[list(perm)])[0, 1])
        n += 1
        if abs(r) >= abs(rho) - 1e-12:
            n_ge += 1
    return rho, n_ge / n


def main():
    kind = sys.argv[1]
    th = ROOT / 'results/transverse_hessian' if kind == 'S' else \
        ROOT / f'results/protonation_sensitivity/shadow_{kind}/results/transverse_hessian'
    ksep, scal = [], {k: [] for k in ('trace', 'min', 'max', 'spectral', 'frobenius')}
    for t in TAGS:
        d = json.loads((th / f'{t}_r255_w015_transverse_hessian.json').read_text())
        K = np.array(d['K_perp_Nm'], float)
        ev = np.linalg.eigvalsh(K)
        ksep.append(d['K_sep_Nm'])
        scal['trace'].append(float(ev.sum()))
        scal['min'].append(float(ev.min()))
        scal['max'].append(float(ev.max()))
        scal['spectral'].append(float(np.abs(ev).max()))
        scal['frobenius'].append(float(np.linalg.norm(K)))
    out = {k: dict(zip(('rho', 'p_exact'), exact_p(ksep, v))) for k, v in scal.items()}
    out['_inputs'] = dict(K_sep=dict(zip(TAGS, ksep)), **{k: dict(zip(TAGS, v)) for k, v in scal.items()})
    dest = ROOT / 'results/protonation_sensitivity' / f'scalar_spearman_{kind}.json'
    dest.write_text(json.dumps(out, indent=1))
    for k in ('trace', 'min', 'max', 'spectral', 'frobenius'):
        print(f'{k:10s} rho {out[k]["rho"]:+.3f}  exact p {out[k]["p_exact"]:.3f}')


if __name__ == '__main__':
    main()
