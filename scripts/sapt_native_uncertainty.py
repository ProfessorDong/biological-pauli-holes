#!/usr/bin/env python3
"""Parametric-bootstrap fit uncertainty for the NATIVE-FRAGMENT SAPT curvatures.

WHY THIS EXISTS
  results/sapt_bio/sapt_uncertainty.json holds bootstrap standard errors for the
  METHANE-SURROGATE series (WT = 15.577 N/m, I553A = 23.419 N/m). That is not the
  series the manuscript quotes. Every k_exch^bio in the text, in Table
  \\ref{tab:kexch} and in the Extended Data scan figure comes from the
  native-fragment campaign (WT = 7.885 N/m, I553A = 13.961 N/m), and no
  uncertainty had ever been computed for it. The figure caption nonetheless
  promised "parametric-bootstrap standard error" and "95% confidence intervals
  firmly excluding zero", which could not be checked against anything on disk.
  This computes them, for the series actually plotted.

THE METHOD
  Identical to bootstrap_curvature() of the superseded sapt_scans_ED_figure.py,
  so the two series are directly comparable: refit the quadratic, take the
  residual scale as std(resid, ddof=3) on the five points, resample Gaussian
  residuals about the fitted curve 5000 times, and take the spread of
  k = 2 c2 * CONV. Same seed.

WHAT THE NUMBER MEANS, AND WHAT IT DOES NOT
  This is the precision of a five-point quadratic fit to deterministic
  single-point SAPT0 energies. It is NOT a physical error bar: it contains no
  basis-set incompleteness, no SAPT0 truncation, no snapshot-to-snapshot
  variation, and no fragment-choice error, and the last of those is known to be
  large (methane surrogates overestimate by 1.5-2x).

  It is also nearly uninformative on its own. The residual scale rises with the
  signal, so sigma_k/k lands between 7.2 and 8.9 per cent for all seven systems.
  A 95% interval therefore excludes zero for every system almost by
  construction, INCLUDING L754A at 0.004 N/m, whose interval [0.0033, 0.0047]
  excludes zero while the curvature itself is three to four orders of magnitude
  below every other system. Reporting "the interval excludes zero" as though it
  were a significance test would be misleading, and the caption says so.

Usage: sapt_native_uncertainty.py     (pauli env)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
SRC = ROOT / 'results/sapt_bio/native_fragment'
OUT = ROOT / 'results/sapt_bio/native_fragment_uncertainty.json'

CONV = 0.694770                 # kcal/mol/A^2 -> N/m
DELTAS = ['-0.2', '-0.1', '0.0', '0.1', '0.2']
N_BOOT = 5000
SEED = 20260803                 # same seed as the methane-series bootstrap
ORDER = ['L754A', 'I552A', 'I538A', 'V750A', 'WT', 'I553A', 'L546A']


def bootstrap_curvature(x, y, rng):
    """Parametric residual bootstrap on the 5-point scan; returns k samples in N/m."""
    A = np.column_stack([np.ones_like(x), x, x ** 2])
    p, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ p
    sigma = float(np.std(resid, ddof=3))
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        p_star, *_ = np.linalg.lstsq(A, A @ p + rng.normal(0.0, sigma, x.size),
                                     rcond=None)
        out[i] = 2.0 * p_star[2] * CONV
    return out


def main():
    rng = np.random.default_rng(SEED)
    res = {}
    print(f'{"system":8s}{"k (N/m)":>10s}{"sigma_k":>10s}{"sigma/k":>9s}'
          f'{"ci95_lo":>10s}{"ci95_hi":>10s}   excludes 0')
    for tag in ORDER:
        d = json.loads((SRC / f'{tag}_native_result.json').read_text())
        x = np.array([float(k) for k in DELTAS])
        y = np.array([d['deltas'][k]['exch'] for k in DELTAS])
        boot = bootstrap_curvature(x, y, rng)
        k = float(d['k_exch_Nm'])
        lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
        res[tag] = dict(k_exch_Nm=k, sigma_k_Nm=float(boot.std()),
                        rel_sigma=float(boot.std() / k), ci95=[lo, hi],
                        excludes_zero=bool(lo > 0),
                        RMS_residual_kcal=d['RMS_residual_kcal'],
                        r_HW_A=d['r_HW_A'], wall_residue=d['wall_residue'])
        print(f'{tag:8s}{k:10.4f}{boot.std():10.4f}{100*boot.std()/k:8.1f}%'
              f'{lo:10.4f}{hi:10.4f}   {"yes" if lo > 0 else "no"}')

    rel = [v['rel_sigma'] for v in res.values()]
    OUT.write_text(json.dumps(dict(
        series='native_fragment', n_boot=N_BOOT, seed=SEED,
        method='parametric residual bootstrap, quadratic on 5 points, ddof=3',
        caveat=('fit precision only; excludes basis-set, SAPT0-truncation, '
                'snapshot and fragment-choice error. sigma_k is close to '
                'proportional to k, so exclusion of zero is near-automatic and '
                'is not a significance test.'),
        rel_sigma_range=[min(rel), max(rel)], systems=res), indent=1))
    print(f'\nsigma_k/k spans {100*min(rel):.1f}% to {100*max(rel):.1f}% across the '
          f'seven systems: the fit precision tracks the signal, so it carries almost\n'
          f'no information beyond "the quadratic fits well".')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
