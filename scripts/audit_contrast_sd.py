#!/usr/bin/env python3
"""Permutation null of the fluctuation test's declared contrast between clamps.

The fluctuation plan (results/ensemble_fluctuation/PREREGISTRATION.md) declares, as a secondary
endpoint, the contrast between the two clamps of the primary descriptor D2 (the standard deviation of
K_sep over the configurations of a clamped run). Its value, -0.257, has been reported without a p value
or interval. This script gives both with the method that scripts/endpoint_robustness.py applies to the
static contrast:

    contrast = |r_partial(compressed)| - |r_partial(reference)|,

each partial correlation controlling for the achieved <r_DA> of its run; one relabeling of ln KIE is
applied to both clamps at once; p is the fraction of the 5040 relabelings with |null| >= |observed|;
the interval is the 2.5th to 97.5th percentile of the null distribution.

Descriptors, covariates and the partial correlation are taken unchanged from
scripts/analyze_ensemble_fluctuation.py (load, partial_corr; D2 = sample SD with ddof = 1 over the finite
configurations). As checks the script first reproduces
  * the static contrast of results/reactive_geometry/endpoint_robustness.json (value, p, null interval),
  * the stored D2 partial correlations of results/ensemble_fluctuation/analysis_result.json.

Writes results/audit_2026_11/contrast_sd.json.
Usage: audit_contrast_sd.py      (pauli env)
"""
import importlib.util as iu
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('PAULI_ROOT') or Path(__file__).resolve().parents[1])
OUT = ROOT / 'results' / 'audit_2026_11' / 'contrast_sd.json'


def module(name, rel):
    spec = iu.spec_from_file_location(name, ROOT / rel)
    m = iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def contrast_null(x1, z1, x2, z2, y, partial):
    def contrast(yy):
        return abs(partial(x1, yy, z1)) - abs(partial(x2, yy, z2))
    obs = contrast(y)
    null = np.array([contrast(np.array(q)) for q in itertools.permutations(y)])
    null = null[np.isfinite(null)]
    p = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    lo, hi = np.percentile(null, [2.5, 97.5])
    return float(obs), p, [float(lo), float(hi)], int(len(null))


def main():
    st = module('static', 'scripts/analyze_reactive_geometry.py')
    en = module('ens', 'scripts/analyze_ensemble_fluctuation.py')
    rob = module('rob', 'scripts/endpoint_robustness.py')

    # check 1: the static contrast, exactly as endpoint_robustness.py computes it
    kx1, geo1 = st.load_kexch('r255'), st.load_geometry('r255')
    kx2, geo2 = st.load_kexch('r340'), st.load_geometry('r340')
    ok = [t for t in st.SYSTEMS if t in kx1 and t in kx2 and t in geo1 and t in geo2]
    y = np.log(np.array([st.KIE[t] for t in ok], float))
    s_obs, s_p, s_ci, s_n = contrast_null(
        np.array([float(kx1[t]) for t in ok]), np.array([geo1[t] for t in ok]),
        np.array([float(kx2[t]) for t in ok]), np.array([geo2[t] for t in ok]), y, rob.partial)
    stored = json.loads((ROOT / 'results/reactive_geometry/endpoint_robustness.json').read_text())['contrast']
    for a, b in ((s_obs, stored['observed']), (s_p, stored['p_exact']),
                 (s_ci[0], stored['null_ci95'][0]), (s_ci[1], stored['null_ci95'][1])):
        if abs(a - b) > 1e-12:
            raise SystemExit(f'static contrast not reproduced: {a} vs {b}')

    # the fluctuation test: D2 at both clamps
    d2, cov = {}, {}
    for clamp in ('r255', 'r340'):
        kx, geo = en.load(clamp)
        sel = [s for s in en.SYSTEMS if s in kx and s in geo]
        d2[clamp], cov[clamp] = {}, {}
        for s in sel:
            v = np.asarray(kx[s], float)
            v = v[np.isfinite(v)]
            d2[clamp][s] = float(v.std(ddof=1))
            cov[clamp][s] = float(geo[s])
    sysf = [s for s in en.SYSTEMS if s in d2['r255'] and s in d2['r340']]
    yf = np.array([math.log(en.KIE[s]) for s in sysf])
    stored_f = json.loads((ROOT / 'results/ensemble_fluctuation/analysis_result.json').read_text())
    for clamp, key in (('r255', 'reactive'), ('r340', 'reference')):
        r = en.partial_corr(np.array([d2[clamp][s] for s in sysf]), yf,
                            np.array([cov[clamp][s] for s in sysf]))
        if abs(r - stored_f[key]['D2_sd']['partial']) > 1e-12:
            raise SystemExit(f'D2 partial at {clamp} not reproduced: {r}')
    f_obs, f_p, f_ci, f_n = contrast_null(
        np.array([d2['r255'][s] for s in sysf]), np.array([cov['r255'][s] for s in sysf]),
        np.array([d2['r340'][s] for s in sysf]), np.array([cov['r340'][s] for s in sysf]),
        yf, en.partial_corr)

    out = dict(
        definition='|partial r at the compressed clamp| - |partial r at the reference clamp|, one joint '
                   'relabeling of ln KIE; p = fraction with |null| >= |observed|; interval = 2.5-97.5 '
                   'percentiles of the null',
        static_contrast_check=dict(observed=s_obs, p_exact=s_p, null_ci95=s_ci, n_perm=s_n,
                                   reproduces='results/reactive_geometry/endpoint_robustness.json'),
        fluctuation_sd_contrast=dict(systems=sysf, observed=f_obs, p_exact=f_p, null_ci95=f_ci,
                                     n_perm=f_n, D2_N_per_m={k: d2[k] for k in d2},
                                     covariate_rDA_A={k: cov[k] for k in cov}))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    print(f'  static contrast reproduced: {s_obs:+.3f}, p {s_p:.4f}, null [{s_ci[0]:+.3f}, {s_ci[1]:+.3f}]')
    print(f'  fluctuation SD contrast: {f_obs:+.3f}, exact p {f_p:.4f} over {f_n} relabelings, '
          f'null 95% [{f_ci[0]:+.3f}, {f_ci[1]:+.3f}]')
    print(f'  wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
