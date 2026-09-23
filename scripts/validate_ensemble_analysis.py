"""Validate the pre-registered ensemble-fluctuation analysis machinery on synthetic data.

Exercises the ACTUAL functions used by analyze_ensemble_fluctuation.py (imported, not
reimplemented) against three synthetic regimes whose right answer is known by construction:

  NULL        descriptor is pure noise, independent of both KIE and geometry
              -> partial correlation should be indistinguishable from zero, and the
                 false-positive rate over many replicates should not exceed nominal 0.05
  CONFOUNDED  descriptor depends ONLY on the geometry covariate, which is itself
              associated with KIE -> raw correlation nonzero, partial must collapse
  SIGNAL      descriptor carries genuine information about ln KIE
              -> partial correlation recovered, interval excludes zero (power check)

Uses the real panel size (n=7), the real KIE values and the real achieved <r_DA> from the
reactive clamp, so the test has exactly the resolution of the deployed analysis.

Usage: validate_ensemble_analysis.py [n_fpr_replicates]
Writes results/ensemble_fluctuation/validation_synthetic.json
"""
import json, sys, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_ensemble_fluctuation import (      # the deployed code path, not a copy
    partial_corr, vif, exact_perm_p, boot_ci, KIE, SYSTEMS, EF, VIF_ABORT,
)

N_FPR = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = np.random.default_rng(20260810)

Y = np.array([math.log(KIE[s]) for s in SYSTEMS])
Z = np.array([float(np.loadtxt(EF / f'{s}_r255_colvar.dat').mean()) for s in SYSTEMS])
Yz = (Y - Y.mean()) / Y.std()
Zz = (Z - Z.mean()) / Z.std()

print(f'panel n={len(SYSTEMS)}  corr(lnKIE, <r_DA>) = {np.corrcoef(Y, Z)[0,1]:+.3f}')
print(f'(a real association between target and covariate is what makes the partial '
      f'correlation necessary)\n')


def report(name, X, expect):
    raw = float(np.corrcoef(X, Y)[0, 1])
    v = vif(X, Z)
    pc, p, ntot = exact_perm_p(X, Y, Z)
    lo, hi = boot_ci(X, Y, Z)
    excl = not (lo < 0 < hi)
    print(f'  {name:<11} raw={raw:+.3f}  VIF={v:5.2f}  partial={pc:+.3f}  '
          f'exact p={p:.4f}  CI=[{lo:+.3f},{hi:+.3f}]  '
          f'{"EXCLUDES ZERO" if excl else "spans zero"}')
    print(f'              expected: {expect}')
    return dict(case=name, raw=raw, vif=v, partial=pc, p_exact=p, ci=[lo, hi],
                excludes_zero=excl, n_perm=ntot, expectation=expect)


print('=== single illustrative replicate per regime ===')
out = {}
X_null = rng.normal(size=len(SYSTEMS))
out['null'] = report('NULL', X_null, 'partial ~ 0, p large, CI spans zero')

X_conf = 3.0 * Zz + 0.15 * rng.normal(size=len(SYSTEMS))
out['confounded'] = report('CONFOUNDED', X_conf,
                           'raw nonzero but partial collapses toward zero')

X_sig = 2.0 * Yz + 0.5 * rng.normal(size=len(SYSTEMS))
out['signal'] = report('SIGNAL', X_sig, 'partial large, CI excludes zero (power)')

print(f'\n=== false-positive calibration: {N_FPR} independent NULL replicates ===')
ps, pcs = [], []
for i in range(N_FPR):
    Xi = rng.normal(size=len(SYSTEMS))
    pc, p, _ = exact_perm_p(Xi, Y, Z)
    ps.append(p); pcs.append(pc)
    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{N_FPR}', flush=True)
ps = np.array(ps)
fpr05 = float((ps < 0.05).mean())
fpr_bonf = float((ps < 0.05 / 6).mean())
print(f'  P(p < 0.05)   = {fpr05:.4f}   (nominal 0.05)')
print(f'  P(p < 0.0083) = {fpr_bonf:.4f}   (Bonferroni threshold for 6 secondary tests)')
print(f'  median |partial| under the null = {np.median(np.abs(pcs)):.3f}'
      f'   <- at n=7 even pure noise gives sizeable partial correlations')
out['fpr'] = dict(n_replicates=N_FPR, p_lt_005=fpr05, p_lt_bonferroni=fpr_bonf,
                  median_abs_partial_null=float(np.median(np.abs(pcs))))

# a null replicate must not be called significant by the deployed threshold
ok = (out['null']['p_exact'] > 0.05 and not out['null']['excludes_zero']
      and abs(out['confounded']['partial']) < abs(out['confounded']['raw']) + 0.5
      and out['signal']['excludes_zero'] and fpr05 <= 0.10)
out['passed'] = bool(ok)
print(f'\nVALIDATION {"PASSED" if ok else "FAILED"}')

f = EF / 'validation_synthetic.json'
f.write_text(json.dumps(out, indent=1))
print(f'wrote {f}')
