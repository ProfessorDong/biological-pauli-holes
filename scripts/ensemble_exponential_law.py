#!/usr/bin/env python3
"""Does the exchange curvature decay exponentially with wall distance in a real ensemble?

WHY THIS EXISTS
  The paper's central structural claim is that direct exchange confinement is local, dying
  as exp(-2 kappa d), and it establishes that rate on two exactly solvable idealizations: a
  hydrogen atom in a paraboloidal cavity and in a hard sphere, which give 2.011 and 2.002 per
  bohr. Those are model cavities containing one electron. Whether the same law describes a
  many-electron molecular fragment against a real protein side chain, sampled at thermal
  equilibrium, has not been tested anywhere in the manuscript.

  The ensemble F-SAPT calculation supplies the test for free. Ten configurations drawn from a
  clamped trajectory hold the donor-acceptor distance fixed near 2.55 A while the wall
  breathes, so the nearest donor-to-wall distance varies over roughly 0.8 A at constant
  reaction coordinate. If the exponential picture is right, the curvature should fall
  log-linearly with that distance, at a rate near the solved one.

WHAT WOULD FALSIFY IT
  A weak or absent correlation, or a fitted rate differing from the solved value by more than
  a factor of about two, would show that the idealized cavities do not describe the biological
  contact and would undercut the locality argument the paper builds on them.

CAVEATS THAT LIMIT THE STRENGTH OF A POSITIVE RESULT
  One system, one clamp, ten configurations. The abscissa is the closest approach between the
  donor fragment and the wall fragment, not the hydrogen-to-wall distance the exact solutions
  use, and the fit assumes the algebraic prefactor is constant over this narrow range when the
  solutions say it carries a power of d. Agreement at the tens-of-percent level should
  therefore be read as consistency, not as a determination of kappa.

Usage: ensemble_exponential_law.py [TAG [clamp]]     (pauli env)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
DF = ROOT / 'results/sapt_bio/donor_fragment'
A0 = 0.5291772109          # angstrom per bohr
SOLVED_PER_A0 = 2.011      # paraboloid fitted rate, from confinement_universality


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'WT'
    clamp = sys.argv[2] if len(sys.argv) > 2 else 'r255'
    d = json.loads((DF / f'{tag}_{clamp}_fsapt_ensemble.json').read_text())
    fr = d['frames']
    x = np.array([r['dmin'] for r in fr])
    y = np.array([r['k_total'] for r in fr])
    assert (y > 0).all(), 'a non-positive curvature cannot be log-fitted'
    ly = np.log(y)

    slope, intercept = np.polyfit(x, ly, 1)
    pr = pearsonr(x, ly)
    rate = -float(slope)
    pred = SOLVED_PER_A0 / A0
    # bootstrap the rate over frames, since n is small
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(10000):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) < 3:
            continue
        boot.append(-np.polyfit(x[i], ly[i], 1)[0])
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f'{tag} {clamp}: ln k_exch against nearest donor-to-wall distance, '
          f'{len(fr)} configurations\n')
    print(f'  fitted decay rate   : {rate:.2f} per angstrom '
          f'(95% bootstrap {lo:.2f} to {hi:.2f})')
    print(f'  exact-solution rate : {pred:.2f} per angstrom '
          f'({SOLVED_PER_A0} per bohr, paraboloid)')
    print(f'  ratio observed/solved: {rate/pred:.2f}')
    print(f'  correlation         : r = {pr.statistic:+.4f}, R^2 = {pr.statistic**2:.3f}, '
          f'p = {pr.pvalue:.2e}')
    print(f'  curvature spans {y.min():.2f} to {y.max():.2f} N/m, a factor '
          f'{y.max()/y.min():.0f}, over {x.max()-x.min():.2f} angstrom of wall motion')
    consistent = lo <= pred <= hi
    print(f'\n  the solved rate lies {"INSIDE" if consistent else "outside"} the bootstrap '
          f'interval of the fitted rate')
    print('  Read as consistency of the exponential picture with a biological contact, '
          'not as a\n  determination of kappa: one system, one clamp, ten configurations, '
          'and a prefactor\n  held constant that the exact solutions say carries a power '
          'of d.')

    (DF / f'{tag}_{clamp}_exponential_law.json').write_text(json.dumps(dict(
        tag=tag, clamp=clamp, n=len(fr), d_angstrom=x.tolist(), k_Nm=y.tolist(),
        fitted_rate_per_A=rate, bootstrap_ci95=[float(lo), float(hi)],
        solved_rate_per_A=pred, solved_rate_per_a0=SOLVED_PER_A0,
        ratio=float(rate / pred), pearson_r=float(pr.statistic),
        r_squared=float(pr.statistic ** 2), p_value=float(pr.pvalue),
        solved_rate_inside_ci=bool(consistent)), indent=1))
    print(f'\nwrote {DF}/{tag}_{clamp}_exponential_law.json')


if __name__ == '__main__':
    main()
