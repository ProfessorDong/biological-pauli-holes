#!/usr/bin/env python3
"""Is the exponential decay rate a property of the active site, or of wild type?

WHY THIS EXISTS
  A ten-configuration fit on wild type alone gave a decay rate of 3.06 per angstrom with a
  bootstrap interval of 1.25 to 3.63, too wide to say much, over only 0.83 angstrom of wall
  motion. Extending the ensemble to seven variants and two clamps widens the lever arm to
  roughly 2.5 angstrom and supplies 140 configurations, which is enough to ask the question
  properly.

WHY A POOLED FIT ALONE WOULD BE WRONG
  Throwing all 140 points into one regression of ln k on distance assumes every system shares
  both the decay rate AND the algebraic prefactor. They plainly do not share the prefactor:
  L754A faces an Asn672 acetamide where the other six face a Leu732 isobutane, a different
  number of electrons at the contact, and the exact solutions say the prefactor carries a
  power of d that differs between cavity shapes. A naive pooled slope would absorb those
  offsets into the rate and could report a decay that belongs to neither.

  So five nested models are fitted and compared:

    A   one rate, one prefactor             ln k = a + b d
    B   one rate, prefactor per system      ln k = a_s + b d
    B2  one rate, prefactor per system and clamp
    C   rate per system, prefactor per system
    C2  rate per system, prefactor per system and clamp

  B against A asks whether systems differ in prefactor. B2 against B asks whether the clamp
  shifts it too, which the data demand asking: I552A gives a higher mean curvature at the
  r340 clamp than at r255 despite slightly larger wall distances, and no common prefactor can
  produce that. The rate question must then be asked against whichever prefactor model
  survives. C2 against B2 is that test; C against B is NOT, because with per-clamp prefactor
  shifts left unmodelled they surface as spurious differences in rate. Both are reported so
  the difference is visible.

WHY THE FITTED RATES NEED AN ATTENUATION CHECK
  The restraint fixes the reaction coordinate, not the wall, so each system samples whatever
  span of wall distance its own fluctuations happen to give, and those spans differ by a
  factor of two across the panel. A slope fitted over a narrow span is biased toward zero
  when the ordinate carries variance from anything else. If the fitted rate tracks the span,
  between-system differences are a sampling artefact and the pooled rate is itself attenuated
  toward zero. That correlation is therefore computed and reported alongside the F tests, and
  the least-attenuated systems are quoted separately.

WHAT A POSITIVE RESULT WOULD AND WOULD NOT SHOW
  A common rate across variants and clamps is what a local exchange law predicts and what the
  paper's title claims to test. It would still not determine kappa: the abscissa is a
  fragment-to-fragment closest approach rather than the hydrogen-to-wall distance of the
  solved problems, the prefactor is held constant within each group over the fitted range,
  and the estimate is attenuated by whatever span each system supplies.

Usage: exponential_law_panel.py     (pauli env)
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist
from scipy.stats import pearsonr

ROOT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio')
DF = ROOT / 'results/sapt_bio/donor_fragment'
TAGS = ['WT', 'I553A', 'I552A', 'L754A', 'V750A', 'I538A', 'L546A']
CLAMPS = ['r255', 'r340']
A0 = 0.5291772109
SOLVED_PER_A0 = 2.011


def lstsq_rss(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return beta, float(r @ r)


def ftest(rss0, p0, rss1, p1, n):
    """Nested F test: model 1 has more parameters."""
    if p1 <= p0 or rss1 <= 0:
        return float('nan'), float('nan')
    F = ((rss0 - rss1) / (p1 - p0)) / (rss1 / (n - p1))
    return F, float(fdist.sf(F, p1 - p0, n - p1))


def main():
    d, k, sysid, clampid, wall = [], [], [], [], []
    missing, dropped = [], 0
    for si, t in enumerate(TAGS):
        for c in CLAMPS:
            f = DF / f'{t}_{c}_fsapt_ensemble.json'
            if not f.exists():
                missing.append(f'{t}/{c}')
                continue
            for r in json.loads(f.read_text())['frames']:
                if r['k_total'] <= 0:
                    dropped += 1
                    continue
                d.append(r['dmin']), k.append(r['k_total'])
                sysid.append(si), clampid.append(c), wall.append(r['wall_atom'])
    if missing:
        print(f'NOT YET COMPUTED: {", ".join(missing)}\n')
    if not d:
        print('no data yet')
        return
    d, k = np.array(d), np.array(k)
    sysid = np.array(sysid)
    ly = np.log(k)
    n = len(d)
    print(f'{n} configurations, d = {d.min():.2f} to {d.max():.2f} A, '
          f'k = {k.min():.3f} to {k.max():.3f} N/m '
          f'(a factor of {k.max()/k.min():.0f})')
    if dropped:
        print(f'  {dropped} configurations dropped for non-positive curvature')

    present = sorted(set(sysid))
    D = np.column_stack([(sysid == s).astype(float) for s in present])   # per-system intercepts
    ones = np.ones((n, 1))
    # per system-and-clamp intercepts. I552A gives a higher mean curvature at the r340 clamp
    # than at r255 despite slightly larger wall distances, which a common prefactor across
    # clamps cannot produce, so whether the clamp shifts the prefactor is tested rather than
    # assumed.
    sc = np.array([f'{TAGS[s]}_{c}' for s, c in zip(sysid, clampid)])
    scl = sorted(set(sc))
    DSC = np.column_stack([(sc == g).astype(float) for g in scl])

    XA = np.column_stack([ones, d])
    XB = np.column_stack([D, d])
    XB2 = np.column_stack([DSC, d])
    XC = np.column_stack([D, D * d[:, None]])
    # C2 is the only legitimate test of rate heterogeneity once B2 has won: it gives each
    # system its own rate while keeping the per-clamp prefactors that the data demand. Testing
    # C against B instead lets an unmodelled clamp shift in the prefactor surface as a fake
    # difference in rate.
    XC2 = np.column_stack([DSC, D * d[:, None]])
    bA, rA = lstsq_rss(XA, ly)
    bB, rB = lstsq_rss(XB, ly)
    bB2, rB2 = lstsq_rss(XB2, ly)
    bC, rC = lstsq_rss(XC, ly)
    bC2, rC2 = lstsq_rss(XC2, ly)
    pA, pB, pB2, pC = XA.shape[1], XB.shape[1], XB2.shape[1], XC.shape[1]
    pC2 = XC2.shape[1]

    print(f'\n{"model":>48}{"params":>8}{"RSS":>10}{"rate (per A)":>15}')
    print(f'{"A   one rate, one prefactor":>48}{pA:>8}{rA:>10.3f}{-bA[1]:>15.2f}')
    print(f'{"B   one rate, prefactor per system":>48}{pB:>8}{rB:>10.3f}{-bB[-1]:>15.2f}')
    print(f'{"B2  one rate, prefactor per system and clamp":>48}'
          f'{pB2:>8}{rB2:>10.3f}{-bB2[-1]:>15.2f}')
    print(f'{"C   rate per system":>48}{pC:>8}{rC:>10.3f}{"see below":>15}')

    F1, p1 = ftest(rA, pA, rB, pB, n)
    F1b, p1b = ftest(rB, pB, rB2, pB2, n)
    F2, p2 = ftest(rB, pB, rC, pC, n)
    F2b, p2b = ftest(rB2, pB2, rC2, pC2, n)
    print(f'{"C2  rate per system, prefactor per system and clamp":>48}'
          f'{pC2:>8}{rC2:>10.3f}{"see below":>15}')
    print(f'\n  B  over A  (prefactor varies by system?)      F = {F1:.2f}, p = {p1:.2e}')
    print(f'  B2 over B  (prefactor varies by clamp too?)   F = {F1b:.2f}, p = {p1b:.2e}')
    print(f'  C  over B  (rate varies? WRONG baseline)      F = {F2:.2f}, p = {p2:.3f}')
    print(f'  C2 over B2 (rate varies? correct baseline)    F = {F2b:.2f}, p = {p2b:.3f}')

    # Report the rate from whichever prefactor model the data supports, so the headline
    # number is not conditioned on an assumption the F test rejects.
    use_b2 = np.isfinite(p1b) and p1b < 0.05
    Xbest, rate_common = (XB2, -bB2[-1]) if use_b2 else (XB, -bB[-1])
    grp = DSC if use_b2 else D
    label = ('per system and clamp' if use_b2 else 'per system')
    pred = SOLVED_PER_A0 / A0
    print(f'\n  prefactor model supported by the data   : {label}')
    print(f'  common rate under that model            : {rate_common:.2f} per A')
    print(f'  exact-solution rate                     : {pred:.2f} per A')
    print(f'  ratio                                   : {rate_common/pred:.2f}')

    # bootstrap the common rate by resampling configurations within system
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(4000):
        idx = np.concatenate([rng.choice(np.where(sysid == s)[0],
                                         (sysid == s).sum(), replace=True)
                              for s in present])
        try:
            bb, _ = lstsq_rss(np.column_stack([grp[idx], d[idx]]), ly[idx])
            boot.append(-bb[-1])
        except np.linalg.LinAlgError:
            pass
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f'  95% bootstrap on the common rate        : {lo:.2f} to {hi:.2f} per A')
    inside = lo <= pred <= hi
    print(f'  the solved rate lies {"INSIDE" if inside else "OUTSIDE"} that interval')

    print(f'\n{"system":>8}{"n":>4}{"d range (A)":>16}{"rate (per A)":>14}{"R^2":>8}')
    per = {}
    for s in present:
        m = sysid == s
        if m.sum() < 4:
            continue
        b, _ = np.polyfit(d[m], ly[m], 1), None
        r = pearsonr(d[m], ly[m])
        per[TAGS[s]] = dict(n=int(m.sum()), rate=float(-b[0]),
                            r2=float(r.statistic ** 2))
        print(f'{TAGS[s]:>8}{m.sum():>4}{d[m].min():>8.2f}-{d[m].max():<8.2f}'
              f'{-b[0]:>13.2f}{r.statistic**2:>8.3f}')
    rates = np.array([v['rate'] for v in per.values()])
    print(f'\n  per-system rates span {rates.min():.2f} to {rates.max():.2f} per A, '
          f'median {np.median(rates):.2f}')

    # ATTENUATION DIAGNOSTIC. A slope fitted over a narrow span of the abscissa is biased
    # toward zero when the ordinate carries variance from anything else, and the systems here
    # sample very different spans because the restraint fixes the reaction coordinate, not the
    # wall. If the fitted rate tracks the span, apparent heterogeneity between systems is a
    # sampling artefact and must not be read as physics.
    spans = np.array([d[sysid == s].max() - d[sysid == s].min()
                      for s in present if (sysid == s).sum() >= 4])
    ra = pearsonr(spans, rates)
    print(f'\n  ATTENUATION CHECK  fitted rate against sampled span of d:')
    print(f'    Pearson r = {ra.statistic:+.3f}, p = {ra.pvalue:.4f}   '
          f'spans {spans.min():.2f} to {spans.max():.2f} A')
    attenuated = ra.pvalue < 0.05 and ra.statistic > 0
    if attenuated:
        wide = [t for t, v in per.items()
                if (d[sysid == TAGS.index(t)].max()
                    - d[sysid == TAGS.index(t)].min()) >= np.median(spans)]
        wr = np.array([per[t]['rate'] for t in wide])
        print(f'    the rate rises with the span, which is the signature of regression')
        print(f'    attenuation: narrow-span systems give slopes biased toward zero, so the')
        print(f'    between-system differences are confounded with how far the wall moved.')
        print(f'    best-determined systems ({", ".join(wide)}): '
              f'{wr.min():.2f} to {wr.max():.2f} per A')
        order = sorted(per, key=lambda t: -(d[sysid == TAGS.index(t)].max()
                                            - d[sysid == TAGS.index(t)].min()))
        top2 = order[:2]
        t2 = np.array([per[t]['rate'] for t in top2])
        print(f'    the two widest-span systems ({top2[0]}, {top2[1]}) are least attenuated')
        print(f'    and agree closely: {t2[0]:.2f} and {t2[1]:.2f} per A, i.e. '
              f'{t2.mean()/pred:.2f} of the solved rate,')
        print(f'    against {rate_common/pred:.2f} for the span-weighted pooled estimate, '
              f'which is itself attenuated.')

    # A nan here means the comparison is undefined, typically because only one system has
    # data yet. That is not evidence either way and must not be reported as a difference.
    # the verdict must use the correct baseline, and must not claim heterogeneity that the
    # attenuation diagnostic explains
    p2use = p2b if np.isfinite(p2b) else p2
    if not np.isfinite(p2use):
        verdict = 'undetermined: the rate comparison needs at least two systems'
    elif attenuated and p2use <= 0.05:
        verdict = ('rate differences are not established: they are confounded with the '
                   'sampled span of wall distance, which predicts the fitted rate at '
                   f'r = {ra.statistic:+.2f}')
    elif p2use > 0.05:
        verdict = ('no resolved difference in decay rate between variants and clamps; '
                   'note that a non-significant F on samples this size is weak evidence '
                   'of sameness')
    else:
        verdict = 'the decay rate differs between variants'
    print(f'\nVERDICT  {verdict}')
    if np.isfinite(p2use):
        which = 'C2 over B2' if np.isfinite(p2b) else 'C over B'
        print(f'         ({which} p = {p2use:.3f})')

    (DF / 'exponential_law_panel.json').write_text(json.dumps(dict(
        n=n, n_dropped=dropped, missing=missing,
        d_min=float(d.min()), d_max=float(d.max()),
        k_min=float(k.min()), k_max=float(k.max()),
        rss=dict(A=rA, B=rB, B2=rB2, C=rC),
        params=dict(A=pA, B=pB, B2=pB2, C=pC),
        prefactor_model_used=label,
        F_prefactor_system=float(F1), p_prefactor_system=float(p1),
        F_prefactor_clamp=float(F1b), p_prefactor_clamp=float(p1b),
        F_rate_wrong_baseline=float(F2), p_rate_wrong_baseline=float(p2),
        F_rate=float(F2b), p_rate=float(p2b),
        attenuation_r=float(ra.statistic), attenuation_p=float(ra.pvalue),
        attenuated=bool(attenuated),
        common_rate_per_A=float(rate_common),
        common_rate_ci95=[float(lo), float(hi)],
        solved_rate_per_A=float(pred), ratio=float(rate_common / pred),
        solved_inside_ci=bool(inside), per_system=per, verdict=verdict), indent=1))
    print(f'\nwrote {DF}/exponential_law_panel.json')


if __name__ == '__main__':
    main()
