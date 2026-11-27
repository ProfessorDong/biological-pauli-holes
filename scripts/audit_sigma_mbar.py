#!/usr/bin/env python3
"""MBAR-reweighted spread of r_DA in the near-attack range, against the KIEs.

Audit producer, 2026-11-25. Reads the primary umbrella data under results/umbrella/ and writes
results/audit_2026_11/sigma_mbar.json only.

Definition. sigma_NA is the standard deviation of r_DA under the unbiased (u = 0) MBAR weights,
restricted to the near-attack range 2.8 <= r_DA <= 3.6 A and renormalized there. It is the
reweighted counterpart of sigma_rDA^end of the endpoint estimator, which uses the same range.
sigma_full is the same standard deviation without the restriction. Weights follow
scripts/mbar_2d_volume.py: log w_n = -log sum_k N_k exp(f_k - u_kn), with f_k from
pymbar.MBAR (initialize='BAR', solver_protocol='robust', states sorted by window center) and
u_kn = (k/2)(r - r_k)^2 / kT.

Sets: 'pooled' combines all 45 windows of the three replicas in one MBAR with every one of the
5000 samples per window; 'pooled_drop20' drops the first 20% of each window; 'rep1' uses
replica 1 alone, all samples. For each set and measure the Pearson R over the seven-system panel
against ln KIE(10 C) and ln[KIE(10 C)/KIE(40 C)], its exhaustive two-sided permutation p over
7! = 5040 relabelings, the Spearman rho, and the leave-one-system-out Q^2 = 1 - PRESS/SS_tot of
the linear fit are reported. KIEs: Hu et al. (2019), Table 1 (Table S3 of the SM).
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import glob, itertools, json, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
from pymbar import MBAR
from scipy.stats import spearmanr

ROOT = Path(_REPO) / 'results' / 'umbrella'
OUT = Path(_REPO) / 'results' / 'audit_2026_11' / 'sigma_mbar.json'
kT = 0.5961
K_BIAS = 12.0
R_LO, R_HI = 2.8, 3.6
SYSTEMS = ['WT', 'V750A', 'I552A', 'I538A', 'L754A', 'L546A', 'I553A']
KIE10 = {'WT': 66, 'V750A': 62, 'I552A': 66, 'I538A': 100, 'L754A': 106, 'L546A': 131, 'I553A': 148}
KIE40 = {'WT': 52, 'V750A': 57, 'I552A': 60, 'I538A': 65, 'L754A': 82, 'L546A': 85, 'I553A': 77}


def weights(dirs, drop):
    c, xs = [], []
    for d in dirs:
        for f in sorted(glob.glob(f'{d}/win_*_colvar.dat'),
                        key=lambda f: float(f.split('win_')[1].split('_')[0])):
            x = np.loadtxt(f)
            if drop:
                x = x[len(x) // 5:]
            c.append(float(f.split('win_')[1].split('_')[0])); xs.append(x)
    order = np.argsort(c, kind='stable')          # neighboring states adjacent, for the BAR start
    c = np.array(c)[order]; xs = [xs[i] for i in order]
    Nk = np.array([len(x) for x in xs]); xn = np.concatenate(xs)
    u = np.array([0.5 * K_BIAS * (xn - ck) ** 2 / kT for ck in c])
    m = MBAR(u, Nk, initialize='BAR', solver_protocol='robust')
    a = m.f_k[:, None] - u
    amax = a.max(0)
    lw = -(amax + np.log(np.sum(Nk[:, None] * np.exp(a - amax), 0)))
    lw -= lw.max(); w = np.exp(lw); w /= w.sum()
    return xn, w, dict(n_windows=int(len(c)), n_samples=int(len(xn)),
                       samples_per_window=sorted(set(int(n) for n in Nk)))


def spread(x, w):
    mu = float(np.sum(w * x))
    return float(np.sqrt(np.sum(w * (x - mu) ** 2))), mu


def stats(xv):
    perms = np.array(list(itertools.permutations(range(7))))
    out = {}
    for lab, y in (('lnKIE10', np.log([KIE10[s] for s in SYSTEMS])),
                   ('lnKIE10_over_KIE40', np.log([KIE10[s] / KIE40[s] for s in SYSTEMS]))):
        x = np.asarray(xv, float)
        r = float(np.corrcoef(x, y)[0, 1])
        rp = np.array([np.corrcoef(x, y[p])[0, 1] for p in perms])
        press = 0.0
        for i in range(7):
            k = np.arange(7) != i
            b, a0 = np.polyfit(x[k], y[k], 1)
            press += (y[i] - (a0 + b * x[i])) ** 2
        out[lab] = dict(pearson_R=r, perm_p_two_sided=float(np.mean(np.abs(rp) >= abs(r) - 1e-12)),
                        spearman_rho=float(spearmanr(x, y)[0]),
                        loo_Q2=float(1 - press / np.sum((y - y.mean()) ** 2)))
    return out


def main():
    sets = {'pooled': (lambda s: [ROOT / s, ROOT / f'{s}_rep2', ROOT / f'{s}_rep3'], False),
            'pooled_drop20': (lambda s: [ROOT / s, ROOT / f'{s}_rep2', ROOT / f'{s}_rep3'], True),
            'rep1': (lambda s: [ROOT / s], False)}
    res = dict(definition=('sigma_NA: SD of r_DA under unbiased MBAR weights restricted to '
                           f'{R_LO} <= r_DA <= {R_HI} A and renormalized; sigma_full: the same without '
                           'the restriction'),
               kie_source='Hu et al. 2019, Table 1 (Table S3 of the SM)', systems=SYSTEMS,
               per_system={}, correlations={})
    for name, (fn, drop) in sets.items():
        D = {}
        for s in SYSTEMS:
            x, w, n = weights(fn(s), drop)
            m = (x >= R_LO) & (x <= R_HI)
            wn = w[m] / w[m].sum()
            sNA, muNA = spread(x[m], wn)
            sF, muF = spread(x, w)
            D[s] = dict(sigma_NA_A=sNA, mean_NA_A=muNA, sigma_full_A=sF, mean_full_A=muF,
                        weight_in_NA_range=float(w[m].sum()), n_samples_in_NA_range=int(m.sum()),
                        kish_ess_in_NA_range=float(1.0 / np.sum(wn ** 2)), **n)
            print(f'  {name:13s} {s:6s}: sigma_NA={sNA:.4f}  sigma_full={sF:.4f}  '
                  f'w(NA)={w[m].sum():.3e}  n(NA)={int(m.sum())}', flush=True)
        res['per_system'][name] = D
        res['correlations'][name] = {k: stats([D[s][k] for s in SYSTEMS]) for k in ('sigma_NA_A', 'sigma_full_A')}
        for k, v in res['correlations'][name].items():
            a = v['lnKIE10']
            print(f'{name:13s} {k:13s}: R={a["pearson_R"]:+.3f} p={a["perm_p_two_sided"]:.3f} '
                  f'rho={a["spearman_rho"]:+.3f} Q2={a["loo_Q2"]:+.2f}', flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
