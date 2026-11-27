#!/usr/bin/env python3
"""Gating descriptors of the umbrella-sampling PMFs, per replica and pooled, against the KIEs.

Audit producer, 2026-11-25. Reads the primary umbrella data under results/umbrella/ and writes
results/audit_2026_11/gating_pmf.json only.

Estimator. The descriptor logic is that of scripts/umbrella_pmf.py: the first 20% of each
window's 1-ps series is dropped, an MBAR histogram FES is built on 45 equal bins over the sampled
range and evaluated on bins holding more than 10 samples, and
  r_DA^0       = bin center of the PMF minimum,
  k_gate       = 2a of a quadratic a r^2 + b r + c fitted to W within 0.5 A of the minimum,
  rms          = (kT/k_gate)^(1/2), the harmonic estimate of the fluctuation amplitude,
  dW31         = W at the bin nearest 3.1 A minus the minimum.
One thing differs, and it is a correction. umbrella_pmf.py through release v1.2.0 passed the two-dimensional bias array
u_kn to FES.generate_fes as the target-state potential u_n. pymbar 4 then flattens it with
kn_to_n, which reads row k, columns 0..N_k-1, as if the array were in the old K x N_max layout. The
target state is therefore not the unbiased one. The unbiased target is u_n = 0, because the bias is
the only difference between the sampled states (the common unbiased potential cancels). This
script uses u_n = 0 throughout, and separately reproduces the stored pmf.dat files with the
original call, to document that they were made with it.

Sets (current trajectories unless stated):
  rep1, rep2, rep3   one replica directory at a time, 15 windows each
  pooled             all 45 windows of the three replicas in one MBAR
  pooled_first_half, pooled_second_half
                     the same with the first or second half of each window's retained series
  stored_original    the pmf.dat files written by umbrella_pmf.py (replica 1; for WT, L754A and
                     I553A they are in _RTX4060_rep1_archive_2026-08-10, the superseded runs)

For every set and descriptor the Pearson R over the seven-system panel against ln KIE(10 C) and
ln[KIE(10 C)/KIE(40 C)] is reported with its exhaustive two-sided permutation p over all
7! = 5040 relabelings, and the Spearman rho. KIEs are ^D k_cat of Hu et al. (2019), Table 1, as
transcribed in Table S3 of the Supplementary Material.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import glob, itertools, json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
from pymbar import FES
from scipy.stats import spearmanr

ROOT = Path(_REPO) / 'results' / 'umbrella'
ARCH = ROOT / '_RTX4060_rep1_archive_2026-08-10'
OUT = Path(_REPO) / 'results' / 'audit_2026_11' / 'gating_pmf.json'
kT = 0.5961                      # kcal/mol at 300 K, as umbrella_pmf.py
K_BIAS = 12.0                    # kcal/mol/A^2, as umbrella_driver.py
SYSTEMS = ['WT', 'V750A', 'I552A', 'I538A', 'L754A', 'L546A', 'I553A']
KIE10 = {'WT': 66, 'V750A': 62, 'I552A': 66, 'I538A': 100, 'L754A': 106, 'L546A': 131, 'I553A': 148}
KIE40 = {'WT': 52, 'V750A': 57, 'I552A': 60, 'I538A': 65, 'L754A': 82, 'L546A': 85, 'I553A': 77}
STORED_DIR = {s: (ARCH / s if s in ('WT', 'L754A', 'I553A') else ROOT / s) for s in SYSTEMS}


def load(dirs, half=None):
    centers, samples = [], []
    for d in dirs:
        files = sorted(glob.glob(f'{d}/win_*_colvar.dat'),
                       key=lambda f: float(f.split('win_')[1].split('_')[0]))
        for f in files:
            x = np.loadtxt(f)
            x = x[len(x) // 5:]
            if half == 'first':
                x = x[:len(x) // 2]
            elif half == 'second':
                x = x[len(x) // 2:]
            centers.append(float(f.split('win_')[1].split('_')[0]))
            samples.append(x)
    return np.array(centers), samples


def descriptors(ctr, W):
    i0 = int(np.argmin(W)); r0 = float(ctr[i0])
    sel = np.abs(ctr - r0) < 0.5
    a = np.polyfit(ctr[sel], W[sel], 2)
    kg = float(2 * a[0])
    rms = float(np.sqrt(kT / kg)) if kg > 0 else None
    i31 = int(np.argmin(np.abs(ctr - 3.1)))
    return dict(r_DA0_A=r0, k_gate_kcal_per_A2=kg, rms_harmonic_A=rms,
                dW_3p1_kcal=float(W[i31] - W.min()), n_bins=int(len(ctr)),
                r_range_A=[float(ctr.min()), float(ctr.max())])


def pmf(dirs, half=None, targets=('unbiased',)):
    centers, samples = load(dirs, half)
    N_k = np.array([len(x) for x in samples]); x_n = np.concatenate(samples)
    u_kn = np.array([0.5 * K_BIAS * (x_n - c) ** 2 / kT for c in centers])
    counts, edges = np.histogram(x_n, bins=45)
    allc = 0.5 * (edges[:-1] + edges[1:]); ctr = allc[counts > 10]
    fes = FES(u_kn, N_k, verbose=False)
    out = {}
    for t in targets:
        target = np.zeros(len(x_n)) if t == 'unbiased' else u_kn      # 'original' = umbrella_pmf.py call
        fes.generate_fes(target, x_n, fes_type='histogram', histogram_parameters={'bin_edges': edges})
        res = fes.get_fes(ctr, reference_point='from-lowest', uncertainty_method='analytical')
        W = res['f_i'] * kT; good = np.isfinite(W)
        c, Wg = ctr[good], W[good] - W[good].min()
        out[t] = dict(ctr=c, W=Wg, desc=descriptors(c, Wg))
    return out, dict(n_windows=int(len(centers)), n_samples=int(len(x_n)))


def corr_table(D):
    y1 = np.log([KIE10[s] for s in SYSTEMS])
    y2 = np.log([KIE10[s] / KIE40[s] for s in SYSTEMS])
    perms = np.array(list(itertools.permutations(range(7))))
    out = {}
    for key in ('r_DA0_A', 'k_gate_kcal_per_A2', 'rms_harmonic_A', 'dW_3p1_kcal'):
        vals = [D[s][key] for s in SYSTEMS]
        if any(v is None for v in vals):
            out[key] = None; continue
        x = np.array(vals, float)
        row = {}
        for lab, y in (('lnKIE10', y1), ('lnKIE10_over_KIE40', y2)):
            if np.std(x) == 0:
                row[lab] = None; continue
            r = float(np.corrcoef(x, y)[0, 1])
            rp = np.array([np.corrcoef(x, y[p])[0, 1] for p in perms])
            p = float(np.mean(np.abs(rp) >= abs(r) - 1e-12))
            rho = float(spearmanr(x, y)[0])                              # average ranks for ties
            row[lab] = dict(pearson_R=r, perm_p_two_sided=p, spearman_rho=rho)
        out[key] = row
    return out


def main():
    result = dict(
        method=('umbrella_pmf.py descriptor logic with the unbiased target u_n = 0; first 20% of '
                'each window dropped; 45-bin MBAR histogram FES; bins with > 10 samples; quadratic '
                'within 0.5 A of the minimum; kT = 0.5961 kcal/mol; k_bias = 12 kcal/mol/A^2'),
        kie_source='Hu et al. 2019, Table 1 (Table S3 of the SM); ^D k_cat at 10 and 40 C',
        systems=SYSTEMS, sets={}, correlations={}, stored_reproduction={})
    # 1. stored pmf.dat files, and their reproduction with the original call
    stored = {}
    for s in SYSTEMS:
        d = STORED_DIR[s]
        st = np.loadtxt(d / 'pmf.dat')
        stored[s] = descriptors(st[:, 0], st[:, 1] - st[:, 1].min())
        res, n = pmf([d], targets=('original', 'unbiased'))
        o = res['original']
        dev = float(np.max(np.abs(np.interp(st[:, 0], o['ctr'], o['W']) - st[:, 1])))
        result['stored_reproduction'][s] = dict(
            directory=str(d.relative_to(Path(_REPO))), max_abs_dW_original_call_vs_stored_kcal=dev,
            original_call=o['desc'], unbiased_target=res['unbiased']['desc'], **n)
        print(f'  stored {s:6s}: original call reproduces pmf.dat to {dev:.1e} kcal/mol; '
              f'r0 {o["desc"]["r_DA0_A"]:.2f} (original) vs {res["unbiased"]["desc"]["r_DA0_A"]:.2f} (unbiased)',
              flush=True)
    result['sets']['stored_original'] = stored
    result['correlations']['stored_original'] = corr_table(stored)
    result['correlations']['stored_trajectories_unbiased'] = corr_table(
        {s: result['stored_reproduction'][s]['unbiased_target'] for s in SYSTEMS})
    # 2. current trajectories, unbiased target
    dirsets = {'rep1': lambda s: [ROOT / s], 'rep2': lambda s: [ROOT / f'{s}_rep2'],
               'rep3': lambda s: [ROOT / f'{s}_rep3'],
               'pooled': lambda s: [ROOT / s, ROOT / f'{s}_rep2', ROOT / f'{s}_rep3']}
    for name, fn in dirsets.items():
        D = {}
        for s in SYSTEMS:
            res, n = pmf(fn(s))
            D[s] = dict(res['unbiased']['desc'], **n)
            if name == 'pooled':
                D[s]['pmf_kcal'] = dict(r_A=res['unbiased']['ctr'].tolist(), W=res['unbiased']['W'].tolist())
            print(f'  {name:6s} {s:6s}: r0={D[s]["r_DA0_A"]:.2f}  kgate={D[s]["k_gate_kcal_per_A2"]:.2f}  '
                  f'dW31={D[s]["dW_3p1_kcal"]:.2f}', flush=True)
        result['sets'][name] = D
        result['correlations'][name] = corr_table(D)
    for half in ('first', 'second'):
        D = {}
        for s in SYSTEMS:
            res, n = pmf(dirsets['pooled'](s), half=half)
            D[s] = dict(res['unbiased']['desc'], **n)
        result['sets'][f'pooled_{half}_half'] = D
        result['correlations'][f'pooled_{half}_half'] = corr_table(D)
    # replica spread of the minimum
    spread = {s: max(result['sets'][r][s]['r_DA0_A'] for r in ('rep1', 'rep2', 'rep3'))
              - min(result['sets'][r][s]['r_DA0_A'] for r in ('rep1', 'rep2', 'rep3')) for s in SYSTEMS}
    result['replica_spread_r_DA0_A'] = spread
    result['replica_spread_r_DA0_max_A'] = max(spread.values())
    # single-replica range of the r_DA0 correlation with ln KIE10
    rr = [result['correlations'][r]['r_DA0_A']['lnKIE10']['pearson_R'] for r in ('rep1', 'rep2', 'rep3')]
    result['single_replica_R_r_DA0_lnKIE10_range'] = [min(rr), max(rr)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    for k, v in result['correlations'].items():
        line = []
        for d in ('r_DA0_A', 'k_gate_kcal_per_A2', 'rms_harmonic_A', 'dW_3p1_kcal'):
            if v[d] is None:
                line.append(f'{d}: n/a'); continue
            a, b = v[d]['lnKIE10'], v[d]['lnKIE10_over_KIE40']
            line.append(f'{d.split("_")[0]}: R={a["pearson_R"]:+.3f} (p={a["perm_p_two_sided"]:.3f}) / '
                        f'ratio R={b["pearson_R"]:+.3f} (p={b["perm_p_two_sided"]:.3f})')
        print(f'{k:28s} ' + ' | '.join(line))
    print(f'replica spread of r_DA0: {spread}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
