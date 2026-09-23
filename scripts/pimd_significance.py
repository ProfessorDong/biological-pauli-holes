#!/usr/bin/env python3
"""Statistical significance of the I553A vs I552A path-multiplicity result.

For each replica R in {1,2,3} of each (system, isotope) combination, compute
S_shannon(R) = -sum p log p over the 24x24 histogram of (r_HO, theta_CHO).
Then bootstrap-resample by drawing one replica per (system, isotope) 10000 times
and compute the distribution of
    Delta ln(N_eff,H/N_eff,D)_I553A - Delta ln(N_eff,H/N_eff,D)_I552A.
Report P(delta > 0), the mean and 95% CI.
"""
from pathlib import Path
import numpy as np, json

DATA = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/pimd_prod')


def load_run(system, iso, rep):
    p = DATA/f'{system}_{iso}_r335_rep{rep}_pimd.npz'
    if not p.exists(): return None
    return np.load(p)


def project(xferH, donor, acceptor):
    donor_b = donor[:, None, :]; acceptor_b = acceptor[:, None, :]
    v_HO = acceptor_b - xferH; v_HC = donor_b - xferH
    r_HO = np.linalg.norm(v_HO, axis=-1); r_HC = np.linalg.norm(v_HC, axis=-1)
    cos_theta = np.einsum('fbi,fbi->fb', v_HO, v_HC) / (r_HO * r_HC + 1e-12)
    theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    return r_HO.flatten(), theta.flatten()


# Compute the (r_HO, theta) sample per replica per (system, iso)
samples = {}  # (system, iso, rep) -> (r_HO, theta) arrays
for s in ['I553A', 'I552A']:
    for iso in ['H', 'D']:
        for rep in [1, 2, 3]:
            d = load_run(s, iso, rep)
            if d is None: continue
            r, t = project(np.asarray(d['xferH_positions']),
                           np.asarray(d['donor_pos']),
                           np.asarray(d['acceptor_pos']))
            samples[(s, iso, rep)] = (r, t)

# Global range so all histograms are directly comparable
r_all = np.concatenate([r for r, _ in samples.values()])
t_all = np.concatenate([t for _, t in samples.values()])
r_range = [np.percentile(r_all, 1), np.percentile(r_all, 99)]
t_range = [np.percentile(t_all, 1), np.percentile(t_all, 99)]
NB = 24


def shannon(r, t):
    H, _, _ = np.histogram2d(r, t, bins=NB, range=[r_range, t_range])
    p = H / (H.sum() + 1e-30)
    nz = p > 0
    return -np.sum(p[nz] * np.log(p[nz]))


# Per-replica entropies
S_per = {}
for k, (r, t) in samples.items():
    S_per[k] = shannon(r, t)

# Bootstrap: draw one replica per (system, iso) with replacement, 10000 times
rng = np.random.default_rng(20260803)
N_BOOT = 10000
delta_boot = np.zeros(N_BOOT)
for i in range(N_BOOT):
    S = {}
    for s in ['I553A', 'I552A']:
        for iso in ['H', 'D']:
            reps_available = [rep for rep in [1, 2, 3] if (s, iso, rep) in samples]
            picked = rng.choice(reps_available)
            S[(s, iso)] = S_per[(s, picked)] if False else S_per[(s, iso, picked)]
    dlnN_I553A = S[('I553A', 'H')] - S[('I553A', 'D')]
    dlnN_I552A = S[('I552A', 'H')] - S[('I552A', 'D')]
    delta_boot[i] = dlnN_I553A - dlnN_I552A


# Also: exhaustive permutation test using the 3x3x3x3 = 81 assignments per row
# — but bootstrap CI is more informative than a null test for a directional prediction.

mean_delta = float(delta_boot.mean())
lo, hi = float(np.percentile(delta_boot, 2.5)), float(np.percentile(delta_boot, 97.5))
p_positive = float((delta_boot > 0).mean())

# Per-system delta-ln-N_eff at the point estimate (each replica averaged)
point = {}
for s in ['I553A', 'I552A']:
    dH = np.mean([S_per[(s, 'H', rep)] for rep in [1, 2, 3] if (s, 'H', rep) in samples])
    dD = np.mean([S_per[(s, 'D', rep)] for rep in [1, 2, 3] if (s, 'D', rep) in samples])
    sH = np.std([S_per[(s, 'H', rep)] for rep in [1, 2, 3] if (s, 'H', rep) in samples], ddof=1)
    sD = np.std([S_per[(s, 'D', rep)] for rep in [1, 2, 3] if (s, 'D', rep) in samples], ddof=1)
    point[s] = {
        'S_H_mean': dH, 'S_D_mean': dD,
        'S_H_sem': sH/np.sqrt(3), 'S_D_sem': sD/np.sqrt(3),
        'ln_N_eff_H_over_D_point': float(dH - dD),
        'N_eff_H': float(np.exp(dH)), 'N_eff_D': float(np.exp(dD)),
    }
d_point = point['I553A']['ln_N_eff_H_over_D_point'] - point['I552A']['ln_N_eff_H_over_D_point']

out = {
    'per_system': point,
    'point_delta_ln_N_eff_isotope_ratio_I553A_minus_I552A': float(d_point),
    'bootstrap_10000': {
        'mean_delta': mean_delta,
        'ci95_lo': lo, 'ci95_hi': hi,
        'P_delta_positive': p_positive,
    },
    'experimental_delta_ln_KIE_I553A_minus_I552A': float(np.log(148) - np.log(66)),
}
with open(DATA/'analysis'/'pimd_significance.json', 'w') as f:
    json.dump(out, f, indent=2)

print(f'Point estimate  Delta ln(N_eff,H/D) I553A - I552A = {d_point:+.4f}')
print(f'Bootstrap 10000: mean={mean_delta:+.4f}  95% CI = [{lo:+.4f}, {hi:+.4f}]')
print(f'P(delta > 0) = {p_positive:.4f}')
print(f'Experimental Delta ln(KIE) = {np.log(148)-np.log(66):+.4f}')
