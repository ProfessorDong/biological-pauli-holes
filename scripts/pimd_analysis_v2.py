#!/usr/bin/env python3
"""Path-multiplicity analysis on the full scaled-up PIMD campaign for I553A vs I552A.

Uses ALL 36 RPMD trajectories:
  I553A/I552A x {H, D} x {r0=3.35 A (reps 1-6), r0=3.60 A (reps 1-3)}
Total: 2 x 2 x (6 + 3) = 36 trajectories, 300 frames x 8 beads = 2400 bead-frame samples
per trajectory, ~86400 samples per (system, isotope) combination pooled.

Reports both point estimate (histogram entropy) and frame-level block-bootstrap
confidence intervals on Delta ln(N_eff,H/N_eff,D) per system and between systems.
"""
from pathlib import Path
import numpy as np, json

DATA = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/pimd_prod')
OUT  = DATA/'analysis'; OUT.mkdir(parents=True, exist_ok=True)


def load_all_frames(system, iso):
    """Concatenate frames across all replicas and BOTH near-attack windows (r0=3.35, r0=3.60)."""
    r_HO_all, theta_all = [], []
    n_trajs = 0
    for r0_tag in ['r335', 'r360']:
        for rep in [1, 2, 3, 4, 5, 6]:
            p = DATA/f'{system}_{iso}_{r0_tag}_rep{rep}_pimd.npz'
            if not p.exists(): continue
            d = np.load(p)
            xH = np.asarray(d['xferH_positions'])
            dpos = np.asarray(d['donor_pos'])[:, None, :]
            apos = np.asarray(d['acceptor_pos'])[:, None, :]
            v_HO = apos - xH; v_HC = dpos - xH
            r_HO = np.linalg.norm(v_HO, axis=-1)
            r_HC = np.linalg.norm(v_HC, axis=-1)
            cos_t = np.einsum('fbi,fbi->fb', v_HO, v_HC) / (r_HO*r_HC + 1e-12)
            th = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
            r_HO_all.append(r_HO); theta_all.append(th); n_trajs += 1
    r_HO = np.concatenate(r_HO_all); theta = np.concatenate(theta_all)
    return r_HO, theta, n_trajs


data = {}
for s in ['I553A', 'I552A']:
    for iso in ['H', 'D']:
        r, t, n = load_all_frames(s, iso)
        data[(s, iso)] = (r, t)
        print(f'  loaded {s} {iso}: n_trajs={n}  n_samples={r.size}  '
              f'<r_HO>={r.mean():.3f} A  <theta>={t.mean():.2f} deg')

# Global (r_HO, theta) range on the pooled data
r_all = np.concatenate([r.flatten() for r, _ in data.values()])
t_all = np.concatenate([t.flatten() for _, t in data.values()])
r_range = [float(np.percentile(r_all, 1)), float(np.percentile(r_all, 99))]
t_range = [float(np.percentile(t_all, 1)), float(np.percentile(t_all, 99))]
NB = 24


def entropy(r, t):
    H, _, _ = np.histogram2d(r.flatten(), t.flatten(), bins=NB, range=[r_range, t_range])
    p = H/(H.sum()+1e-30)
    nz = p > 0
    return -np.sum(p[nz]*np.log(p[nz]))


S_point = {k: entropy(v[0], v[1]) for k, v in data.items()}
d_I553A_pt = S_point[('I553A','H')] - S_point[('I553A','D')]
d_I552A_pt = S_point[('I552A','H')] - S_point[('I552A','D')]
DD_pt = d_I553A_pt - d_I552A_pt


# Frame-level block bootstrap: block=30 frames = 1.5 ps blocks
BLOCK = 30
N_BOOT = 5000
rng = np.random.default_rng(20260803)


def block_boot(r, t):
    F, N = r.shape
    nblocks = F // BLOCK
    out = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        start = rng.integers(0, F - BLOCK + 1, size=nblocks)
        idx = (start[:, None] + np.arange(BLOCK)[None, :]).flatten()[:F]
        out[b] = entropy(r[idx], t[idx])
    return out


boot = {k: block_boot(v[0], v[1]) for k, v in data.items()}
d_I553A = boot[('I553A','H')] - boot[('I553A','D')]
d_I552A = boot[('I552A','H')] - boot[('I552A','D')]
DD      = d_I553A - d_I552A


def summ(x):
    return dict(mean=float(x.mean()), sem=float(x.std()),
                ci95_lo=float(np.percentile(x, 2.5)),
                ci95_hi=float(np.percentile(x, 97.5)),
                p_positive=float((x > 0).mean()))


result = {
    'campaign_scale': {
        'r335_reps_per_condition': 6, 'r360_reps_per_condition': 3,
        'total_trajectories': sum(1 for k, _ in data.items()) * 9,
        'ps_prod_per_trajectory': 15.0, 'nbeads': 8,
        'samples_per_condition': int(data[('I553A','H')][0].size),
        'block_bootstrap': dict(block_frames=BLOCK, n_boot=N_BOOT, ns_prod_pooled=15.0*9),
        'histogram_bins': NB, 'r_HO_range_A': r_range, 'theta_range_deg': t_range,
    },
    'per_condition_point': {
        f'{k[0]}_{k[1]}': {
            'S_shannon_nats': float(S_point[k]),
            'N_eff': float(np.exp(S_point[k])),
            'r_HO_mean_A': float(v[0].mean()),
            'r_HO_std_A':  float(v[0].std()),
            'theta_mean_deg': float(v[1].mean()),
            'theta_std_deg':  float(v[1].std()),
        }
        for k, v in data.items()
    },
    'per_system': {
        'I553A': {
            'ln_N_eff_H_over_D_point': float(d_I553A_pt),
            'ln_N_eff_H_over_D_bootstrap': summ(d_I553A),
            'KIE': 148, 'ln_KIE': float(np.log(148)),
        },
        'I552A': {
            'ln_N_eff_H_over_D_point': float(d_I552A_pt),
            'ln_N_eff_H_over_D_bootstrap': summ(d_I552A),
            'KIE': 66, 'ln_KIE': float(np.log(66)),
        },
    },
    'between_systems_verdict': {
        'point_delta_ln_N_eff_isotope_ratio_I553A_minus_I552A': float(DD_pt),
        'bootstrap': summ(DD),
        'experimental_delta_ln_KIE_I553A_minus_I552A': float(np.log(148)-np.log(66)),
        'sign_matches': bool(DD_pt > 0),
        'path_multiplicity_fraction_of_experimental_ladder': float(DD_pt / (np.log(148)-np.log(66))),
    }
}

with open(OUT/'pimd_analysis_v2.json', 'w') as f:
    json.dump(result, f, indent=2)

print()
print('=== SCALED-UP RESULTS ===')
print(f'  I553A: N_eff,H={np.exp(S_point[("I553A","H")]):.1f}  N_eff,D={np.exp(S_point[("I553A","D")]):.1f}')
print(f'         ln(N_eff,H/D) = {d_I553A_pt:+.4f}   bootstrap: {d_I553A.mean():+.4f} +/- {d_I553A.std():.4f}  '
      f'95% CI [{np.percentile(d_I553A,2.5):+.4f}, {np.percentile(d_I553A,97.5):+.4f}]  '
      f'P(>0)={(d_I553A>0).mean():.4f}')
print(f'  I552A: N_eff,H={np.exp(S_point[("I552A","H")]):.1f}  N_eff,D={np.exp(S_point[("I552A","D")]):.1f}')
print(f'         ln(N_eff,H/D) = {d_I552A_pt:+.4f}   bootstrap: {d_I552A.mean():+.4f} +/- {d_I552A.std():.4f}  '
      f'95% CI [{np.percentile(d_I552A,2.5):+.4f}, {np.percentile(d_I552A,97.5):+.4f}]  '
      f'P(>0)={(d_I552A>0).mean():.4f}')
print()
print(f'  BETWEEN-SYSTEMS DD = {DD_pt:+.4f}')
print(f'    bootstrap: mean={DD.mean():+.4f}  SEM={DD.std():.4f}  '
      f'95% CI = [{np.percentile(DD,2.5):+.4f}, {np.percentile(DD,97.5):+.4f}]  '
      f'P(sign>0) = {(DD>0).mean():.4f}')
print(f'    experimental: Delta ln(KIE) = +{np.log(148)-np.log(66):.4f}')
print(f'    path-multiplicity fraction of ladder = {DD_pt / (np.log(148)-np.log(66))*100:.2f}%')
print()
print(f'wrote {OUT/"pimd_analysis_v2.json"}')
