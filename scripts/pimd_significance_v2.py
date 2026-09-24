#!/usr/bin/env python3
"""Frame-level block-bootstrap significance test of the I553A vs I552A path-multiplicity result.

The 3-replica bootstrap has effective sample size ~3, way too coarse. Here we pool all frames
x beads (~7200 samples per condition) and block-bootstrap over frames with block size ~30 frames
(1.5 ps blocks) to respect the autocorrelation of the PIMD reactive-basin sampling. This gives
the honest error bar on Delta ln(N_eff,H/D) per system and on the between-systems difference.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

from pathlib import Path
import numpy as np, json

DATA = Path(_REPO + '/results/pimd_prod')
OUT = DATA/'analysis'; OUT.mkdir(parents=True, exist_ok=True)


def load_projected(system, iso):
    """Concatenate all replica frames for one (system, iso). Return (n_frames, n_beads) arrays
    of (r_HO, theta_CHO), preserving the temporal frame ordering per replica."""
    r_HO_all, theta_all, rep_ids, frame_ids = [], [], [], []
    for rep in [1, 2, 3]:
        p = DATA/f'{system}_{iso}_r335_rep{rep}_pimd.npz'
        if not p.exists(): continue
        d = np.load(p)
        xH = np.asarray(d['xferH_positions'])   # (F, N, 3)
        dpos = np.asarray(d['donor_pos'])[:, None, :]
        apos = np.asarray(d['acceptor_pos'])[:, None, :]
        v_HO = apos - xH
        v_HC = dpos - xH
        r_HO = np.linalg.norm(v_HO, axis=-1)
        r_HC = np.linalg.norm(v_HC, axis=-1)
        cos_theta = np.einsum('fbi,fbi->fb', v_HO, v_HC) / (r_HO*r_HC + 1e-12)
        theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        F = r_HO.shape[0]
        r_HO_all.append(r_HO); theta_all.append(theta)
        rep_ids.append(np.full(F, rep, dtype=np.int8))
        frame_ids.append(np.arange(F))
    return np.concatenate(r_HO_all), np.concatenate(theta_all), np.concatenate(rep_ids), np.concatenate(frame_ids)


# Load
data = {}
for s in ['I553A', 'I552A']:
    for iso in ['H', 'D']:
        data[(s, iso)] = load_projected(s, iso)

# Global range from all data
r_all = np.concatenate([d[0].flatten() for d in data.values()])
t_all = np.concatenate([d[1].flatten() for d in data.values()])
r_range = [float(np.percentile(r_all, 1)), float(np.percentile(r_all, 99))]
t_range = [float(np.percentile(t_all, 1)), float(np.percentile(t_all, 99))]
NB = 24


def entropy_from_frames(r_HO, theta):
    """Shannon entropy of the (r_HO, theta) marginal, flattening all beads."""
    H, _, _ = np.histogram2d(r_HO.flatten(), theta.flatten(), bins=NB, range=[r_range, t_range])
    p = H / (H.sum() + 1e-30)
    nz = p > 0
    return -np.sum(p[nz] * np.log(p[nz]))


# Point estimate
S_point = {k: entropy_from_frames(v[0], v[1]) for k, v in data.items()}
d_I553A_point = S_point[('I553A', 'H')] - S_point[('I553A', 'D')]
d_I552A_point = S_point[('I552A', 'H')] - S_point[('I552A', 'D')]
DD_point = d_I553A_point - d_I552A_point

# Block bootstrap: resample frames with replacement in blocks of size BLOCK
BLOCK = 30
N_BOOT = 5000
rng = np.random.default_rng(20260803)

def block_bootstrap_entropy(r, t, n_boot):
    F, N = r.shape
    nblocks = F // BLOCK
    out = np.zeros(n_boot)
    for b in range(n_boot):
        start = rng.integers(0, F - BLOCK + 1, size=nblocks)
        idx = (start[:, None] + np.arange(BLOCK)[None, :]).flatten()
        idx = idx[:F]                                    # trim to keep length constant
        rb = r[idx]; tb = t[idx]
        out[b] = entropy_from_frames(rb, tb)
    return out

boot_S = {k: block_bootstrap_entropy(v[0], v[1], N_BOOT) for k, v in data.items()}

# Delta ln(N_eff,H/D) per system, and the between-system difference
d_I553A = boot_S[('I553A', 'H')] - boot_S[('I553A', 'D')]
d_I552A = boot_S[('I552A', 'H')] - boot_S[('I552A', 'D')]
DD = d_I553A - d_I552A

def summarise(x, name):
    return {name+'_mean': float(x.mean()), name+'_sem': float(x.std()),
            name+'_ci95_lo': float(np.percentile(x, 2.5)),
            name+'_ci95_hi': float(np.percentile(x, 97.5))}

out = {
    'BLOCK': BLOCK, 'N_BOOT': N_BOOT, 'NB': NB,
    'r_HO_range': r_range, 'theta_range_deg': t_range,
    'point_estimate': {
        'S_I553A_H': float(S_point[('I553A','H')]),
        'S_I553A_D': float(S_point[('I553A','D')]),
        'S_I552A_H': float(S_point[('I552A','H')]),
        'S_I552A_D': float(S_point[('I552A','D')]),
        'N_eff_I553A_H': float(np.exp(S_point[('I553A','H')])),
        'N_eff_I553A_D': float(np.exp(S_point[('I553A','D')])),
        'N_eff_I552A_H': float(np.exp(S_point[('I552A','H')])),
        'N_eff_I552A_D': float(np.exp(S_point[('I552A','D')])),
        'delta_ln_N_eff_I553A_point': float(d_I553A_point),
        'delta_ln_N_eff_I552A_point': float(d_I552A_point),
        'between_systems_point': float(DD_point),
    },
    'bootstrap': {
        **summarise(d_I553A, 'delta_ln_N_eff_I553A'),
        **summarise(d_I552A, 'delta_ln_N_eff_I552A'),
        **summarise(DD, 'DD_I553A_minus_I552A'),
        'P_delta_I553A_positive': float((d_I553A > 0).mean()),
        'P_delta_I552A_positive': float((d_I552A > 0).mean()),
        'P_DD_positive': float((DD > 0).mean()),
    },
    'experimental_delta_ln_KIE_I553A_minus_I552A': float(np.log(148) - np.log(66)),
}

with open(OUT/'pimd_significance_v2.json', 'w') as f:
    json.dump(out, f, indent=2)

# Console summary
print(f'Point estimates (Shannon entropy, nats; N_eff = exp(S)):')
print(f'  I553A: S_H={S_point[("I553A","H")]:.4f}  S_D={S_point[("I553A","D")]:.4f}  '
      f'N_eff,H={np.exp(S_point[("I553A","H")]):.1f}  N_eff,D={np.exp(S_point[("I553A","D")]):.1f}  '
      f'ln(N_eff,H/D)={d_I553A_point:+.4f}')
print(f'  I552A: S_H={S_point[("I552A","H")]:.4f}  S_D={S_point[("I552A","D")]:.4f}  '
      f'N_eff,H={np.exp(S_point[("I552A","H")]):.1f}  N_eff,D={np.exp(S_point[("I552A","D")]):.1f}  '
      f'ln(N_eff,H/D)={d_I552A_point:+.4f}')
print()
print(f'Between-systems point:  DD = {DD_point:+.4f}   (experimental: +{np.log(148)-np.log(66):.4f})')
print()
print(f'Block bootstrap (block={BLOCK} frames, n_boot={N_BOOT}):')
print(f'  Delta ln(N_eff,H/D) I553A  = {d_I553A.mean():+.4f} +/- {d_I553A.std():.4f}  '
      f'95% CI = [{np.percentile(d_I553A,2.5):+.4f}, {np.percentile(d_I553A,97.5):+.4f}]  '
      f'P(>0) = {(d_I553A>0).mean():.4f}')
print(f'  Delta ln(N_eff,H/D) I552A  = {d_I552A.mean():+.4f} +/- {d_I552A.std():.4f}  '
      f'95% CI = [{np.percentile(d_I552A,2.5):+.4f}, {np.percentile(d_I552A,97.5):+.4f}]  '
      f'P(>0) = {(d_I552A>0).mean():.4f}')
print(f'  DD = I553A - I552A         = {DD.mean():+.4f} +/- {DD.std():.4f}  '
      f'95% CI = [{np.percentile(DD,2.5):+.4f}, {np.percentile(DD,97.5):+.4f}]  '
      f'P(>0) = {(DD>0).mean():.4f}')
