#!/usr/bin/env python3
"""Path-multiplicity analysis for the I553A/I552A PIMD campaign.

Reads the .npz files produced by pimd_window.py for I553A and I552A x {H, D} x replicas.
For each system-isotope combination it:

  1. Projects every bead position at every recorded frame onto the reactive plane
     (r_HO, theta_CHO), giving an isotope-specific quantum sample of the transferring
     particle in the reactive geometry.
  2. Estimates the differential entropy H[rho_i(r_HO, theta_CHO)] by histogram + KDE.
  3. Defines N_eff,i = exp(H[rho_i]) (Renyi-1 entropy exponentiation) and reports
     the isotope-specific path-multiplicity contribution
        Delta S_2^(H-D) / kB = ln(N_eff,H) - ln(N_eff,D).
  4. Compares I553A vs I552A: the framework's central prediction is that the distal
     mutation I553A supports a larger Delta S_2^(H-D) than the proximal I552A
     (which sits at the WT KIE of 66 despite being closer to the reactive cavity).

The bead-cloud sigma is reported alongside as a sanity-check descriptor: for a
transferring particle in a harmonic well, sigma_H/sigma_D -> sqrt(m_D/m_H) = sqrt(2)
= 1.414 in the harmonic limit.
"""
from pathlib import Path
import numpy as np
import json

DATA = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/pimd_prod')
OUT = Path('/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/pimd_prod/analysis')
OUT.mkdir(parents=True, exist_ok=True)

SYSTEMS = ['I553A', 'I552A']
ISOTOPES = ['H', 'D']
REPLICAS = [1, 2, 3]
KIE = {'I553A': 148, 'I552A': 66}


def load_run(system, iso, rep):
    """Load a single PIMD run's .npz; returns None if missing/corrupt."""
    p = DATA / f'{system}_{iso}_r335_rep{rep}_pimd.npz'
    if not p.exists():
        return None
    try:
        return np.load(p)
    except Exception as ex:
        print(f'  warn: could not load {p}: {ex}')
        return None


def project_to_reactive_plane(xferH, donor, acceptor):
    """For each frame f and bead b: compute r_HO (H..O distance) and theta_CHO
    (C-H-O angle in degrees). Returns arrays (n_frames, n_beads) each.
    xferH: (n_frames, n_beads, 3);  donor, acceptor: (n_frames, 3).
    """
    donor_b = donor[:, None, :]      # (F, 1, 3)
    acceptor_b = acceptor[:, None, :]
    v_HO = acceptor_b - xferH        # H -> O
    v_HC = donor_b - xferH           # H -> C
    r_HO = np.linalg.norm(v_HO, axis=-1)              # (F, N)
    r_HC = np.linalg.norm(v_HC, axis=-1)
    cos_theta = np.einsum('fbi,fbi->fb', v_HO, v_HC) / (r_HO * r_HC + 1e-12)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))          # (F, N)  in degrees
    return r_HO, theta


def differential_entropy_2d(x, y, nbins=32, range_xy=None):
    """Differential entropy H[rho(x,y)] estimated by 2D histogram + small floor.
    Uses natural log (in nats). H[rho] = - sum p log p - log(dx*dy) if we want the
    differential entropy in the same units.
    """
    if range_xy is None:
        range_xy = [[x.min(), x.max()], [y.min(), y.max()]]
    H2d, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=range_xy)
    dx = xedges[1] - xedges[0]; dy = yedges[1] - yedges[0]
    p = H2d.astype(np.float64)
    p /= p.sum() + 1e-30
    nz = p > 0
    S_shannon = -np.sum(p[nz] * np.log(p[nz]))       # nats, dimensionless
    S_diff = S_shannon + np.log(dx * dy)             # nats, includes cell scale
    return S_shannon, S_diff


def process_run(system, iso, rep):
    d = load_run(system, iso, rep)
    if d is None:
        return None
    xferH = np.asarray(d['xferH_positions'])
    donor = np.asarray(d['donor_pos'])
    acceptor = np.asarray(d['acceptor_pos'])
    r_HO, theta = project_to_reactive_plane(xferH, donor, acceptor)
    # flatten across frames and beads
    r_HO_all = r_HO.flatten()
    theta_all = theta.flatten()
    # bead-cloud sigma (deviation of each bead from the per-frame centroid, in A)
    centroid = xferH.mean(axis=1, keepdims=True)
    sigma_bead = np.linalg.norm(xferH - centroid, axis=-1).mean()  # scalar in A
    return {
        'system': system, 'iso': iso, 'rep': rep,
        'n_frames': int(xferH.shape[0]), 'n_beads': int(xferH.shape[1]),
        'r_HO_all': r_HO_all, 'theta_all': theta_all,
        'sigma_bead_A': float(sigma_bead),
        'r_DA_mean': float(np.asarray(d['r_DA']).mean()),
        'r_DA_std':  float(np.asarray(d['r_DA']).std()),
    }


def combine_replicas(runs, system, iso):
    kept = [r for r in runs if r is not None and r['system']==system and r['iso']==iso]
    if not kept:
        return None
    r_HO_all = np.concatenate([k['r_HO_all'] for k in kept])
    theta_all = np.concatenate([k['theta_all'] for k in kept])
    sigma_bead_reps = [k['sigma_bead_A'] for k in kept]
    return {
        'system': system, 'iso': iso, 'n_replicas': len(kept),
        'r_HO_all': r_HO_all, 'theta_all': theta_all,
        'sigma_bead_mean_A': float(np.mean(sigma_bead_reps)),
        'sigma_bead_sem_A':  float(np.std(sigma_bead_reps)/max(1, np.sqrt(len(sigma_bead_reps)))),
        'n_samples': int(len(r_HO_all)),
    }


def main():
    runs = []
    for s in SYSTEMS:
        for iso in ISOTOPES:
            for rep in REPLICAS:
                r = process_run(s, iso, rep)
                if r is not None:
                    runs.append(r)
                    print(f'  loaded {s} {iso} rep{rep}: nframes={r["n_frames"]} '
                          f'sigma_bead={r["sigma_bead_A"]:.4f} A  r_DA=<{r["r_DA_mean"]:.3f}>')
                else:
                    print(f'  missing {s} {iso} rep{rep}')

    # Combine per (system, isotope)
    combos = {}
    for s in SYSTEMS:
        for iso in ISOTOPES:
            c = combine_replicas(runs, s, iso)
            if c is not None:
                combos[(s, iso)] = c

    # Use a common (r_HO, theta) range across all (S, iso) so entropies are comparable
    if not combos:
        print('no runs to analyse'); return
    r_all = np.concatenate([c['r_HO_all'] for c in combos.values()])
    t_all = np.concatenate([c['theta_all'] for c in combos.values()])
    r_range = [float(np.percentile(r_all, 1)), float(np.percentile(r_all, 99))]
    t_range = [float(np.percentile(t_all, 1)), float(np.percentile(t_all, 99))]

    NBINS = 24
    result = {}
    for key, c in combos.items():
        S_shannon, S_diff = differential_entropy_2d(c['r_HO_all'], c['theta_all'],
                                                    nbins=NBINS, range_xy=[r_range, t_range])
        N_eff = float(np.exp(S_shannon))
        result[f'{key[0]}_{key[1]}'] = {
            'n_replicas': c['n_replicas'],
            'n_samples': c['n_samples'],
            'sigma_bead_mean_A': c['sigma_bead_mean_A'],
            'sigma_bead_sem_A': c['sigma_bead_sem_A'],
            'r_HO_mean_A': float(c['r_HO_all'].mean()),
            'r_HO_std_A':  float(c['r_HO_all'].std()),
            'theta_mean_deg': float(c['theta_all'].mean()),
            'theta_std_deg':  float(c['theta_all'].std()),
            'S_shannon_nats': float(S_shannon),
            'S_diff_nats':    float(S_diff),
            'N_eff':          N_eff,
        }

    # Path-multiplicity contribution to ln(KIE)
    per_system = {}
    for s in SYSTEMS:
        if (s, 'H') in combos and (s, 'D') in combos:
            NH = result[f'{s}_H']['N_eff']; ND = result[f'{s}_D']['N_eff']
            sH = result[f'{s}_H']['S_shannon_nats']; sD = result[f'{s}_D']['S_shannon_nats']
            per_system[s] = {
                'N_eff_H': NH, 'N_eff_D': ND,
                'ln_N_eff_H_over_D': float(sH - sD),
                'sigma_bead_H_over_D': result[f'{s}_H']['sigma_bead_mean_A'] / result[f'{s}_D']['sigma_bead_mean_A'],
                'KIE_experimental': KIE[s],
                'ln_KIE_experimental': float(np.log(KIE[s])),
            }

    # The framework's central positive result:
    # Delta ln(KIE) (I553A vs I552A) = 0.81  (KIE 148 vs 66)
    # Prediction: Delta ln(N_eff,H/N_eff,D) is positive and of comparable size
    verdict = {}
    if 'I553A' in per_system and 'I552A' in per_system:
        d_ln_kie = per_system['I553A']['ln_KIE_experimental'] - per_system['I552A']['ln_KIE_experimental']
        d_ln_Neff = per_system['I553A']['ln_N_eff_H_over_D'] - per_system['I552A']['ln_N_eff_H_over_D']
        d_sigma = (per_system['I553A']['sigma_bead_H_over_D']
                   - per_system['I552A']['sigma_bead_H_over_D'])
        verdict = {
            'Delta_ln_KIE_I553A_minus_I552A': float(d_ln_kie),
            'Delta_ln_N_eff_isotope_ratio_I553A_minus_I552A': float(d_ln_Neff),
            'Delta_sigma_bead_isotope_ratio_I553A_minus_I552A_A': float(d_sigma),
            'sign_matches_KIE': bool(d_ln_Neff > 0),
        }
        print()
        print('=== POSITIVE-RESULT VERDICT ===')
        print(f'  Delta ln(KIE) I553A - I552A          = {d_ln_kie:+.3f}  (experimental)')
        print(f'  Delta ln(N_eff,H/N_eff,D)            = {d_ln_Neff:+.3f}  (this PIMD test)')
        print(f'  Delta (sigma_H/sigma_D)              = {d_sigma:+.3f}')
        print(f'  Sign of path-multiplicity matches KIE direction: {verdict["sign_matches_KIE"]}')

    with open(OUT/'pimd_analysis.json', 'w') as f:
        json.dump({'per_run_result': result, 'per_system': per_system, 'verdict': verdict}, f, indent=2)
    print(f'\nwrote {OUT/"pimd_analysis.json"}')


if __name__ == '__main__':
    main()
