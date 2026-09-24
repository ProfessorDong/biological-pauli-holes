#!/usr/bin/env python3
"""A3-proper analysis: MBAR-reweighted joint (r_HO, theta_CHO) volume from the short
re-run per-frame data.

For each of the 8 SLO systems: pool all 15 near-attack window r_DA/r_HO/theta_CHO time
series; run MBAR on r_DA with harmonic-umbrella u_kn; compute equilibrium (u=0) weights
directly from f_k and u_kn (skip pymbar's expectations API, which triggers a numpy
eigh non-convergence on ill-conditioned Theta); restrict to the near-attack range
r_DA in [2.8, 3.6] A; compute the joint weighted 2x2 covariance on (r_HO, theta_CHO)
and report sqrt(det Sigma). Regress against ln(KIE_10C).
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

from pathlib import Path
import numpy as np, json
from pymbar import MBAR

DATA = Path(_REPO + '/results/umbrella_reactplane')
OUT  = DATA/'analysis'; OUT.mkdir(parents=True, exist_ok=True)

SYSTEMS = ['WT','V750A','I552A','I538A','L754A','L546A','I553A','DM']
KIE = {'WT':66,'V750A':62,'I552A':66,'I538A':100,'L754A':106,'L546A':131,'I553A':148,'DM':537}
K_BIAS_kcal = 12.0                             # kcal/mol/A^2
K_BIAS = K_BIAS_kcal * 4.184                   # kJ/mol/A^2
KT_kJ = 8.314e-3 * 300.0                       # kJ/mol at 300 K

WINDOWS = [2.70,2.95,3.20,3.45,3.70,3.95,4.20,4.45,4.70,4.95,5.20,5.45,5.70,5.95,6.20]


def load_traj(system):
    traj = []
    for w in WINDOWS:
        p = DATA/f'{system}_win_{w:.2f}_reactplane.npz'
        if not p.exists(): continue
        d = np.load(p)
        traj.append((float(w), d['r_DA'].astype(np.float64),
                     d['r_HO'].astype(np.float64), d['theta_CHO'].astype(np.float64)))
    return traj


def mbar_weights_unbiased(system):
    """Return per-frame equilibrium (u=0) weights via direct log-weight formula
    log_w_n = -log(sum_k N_k exp(f_k - u_kn[k,n])), normalised to sum(w) = 1."""
    traj = load_traj(system)
    if not traj: return None
    K = len(traj)
    Nk = np.array([len(t[1]) for t in traj])
    N = int(Nk.sum())
    r_DA_all = np.concatenate([t[1] for t in traj])
    r_HO_all = np.concatenate([t[2] for t in traj])
    theta_all = np.concatenate([t[3] for t in traj])
    x_centres = [t[0] for t in traj]
    u_kn = np.zeros((K, N))
    for k, w_k in enumerate(x_centres):
        u_kn[k] = 0.5 * K_BIAS * (r_DA_all - w_k)**2 / KT_kJ
    mbar = MBAR(u_kn, Nk, initialize='BAR', solver_protocol='robust')
    f_k = mbar.f_k                                                # free energies (kT units)
    # Log-denominator: log(sum_k N_k * exp(f_k - u_kn[k,n]))
    x = f_k[:, None] - u_kn                                       # shape (K, N)
    x_max = np.max(x, axis=0)                                      # log-sum-exp trick
    log_denom = x_max + np.log(np.sum(Nk[:, None] * np.exp(x - x_max), axis=0))
    log_w = -log_denom                                             # log unbiased weight
    log_w -= (np.max(log_w) + np.log(np.sum(np.exp(log_w - np.max(log_w)))))
    weights = np.exp(log_w)
    return dict(r_DA=r_DA_all, r_HO=r_HO_all, theta=theta_all, weights=weights,
                x_centres=x_centres, Nk=Nk, f_k=f_k)


def joint_covariance_reactplane(system, r_lo=2.8, r_hi=3.6):
    d = mbar_weights_unbiased(system)
    if d is None: return None
    mask = (d['r_DA'] >= r_lo) & (d['r_DA'] <= r_hi)
    n_in = int(mask.sum())
    w_in = d['weights'][mask]
    if n_in < 20 or w_in.sum() < 1e-12: return None
    w = w_in / w_in.sum()
    r = d['r_HO'][mask]; t = d['theta'][mask]
    mean_r = float((w * r).sum()); mean_t = float((w * t).sum())
    var_r = float((w * (r - mean_r)**2).sum())
    var_t = float((w * (t - mean_t)**2).sum())
    cov   = float((w * (r - mean_r) * (t - mean_t)).sum())
    Sigma = np.array([[var_r, cov], [cov, var_t]])
    detS = float(np.linalg.det(Sigma))
    return dict(r_HO_mean=mean_r, theta_mean=mean_t,
                sigma_r=float(np.sqrt(var_r)), sigma_t=float(np.sqrt(var_t)),
                corr=float(cov/np.sqrt(var_r*var_t + 1e-30)),
                det_Sigma=detS, sqrt_det=float(np.sqrt(max(detS, 0.0))),
                n_frames_in_range=n_in, n_frames_total=int(len(d['r_DA'])),
                w_sum_in_range=float(w_in.sum()))


if __name__ == '__main__':
    results = {}
    for s in SYSTEMS:
        try:
            r = joint_covariance_reactplane(s)
        except Exception as ex:
            print(f'  {s}: FAILED {ex}')
            continue
        if r is None:
            print(f'  {s}: not enough near-attack samples'); continue
        r['KIE'] = KIE[s]
        results[s] = r
        print(f'  {s:6s}: KIE={KIE[s]:3d}  <r_HO>={r["r_HO_mean"]:.3f}  '
              f'<theta>={r["theta_mean"]:.2f}  sigma_r={r["sigma_r"]:.4f}  '
              f'sigma_t={r["sigma_t"]:.4f}  corr={r["corr"]:+.3f}  '
              f'sqrt(det)={r["sqrt_det"]:.5f}  n={r["n_frames_in_range"]}')

    with open(OUT/'mbar_2d_volume.json', 'w') as f:
        json.dump(results, f, indent=2)

    jbc = ['WT','V750A','I552A','I538A','L754A','L546A','I553A']
    xs = np.array([results[s]['sqrt_det'] for s in jbc if s in results])
    ys = np.array([np.log(results[s]['KIE']) for s in jbc if s in results])
    if len(xs) >= 3:
        R = float(np.corrcoef(xs, ys)[0, 1])
        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(xs, ys)
        except Exception:
            rho, p = float('nan'), float('nan')
        print(f'\nMBAR 2D joint volume vs ln(KIE) [n={len(xs)} JBC]: Pearson R={R:+.3f}   '
              f'Spearman rho={rho:+.3f}  p={p:.3f}')
        with open(OUT/'mbar_2d_regression.json', 'w') as f:
            json.dump(dict(pearson_R=R, spearman_rho=float(rho), spearman_p=float(p),
                           n=int(len(xs)), systems=[s for s in jbc if s in results],
                           sqrt_det_values=xs.tolist(), ln_KIE_values=ys.tolist()), f, indent=2)
        print(f'wrote {OUT/"mbar_2d_regression.json"}')
