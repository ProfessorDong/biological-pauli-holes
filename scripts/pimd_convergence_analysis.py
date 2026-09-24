#!/usr/bin/env python3
"""Analyse the B1 RPMD bead-convergence sweep on I553A/I552A.

For each (system, isotope, bead count P) cell, aggregate all replicas and windows,
compute the differential-entropy occupancy Ñ_i = exp H[ρ_i(r_HO, θ_CHO)], the isotope
ratio ln(Ñ_H/Ñ_D) per system, and the between-mutant contrast
ΔΔ = ln(Ñ_H/Ñ_D)_I553A - ln(Ñ_H/Ñ_D)_I552A. Compare across P ∈ {8, 16, 32} to test
whether the previously-reported P=8 null result on the mutant contrast persists at
higher bead count (indicating true converged null) or shifts (indicating the P=8
result was an under-convergence artefact).

Data sources:
- P=8:  results/pimd_prod/*_pimd.npz          (from the reported campaign, 36 trajs)
- P=16, P=32: results/pimd_convergence/*_pimd.npz  (this convergence sweep, 48 trajs)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

from pathlib import Path
import numpy as np, json

ROOT = Path(_REPO + '/results')
P8_DIR   = ROOT/'pimd_prod'
BIG_DIR  = ROOT/'pimd_convergence'
OUT      = ROOT/'pimd_convergence'/'analysis'
OUT.mkdir(parents=True, exist_ok=True)

# MATCHED DESIGN (audit fix, 2026-08-15)
# --------------------------------------
# The P=8 production campaign has 6 replicas at r335 and 3 at r360; the B1
# convergence campaign has 3 and 3. Pooling all of them put a 2:1 window weight on
# the P=8 cell and a 1:1 weight on P=16 and P=32, so the abscissa of this figure
# varied the WINDOW MIXTURE as well as the bead count. That is fatal for a
# convergence plot: recomputing P=8 on the matched 3+3 subset moves Ntilde by up to
# +31 units, moves DD by -0.039 (a sign flip), and REVERSES the apparent monotonic
# rise of Ntilde_H for I553A (241.4 -> 226.7 -> 251.1 instead of 213.2 -> 226.7 ->
# 251.1). Every P now uses 3 replicas per window so that only P varies.
MATCHED_DESIGN = True
REPS_PER_WINDOW = 3
P8_R_WINS   = ['r335', 'r360']
P16_R_WINS  = ['r335', 'r360']
P32_R_WINS  = ['r335', 'r360']


def load_trajs(pdir, system, iso, r_wins, P):
    """Return concatenated (r_HO, theta) arrays from all matching npz files."""
    r_all, t_all = [], []
    reps_available = []
    for rw in r_wins:
        max_rep = REPS_PER_WINDOW if MATCHED_DESIGN else (
            6 if (P == 8 and rw == 'r335') else 3)
        for rep in range(1, max_rep+1):
            if P == 8:
                fp = pdir/f'{system}_{iso}_{rw}_rep{rep}_pimd.npz'
            else:
                fp = pdir/f'{system}_{iso}_{rw}_P{P}_rep{rep}_pimd.npz'
            if not fp.exists(): continue
            d = np.load(fp)
            xH = np.asarray(d['xferH_positions'])
            dpos = np.asarray(d['donor_pos'])[:, None, :]
            apos = np.asarray(d['acceptor_pos'])[:, None, :]
            v_HO = apos - xH; v_HC = dpos - xH
            r_HO = np.linalg.norm(v_HO, axis=-1)
            r_HC = np.linalg.norm(v_HC, axis=-1)
            cos_t = np.einsum('fbi,fbi->fb', v_HO, v_HC) / (r_HO*r_HC + 1e-12)
            th = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
            r_all.append(r_HO.flatten()); t_all.append(th.flatten())
            reps_available.append((rw, rep))
    if not r_all: return None, None, reps_available
    return np.concatenate(r_all), np.concatenate(t_all), reps_available


# Pool data first to determine global histogram range
all_r, all_t = [], []
for sys_name in ['I553A', 'I552A']:
    for iso in ['H', 'D']:
        for P, pdir, wins in [(8, P8_DIR, P8_R_WINS), (16, BIG_DIR, P16_R_WINS), (32, BIG_DIR, P32_R_WINS)]:
            r, t, _ = load_trajs(pdir, sys_name, iso, wins, P)
            if r is not None:
                all_r.append(r); all_t.append(t)
r_all = np.concatenate(all_r); t_all = np.concatenate(all_t)
r_range = (float(np.percentile(r_all, 1)), float(np.percentile(r_all, 99)))
t_range = (float(np.percentile(t_all, 1)), float(np.percentile(t_all, 99)))
NB = 24
print(f'Global range: r_HO in [{r_range[0]:.3f}, {r_range[1]:.3f}] A,  '
      f'theta in [{t_range[0]:.1f}, {t_range[1]:.1f}] deg')
print(f'Histogram bins: {NB}x{NB}')
print()


def entropy(r, t):
    H, _, _ = np.histogram2d(r, t, bins=NB, range=[r_range, t_range])
    p = H/(H.sum()+1e-30); nz = p > 0
    return -np.sum(p[nz]*np.log(p[nz]))


def entropy_mm(r, t):
    """Plug-in entropy and its Miller-Madow bias correction, +(m-1)/(2N).

    The plug-in estimator is biased LOW, and the bias shrinks as the sample count
    grows. N here is frames x beads, so it rises 8-fold from P=8 to P=32 simply
    because a 32-bead ring polymer contributes four times as many bead positions
    per frame as an 8-bead one. Part of the rise of Ntilde with P in panel (a) is
    therefore an estimator artefact and not delocalization. Beads within a frame
    are strongly correlated, so N overstates the independent sample count and this
    correction is a LOWER bound on the artefact.

    The isotope RATIO is largely protected: H and D have identical N at a given P,
    so the bias cancels to first order in ln(Ntilde_H/Ntilde_D).
    """
    H, _, _ = np.histogram2d(r, t, bins=NB, range=[r_range, t_range])
    N = H.sum()
    p = H/(N+1e-30); nz = p > 0
    S = -np.sum(p[nz]*np.log(p[nz]))
    m = int(nz.sum())
    return float(S), float(S + (m - 1)/(2.0*N)), m, int(N)


results = {}
for P, pdir, wins in [(8, P8_DIR, P8_R_WINS), (16, BIG_DIR, P16_R_WINS), (32, BIG_DIR, P32_R_WINS)]:
    for sys_name in ['I553A', 'I552A']:
        for iso in ['H', 'D']:
            r, t, reps = load_trajs(pdir, sys_name, iso, wins, P)
            if r is None:
                continue
            S, S_mm, m_occ, n_tot = entropy_mm(r, t)
            key = f'P{P}_{sys_name}_{iso}'
            results[key] = dict(
                P=P, system=sys_name, isotope=iso,
                n_trajectories=len(reps),
                n_samples=int(r.size),
                r_HO_mean_A=float(r.mean()), r_HO_std_A=float(r.std()),
                theta_mean_deg=float(t.mean()), theta_std_deg=float(t.std()),
                S_shannon=float(S), N_eff=float(np.exp(S)),
                S_miller_madow=S_mm, N_eff_miller_madow=float(np.exp(S_mm)),
                occupied_bins=m_occ, n_bead_samples=n_tot,
            )


# Table
print(f'{"P":>4} {"sys":>7} {"iso":>3} {"ntraj":>5} {"nsamp":>7} '
      f'{"<r_HO>":>7} {"<theta>":>8} {"S":>7} {"N_eff":>7}')
for key, v in results.items():
    print(f'{v["P"]:>4} {v["system"]:>7} {v["isotope"]:>3} '
          f'{v["n_trajectories"]:>5} {v["n_samples"]:>7} '
          f'{v["r_HO_mean_A"]:>7.3f} {v["theta_mean_deg"]:>8.2f} '
          f'{v["S_shannon"]:>7.4f} {v["N_eff"]:>7.1f}')


# Per-system isotope shifts and between-mutant contrast per P
print(f'\n{"P":>4} {"ln(N_H/N_D) I553A":>18} {"ln(N_H/N_D) I552A":>18} '
      f'{"ΔΔ_I553A-I552A":>18}')
convergence = {}
for P in [8, 16, 32]:
    try:
        dH_I553A = results[f'P{P}_I553A_H']['S_shannon'] - results[f'P{P}_I553A_D']['S_shannon']
        dH_I552A = results[f'P{P}_I552A_H']['S_shannon'] - results[f'P{P}_I552A_D']['S_shannon']
        DD = dH_I553A - dH_I552A
        convergence[f'P{P}'] = dict(
            delta_ln_Neff_I553A=float(dH_I553A),
            delta_ln_Neff_I552A=float(dH_I552A),
            DD_I553A_minus_I552A=float(DD),
        )
        print(f'{P:>4} {dH_I553A:>+18.4f} {dH_I552A:>+18.4f} {DD:>+18.4f}')
    except KeyError as e:
        print(f'{P:>4} missing: {e}')


# Frame-level block bootstrap on ΔΔ for each P (3000 resamples, 30-frame blocks)
def block_boot_DD(P, pdir, wins, N_BOOT=3000, BLOCK=30, rng=None):
    if rng is None: rng = np.random.default_rng(20260806)
    # Re-load per-condition arrays and record trajectory boundaries
    per_cond = {}
    for sys_name in ['I553A', 'I552A']:
        for iso in ['H', 'D']:
            r, t, reps = load_trajs(pdir, sys_name, iso, wins, P)
            if r is None: continue
            # Reshape to (F, N_beads) — we lost the reshape; reload per-file
            frames = []
            for rw in wins:
                max_rep = REPS_PER_WINDOW if MATCHED_DESIGN else (
                    6 if (P == 8 and rw == 'r335') else 3)
                for rep in range(1, max_rep+1):
                    if P == 8:
                        fp = pdir/f'{sys_name}_{iso}_{rw}_rep{rep}_pimd.npz'
                    else:
                        fp = pdir/f'{sys_name}_{iso}_{rw}_P{P}_rep{rep}_pimd.npz'
                    if not fp.exists(): continue
                    d = np.load(fp)
                    xH = np.asarray(d['xferH_positions']); F, N, _ = xH.shape
                    dpos = np.asarray(d['donor_pos'])[:, None, :]
                    apos = np.asarray(d['acceptor_pos'])[:, None, :]
                    v_HO = apos - xH; v_HC = dpos - xH
                    r_HO = np.linalg.norm(v_HO, axis=-1)
                    r_HC = np.linalg.norm(v_HC, axis=-1)
                    cos_t = np.einsum('fbi,fbi->fb', v_HO, v_HC) / (r_HO*r_HC + 1e-12)
                    th = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
                    frames.append((r_HO, th))
            per_cond[(sys_name, iso)] = frames
    if not per_cond or any((s, i) not in per_cond for s in ['I553A','I552A'] for i in ['H','D']):
        return None
    def sample_entropy(cond_frames):
        # Block-bootstrap frames within each trajectory, keep all beads together
        r_boot, t_boot = [], []
        for r_HO, th in cond_frames:
            F, N = r_HO.shape
            nblocks = max(F // BLOCK, 1)
            starts = rng.integers(0, F - BLOCK + 1, size=nblocks)
            idx = (starts[:, None] + np.arange(BLOCK)[None, :]).flatten()[:F]
            r_boot.append(r_HO[idx].flatten()); t_boot.append(th[idx].flatten())
        r_boot = np.concatenate(r_boot); t_boot = np.concatenate(t_boot)
        return entropy(r_boot, t_boot)
    DD_samples = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        S_I553A_H = sample_entropy(per_cond[('I553A','H')])
        S_I553A_D = sample_entropy(per_cond[('I553A','D')])
        S_I552A_H = sample_entropy(per_cond[('I552A','H')])
        S_I552A_D = sample_entropy(per_cond[('I552A','D')])
        DD_samples[b] = (S_I553A_H - S_I553A_D) - (S_I552A_H - S_I552A_D)
    return DD_samples


print(f'\n=== Block bootstrap on ΔΔ (30-frame blocks, 3000 resamples) ===')
for P, pdir, wins in [(8, P8_DIR, P8_R_WINS), (16, BIG_DIR, P16_R_WINS), (32, BIG_DIR, P32_R_WINS)]:
    DDb = block_boot_DD(P, pdir, wins)
    if DDb is None:
        continue
    convergence[f'P{P}'].update(dict(
        DD_bootstrap_mean=float(DDb.mean()),
        DD_bootstrap_sem=float(DDb.std()),
        DD_ci95_lo=float(np.percentile(DDb, 2.5)),
        DD_ci95_hi=float(np.percentile(DDb, 97.5)),
        P_sign_positive=float((DDb > 0).mean()),
    ))
    print(f'P={P}: DD = {DDb.mean():+.4f} ± {DDb.std():.4f}  '
          f'95%CI [{np.percentile(DDb,2.5):+.4f}, {np.percentile(DDb,97.5):+.4f}]  '
          f'P(sign>0)={(DDb>0).mean():.3f}')

with open(OUT/'B1_convergence_analysis.json', 'w') as f:
    json.dump(dict(
        per_condition=results,
        convergence=convergence,
        design=dict(matched=MATCHED_DESIGN, reps_per_window=REPS_PER_WINDOW,
                    note=('every P uses 3 replicas at r335 and 3 at r360 so that '
                          'only the bead count varies; the P=8 cell therefore uses '
                          'a 6-trajectory subset of the 9-trajectory production '
                          'campaign, and its Ntilde differs from the values quoted '
                          'in the RPMD section, which use all 9 and a different '
                          'histogram range')),
        histogram_range={'r_HO': r_range, 'theta_deg': t_range, 'bins': NB},
    ), f, indent=2)
print(f'\nwrote {OUT/"B1_convergence_analysis.json"}')
