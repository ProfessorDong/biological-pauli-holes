#!/usr/bin/env python3
"""Histogram-range dependence of the RPMD isotope shift ln(N_H/N_D) on the I553A/I552A pair.

Audit producer, 2026-11-25. Reads results/pimd_prod/ and results/pimd_convergence/ (primary
data) and writes results/audit_2026_11/rpmd_range.json only.

np.histogram2d(range=...) discards samples outside the range. The reported analyses
(scripts/pimd_analysis_v2.py for the 36-trajectory production campaign, and
scripts/pimd_convergence_analysis.py for the matched bead sweep) set the range to the 1st and
99th percentiles of the pooled data, 24 x 24 bins. This script
  1. reproduces the reported point estimates with that range,
  2. records the fraction of samples each cell loses to it,
  3. recomputes the estimates with wider ranges at the SAME bin width (0.5-99.5, 0.1-99.9 and
     0-100 percentiles; 0-100 keeps every sample), and
  4. repeats the frame-level block bootstrap of each analysis (production: 30-frame blocks over
     the concatenated frames, 5000 resamples, seed 20260803; sweep: 30-frame blocks within each
     trajectory, 3000 resamples, seed 20260806) for the reported range and for all samples kept,
     with the same random block draws, so the two ranges are compared on identical resamples.
N_i = exp(-sum_b p_b ln p_b) is the exponentiated Shannon entropy of the binned bead positions,
an effective number of occupied bins; ln(N_H/N_D) is the difference of the two entropies.
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
from pathlib import Path
import numpy as np

RES = Path(_REPO) / 'results'
PROD, CONV = RES / 'pimd_prod', RES / 'pimd_convergence'
OUT = RES / 'audit_2026_11' / 'rpmd_range.json'
NB = 24
CELLS = [('I553A', 'H'), ('I553A', 'D'), ('I552A', 'H'), ('I552A', 'D')]
RANGES = [(1.0, 99.0), (0.5, 99.5), (0.1, 99.9), (0.0, 100.0)]


def geom(fp):
    d = np.load(fp)
    xH = np.asarray(d['xferH_positions'])
    dp = np.asarray(d['donor_pos'])[:, None, :]; ap = np.asarray(d['acceptor_pos'])[:, None, :]
    vHO = ap - xH; vHC = dp - xH
    rHO = np.linalg.norm(vHO, axis=-1); rHC = np.linalg.norm(vHC, axis=-1)
    th = np.degrees(np.arccos(np.clip(np.einsum('fbi,fbi->fb', vHO, vHC) / (rHO * rHC + 1e-12), -1, 1)))
    return rHO, th                                     # (frames, beads)


def prod_trajs(s, iso):
    out = []
    for w in ('r335', 'r360'):
        for rep in range(1, 7):
            fp = PROD / f'{s}_{iso}_{w}_rep{rep}_pimd.npz'
            if fp.exists():
                out.append(geom(fp))
    return out


def sweep_trajs(s, iso, P):
    out = []
    for w in ('r335', 'r360'):
        for rep in (1, 2, 3):
            fp = (PROD / f'{s}_{iso}_{w}_rep{rep}_pimd.npz') if P == 8 else (CONV / f'{s}_{iso}_{w}_P{P}_rep{rep}_pimd.npz')
            if fp.exists():
                out.append(geom(fp))
    return out


class Hist:
    """Histogram geometry: a percentile range of the pooled data, at the bin width of the 1-99 grid."""
    def __init__(self, rall, tall, lo, hi):
        r1 = [np.percentile(rall, 1), np.percentile(rall, 99)]; t1 = [np.percentile(tall, 1), np.percentile(tall, 99)]
        wr, wt = (r1[1] - r1[0]) / NB, (t1[1] - t1[0]) / NB
        self.rr = [float(np.percentile(rall, lo)), float(np.percentile(rall, hi))]
        self.tr = [float(np.percentile(tall, lo)), float(np.percentile(tall, hi))]
        if (lo, hi) == (1.0, 99.0):
            self.nb = [NB, NB]
        else:
            self.nb = [max(1, int(round((self.rr[1] - self.rr[0]) / wr))), max(1, int(round((self.tr[1] - self.tr[0]) / wt)))]

    def entropy(self, r, t):
        H, _, _ = np.histogram2d(r.ravel(), t.ravel(), bins=self.nb, range=[self.rr, self.tr])
        n = H.sum(); p = H / (n + 1e-30); nz = p > 0
        return float(-np.sum(p[nz] * np.log(p[nz]))), float(1.0 - n / r.size)


def summ(x):
    return dict(mean=float(x.mean()), sd=float(x.std()), ci95=[float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
                p_positive=float((x > 0).mean()))


def production():
    data = {}
    for c in CELLS:
        tr = prod_trajs(*c)
        data[c] = (np.concatenate([a for a, _ in tr]), np.concatenate([b for _, b in tr]), len(tr))
    rall = np.concatenate([v[0].ravel() for v in data.values()]); tall = np.concatenate([v[1].ravel() for v in data.values()])
    out = dict(n_trajectories={f'{s}_{i}': data[(s, i)][2] for s, i in CELLS},
               samples_per_cell={f'{s}_{i}': int(data[(s, i)][0].size) for s, i in CELLS}, ranges={})
    hists = {}
    for lo, hi in RANGES:
        h = Hist(rall, tall, lo, hi); hists[(lo, hi)] = h
        S = {c: h.entropy(data[c][0], data[c][1]) for c in CELLS}
        d53 = S[('I553A', 'H')][0] - S[('I553A', 'D')][0]; d52 = S[('I552A', 'H')][0] - S[('I552A', 'D')][0]
        out['ranges'][f'{lo:g}-{hi:g}'] = dict(
            bins=h.nb, r_HO_range_A=h.rr, theta_range_deg=h.tr,
            discarded_fraction={f'{s}_{i}': S[(s, i)][1] for s, i in CELLS},
            ln_N_H_over_D={'I553A': d53, 'I552A': d52}, contrast_I553A_minus_I552A=d53 - d52)
        print(f'  production {lo:g}-{hi:g}: bins {h.nb}  I553A {d53:+.4f}  I552A {d52:+.4f}  DD {d53 - d52:+.4f}  '
              f'dropped {[round(S[c][1], 4) for c in CELLS]}', flush=True)
    # block bootstrap, pimd_analysis_v2.py scheme, identical draws for both ranges
    BLOCK, NBOOT = 30, 5000
    for key in ((1.0, 99.0), (0.0, 100.0)):
        rng = np.random.default_rng(20260803)
        h = hists[key]; boot = {}
        for c in CELLS:
            r, t, _ = data[c]; F = r.shape[0]; nbk = F // BLOCK; vals = np.zeros(NBOOT)
            for b in range(NBOOT):
                st = rng.integers(0, F - BLOCK + 1, size=nbk)
                idx = (st[:, None] + np.arange(BLOCK)[None, :]).flatten()[:F]
                vals[b] = h.entropy(r[idx], t[idx])[0]
            boot[c] = vals
        d53 = boot[('I553A', 'H')] - boot[('I553A', 'D')]; d52 = boot[('I552A', 'H')] - boot[('I552A', 'D')]
        out['ranges'][f'{key[0]:g}-{key[1]:g}']['block_bootstrap'] = dict(
            block_frames=BLOCK, n_boot=NBOOT, seed=20260803,
            I553A=summ(d53), I552A=summ(d52), contrast=summ(d53 - d52))
        print(f'  production bootstrap {key}: I553A {summ(d53)}  I552A {summ(d52)}  DD {summ(d53 - d52)}', flush=True)
    return out


def sweep():
    data = {}
    for P in (8, 16, 32):
        for c in CELLS:
            data[(P,) + c] = sweep_trajs(c[0], c[1], P)
    rall = np.concatenate([a.ravel() for v in data.values() for a, _ in v])
    tall = np.concatenate([b.ravel() for v in data.values() for _, b in v])
    out = dict(n_trajectories={f'P{k[0]}_{k[1]}_{k[2]}': len(v) for k, v in data.items()}, ranges={})
    for lo, hi in ((1.0, 99.0), (0.0, 100.0)):
        h = Hist(rall, tall, lo, hi); per = {}
        for P in (8, 16, 32):
            S = {c: h.entropy(np.concatenate([a.ravel() for a, _ in data[(P,) + c]]),
                              np.concatenate([b.ravel() for _, b in data[(P,) + c]])) for c in CELLS}
            d53 = S[('I553A', 'H')][0] - S[('I553A', 'D')][0]; d52 = S[('I552A', 'H')][0] - S[('I552A', 'D')][0]
            # bootstrap as pimd_convergence_analysis.py: blocks within each trajectory, one rng per P
            rng = np.random.default_rng(20260806); NBOOT, BLOCK = 3000, 30; DD = np.zeros(NBOOT)
            for b in range(NBOOT):
                ent = {}
                for c in CELLS:
                    rb, tb = [], []
                    for r, t in data[(P,) + c]:
                        F = r.shape[0]; st = rng.integers(0, F - BLOCK + 1, size=max(F // BLOCK, 1))
                        idx = (st[:, None] + np.arange(BLOCK)[None, :]).flatten()[:F]
                        rb.append(r[idx].ravel()); tb.append(t[idx].ravel())
                    ent[c] = h.entropy(np.concatenate(rb), np.concatenate(tb))[0]
                DD[b] = (ent[('I553A', 'H')] - ent[('I553A', 'D')]) - (ent[('I552A', 'H')] - ent[('I552A', 'D')])
            per[f'P{P}'] = dict(ln_N_H_over_D={'I553A': d53, 'I552A': d52}, contrast_I553A_minus_I552A=d53 - d52,
                                discarded_fraction={f'{s}_{i}': S[(s, i)][1] for s, i in CELLS},
                                contrast_block_bootstrap=dict(block_frames=BLOCK, n_boot=NBOOT, seed=20260806, **summ(DD)))
            print(f'  sweep {lo:g}-{hi:g} P{P}: I553A {d53:+.4f}  I552A {d52:+.4f}  DD {d53 - d52:+.4f}  '
                  f'CI {summ(DD)["ci95"]}  dropped {[round(S[c][1], 4) for c in CELLS]}', flush=True)
        out['ranges'][f'{lo:g}-{hi:g}'] = dict(bins=h.nb, r_HO_range_A=h.rr, theta_range_deg=h.tr, per_P=per)
    return out


def main():
    res = dict(note=('N = exp of the Shannon entropy of the binned bead positions; wider ranges keep the '
                     'bin width of the 1-99 percentile 24 x 24 grid; 0-100 keeps every sample'),
               production_P8=production(), bead_sweep_matched=sweep())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
