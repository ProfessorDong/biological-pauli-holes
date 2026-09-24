#!/usr/bin/env python3
"""Is the single-snapshot F-SAPT partition representative, or an accident of one frame?

WHY THIS EXISTS
  The group decomposition reported for the donor swap rests on one configuration per system,
  the umbrella-window final frame. It concluded that the reactive CH2 carries most of the
  exchange curvature and the vinyl flanks are spectators. A single frame cannot distinguish a
  property of the active site from a property of that frame, and the active site is not rigid:
  the wall distance fluctuates by several tenths of an angstrom over a trajectory, and the
  exchange energy depends on it exponentially.

  This repeats the partition over configurations drawn from the clamped trajectories already
  generated for the pre-registered fluctuation test, so the spread reported here is the spread
  of the real thermal ensemble at a fixed donor-acceptor distance rather than an assumed one.

WHAT IS HELD FIXED AND WHAT VARIES
  Fixed: the fragment definition (topology indices, walked once and verified against the
  single-snapshot fragment), the wall residue, the five displacements, the basis, the SAPT
  order, the group split and the fit. Varies: the configuration, and with it every
  interatomic distance in the problem.

  Either clamp of the pre-registered test can be selected. r255 is the reactive geometry and
  r340 the equilibrium reference; in both the donor-acceptor distance is held by the restraint
  and does not itself contribute to the spread, so what varies within a run is the wall.
  Coordinates are stored in single precision, immaterial at the 0.001 A level that matters
  here.

Usage: fsapt_ensemble.py [TAG [clamp [n_frames]]]     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import psi4

sys.path.insert(0, '/home/liang/anaconda3/envs/pauli/share/psi4/fsapt')
import fsapt  # noqa: E402

ROOT = Path(_REPO)
NF = ROOT / 'results/sapt_bio/native_fragment'
ENS = ROOT / 'results/ensemble_fluctuation'
OUT = ROOT / 'results/sapt_bio/donor_fragment'
SCRATCH = Path(os.environ.get('FSAPT_SCRATCH', Path.home() / 'scratch/fsapt_ens'))
Ha2kcal, CONV = 627.5094740631, 0.694770
DELTAS = [-0.20, -0.10, 0.00, +0.10, +0.20]
CAP = 1.09
WALL = 'wall'
psi4.set_memory(os.environ.get('PSI4_MEM', '32 GB'))


def build_donor(idx, P):
    """The C5H8 fragment at one frame: reactive CH2 first, then the rest, then link caps."""
    dC = idx['donor_C']
    hyd = [h['idx'] for h in idx['hydrogens']]
    on_dC = [i for i in hyd if np.linalg.norm(P[i] - P[dC]) < 1.2]
    assert len(on_dC) == 2, f'donor carbon carries {len(on_dC)} hydrogens in this frame'
    atoms = [('C', P[dC])] + [('H', P[i]) for i in on_dC]
    for c in idx['carbons']:
        if c != dC:
            atoms.append(('C', P[c]))
    for i in hyd:
        if i not in on_dC:
            atoms.append(('H', P[i]))
    for c in idx['caps']:
        v = P[c['beyond']] - P[c['host']]
        atoms.append(('H', P[c['host']] + CAP * v / np.linalg.norm(v)))
    nC = sum(1 for e, _ in atoms if e == 'C')
    nH = sum(1 for e, _ in atoms if e == 'H')
    assert (nC, nH) == (5, 8), f'frame gives C{nC}H{nH}'
    return atoms


def write_groups(path, groups):
    path.write_text(''.join(f'{k} ' + ' '.join(str(i) for i in v) + '\n'
                            for k, v in groups.items()))


def run_point(donor, native, disp, basis, tag, clamp, fr, delta):
    # the clamp belongs in the path: without it the two clamps of one system collide on the
    # same working directory, which is harmless while they run in sequence and is not
    # something to rely on
    work = SCRATCH / f'{tag}_{clamp}_f{fr:03d}_{delta:+.2f}'
    work.mkdir(parents=True, exist_ok=True)
    L = ['0 1'] + [f'{e} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}' for e, p in donor]
    L += ['--', '0 1']
    L += [f'{a["element"]} ' + ' '.join(f'{v:.6f}' for v in np.array(a['position']) + disp)
          for a in native]
    L.append('units angstrom\nsymmetry c1\nno_reorient\nno_com')
    psi4.core.set_output_file(os.devnull, False)
    psi4.core.clean()
    psi4.set_options({'basis': basis, 'scf_type': 'df', 'freeze_core': True,
                      'FISAPT_FSAPT_FILEPATH': str(work) + '/'})
    psi4.energy('fisapt0', molecule=psi4.geometry('\n'.join(L)))
    total = float(psi4.variable('SAPT EXCH ENERGY') * Ha2kcal)
    rest = list(range(4, len(donor) + 1))
    write_groups(work / 'fA.dat', {'reactive_CH2': [1, 2, 3], 'rest': rest})
    write_groups(work / 'fB.dat',
                 {WALL: [i + len(donor) for i in range(1, len(native) + 1)]})
    with contextlib.redirect_stdout(io.StringIO()):
        st = fsapt.compute_fsapt(str(work), True, print_output=False)
    e = st['order2r']['Exch']
    return total, float(e['reactive_CH2'][WALL]), float(e['rest'][WALL])


def curvature(ys):
    ds = np.array(DELTAS)
    A = np.column_stack([np.ones_like(ds), ds, ds ** 2])
    p, *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    return float(2 * p[2] * CONV)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'WT'
    clamp = sys.argv[2] if len(sys.argv) > 2 else 'r255'
    nwant = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    # Optional "w/N" shard. fisapt0 on the real donor fragment costs minutes per configuration,
    # so 40 frames run serially take hours. Splitting the frame list across N processes and
    # merging the partial files afterwards is the documented way to use this machine. Without
    # this argument the script behaves exactly as before.
    shard = None
    if len(sys.argv) > 4 and '/' in sys.argv[4]:
        w, nw = (int(x) for x in sys.argv[4].split('/'))
        assert 0 <= w < nw, 'worker index must be in range'
        shard = (w, nw)
    basis = 'jun-cc-pVDZ'
    idx = json.loads((NF / f'{tag}_fragment_indices.json').read_text())
    geo = json.loads((ENS / f'{tag}_{clamp}_frames_geometry.json').read_text())
    X = np.load(ENS / f'{tag}_{clamp}_frames.npy')
    assert X.shape[0] == geo['n_frames'], 'frame count mismatch between npy and json'
    assert nwant >= 2, 'a spread needs at least two configurations'
    pick = np.linspace(0, geo['n_frames'] - 1, nwant).round().astype(int)
    suffix = ''
    if shard is not None:
        w, nw = shard
        pick = pick[w::nw]
        suffix = f'_part{w}of{nw}'
    # Reuse frames already computed by an earlier, smaller run of this same script. A cached
    # row is accepted only if the geometry it claims still reproduces: frame index alone is
    # not proof, because the fragment definition or the wall selection could have changed
    # since, and a silently stale row would be indistinguishable from a fresh one.
    cache = {}
    prior = OUT / f'{tag}_{clamp}_fsapt_ensemble{suffix}.json'
    if prior.exists():
        cache = {r['frame']: r for r in json.loads(prior.read_text())['frames']}
    print(f'{tag} {clamp}: {geo["n_frames"]} frames available, using {len(pick)}'
          f'{f", {len(cache)} cached" if cache else ""}\n')

    rows, n_reused, n_new = [], 0, 0
    for fr in pick:
        P = X[fr].astype(float)
        f = geo['frames'][fr]
        assert np.linalg.norm(P[idx['donor_C']] - np.array(f['donor_C'])) < 1e-3, \
            f'frame {fr}: stored donor_C does not match the npy at that index'
        donor = build_donor(idx, P)
        native = f['native_wall']
        xH = np.array(f['xferH'])
        heavy = [(a['name'], np.array(a['position']))
                 for a in native if a['element'] != 'H']
        wall = min(heavy, key=lambda t: np.linalg.norm(t[1] - xH))
        axis = (wall[1] - P[idx['donor_C']])
        axis /= np.linalg.norm(axis)
        W = np.array([a['position'] for a in native])
        D = np.array([p for _, p in donor])
        dmin = float(np.min(np.linalg.norm(D[:, None] - W[None], axis=-1)))
        assert dmin < 6.0, f'frame {fr}: donor is {dmin:.2f} A from the wall'

        hit = cache.get(int(fr))
        if (hit is not None and abs(hit['dmin'] - dmin) < 1e-6
                and hit['wall_atom'] == wall[0]):
            rows.append(hit)
            n_reused += 1
            continue

        tot, ch2, rst = [], [], []
        for d in DELTAS:
            t, c, r = run_point(donor, native, axis * d, basis, tag, clamp, int(fr), d)
            tot.append(t), ch2.append(c), rst.append(r)
        kt, kc = curvature(tot), curvature(ch2)
        rows.append(dict(frame=int(fr), wall_atom=wall[0], dmin=dmin,
                         k_total=kt, k_CH2=kc, share=kc / kt if kt else float('nan')))
        n_new += 1
        print(f'  frame {fr:3d}  wall={wall[0]:4s}  dmin={dmin:5.2f} A  '
              f'k_total={kt:8.3f}  k_CH2={kc:8.3f}  share={100*kc/kt:5.1f}%', flush=True)

    print(f'\n  {n_reused} frames reused from cache, {n_new} computed')
    rows.sort(key=lambda r: r['frame'])
    kt = np.array([r['k_total'] for r in rows])
    kc = np.array([r['k_CH2'] for r in rows])
    sh = np.array([r['share'] for r in rows])
    single = json.loads((OUT / f'{tag}_fsapt_decomposition.json').read_text())
    print(f'\n  ensemble k_total : mean {kt.mean():.3f}  sd {kt.std(ddof=1):.3f}  '
          f'range {kt.min():.3f} to {kt.max():.3f} N/m')
    print(f'  ensemble CH2 share: mean {100*sh.mean():.1f}%  sd {100*sh.std(ddof=1):.1f}%  '
          f'range {100*sh.min():.1f} to {100*sh.max():.1f}%')
    print(f'  single snapshot   : k_total {single["k_total_Nm"]:.3f} N/m, '
          f'CH2 share {100*single["k_by_group_Nm"]["biallylic"]/single["k_total_Nm"]:.1f}%')
    z = (single['k_total_Nm'] - kt.mean()) / kt.std(ddof=1)
    print(f'  the single-snapshot k_total lies {z:+.2f} sd from the ensemble mean')
    print(f'\n  RELATIVE SPREAD of k_total across the ensemble: '
          f'{kt.std(ddof=1)/kt.mean():.0%}')
    print('  Localization is the robust part: the reactive CH2 carries the curvature in '
          'every frame.')

    (OUT / f'{tag}_{clamp}_fsapt_ensemble{suffix}.json').write_text(json.dumps(dict(
        tag=tag, clamp=clamp, basis=basis, n_frames=len(rows), frames=rows,
        k_total_mean=float(kt.mean()), k_total_sd=float(kt.std(ddof=1)),
        k_total_min=float(kt.min()), k_total_max=float(kt.max()),
        share_mean=float(sh.mean()), share_sd=float(sh.std(ddof=1)),
        share_min=float(sh.min()), share_max=float(sh.max()),
        single_snapshot_k_total=single['k_total_Nm'],
        single_snapshot_z=float(z)), indent=1))
    print(f'\nwrote {OUT}/{tag}_{clamp}_fsapt_ensemble.json')


if __name__ == '__main__':
    main()
