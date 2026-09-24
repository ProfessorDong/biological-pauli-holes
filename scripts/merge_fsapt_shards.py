#!/usr/bin/env python3
"""Merge sharded F-SAPT partial files into the single ensemble file the analyses read.

Each worker of `fsapt_ensemble.py TAG CLAMP N w/NW` writes `..._fsapt_ensemble_part{w}of{NW}.json`
holding its own slice of frames and summary statistics computed over that slice only. This merges
the slices, recomputes the summary over the full set, and writes the canonical file.

It refuses to write unless the shards together cover exactly the frame list the unsharded run
would have used, with no duplicates and no gaps, because a silently short ensemble is the failure
mode that matters here: the analyses downstream would simply average fewer points and report a
number that looks fine.

Usage: merge_fsapt_shards.py TAG CLAMP [n_frames]     (pauli env)
"""
import os as _os
_REPO = _os.environ.get('PAULI_ROOT') or _os.path.abspath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(_REPO)
OUT = ROOT / 'results/sapt_bio/donor_fragment'
ENS = ROOT / 'results/ensemble_fluctuation'


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else 'V750A'
    clamp = sys.argv[2] if len(sys.argv) > 2 else 'r255'
    nwant = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    parts = sorted(OUT.glob(f'{tag}_{clamp}_fsapt_ensemble_part*of*.json'))
    assert parts, f'no shards found for {tag} {clamp}'
    rows, seen = [], set()
    for f in parts:
        d = json.loads(f.read_text())
        for r in d['frames']:
            assert r['frame'] not in seen, f'frame {r["frame"]} appears in more than one shard'
            seen.add(r['frame'])
            rows.append(r)

    geo = json.loads((ENS / f'{tag}_{clamp}_frames_geometry.json').read_text())
    expect = set(np.linspace(0, geo['n_frames'] - 1, nwant).round().astype(int).tolist())
    missing = sorted(expect - seen)
    extra = sorted(seen - expect)
    assert not missing, f'{len(missing)} frames missing from the shards: {missing}'
    assert not extra, f'shards contain frames outside the requested list: {extra}'

    rows.sort(key=lambda r: r['frame'])
    kt = np.array([r['k_total'] for r in rows])
    kc = np.array([r['k_CH2'] for r in rows])
    share = kc / kt
    single = json.loads((OUT / f'{tag}_fsapt_decomposition.json').read_text())
    s_k = single.get('k_total_Nm', single.get('k_methane_Nm', float('nan')))
    z = float((s_k - kt.mean()) / kt.std(ddof=1)) if kt.std(ddof=1) else float('nan')

    dest = OUT / f'{tag}_{clamp}_fsapt_ensemble.json'
    dest.write_text(json.dumps(dict(
        tag=tag, clamp=clamp, basis='jun-cc-pVDZ', n_frames=len(rows), frames=rows,
        k_total_mean=float(kt.mean()), k_total_sd=float(kt.std(ddof=1)),
        k_total_min=float(kt.min()), k_total_max=float(kt.max()),
        share_mean=float(share.mean()), share_sd=float(share.std(ddof=1)),
        share_min=float(share.min()), share_max=float(share.max()),
        single_snapshot_k_total=s_k, single_snapshot_z=z,
        merged_from=[p.name for p in parts]), indent=1))
    print(f'  {tag} {clamp}: merged {len(parts)} shards, {len(rows)} frames, '
          f'k_total mean {kt.mean():.3f} sd {kt.std(ddof=1):.3f}')
    print(f'  wrote {dest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
