#!/usr/bin/env bash
# Finish the stage-5 MD for the two systems the crash left without data, in 1 ns segments.
# Each segment is seeded from the previous one's restart, so a crash costs one segment, not a system.
# Segments are concatenated into <TAG>_frames.npy, the single file stage5_extract.py reads.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -u
ROOT=${_REPO}
SL=/home/liang/anaconda3/envs/slomd/bin/python
OUT=$ROOT/results/stage5_configs
LOG=$OUT/chunked.log

run_one() {  # tag prmtop donor acceptor
  local tag=$1
  bash "$ROOT/scripts/stage5_md_chunked.sh" "$@" 3 1.0 || { echo "  $tag INCOMPLETE" | tee -a "$LOG"; return 1; }
  "$SL" - "$tag" <<'PY' | tee -a "$LOG"
import sys, numpy as np, pathlib
tag = sys.argv[1]
out = pathlib.Path('${_REPO}/results/stage5_configs')
segs = sorted(out.glob(f'{tag}_seg*_frames.npy'))
arrs = [np.load(s) for s in segs]
allf = np.concatenate(arrs, axis=0)
np.save(out / f'{tag}_frames.npy', allf)
print(f"  {tag}: {len(segs)} segments -> {allf.shape[0]} frames {allf.shape}")
# restraint check, same quantity the completed systems report
cv = np.concatenate([np.loadtxt(out / f'{s.name.replace("_frames.npy","")}_colvar.dat') for s in segs])
print(f"  {tag}: <r_DA> = {cv.mean():.3f} +/- {cv.std():.3f} A  (n={len(cv)})")
PY
}

echo "=== $(date -Iseconds) finishing stage-5 MD: I538A, L546A ===" | tee -a "$LOG"
run_one I538A SLO_I538A_sub_solv 12993 12976
run_one L546A SLO_L546A_sub_solv 12993 12976
echo "=== $(date -Iseconds) stage-5 MD complete ===" | tee -a "$LOG"
