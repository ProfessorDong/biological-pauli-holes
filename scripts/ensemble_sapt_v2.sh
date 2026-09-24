#!/bin/bash
# Ensemble SAPT, v2: ONE PROCESS PER FRAME.
#
# v1 (sapt_frames.py) ran 40 frames in a persistent psi4 session to save startup
# cost. It wedged after 14.9 h without completing frame 0 of WT r340; the same
# frame finishes in <30 s with a fresh process, so the geometry was fine and the
# accumulated session state was not. We revert to the pattern already proven by
# the static test (14/14 successes at ~25 s each) and accept the startup cost.
#
# Each frame is additionally wrapped in `timeout 300` so a single bad SCF costs
# five minutes rather than a night, and the driver reports how many timed out.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -uo pipefail
ROOT=${_REPO}
SL="${SLOMD_PYTHON:-/home/liang/anaconda3/envs/slomd/bin/python}"
P4=${PAULI_PYTHON:-/home/liang/anaconda3/envs/pauli/bin/python}
EF="$ROOT/results/ensemble_fluctuation"
LOG="$EF/sapt_v2.log"
NFR=40
echo "=== ensemble SAPT v2 start $(date -Iseconds) ===" | tee -a "$LOG"

for CLAMP in r255 r340; do
  for TAG in WT V750A I552A I538A L754A L546A I553A; do
    AGG="$EF/${TAG}_${CLAMP}_kexch_frames.json"
    [[ -f "$AGG" ]] && { echo "  skip $TAG $CLAMP (v1 result exists)" | tee -a "$LOG"; continue; }
    [[ -f "$EF/${TAG}_${CLAMP}_frames.npy" ]] || { echo "  SKIP $TAG $CLAMP: no frames" | tee -a "$LOG"; continue; }
    [[ -f "$EF/${TAG}_${CLAMP}_frames_geometry.json" ]] || \
      "$SL" "$ROOT/scripts/extract_frames_geometry.py" "$TAG" "$CLAMP" 2>&1 | tee -a "$LOG"
    echo "--- $(date -Iseconds) $TAG $CLAMP ($NFR frames) ---" | tee -a "$LOG"
    nto=0
    for ((k=0;k<NFR;k++)); do
      f=$(printf "%s/%s_%s_f%03d.json" "$EF" "$TAG" "$CLAMP" $k)
      [[ -f "$f" ]] && continue
      timeout 300 "$P4" "$ROOT/scripts/sapt_one_frame.py" "$TAG" "$CLAMP" "$k" >/dev/null 2>&1 \
        || { nto=$((nto+1)); echo "    frame $k TIMEOUT/FAIL" | tee -a "$LOG"; }
    done
    # aggregate this system's frames
    "$SL" - "$TAG" "$CLAMP" <<'PY' 2>&1 | tee -a "$LOG"
import json,sys,glob
from pathlib import Path
TAG,CLAMP=sys.argv[1],sys.argv[2]
EF=Path('${_REPO}/results/ensemble_fluctuation')
ks=[]
# CORRECTION 2026-09-17: 'f*' also matched {TAG}_{CLAMP}_frames_geometry.json, which has
# no k_exch_Nm key, so this block raised KeyError and no per-system aggregate was written.
# The bug was latent: every other system skipped this path because its aggregate already
# existed from the v1 run. Match the frame index digits explicitly.
for f in sorted(EF.glob(f'{TAG}_{CLAMP}_f[0-9][0-9][0-9].json')):
    ks.append(json.loads(f.read_text())['k_exch_Nm'])
if ks:
    import statistics as st
    (EF/f'{TAG}_{CLAMP}_kexch_frames.json').write_text(json.dumps(
        dict(TAG=TAG,clamp=CLAMP,n_ok=len(ks),k_exch_Nm=ks),indent=1))
    print(f'  {TAG} {CLAMP}: n={len(ks)}  <k>={st.mean(ks):.4g}  sd={st.stdev(ks) if len(ks)>1 else 0:.4g} N/m')
PY
    [[ $nto -gt 0 ]] && echo "    ($nto frames timed out)" | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
echo "=== ensemble SAPT v2 done $(date -Iseconds) ===" | tee -a "$LOG"
"$SL" - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
EF=Path('${_REPO}/results/ensemble_fluctuation')
for clamp in ('r255','r340'):
    out={}
    for f in sorted(EF.glob(f'*_{clamp}_kexch_frames.json')):
        d=json.loads(f.read_text()); out[d['TAG']]=d['k_exch_Nm']
    (EF/f'kexch_frames_{clamp}.json').write_text(json.dumps(out,indent=1))
    print(f'  kexch_frames_{clamp}.json: {len(out)}/7 systems, frames {[len(v) for v in out.values()]}')
PY
