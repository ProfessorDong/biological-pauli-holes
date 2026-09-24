#!/bin/bash
# Ensemble-fluctuation SAPT campaign: k_exch for every saved frame.
#
# Waits for the frame-generating MD campaign to finish, then runs the audited
# SAPT0/jun-cc-pVDZ descriptor once per frame. 7 systems x 2 clamps x 40 frames
# x 5 scan points = 2800 SAPT calculations, ~3 h. Collates into the two files the
# pre-registered analysis reads.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -uo pipefail
ROOT=${_REPO}
SL=/home/liang/anaconda3/envs/slomd/bin/python
P4=/home/liang/anaconda3/envs/pauli/bin/python
EF="$ROOT/results/ensemble_fluctuation"
LOG="$EF/sapt_campaign.log"
mkdir -p "$EF"

echo "=== waiting for frame MD to finish ($(date -Iseconds)) ===" | tee -a "$LOG"
while pgrep -f ensemble_fluctuation_campaign.sh >/dev/null 2>&1; do sleep 60; done
echo "=== MD done; $(ls "$EF"/*_frames.npy 2>/dev/null | wc -l)/14 frame sets present ===" | tee -a "$LOG"

echo "=== ensemble SAPT start $(date -Iseconds) ===" | tee -a "$LOG"
for CLAMP in r255 r340; do
  for TAG in WT V750A I552A I538A L754A L546A I553A; do
    OUTF="$EF/${TAG}_${CLAMP}_kexch_frames.json"
    [[ -f "$OUTF" ]] && { echo "  skip $TAG $CLAMP (done)" | tee -a "$LOG"; continue; }
    [[ -f "$EF/${TAG}_${CLAMP}_frames.npy" ]] || {
      echo "  SKIP $TAG $CLAMP: no frames.npy" | tee -a "$LOG"; continue; }
    echo "" | tee -a "$LOG"
    echo "--- $(date -Iseconds) $TAG $CLAMP ---" | tee -a "$LOG"
    "$SL" "$ROOT/scripts/extract_frames_geometry.py" "$TAG" "$CLAMP" 2>&1 | tee -a "$LOG"
    [[ -f "$EF/${TAG}_${CLAMP}_frames_geometry.json" ]] || {
      echo "  extract FAILED" | tee -a "$LOG"; continue; }
    "$P4" "$ROOT/scripts/sapt_frames.py" "$TAG" "$CLAMP" 2>&1 | tee -a "$LOG" | tail -4
  done
done

echo "" | tee -a "$LOG"
echo "=== ensemble SAPT done $(date -Iseconds) ===" | tee -a "$LOG"

# collate into the exact filenames the pre-registered analysis expects
"$SL" - <<'PY' 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
EF = Path('${_REPO}/results/ensemble_fluctuation')
for clamp in ('r255','r340'):
    out = {}
    for f in sorted(EF.glob(f'*_{clamp}_kexch_frames.json')):
        d = json.loads(f.read_text())
        out[d['TAG']] = d['k_exch_Nm']
    (EF/f'kexch_frames_{clamp}.json').write_text(json.dumps(out, indent=1))
    n = {k: len(v) for k, v in out.items()}
    print(f'  kexch_frames_{clamp}.json: {len(out)}/7 systems, frames per system {n}')
PY
