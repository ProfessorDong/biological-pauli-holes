#!/usr/bin/env bash
# Wait for the stage-5 MD to finish, extract configurations, then run the SAPT pool.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -u
ROOT=${_REPO}
SL=/home/liang/anaconda3/envs/slomd/bin/python
LOG=$ROOT/results/stage5_configs/chain.log
: > "$LOG"
while pgrep -f umbrella_window_frames >/dev/null 2>&1; do sleep 60; done
echo "MD finished $(date -Iseconds)" >> "$LOG"
for t in WT I553A I552A L754A V750A I538A L546A; do
  if [[ -f "$ROOT/results/stage5_configs/${t}_frames.npy" ]]; then
    "$SL" "$ROOT/scripts/stage5_extract.py" "$t" 16 >> "$LOG" 2>&1
  else
    echo "  MISSING frames for $t" >> "$LOG"
  fi
done
echo "extraction done $(date -Iseconds)" >> "$LOG"
bash "$ROOT/scripts/stage5_run.sh" 14 >> "$LOG" 2>&1
echo "ALL STAGE 5 DONE $(date -Iseconds)" >> "$LOG"
