#!/bin/bash
# Waits for the ensemble SAPT campaign to finish, then runs the PRE-REGISTERED
# analysis exactly once, unmodified. Chained so the result does not depend on an
# interactive session staying alive.
set -uo pipefail
ROOT=/home/liang/Workspace/WritePaper/CatalysisQuamBio
EF="$ROOT/results/ensemble_fluctuation"
LOG="$EF/analysis_run.log"

echo "=== waiting for ensemble SAPT ($(date -Iseconds)) ===" | tee -a "$LOG"
while pgrep -f ensemble_sapt_campaign.sh >/dev/null 2>&1; do sleep 60; done

# require both collated inputs before analysing
for f in kexch_frames_r255.json kexch_frames_r340.json; do
  if [[ ! -f "$EF/$f" ]]; then
    echo "MISSING $f -- SAPT did not collate; analysis NOT run" | tee -a "$LOG"; exit 1
  fi
done
echo "=== inputs present; running pre-registered analysis $(date -Iseconds) ===" | tee -a "$LOG"
/home/liang/anaconda3/envs/slomd/bin/python "$ROOT/scripts/analyze_ensemble_fluctuation.py" 2>&1 | tee -a "$LOG"
echo "=== analysis complete $(date -Iseconds) ===" | tee -a "$LOG"
