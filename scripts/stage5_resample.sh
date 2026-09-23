#!/usr/bin/env bash
# Stage 5: regenerate an INDEPENDENTLY SAMPLED clamped ensemble, with full coordinate frames.
#
# The published ensemble campaign stored only donor, transferring H and wall for each frame, not
# the acceptor oxygen, so the transverse plane that defines K_perp cannot be built on those frames
# and the trajectories themselves are gone. This re-runs the SAME protocol used for the published
# clamped runs (k = 150 kcal/mol/A^2, r0 = 2.55 A, 0.3 ns equilibration, 3.0 ns production) with a
# fresh RNG seed, writing full coordinates every 50 ps. The result is an independent ensemble,
# which is a stronger sensitivity test than re-using the published frames would have been.
set -u
ROOT=/home/liang/Workspace/WritePaper/CatalysisQuamBio
SL=/home/liang/anaconda3/envs/slomd/bin/python
WIN=$ROOT/scripts/umbrella_window_frames.py
MD=$ROOT/md/mcpb
OUT=$ROOT/results/stage5_configs
LOG=$OUT/campaign.log
mkdir -p "$OUT"
echo "=== stage 5 resampling start $(date -Iseconds) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tee -a "$LOG"

#      tag     prmtop                donor  acceptor
run () {
  tag=$1; prm=$2; d=$3; a=$4
  if [[ -f "$OUT/${tag}_frames.npy" ]]; then echo "  skip $tag (exists)" | tee -a "$LOG"; return; fi
  echo "--- $(date -Iseconds) $tag ---" | tee -a "$LOG"
  timeout 5400 "$SL" "$WIN" --prmtop "$MD/$prm.prmtop" \
      --seed "$ROOT/results/reactive_geometry/${tag}_r255_final.rst7" \
      --donor "$d" --acceptor "$a" --r0 2.55 --k 150.0 \
      --ns-eq 0.3 --ns-prod 3.0 --frame-stride 50 \
      --out "$OUT/$tag" --rngseed $((RANDOM+1)) 2>&1 | tee -a "$LOG" | tail -3
}

run WT    SLO_sub_solv       13002 12985
run I553A SLO_I553A_sub_solv 12993 12976
run I552A SLO_I552A_sub_solv 12993 12976
run L754A SLO_L754A_sub_solv 12993 12976
run V750A SLO_V750A_sub_solv 12996 12979
run I538A SLO_I538A_sub_solv 12993 12976
run L546A SLO_L546A_sub_solv 12993 12976
echo "=== done $(date -Iseconds) ===" | tee -a "$LOG"
