#!/usr/bin/env bash
# Crash-resilient stage-5 MD: run each system as a chain of short segments.
#
# The machine has died twice, both times during sustained GPU MD, with no kernel diagnostic.
# umbrella_window_frames.py writes its frames only after the production loop finishes, so a crash
# costs the whole system. Running in segments, each seeded from the previous segment's restart,
# bounds the loss to one segment and makes resume automatic: a segment whose frames file already
# exists is skipped.
#
# Usage: stage5_md_chunked.sh TAG PRMTOP DONOR ACCEPTOR N_SEGMENTS [NS_PER_SEG]
set -u
ROOT=/home/liang/Workspace/WritePaper/CatalysisQuamBio
SL=/home/liang/anaconda3/envs/slomd/bin/python
WIN=$ROOT/scripts/umbrella_window_frames.py
MD=$ROOT/md/mcpb
OUT=$ROOT/results/stage5_configs
LOG=$OUT/chunked.log

tag=$1; prm=$2; d=$3; a=$4; nseg=$5; ns=${6:-1.0}
seed=$ROOT/results/reactive_geometry/${tag}_r255_final.rst7

for ((i=0; i<nseg; i++)); do
  seg=$(printf '%s_seg%02d' "$tag" "$i")
  if [[ -f "$OUT/${seg}_frames.npy" ]]; then
    echo "  skip $seg (exists)" | tee -a "$LOG"
    seed=$OUT/${seg}_final.rst7
    continue
  fi
  echo "--- $(date -Iseconds) $seg  (seed $(basename "$seed")) ---" | tee -a "$LOG"
  # 0.05 ns equilibration only after the first segment: the state is already equilibrated.
  eq=$( ((i==0)) && echo 0.3 || echo 0.05 )
  timeout 1800 "$SL" "$WIN" --prmtop "$MD/$prm.prmtop" --seed "$seed" \
      --donor "$d" --acceptor "$a" --r0 2.55 --k 150.0 \
      --ns-eq "$eq" --ns-prod "$ns" --frame-stride 50 \
      --out "$OUT/$seg" --rngseed $((RANDOM+1)) 2>&1 | tee -a "$LOG" | tail -2
  if [[ ! -f "$OUT/${seg}_frames.npy" ]]; then
    echo "  FAILED at $seg, stopping; rerun this script to resume here" | tee -a "$LOG"; exit 1
  fi
  seed=$OUT/${seg}_final.rst7
done
echo "  $tag complete: $nseg segments" | tee -a "$LOG"
