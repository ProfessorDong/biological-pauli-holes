#!/usr/bin/env bash
# Fixed-size worker pool over every stage-5 configuration.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -u
ROOT=${_REPO}
PY="${PAULI_PYTHON:-/home/liang/anaconda3/envs/pauli/bin/python}"
N=${1:-14}
export OMP_NUM_THREADS=1
mapfile -t JOBS < <(ls $ROOT/results/stage5_configs/*/*_native_donor.json 2>/dev/null)
echo "queue: ${#JOBS[@]} configurations, $N workers"
for f in "${JOBS[@]}"; do
  while [[ $(jobs -rp | wc -l) -ge $N ]]; do sleep 5; done
  "$PY" -u "$ROOT/scripts/stage5_transverse.py" "$f" 0.15 5 \
      >> "$ROOT/results/stage5_configs/sapt.log" 2>&1 &
done
wait
echo "all configurations done $(date -Iseconds)"
