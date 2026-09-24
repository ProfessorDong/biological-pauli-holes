#!/bin/bash
# Stage 4 checks 3, 4, 5, 6, queued behind the grid sweeps.
# Each run is one process per grid point (psi4 is unreliable in a forked pool here), resumable
# because a point whose file exists is skipped. Worker counts are set by per-point memory:
# aug-cc-pVTZ has nbf 828 and needs the most, so it gets the fewest workers.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -u
ROOT=${_REPO}
PY="${PAULI_PYTHON:-/home/liang/anaconda3/envs/pauli/bin/python}"
L="$1"
export OMP_NUM_THREADS=1

run() {  # label tag method basis donor wall half npa nworkers mem
  local label=$1 tag=$2 meth=$3 bas=$4 don=$5 wal=$6 half=$7 npa=$8 nw=$9 mem=${10}
  echo "--- $(date -Iseconds) $label ($meth/$bas donor=$don wall=$wal ${npa}x${npa}) ---" >> "$L/remaining.log"
  local n=$((npa-1))
  for i in $(seq 0 $n); do for j in $(seq 0 $n); do
    while [ "$(pgrep -cf 'stage4[_]point.py')" -ge "$nw" ]; do sleep 5; done
    "$PY" "$ROOT/scripts/stage4_point.py" "$label" "$tag" "$meth" "$bas" "$don" "$wal" \
        "$half" "$npa" "$i" "$j" "$mem" >> "$L/$label.log" 2>&1 &
  done; done
  wait
  "$PY" "$ROOT/scripts/stage4_fit.py" "$label" >> "$L/remaining.log" 2>&1
}

# wait for the grid sweeps to clear before adding load
while pgrep -f 'transverse_native[_]donor' > /dev/null; do sleep 60; done
echo "=== grid sweeps clear, starting remaining checks $(date -Iseconds) ===" >> "$L/remaining.log"

# check 3: orbital basis on K_perp (jun-cc-pVDZ is the published panel, already have it)
run s4_basis_augdz  WT sapt0 aug-cc-pVDZ native sidechain 0.15 5 8 '10 GB'
run s4_basis_augtz  WT sapt0 aug-cc-pVTZ native sidechain 0.15 5 4 '26 GB'
# check 4: SAPT order, WT, one geometry
run s4_order_sapt2p WT sapt2+ jun-cc-pVDZ native sidechain 0.15 5 4 '26 GB'
# check 5: wall fragment size, WT and L754A
run s4_wall_bb_WT    WT    sapt0 jun-cc-pVDZ native backbone 0.15 5 8 '10 GB'
run s4_wall_bb_L754A L754A sapt0 jun-cc-pVDZ native backbone 0.15 5 8 '10 GB'
# check 6: donor fragment size, WT
run s4_donor_shell2_WT WT sapt0 jun-cc-pVDZ shell2 sidechain 0.15 5 8 '10 GB'
echo "=== ALL STAGE 4 REMAINING DONE $(date -Iseconds) ===" >> "$L/remaining.log"
