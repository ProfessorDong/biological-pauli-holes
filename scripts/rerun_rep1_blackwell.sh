#!/bin/bash
# Re-run rep 1 of the DAD PMF for WT / I553A / L754A / DM on the RTX PRO 5000
# Blackwell, so that ALL production umbrella sampling in the paper comes from one
# machine with one software stack.
#
# The superseded rep-1 trajectories (RTX 4060, 2026-07-12..14) are archived under
# results/umbrella/_RTX4060_rep1_archive_2026-08-10/ together with the
# hardware-consistency test showing the two GPUs are statistically
# indistinguishable (ratio 1.08, 0/4 outliers). This re-run is for provenance
# cleanliness, not because those results were wrong.
#
# Arguments are IDENTICAL to run_pathA_replicas.sh (same eq structures, same
# donor/acceptor indices, same eq r_DA), so rep1/rep2/rep3 remain directly
# comparable; only the sampling hardware is now uniform.
#
# 4 systems x 15 windows x 5 ns ~= 330 ns. Expect ~15 min/window => ~15 h.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -uo pipefail
SL=/home/liang/anaconda3/envs/slomd/bin/python
DRV=${_REPO}/scripts/umbrella_driver.py
cd ${_REPO}/md/mcpb

LOG=${_REPO}/results/umbrella/rep1_blackwell.log
mkdir -p "$(dirname "$LOG")"
echo "=== rep1 Blackwell re-run start: $(date -Iseconds) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | tee -a "$LOG"

run_rep () {
    tag=$1; prm=$2; rst=$3; d=$4; a=$5; r=$6; rep=$7
    echo "--- $(date -Iseconds)  $tag rep=$rep  d=$d a=$a  eqr=$r ---" | tee -a "$LOG"
    "$SL" "$DRV" "$tag" "$prm" "$rst" "$d" "$a" "$r" "$rep" 2>&1 | tee -a "$LOG"
}

run_rep WT    SLO_sub_solv.prmtop       SLO_sub_eq.rst7       13002 12985 5.07 1
run_rep I553A SLO_I553A_sub_solv.prmtop SLO_I553A_sub_eq.rst7 12993 12976 4.75 1
run_rep L754A SLO_L754A_sub_solv.prmtop SLO_L754A_sub_eq.rst7 12993 12976 5.13 1
run_rep DM    SLO_DM_sub_solv.prmtop    SLO_DM_sub_eq.rst7    12984 12967 5.53 1

echo "" | tee -a "$LOG"
echo "=== rep1 Blackwell re-run done: $(date -Iseconds) ===" | tee -a "$LOG"
for s in WT I553A L754A DM; do
  n=$(ls ${_REPO}/results/umbrella/$s/win_*_colvar.dat 2>/dev/null | wc -l)
  echo "  $s: $n / 15 windows" | tee -a "$LOG"
done
