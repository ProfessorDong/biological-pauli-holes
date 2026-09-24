#!/bin/bash
# Resume Path A after the L754A_rep2 window-6.20 rst7-save crash (parmed py3.14 bug fixed
# in umbrella_window.py — now uses native rst7 writer). Skips any window whose colvar.dat
# already exists.
#   L754A_rep2 (6 remaining) -> DM_rep2 (15) -> rep3 for WT/I553A/L754A/DM (15 x 4 = 60)
# Then chains to run_pathA_followup.sh.
_REPO="${PAULI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"

set -euo pipefail
SL="${SLOMD_PYTHON:-/home/liang/anaconda3/envs/slomd/bin/python}"
DRV=${_REPO}/scripts/umbrella_driver.py
MCPB=${_REPO}/md/mcpb
LOG=${_REPO}/results/umbrella/pathA_replicas.log
cd "$MCPB"
echo "=== PATH A RESUME start: $(date -Iseconds) ===" | tee -a "$LOG"

run_rep () {
    tag=$1; prm=$2; rst=$3; d=$4; a=$5; r=$6; rep=$7
    echo "--- $(date -Iseconds)  $tag rep=$rep  d=$d a=$a  eqr=$r ---" | tee -a "$LOG"
    "$SL" "$DRV" "$tag" "$prm" "$rst" "$d" "$a" "$r" "$rep" 2>&1 | tee -a "$LOG"
}

# (a) finish L754A_rep2 (only missing windows will run)
run_rep L754A SLO_L754A_sub_solv.prmtop SLO_L754A_sub_eq.rst7 12993 12976 5.13 2
# (b) DM_rep2
run_rep DM SLO_DM_sub_solv.prmtop SLO_DM_sub_eq.rst7 12984 12967 5.53 2
# (c) rep 3 for all four
run_rep WT    SLO_sub_solv.prmtop       SLO_sub_eq.rst7       13002 12985 5.07 3
run_rep I553A SLO_I553A_sub_solv.prmtop SLO_I553A_sub_eq.rst7 12993 12976 4.75 3
run_rep L754A SLO_L754A_sub_solv.prmtop SLO_L754A_sub_eq.rst7 12993 12976 5.13 3
run_rep DM    SLO_DM_sub_solv.prmtop    SLO_DM_sub_eq.rst7    12984 12967 5.53 3

echo "=== PATH A RESUME done: $(date -Iseconds) ===" | tee -a "$LOG"
echo "=== Chaining into pathA_followup.sh ===" | tee -a "$LOG"
exec ${_REPO}/scripts/run_pathA_followup.sh
