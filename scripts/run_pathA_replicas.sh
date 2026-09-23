#!/bin/bash
# Path A step (1): run replicas 2 and 3 of the DAD PMF for WT + I553A + L754A + DM.
# Sequential (one GPU). Each system-replica = 15 windows x 5 ns HMR ~= 82.5 ns.
# Total for step (1): 8 driver invocations, ~660 ns of production.
set -euo pipefail
SL=/home/liang/anaconda3/envs/slomd/bin/python
DRV=/home/liang/Workspace/WritePaper/CatalysisQuamBio/scripts/umbrella_driver.py
cd /home/liang/Workspace/WritePaper/CatalysisQuamBio/md/mcpb

LOG=/home/liang/Workspace/WritePaper/CatalysisQuamBio/results/umbrella/pathA_replicas.log
mkdir -p "$(dirname "$LOG")"
echo "=== PATH A replicas start: $(date -Iseconds) ===" | tee -a "$LOG"

# args: <tag> <prmtop> <eq_rst7> <donor> <acceptor> <eq_r_DA_A> <rep>
run_rep () {
    tag=$1; prm=$2; rst=$3; d=$4; a=$5; r=$6; rep=$7
    echo "--- $(date -Iseconds)  $tag rep=$rep  d=$d a=$a  eqr=$r ---" | tee -a "$LOG"
    "$SL" "$DRV" "$tag" "$prm" "$rst" "$d" "$a" "$r" "$rep" 2>&1 | tee -a "$LOG"
}

for rep in 2 3; do
    run_rep WT    SLO_sub_solv.prmtop       SLO_sub_eq.rst7       13002 12985 5.07 "$rep"
    run_rep I553A SLO_I553A_sub_solv.prmtop SLO_I553A_sub_eq.rst7 12993 12976 4.75 "$rep"
    run_rep L754A SLO_L754A_sub_solv.prmtop SLO_L754A_sub_eq.rst7 12993 12976 5.13 "$rep"
    run_rep DM    SLO_DM_sub_solv.prmtop    SLO_DM_sub_eq.rst7    12984 12967 5.53 "$rep"
done

echo "=== PATH A replicas done: $(date -Iseconds) ===" | tee -a "$LOG"
