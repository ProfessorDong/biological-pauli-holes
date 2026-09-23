#!/bin/bash
# Run umbrella-sampling PMF for the 3 mutants sequentially (after WT is validated).
cd /home/dong/Workspace/WritePaper/CatalysisQuamBio/md/mcpb
PY=/home/dong/anaconda3/envs/slomd/bin/python
DRV=/home/dong/Workspace/WritePaper/CatalysisQuamBio/scripts/umbrella_driver.py
# tag  prmtop                     eq_rst7                  donor acceptor eqr_DA
run() { echo "=== $1 umbrella $(date '+%F %T') ==="; $PY $DRV "$@" > umbrella_$1.log 2>&1; echo "=== $1 DONE $(date '+%F %T') ==="; }
run I553A SLO_I553A_sub_solv.prmtop SLO_I553A_sub_eq.rst7 12993 12976 4.75
run L754A SLO_L754A_sub_solv.prmtop SLO_L754A_sub_eq.rst7 12993 12976 5.13
run DM    SLO_DM_sub_solv.prmtop    SLO_DM_sub_eq.rst7    12984 12967 5.53
echo "ALL_MUTANT_UMBRELLA_DONE $(date '+%F %T')"
