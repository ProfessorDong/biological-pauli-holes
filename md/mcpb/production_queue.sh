#!/bin/bash
cd /home/dong/Workspace/WritePaper/CatalysisQuamBio/md/mcpb
export AMBERHOME=/home/dong/anaconda3/envs/slomd
PY=/home/dong/anaconda3/envs/slomd/bin/python
MD=/home/dong/Workspace/WritePaper/CatalysisQuamBio/scripts/md_metad.py
RES=/home/dong/Workspace/WritePaper/CatalysisQuamBio/results/metad
TAGS=(WT I553A L754A DM)
PRE=(SLO_sub SLO_I553A_sub SLO_L754A_sub SLO_DM_sub)
DON=(13002 12993 12993 12984)
ACC=(12985 12976 12976 12967)
for i in 0 1 2 3; do
  t=${TAGS[$i]}; mkdir -p $RES/$t
  echo "QUEUE_START $t $(date '+%F %T')"
  $PY $MD --prmtop ${PRE[$i]}_solv.prmtop --inpcrd ${PRE[$i]}_eq.rst7 \
     --donor ${DON[$i]} --acceptor ${ACC[$i]} --ns 150 --hmr --replica 1 --seed 1 \
     --out $RES/$t/rep1 > metad_${t}_rep1.log 2>&1
  echo "QUEUE_DONE $t $(date '+%F %T') conv=$(tail -1 $RES/$t/rep1_conv.csv 2>/dev/null)"
done
echo "SIGNAL_FIRST_CAMPAIGN_COMPLETE $(date '+%F %T')"
