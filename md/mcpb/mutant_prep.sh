#!/bin/bash
cd /home/dong/Workspace/WritePaper/CatalysisQuamBio/md/mcpb
export AMBERHOME=/home/dong/anaconda3/envs/slomd
PY=/home/dong/anaconda3/envs/slomd/bin/python
for tag in I553A L754A DM; do
  echo "=== $tag minimize ==="; $PY minimize_sub.py SLO_${tag}_sub > min_${tag}.log 2>&1
  echo "=== $tag equilibrate ==="; $PY equilibrate_sub.py 1.0 SLO_${tag}_sub > eq_${tag}.log 2>&1
  echo "=== $tag DONE: $(grep -h 'equilibrated:' eq_${tag}.log 2>/dev/null | tail -1) ==="
done
echo "ALL MUTANTS PREPARED"
