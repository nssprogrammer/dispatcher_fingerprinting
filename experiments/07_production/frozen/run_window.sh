#!/bin/bash
# usage: run_window.sh W0 ; each step in its own process (memory), same injection seed
W=$1; F=${2:-10}; cd /tmp/claude-0/-home-claude/78f29ddf-32f5-54b2-bcc0-86fa7c435eb9/scratchpad/nb10
for step in "R1,R2" "R3" "R4" "R5,R8,R9" "R7" "R6"; do
  tag=$(echo $step | tr ',' '_')
  python3 analyze_window.py data/$W.parquet data/${W}_${tag}.json --floor $F --heavy_max 2000 --nmax 100000 --only $step > data/${W}_${tag}.log 2>&1
  echo "$W $step exit $?"
done
