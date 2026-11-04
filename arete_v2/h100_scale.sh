#!/bin/bash
# Consolidated H100 job: replicate the frozen configuration on two 7B models.
set -uo pipefail
cd "$HOME/arete_run"
mkdir -p results_v2 "$HOME/checkpoints/results_v2"
PY="${PY:-python3}"
for M in Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct; do
  echo "=== $M $(date)"
  $PY -m arete_v2.scale_run --model "$M" --bs 96 --windows 10 || echo "FAILED $M"
  cp -f results_v2/*.json "$HOME/checkpoints/results_v2/" 2>/dev/null; sync
done
echo JOB_DONE "$(date)"
