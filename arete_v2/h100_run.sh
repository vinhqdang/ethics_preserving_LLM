#!/bin/bash
# Single consolidated job for the H100 VM. Run from the repo root (~/arete_run).
set -euo pipefail
M="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"; L="${LAYER:-16}"
PY="${PY:-python3}"
mkdir -p results_v2 "$HOME/checkpoints"
$PY -m arete_v2.extract --model "$M" --layer "$L" --bs 32 --bf16
$PY -m arete_v2.detect_eval --model "$M" --layer "$L"
$PY -m arete_v2.gcg --model "$M" --n 25 --steps 100
G="results_v2/gcg_${M//\//__}.json"
$PY -m arete_v2.gen_eval --model "$M" --layer "$L" --bf16 --limit 100 --seeds 0 1 2 --bs 64 --alpha-rel 0.4 --gcg "$G"
mkdir -p "$HOME/checkpoints/arete_v2" && cp -r results_v2 cache_v2 "$HOME/checkpoints/arete_v2/" && sync
echo JOB_DONE
