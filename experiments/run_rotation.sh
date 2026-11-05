#!/usr/bin/env bash
# Runs the five rotation conditions. Set DRY_RUN=1 to print the commands instead of running them.
set -euo pipefail

MODEL="${MODEL:-opt125}"
SEEDS="${SEEDS:-0 1 2 3 4}"
ITERATIONS="${ITERATIONS:-10}"
MAIN_DIR="${MAIN_DIR:-versions/rotation}"

run() {
  local name="$1" seed="$2"
  shift 2
  local command=(python evolution/train_gen_loop.py --run_name "${name}_seed${seed}" --main_dir "$MAIN_DIR"
    --model_tag "$MODEL" --number_loops "$ITERATIONS" --seed "$seed" --partition_seed "$seed"
    --log_checkpoint_metrics "$@")
  echo "${command[*]}"
  if [[ -z "${DRY_RUN:-}" ]]; then "${command[@]}"; fi
}

for seed in $SEEDS; do
  run d1_baseline "$seed" --number_models 1
  run d8_baseline "$seed" --number_models 8
  run fixed_subset "$seed" --number_models 1 --arm fixed_subset
  run rotated_own "$seed" --number_models 1 --arm rotated_own
  run rotated_real "$seed" --number_models 1 --arm rotated_real
done
