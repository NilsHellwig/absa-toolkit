#!/bin/bash

DATASET_NAMES=("rest16" "rest15" "flightabsa" "coursera" "hotels")
TASKS=("asqp" "tasd")
N_SEEDS_RUNS=5

for seed_run in $(seq 0 $((N_SEEDS_RUNS - 1))); do
    for dataset_name in "${DATASET_NAMES[@]}"; do
        for task in "${TASKS[@]}"; do
            python script_llm.py \
                --dataset_name "$dataset_name" \
                --task "$task" \
                --seed_run "$seed_run"
        done
    done
done

