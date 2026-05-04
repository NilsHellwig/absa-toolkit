#!/bin/bash

DATASET_NAMES=("rest16")
TASKS=("asqp")
N_SEEDS_RUNS=1

# Ensure output directory exists
mkdir -p fine_tuning_results_gemma_4

for seed_run in $(seq 0 $((N_SEEDS_RUNS - 1))); do
    for dataset_name in "${DATASET_NAMES[@]}"; do
        for task in "${TASKS[@]}"; do
            echo "Running experiment: $dataset_name $task Seed: $seed_run"
            
            # Step 1: Train (using vllm_unsloth env)
            echo "Starting Training..."
            # Verwende absolute Pfade zum Python-Interpreter der Conda-Envs
            # Damit siehst du den Output sofort im Terminal
            ~/miniconda3/envs/vllm_unsloth/bin/python train_gemma4.py \
                --dataset_name "$dataset_name" \
                --task "$task" \
                --seed_run "$seed_run"
            
            # Warte kurz und kille alle verbleibenden GPU-Prozesse vor der Evaluation
            sleep 5
            GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
            if [ ! -z "$GPU_PIDS" ]; then
                echo "Killing GPU processes: $GPU_PIDS"
                echo "$GPU_PIDS" | xargs -r kill -9
            fi
            sleep 5

            # Step 2: Test (using vllm env)
            # echo "Starting Evaluation..."
            # ~/miniconda3/envs/vllm/bin/python test_gemma4.py \
            #     --dataset_name "$dataset_name" \
            #     --task "$task" \
            #     --seed_run "$seed_run"
            
            # Cleanup model_temp after each run to save space and ensure fresh start
            # if [ -d "model_temp" ]; then
            #     rm -rf "model_temp"
            #     echo "Cleaned up model_temp"
            # fi
            
            echo "Finished $dataset_name $task Seed: $seed_run"
            echo "-----------------------------------"
        done
    done
done
