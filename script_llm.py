from llm import train_and_evaluate
import sys
import os
import json
import random
import argparse

import sys
TOOLKIT_PATH = '/home/hellwig/absa-toolkit'
sys.path.append(TOOLKIT_PATH)
if TOOLKIT_PATH:
    from helper import *

DATASET_NAMES = ["rest16", "rest15", "flightabsa", "coursera", "hotels"]
TASKS = ["asqp", "tasd"]
N_SEEDS_RUNS = 5


def run_training_pipeline_real(dataset_name, task, seed_run):
    train_data = get_dataset(dataset_name, "train", task, TOOLKIT_PATH+"/data")
    print(len(train_data), "training examples loaded")

    test_dataset = get_dataset(
        dataset_name, "test",
        task,
        TOOLKIT_PATH+"/data"
    )

    results = train_and_evaluate(
        train_data,
        test_dataset,
        task=task,
        dataset_name=dataset_name,
        seed=seed_run,
        num_train_epochs=5
    )
    
    # delete model_temp and free trash
    import shutil
    model_temp_path = TOOLKIT_PATH + "/model_temp"
    if os.path.exists(model_temp_path):
        shutil.rmtree(model_temp_path)
        print(f"Deleted {model_temp_path}")
    
    clear_memory()

    return results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run LLM training and evaluation')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--task', type=str, help='Task type')
    parser.add_argument('--seed_run', type=int, help='Seed run number')
    args = parser.parse_args()

    setup_gpu_environment()
    clear_memory()

    dataset_name = args.dataset_name
    task = args.task
    seed_run = args.seed_run
    
    path_results = f"fine_tuning_results/results_llm_{dataset_name}_{task}_{seed_run}.json"
    
    if os.path.exists(path_results):
        print(
            f"Results for {dataset_name} {task} seed {seed_run} already exist. Skipping.")
        return

    results = run_training_pipeline_real(dataset_name, task, seed_run)

    with open(path_results, "w") as f_out:
        json.dump(results, f_out)

    print(f"Results saved to {path_results}")


if __name__ == "__main__":
    main()
