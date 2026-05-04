import sys
import os
import json
import argparse
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

TOOLKIT_PATH = '/home/hellwig/absa-toolkit'
sys.path.append(TOOLKIT_PATH)
from helper import get_dataset, get_prompt, get_unique_aspect_categories_in_list, setup_gpu_environment, clear_memory, parse_label_string
from gpu_monitor import GPUMonitor
from paraphrase import compute_f1_scores

def main():
    parser = argparse.ArgumentParser(description='Test Gemma 4 LoRA adapter')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--seed_run', type=int, required=True)
    args = parser.parse_args()

    setup_gpu_environment()
    clear_memory()
    
    # Force v0 engine for stability as seen in debug.ipynb
    os.environ["VLLM_USE_V1"] = "0"

    model_name_or_path = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
    lora_path = "model_temp"
    max_seq_length = 512
    lora_rank = 64

    test_data_raw = get_dataset(args.dataset_name, "test", args.task, TOOLKIT_PATH+"/data")
    
    gpu_monitor_eval = GPUMonitor()
    gpu_monitor_eval.start()

    llm = LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        enable_lora=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.9, # Adjusted for safety
        trust_remote_code=True
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_seq_length)

    test_labels = [example["label"] for example in test_data_raw]
    unique_aspect_categories = get_unique_aspect_categories_in_list(test_labels)

    conversations = []
    for example in test_data_raw:
        prompt = get_prompt(
            dataset_name=args.dataset_name,
            task=args.task,
            text_pred=example["text"],
            examples=[],
            unique_aspect_categories=unique_aspect_categories
        )
        # Manuelle Formatierung exakt wie im Training (train_gemma4.py)
        # Beachtung der fehlenden Pipe in <|turn> falls das beabsichtigt war
        formatted_prompt = f"<|turn>user\n{prompt}<|turn>model\n"
        conversations.append(formatted_prompt)

    # vLLM prompt formatting for Gemma 4
    # Wir nutzen llm.generate statt llm.chat, da wir den Prompt bereits fertig formatiert haben
    outputs = llm.generate(
        prompts=conversations,
        sampling_params=sampling_params,
        lora_request=LoRARequest("adapter", 1, lora_path)
    )

    all_preds = []
    for idx, output in enumerate(outputs):
        try:
            raw_output = output.outputs[0].text
            try:
               parsed = parse_label_string(raw_output, task=args.task)
            except Exception as e:
                print(f"Error parsing output {idx}: {e}")
                parsed = []
            all_preds.append([list(tupl) for tupl in parsed])
        except Exception as e:
            print(f"Error parsing output {idx}: {e}")
            all_preds.append([])

    avg_gpu_power_eval_W, total_time_eval = gpu_monitor_eval.stop()

    all_gold = [[list(tupl) for tupl in example["label"]] for example in test_data_raw]

    path_results = f"fine_tuning_results_gemma_4/results_llm_{args.dataset_name}_{args.task}_{args.seed_run}.json"
    
    # Load existing training results if they exist to avoid overwriting
    results = {}
    if os.path.exists(path_results):
        with open(path_results, "r") as f_in:
            results = json.load(f_in)
            print(f"Loaded existing results from {path_results}")

    results.update({
        'total_time_eval': total_time_eval,
        'avg_gpu_power_eval_W': avg_gpu_power_eval_W,
        'all_preds': all_preds,
        'compute_f1_scores': compute_f1_scores(all_preds, all_gold),
        'all_labels': [example["label"] for example in test_data_raw]
    })

    os.makedirs(os.path.dirname(path_results), exist_ok=True)
    with open(path_results, "w") as f_out:
        json.dump(results, f_out)

    print(f"Combined results saved to {path_results}")

    # Delete model_temp after use
    import shutil
    if os.path.exists(lora_path):
        # shutil.rmtree(lora_path) # Complying with "löscht sie wieder" requirement
        # print(f"Deleted {lora_path}")
        pass

if __name__ == "__main__":
    main()
