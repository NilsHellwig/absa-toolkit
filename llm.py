from datasets import Dataset
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only
from paraphrase import compute_f1_scores
from trl import SFTTrainer, SFTConfig
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from unsloth import FastModel
import torch
import numpy as np
import os
import sys

# Disable tokenizers parallelism warning when using DataLoader with multiple workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
TOOLKIT_PATH = '/home/hellwig/absa-toolkit'
sys.path.append(TOOLKIT_PATH)
if TOOLKIT_PATH:    
   from helper import *
   from gpu_monitor import GPUMonitor

german_datasets = ["gerest"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    # When running on CuDNN backend, make operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed} for reproducibility")


def train_and_evaluate(
    train_data_raw,
    test_data_raw,
    task="tasd",
    dataset_name="rest15",
    model_name_or_path="unsloth/gemma-3-27b-it-bnb-4bit",
    num_train_epochs=5,
    train_batch_size=4,
    max_seq_length=512,
    learning_rate=2e-4,
    seed=42,
    lora_rank=64,
    lora_alpha=16,
):
    # Set random seed for reproducibility
    set_seed(seed)

    # Start GPU monitoring for training
    gpu_monitor_train = GPUMonitor()
    gpu_monitor_train.start()

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        # load_in_4bit=True,
        # load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=seed,
    )

    # Calculate unique aspect categories from training data
    train_labels = [example["label"] for example in train_data_raw]
    unique_aspect_categories_train = get_unique_aspect_categories_in_list(
        train_labels)

    for example in train_data_raw:
        prompt = get_prompt(
            dataset_name=dataset_name,
            task=task,
            text_pred=example["text"],
            examples=[],
            unique_aspect_categories=unique_aspect_categories_train
        )
        example["text"] = "<start_of_turn>user\n" + prompt + \
            "<end_of_turn>\n<start_of_turn>model\n" + \
            str(example["label"]) + "<end_of_turn>"
    print(example["text"])

    train_data = Dataset.from_list(train_data_raw)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=seed,
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    trainer.train()

    avg_gpu_power_train_W, total_time_train = gpu_monitor_train.stop()

    results = {}

    results['total_time_train'] = total_time_train
    results['avg_gpu_power_train_W'] = avg_gpu_power_train_W

    # Save model and tokenizer to model_temp
    print("Saving model and tokenizer to model_temp...")
    model.save_pretrained("model_temp")
    tokenizer.save_pretrained("model_temp")

    # Start GPU monitoring for evaluation
    gpu_monitor_eval = GPUMonitor()
    gpu_monitor_eval.start()

    # perform evaluation here using vLLM

    print("Initializing vLLM for evaluation...")
    # Initialize vLLM with the base model
    llm = LLM(
        model=model_name_or_path,
        tokenizer=model_name_or_path,
        enable_lora=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7
    )

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=max_seq_length)

    # Calculate unique aspect categories from test data
    test_labels = [example["label"] for example in test_data_raw]
    unique_aspect_categories = get_unique_aspect_categories_in_list(
        test_labels)

    print(len(test_data_raw), "examples to evaluate")

    # Prepare prompts for batch inference
    prompts = []
    for example in test_data_raw:
        prompt = get_prompt(
            dataset_name=dataset_name,
            task=task,
            text_pred=example["text"],
            examples=[],
            unique_aspect_categories=unique_aspect_categories
        )
        prompt = "<start_of_turn>user\n" + prompt + \
            "<end_of_turn>\n<start_of_turn>model\n"
        prompts.append(prompt)

    # Generate with the LoRA adapter
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest("adapter", 1, "model_temp")
    )

    all_preds = []

    for idx, output in enumerate(outputs):
        try:
            raw_output = output.outputs[0].text
            raw_output = parse_label_string(raw_output, task=task)
            raw_output = [list(tupl) for tupl in raw_output]
        except Exception as e:
            print(f"Error parsing output {idx}: {e}")
            raw_output = []

        all_preds.append(raw_output)

    # Stop GPU monitoring for evaluation and get results
    avg_gpu_power_eval_W, total_time_eval = gpu_monitor_eval.stop()

    all_gold = [[list(tupl) for tupl in example["label"]]
                for example in test_data_raw]

    results['total_time_eval'] = total_time_eval
    results['avg_gpu_power_eval_W'] = avg_gpu_power_eval_W
    results["all_preds"] = all_preds
    results["compute_f1_scores"] = compute_f1_scores(all_preds, all_gold)
    results["all_labels"] = [example["label"] for example in test_data_raw]

    return results
