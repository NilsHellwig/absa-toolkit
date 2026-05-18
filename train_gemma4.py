from gpu_monitor import GPUMonitor
from helper import get_dataset, get_prompt, get_unique_aspect_categories_in_list, setup_gpu_environment, clear_memory
import sys
import os
import argparse
import json
import torch
import numpy as np
from datasets import Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

TOOLKIT_PATH = '/home/hellwig/absa-toolkit'
sys.path.append(TOOLKIT_PATH)


def set_seed(seed=42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser(description='Train Gemma 4 LoRA adapter')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--seed_run', type=int, required=True)
    args = parser.parse_args()

    setup_gpu_environment()
    clear_memory()
    set_seed(args.seed_run)

    model_name_or_path = "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
    max_seq_length = 512
    lora_rank = 64
    lora_alpha = 16

    train_data_raw = get_dataset(
        args.dataset_name, "train", args.task, TOOLKIT_PATH+"/data")
    # train_data_raw = train_data_raw[:100] # Optional: for testing

    gpu_monitor_train = GPUMonitor()
    gpu_monitor_train.start()

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
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
        random_state=args.seed_run,
    )

    train_labels = [example["label"] for example in train_data_raw]
    unique_aspect_categories_train = get_unique_aspect_categories_in_list(
        train_labels)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-4",
    )

    def formatting_prompts_func(examples):
        texts = []
        for text, label in zip(examples["text"], examples["label"]):
            prompt = get_prompt(
                dataset_name=args.dataset_name,
                task=args.task,
                text_pred=text,
                examples=[],
                unique_aspect_categories=unique_aspect_categories_train
            )
            messages = [
                {"role": "user", "content": prompt},
                {"role": "model", "content": str(label)},
            ]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            # Entferne <bos>, Unsloth/FastModel fügt es später wieder hinzu
            formatted = formatted.removeprefix("<bos>")
            texts.append(formatted)
        print("Sample formatted prompt:", texts[0])
        return {"text": texts}

    train_data = Dataset.from_list(train_data_raw)
    train_data = train_data.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=args.seed_run,
            report_to="none",
            output_dir="unsloth_training_checkpoints",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    trainer.train()
    avg_gpu_power_train_W, total_time_train = gpu_monitor_train.stop()

    results_train = {
        'total_time_train': total_time_train,
        'avg_gpu_power_train_W': avg_gpu_power_train_W
    }

    path_results = f"fine_tuning_results_gemma_4/results_llm_{args.dataset_name}_{args.task}_{args.seed_run}.json"
    os.makedirs(os.path.dirname(path_results), exist_ok=True)
    with open(path_results, "w") as f_out:
        json.dump(results_train, f_out)
    print(f"Training results saved to {path_results}")

    model.save_pretrained("model_temp")
    tokenizer.save_pretrained("model_temp")

    print("Model saved to model_temp")


if __name__ == "__main__":
    main()
