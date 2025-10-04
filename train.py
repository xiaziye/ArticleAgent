"""
Fine-tuning script for Qwen2.5-1.5B-Instruct on academic concept path extraction.
Supports YAML config and command-line overrides.

Usage:
    python train.py --config config/qwen2.5_finetune.yaml
"""

import os
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from datasets import Dataset
import shutil
import glob


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class KeepLatestCheckpointsCallback(TrainerCallback):
    def __init__(self, keep_limit: int = 2):
        self.keep_limit = keep_limit

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ckpt_dirs = [d for d in glob.glob(os.path.join(args.output_dir, "checkpoint-*")) if os.path.isdir(d)]
            ckpt_dirs.sort(key=os.path.getmtime, reverse=True)
            for d in ckpt_dirs[self.keep_limit:]:
                shutil.rmtree(d, ignore_errors=True)


def load_json_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_example(example: Dict, tokenizer, max_length: int = 2048) -> Dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
        {"role": "assistant", "content": example['output']}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(text, max_length=max_length, truncation=True, padding=False)

    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, max_length=max_length, truncation=True)["input_ids"]
    prompt_len = len(prompt_ids)

    labels = tokenized["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args_cmd = parser.parse_args()

    # Load config
    with open(args_cmd.config) as f:
        config = yaml.safe_load(f)

    # Paths
    model_path = config["model"]["name_or_path"]
    train_file = config["data"]["train_file"]
    val_file = config["data"]["val_file"]
    output_dir = config["training"]["output_dir"]

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16_setting = config["model"]["use_bf16"]
    if bf16_setting == "auto":
        use_bf16 = torch.cuda.is_bf16_supported()
    else:
        use_bf16 = bf16_setting

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        trust_remote_code=True,
        ignore_mismatched_sizes=True
    )
    model.config.use_cache = False

    # Load data
    train_data = load_json_data(train_file)
    if val_file and os.path.exists(val_file):
        val_data = load_json_data(val_file)
    else:
        split = int(len(train_data) * (1 - config["data"]["val_split_ratio"]))
        train_data, val_data = train_data[:split], train_data[split:]

    # Create datasets
    max_len = config["data"]["max_seq_length"]
    train_dataset = Dataset.from_list(train_data).map(
        lambda ex: process_example(ex, tokenizer, max_len),
        remove_columns=list(train_data[0].keys()),
        batched=False
    )
    val_dataset = Dataset.from_list(val_data).map(
        lambda ex: process_example(ex, tokenizer, max_len),
        remove_columns=list(val_data[0].keys()),
        batched=False
    )

    # Training args
    train_cfg = config["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        optim=train_cfg["optim"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=train_cfg["remove_unused_columns"],
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8),
        callbacks=[KeepLatestCheckpointsCallback(config["callbacks"]["keep_latest_checkpoints"])]
    )

    trainer.train()
    final_path = Path(output_dir) / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Model saved to {final_path}")


if __name__ == "__main__":
    main()
