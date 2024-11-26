import torch
import pandas as pd
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
import argparse


# Argument Parsing for Flexible Configuration
def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning with Full, LoRA, or QLoRA.")
    parser.add_argument("--model_path", type=str, default='./model_load/llama3.2_1B/', help="Path to pre-trained model.")
    parser.add_argument("--data_path", type=str,  default='./alpaca_data_cleaned_archive.json', help="Path to training data (JSON).")
    parser.add_argument("--method", type=str, default="qlora", choices=["full", "lora", "qlora"],
                        help="Choose fine-tuning method: full, lora, qlora.")
    parser.add_argument("--output_dir", type=str, default="./output/lora_weights/", help="Directory to save the outputs.")
    return parser.parse_args()


# Loading Tokenizer and Model
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                                                 use_cache=False, torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    return model, tokenizer


# LoRA/QLoRA Configuration
def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False
    )


# Data Processing Function
def load_and_process_data(data_path, tokenizer, max_length=384):
    df = pd.read_json(data_path)
    ds = Dataset.from_pandas(df)

    def process_func(example):
        instruction = tokenizer(
            f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False
        )
        response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return ds.map(process_func, remove_columns=ds.column_names)


# Apply LoRA/QLoRA
def apply_lora_if_selected(model, method):
    if method in ["lora", "qlora"]:
        config = get_lora_config()
        model = get_peft_model(model, config)
    return model


# Define Training Arguments
def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        gradient_checkpointing=True,
        save_on_each_node=True,
    )


# Main Training Function
def main():
    args = parse_arguments()
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model = apply_lora_if_selected(model, args.method)

    # Load and Process Dataset
    dataset = load_and_process_data(args.data_path, tokenizer)

    # Define Training Arguments and Trainer
    training_args = get_training_args(args.output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    # Start Training
    trainer.train()

    # Save Model and Tokenizer
    output_model_dir = f"{args.output_dir}/{args.method}_weights/"
    trainer.model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Model and tokenizer saved to {output_model_dir}")


if __name__ == "__main__":
    main()
