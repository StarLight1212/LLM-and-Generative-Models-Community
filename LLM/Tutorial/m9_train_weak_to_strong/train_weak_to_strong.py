import json
import os
from typing import Dict, List, Optional, Sequence, Union

import fire
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType, get_peft_model

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from weak_to_strong.datasets import VALID_DATASETS, tokenize_dataset
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model

# NOTE 学习率未特别调整，在训练批量大小32下效果尚可
MODEL_CONFIGS = [
    ModelConfig(
        name="gpt2",
        default_lr=5e-5,
        eval_batch_size=32,
        custom_kwargs={
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
    ),
    # 其他模型配置省略...
    ModelConfig(
        name="Llama3.2",
        default_lr=1e-5,
        eval_batch_size=2,
        gradient_checkpointing=True,
        model_parallel=True,
        custom_kwargs={
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        },
        default_optimizer="adafactor",
    ),
]

MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}

loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
}

VALID_LOSSES: List[str] = list(loss_dict.keys())


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Weak to Strong Training with LLaMA3.2 and LoRA.")
    parser.add_argument("--batch_size", type=int, default=32, help="训练批次大小")
    parser.add_argument("--max_ctx", type=int, default=1024, help="最大上下文长度")
    parser.add_argument("--ds_name", type=str, default="alpaca_data", help="数据集名称")
    parser.add_argument("--transfer_loss", type=str, default="xent,logconf", help="转移损失函数")
    parser.add_argument("--n_docs", type=int, default=10000, help="训练文档数量")
    parser.add_argument("--n_test_docs", type=int, default=200, help="测试文档数量")
    parser.add_argument("--weak_model_size", type=str, default="gpt2", help="弱模型大小")
    parser.add_argument("--weak_lr", type=float, default=None, help="弱模型学习率")
    parser.add_argument("--strong_model_size", type=str, default="Llama3.2", help="强模型大小")
    parser.add_argument("--strong_lr", type=float, default=None, help="强模型学习率")
    parser.add_argument("--transfer_lr", type=float, default=None, help="转移学习率")
    parser.add_argument("--weak_optim", type=str, default=None, help="弱模型优化器")
    parser.add_argument("--strong_optim", type=str, default=None, help="强模型优化器")
    parser.add_argument("--transfer_optim", type=str, default=None, help="转移优化器")
    parser.add_argument("--gt_epochs", type=int, default=2, help="全监督训练轮数")
    parser.add_argument("--transfer_epochs", type=int, default=None, help="转移训练轮数")
    parser.add_argument("--force_retrain", action='store_true', help="是否强制重新训练")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--minibatch_size_per_device", type=int, default=None, help="每设备小批量大小")
    parser.add_argument("--train_with_dropout", action='store_true', help="是否使用 dropout 训练")
    parser.add_argument("--results_folder", type=str, default="/tmp/results", help="结果保存文件夹")
    parser.add_argument("--linear_probe", action='store_true', help="是否使用线性探针")
    parser.add_argument("--lr_schedule", type=str, default="cosine_anneal", help="学习率调度")
    parser.add_argument("--log_prefix", type=str, default="", help="日志前缀")
    parser.add_argument("--eval_every", type=int, default=100000000, help="每多少步评估一次")
    return parser.parse_args()


def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        inference_mode=False
    )


def load_model_and_tokenizer(model_path, method):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    if method in ["lora", "qlora"]:
        config = get_lora_config()
        model = get_peft_model(model, config)
    model.enable_input_require_grads()
    return model, tokenizer

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

def main():
    args = parse_arguments()
    assert args.ds_name == "alpaca_data", "当前脚本仅支持 'alpaca_data' 数据集"
    assert os.path.exists(args.results_folder), f"结果文件夹 {args.results_folder} 不存在"

    # 加载弱模型配置
    assert args.weak_model_size in MODELS_DICT, f"未知的弱模型大小 {args.weak_model_size}"
    weak_model_config = MODELS_DICT[args.weak_model_size]

    # 加载强模型配置
    assert args.strong_model_size in MODELS_DICT, f"未知的强模型大小 {args.strong_model_size}"
    strong_model_config = MODELS_DICT[args.strong_model_size]

    # 设置学习率
    weak_lr = args.weak_lr if args.weak_lr else weak_model_config.default_lr
    strong_lr = args.strong_lr if args.strong_lr else strong_model_config.default_lr
    transfer_lr = args.transfer_lr if args.transfer_lr else strong_lr

    # 设置优化器
    weak_optim = args.weak_optim if args.weak_optim else weak_model_config.default_optimizer
    strong_optim = args.strong_optim if args.strong_optim else strong_model_config.default_optimizer
    transfer_optim = args.transfer_optim if args.transfer_optim else strong_optim

    weak_eval_batch_size = weak_model_config.eval_batch_size
    strong_eval_batch_size = strong_model_config.eval_batch_size

    # 加载数据集
    dataset = load_dataset("json", data_files={"train": args.data_path}, split="train")
    dataset = dataset.train_test_split(test_size=args.n_test_docs / args.n_docs, seed=args.seed)
    train_ds, test_ds = dataset["train"], dataset["test"]

    # 训练弱模型
    print(f"训练弱模型，大小 {args.weak_model_size}")
    weak_model, weak_tokenizer = load_model_and_tokenizer(args.weak_model_size, method=args.method)
    weak_dataset = load_and_process_data(args.data_path, weak_tokenizer, max_length=args.max_ctx)
    training_args = get_training_args(os.path.join(args.results_folder, "weak_model"))
    trainer = Trainer(
        model=weak_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=weak_tokenizer, padding=True)
    )
    trainer.train()
    weak_model.save_pretrained(os.path.join(args.results_folder, "weak_model"))

    # 训练强模型
    print(f"训练强模型，大小 {args.strong_model_size}")
    strong_model, strong_tokenizer = load_model_and_tokenizer(args.strong_model_size, method=args.method)
    strong_dataset = load_and_process_data(args.data_path, strong_tokenizer, max_length=args.max_ctx)
    training_args = get_training_args(os.path.join(args.results_folder, "strong_model"))
    trainer = Trainer(
        model=strong_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=strong_tokenizer, padding=True)
    )
    trainer.train()
    strong_model.save_pretrained(os.path.join(args.results_folder, "strong_model"))

    # 保存结果
    results = {
        "weak_model": args.weak_model_size,
        "strong_model": args.strong_model_size,
    }
    with open(os.path.join(args.results_folder, "training_summary.json"), "w") as f:
        json.dump(results, f)
    print("训练完成，结果已保存。")

if __name__ == "__main__":
    fire.Fire(main)
