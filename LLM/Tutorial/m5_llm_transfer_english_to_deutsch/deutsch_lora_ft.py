import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
import argparse
from tokenizers import ByteLevelBPETokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json


# Argument Parsing for Flexible Configuration
def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Fine-Tuning with Full, LoRA, or QLoRA.")
    parser.add_argument("--model_path", type=str, default='./model_load/llama3.2_1B/',
                        help="Path to pre-trained model.")
    parser.add_argument("--data_path", type=str, default='./deutsch_data.json', help="Path to training data (JSON).")
    parser.add_argument("--german_corpus_path", type=str, default='./deutsch_data.json',
                        help="Path to German literary corpus.")
    parser.add_argument("--method", type=str, default="qlora", choices=["full", "lora", "qlora"],
                        help="Choose fine-tuning method: full, lora, qlora.")
    parser.add_argument("--output_dir", type=str, default="./output/lora_weights/",
                        help="Directory to save the outputs.")
    return parser.parse_args()


# Loading and Expanding Tokenizer
def load_and_expand_tokenizer(model_path, german_corpus_path, tokenizer_save_path="./tokenizer_expanded/"):
    # 加载基础分词器
    base_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    
    # 从JSON文件中提取德语文本
    with open(german_corpus_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 收集所有德语文本
    german_texts = []
    for item in data:
        german_texts.extend([
            item['instruction'],
            item['output']
        ])
    
    # 训练新的tokenizer
    new_tokenizer = ByteLevelBPETokenizer()
    new_tokenizer.train_from_iterator(
        german_texts,
        vocab_size=5000,  # 新词汇量
        min_frequency=2,  # 最小词频
        special_tokens=["<s>", "<pad>", "", "<unk>", "<mask>"]
    )
    
    # 获取新词汇并添加到基础分词器
    new_vocab = set(new_tokenizer.get_vocab().keys())
    base_vocab = set(base_tokenizer.get_vocab().keys())
    new_tokens = list(new_vocab - base_vocab)
    
    # 添加新词汇
    base_tokenizer.add_tokens(new_tokens)
    print(f"添加了 {len(new_tokens)} 个新词汇")
    
    # 保存扩展后的分词器
    base_tokenizer.save_pretrained(tokenizer_save_path)
    return base_tokenizer


# Loading Model and Tokenizer
def load_model_and_tokenizer(model_path, german_corpus_path):
    tokenizer = load_and_expand_tokenizer(model_path, german_corpus_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bfloat16
    )
    model.resize_token_embeddings(len(tokenizer))
    model.enable_input_require_grads()
    return model, tokenizer


# LoRA/QLoRA Configuration
def get_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 针对Llama模型的关键层
        r=16,  # LoRA秩，增加以提升模型性能
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        inference_mode=False
    )


# Data Processing Function
def load_and_process_data(data_path, tokenizer, max_length=512):  # 增加最大长度以容纳更多上下文
    df = pd.read_json(data_path)
    ds = Dataset.from_pandas(df)

    def process_func(example):
        # 改进提示模板以更好地处理德语指令
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}\n"
        
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 创建标签，将instruction部分的标签设为-100
        instruction_len = len(tokenizer(f"### Instruction:\n{example['instruction']}\n\n### Response:\n").input_ids)
        labels = [-100] * instruction_len + encoded.input_ids[0][instruction_len:].tolist()
        
        return {
            "input_ids": encoded.input_ids[0],
            "attention_mask": encoded.attention_mask[0],
            "labels": labels
        }

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
        per_device_train_batch_size=2,  # 减小批次大小以适应显存
        gradient_accumulation_steps=8,  # 增加梯度累积步数
        learning_rate=2e-4,  # 略微提高学习率
        num_train_epochs=5,  # 增加训练轮次
        logging_steps=5,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        save_total_limit=3,
        gradient_checkpointing=True,
        fp16=True,  # 启用混合精度训练
        warmup_steps=100,
        weight_decay=0.01,
    )


# Main Training Function
def main():
    args = parse_arguments()
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.german_corpus_path)
    model = apply_lora_if_selected(model, args.method)

    # Load and Process Dataset
    dataset = load_and_process_data(args.data_path, tokenizer)

    # Define Training Arguments和Trainer
    training_args = get_training_args(args.output_dir)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    try:
        # Start Training
        trainer.train()

        # Save Model and Tokenizer
        output_model_dir = f"{args.output_dir}/{args.method}_weights/"
        trainer.model.save_pretrained(output_model_dir)
        tokenizer.save_pretrained(output_model_dir)
        print(f"模型和分词器已保存到 {output_model_dir}")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")


if __name__ == "__main__":
    main()
