import os
import torch
import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    TrainingArguments, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    DataCollatorForSeq2Seq
)
import matplotlib
matplotlib.use('Agg')
from mergoo.compose_experts import ComposeExperts
import matplotlib.pyplot as plt  # 用于可视化
import unittest  # 用于单元测试


# 参数解析
def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM MoE Fine Tuning.")
    parser.add_argument("--model_path", type=str, default="./model_load/llama_moe", help="Path to initialized MoE model.")
    parser.add_argument("--data_path", type=str, default='./alpaca_data_cleaned_archive.json', help="Path to training data (JSON).")
    parser.add_argument("--method", type=str, default="qlora", choices=["full", "lora", "qlora"],
                        help="Choose fine-tuning method: full, lora, qlora.")
    parser.add_argument("--output_dir", type=str, default="output/llama_moe/", help="Directory to save the outputs.")
    parser.add_argument("--experts_config", type=str, default="experts_config.json", help="Path to experts configuration file.")
    return parser.parse_args()


# 数据处理
def load_and_process_data(data_path, tokenizer, max_length=512):
    df = pd.read_json(data_path)
    ds = Dataset.from_pandas(df)

    def process_func(example):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}\n"
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        instruction_len = len(tokenizer(f"### Instruction:\n{example['instruction']}\n\n### Response:\n").input_ids)
        labels = [-100] * instruction_len + encoded.input_ids[0][instruction_len:].tolist()
        return {
            "input_ids": encoded.input_ids[0],
            "attention_mask": encoded.attention_mask[0],
            "labels": labels
        }

    return ds.map(process_func, remove_columns=ds.column_names)


# 加载模型和分词器
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.enable_input_require_grads()
    return model, tokenizer


# 配置专家
def configure_experts(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"专家配置文件未找到: {config_path}")
    import json
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


# 初始化并保存专家合并
def initialize_expert_merger(config):
    checkpoint_path = config.get("checkpoint_path", "model_load/llama_moe")
    if not os.path.exists(checkpoint_path):
        expertmerger = ComposeExperts(config, torch_dtype=torch.float16)
        expertmerger.compose()
        expertmerger.save_checkpoint(checkpoint_path)
        print(f"已保存检查点: {checkpoint_path}")
    else:
        print(f"路径已存在，跳过保存检查点: {checkpoint_path}")


# 设置模型参数
def set_model_parameters(model):
    n_weights, n_router_weights = 0, 0
    for name, weight in model.named_parameters():
        if "gate" not in name:
            weight.requires_grad_(False)
            n_router_weights += 1
        n_weights += 1
    print('总参数数量: ', n_weights)
    print('路由器参数数量: ', n_router_weights)


# 获取训练参数
def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=1e-5,
        save_total_limit=1,
        num_train_epochs=3,
        eval_steps=5000,
        logging_strategy="steps",
        logging_steps=25,
        gradient_accumulation_steps=4,
        bf16=True,
        report_to=[],  # 禁用wandb
        disable_tqdm=False
    )


# 可视化专家模型 (伪代码)
def visualize_experts(config):
    experts = config.get("experts", [])
    expert_names = [expert["expert_name"] for expert in experts]
    expert_models = [expert["model_id"] for expert in experts]
    
    plt.figure(figsize=(10, 6))
    plt.bar(expert_names, range(len(expert_names)), color='skyblue')
    plt.xlabel('专家名称')
    plt.ylabel('模型ID索引')
    plt.title('合并后的专家模型可视化')
    plt.show()
    # 注意：实际可视化可能需要更复杂的逻辑


# 推理测试函数
def inference_test(model, tokenizer, prompt, max_length=512):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# LoRA 微调支持准备
def prepare_lora_finetuning(model):
    # TODO: 实现LoRA微调的准备工作
    pass


# 单元测试
class TestLlamaMoETrainer(unittest.TestCase):
    def test_load_and_process_data(self):
        tokenizer = AutoTokenizer.from_pretrained("./model_load/llama_moe")
        dataset = load_and_process_data('./alpaca_data_cleaned_archive.json', tokenizer)
        self.assertIsInstance(dataset, Dataset)
        self.assertGreater(len(dataset), 0)

    def test_load_model_and_tokenizer(self):
        model, tokenizer = load_model_and_tokenizer("./model_load/llama_moe")
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)

    def test_configure_experts(self):
        config = configure_experts("experts_config.json")
        self.assertIn("model_type", config)
        self.assertIn("experts", config)

    def test_inference_test(self):
        model, tokenizer = load_model_and_tokenizer("./model_load/llama_moe")
        prompt = "### Instruction:\nBeschreibe einen typischen deutschen Tag.\n\n### Response:\n"
        response = inference_test(model, tokenizer, prompt)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)


def main():
    args = parse_arguments()
    config = configure_experts(args.experts_config)
    initialize_expert_merger(config)
    visualize_experts(config)  # 可视化专家模型
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    set_model_parameters(model)
    dataset = load_and_process_data(args.data_path, tokenizer)
    trainer_args = get_training_args(args.output_dir)

    trainer = Trainer(
        model,
        args=trainer_args,
        train_dataset=dataset,
        # eval_dataset=dataset_test,  # 如果有验证集，可以在此处添加
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    trainer.train()

    # 推理测试
    test_prompt = "### Instruction:\nErkläre das deutsche Gesundheitssystem.\n\n### Response:\n"
    response = inference_test(model, tokenizer, test_prompt)
    print("推理测试结果:\n", response)


if __name__ == "__main__":
    main()
    # 运行单元测试
    unittest.main(argv=[''], exit=False)