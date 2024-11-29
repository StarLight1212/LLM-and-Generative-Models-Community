#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from: https://github.com/dvlab-research/LongLoRA/tree/main

"""
LLaMA/GPT-NeoX模型训练脚本
基于Landmark-Attention实现的高效训练方案
支持Flash Attention和低秩适应(LoRA)训练
"""

import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence
import logging

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    Trainer, 
    DataCollatorForLanguageModeling,
    TrainingArguments as HfTrainingArguments
)
from llama_attn_replace import replace_llama_attn
from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
from datasets import load_dataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量定义
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

SUPPORTED_MODEL_TYPES = ["llama", "gpt-neox"]

@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-1.4b-deduped",
        metadata={"help": "预训练模型的路径或标识符"}
    )
    model_type: Optional[str] = field(
        default="llama",
        metadata={"help": f"模型类型，当前支持: {', '.join(SUPPORTED_MODEL_TYPES)}"}
    )

@dataclass
class TrainingArguments(HfTrainingArguments):
    """训练相关参数"""
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "缓存目录路径"}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "优化器类型"}
    )
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "最大序列长度，超出部分将被截断"}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "是否使用Flash Attention"}
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "是否使用完整注意力机制"}
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "是否使用低秩适应训练"}
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "除LoRA权重外的可训练参数"}
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
) -> None:
    """
    调整tokenizer和embedding大小
    
    Args:
        special_tokens_dict: 特殊token字典
        tokenizer: 分词器
        model: 模型
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # 计算新token的初始embedding
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        # 使用平均值初始化新token的embedding
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def tokenize_fn(tokenizer, example):
    """
    数据集tokenization函数
    
    Args:
        tokenizer: 分词器
        example: 数据样本
    
    Returns:
        tokenized_output: 分词后的输出
    """
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}

def setup_model_and_tokenizer(model_args, training_args):
    """
    设置模型和分词器
    
    Args:
        model_args: 模型参数
        training_args: 训练参数
    
    Returns:
        model: 设置好的模型
        tokenizer: 设置好的分词器
    """
    # 验证模型类型
    if model_args.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"不支持的模型类型: {model_args.model_type}")
    
    # 替换注意力机制
    if model_args.model_type == "gpt-neox":
        replace_gpt_neox_attn(training_args.use_flash_attn, training_args.use_full_attn)
    else:
        replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # 加载配置
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # 设置RoPE缩放
    orig_rope_scaling = getattr(config, "rope_scaling", None) or {"factor": 1}
    orig_rope_scaling_factor = orig_rope_scaling.get("factor", 1)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)

    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # 加载模型和分词器
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    except Exception as e:
        logger.error(f"加载模型或分词器时出错: {str(e)}")
        raise

    return model, tokenizer

def train():
    """主训练函数"""
    # 解析参数
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    try:
        # 设置模型和分词器
        model, tokenizer = setup_model_and_tokenizer(model_args, training_args)

        # 添加特殊token
        special_tokens_dict = {
            k: v for k, v in {
                "pad_token": DEFAULT_PAD_TOKEN if tokenizer.pad_token is None else None,
                "eos_token": DEFAULT_EOS_TOKEN if tokenizer.eos_token is None else None,
                "bos_token": DEFAULT_BOS_TOKEN if tokenizer.bos_token is None else None,
                "unk_token": DEFAULT_UNK_TOKEN if tokenizer.unk_token is None else None,
            }.items() if v is not None
        }

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

        # 分布式训练同步
        rank = int(os.environ.get('RANK', -1))
        if rank > 0:
            barrier()

        # 加载数据集
        dataset = load_dataset(
            "togethercomputer/RedPajama-Data-1T-Sample",
            cache_dir=training_args.cache_dir
        )
        dataset = dataset.map(
            partial(tokenize_fn, tokenizer),
            batched=True,
            num_proc=128,
            remove_columns=["text", "meta"]
        )

        if rank == 0:
            barrier()

        logger.info(f"数据集信息:\n{dataset}")

        # 配置训练
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        if training_args.low_rank_training:
            # 配置LoRA
            targets = (
                ["query_key_value", "dense"]
                if model_args.model_type == "gpt-neox"
                else ["q_proj", "k_proj", "v_proj", "o_proj"]
            )

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=targets,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            
            # 设置可训练参数
            trainable_params = training_args.trainable_params.split(",")
            for name, param in model.named_parameters():
                if any(k in name for k in trainable_params):
                    param.requires_grad_()

        # 配置模型训练设置
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

        # 创建训练器
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=None,
            data_collator=data_collator
        )

        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存模型
        logger.info("保存模型...")
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)
        logger.info(f"模型已保存到: {training_args.output_dir}")

    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    train()
