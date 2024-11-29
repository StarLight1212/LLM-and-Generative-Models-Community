#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLaMA模型推理脚本
该脚本用于加载LLaMA模型并进行文本生成推理
支持上下文扩展、Flash Attention和模型量化等功能
"""

import os
import sys
import math
import torch
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import (
    GenerationConfig, 
    TextStreamer, 
    BitsAndBytesConfig
)
from llama_attn_replace import replace_llama_attn

# 定义不同场景下的提示模板
PROMPT_DICT = {
    # 基础提示模板
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    # LLaMA-2专用提示模板(带系统提示)
    "prompt_no_input_llama2": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    # LLaMA-2简单提示模板
    "prompt_llama2": "[INST]{instruction}[/INST]"
}

def parse_config():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LLaMA模型推理参数配置')
    parser.add_argument('--material', type=str, default="", help='输入材料文件路径')
    parser.add_argument('--question', type=str, default="", help='问题内容')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf",
                      help='基础模型路径')
    parser.add_argument('--cache_dir', type=str, default="./cache",
                      help='模型缓存目录')
    parser.add_argument('--context_size', type=int, default=-1,
                      help='上下文窗口大小')
    parser.add_argument('--flash_attn', type=bool, default=False,
                      help='是否使用Flash Attention')
    parser.add_argument('--temperature', type=float, default=0.6,
                      help='生成温度参数')
    parser.add_argument('--top_p', type=float, default=0.9,
                      help='top-p采样参数')
    parser.add_argument('--max_gen_len', type=int, default=512,
                      help='最大生成长度')
    return parser.parse_args()

def read_txt_file(material_txt: str) -> str:
    """
    读取文本文件内容
    
    Args:
        material_txt: 文本文件路径
    
    Returns:
        文件内容字符串
    
    Raises:
        ValueError: 如果文件格式不是txt
    """
    if not material_txt.endswith('.txt'):
        raise ValueError("仅支持txt文件格式")
    
    try:
        with open(material_txt, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"读取文件出错: {str(e)}")
        return ""

def build_generator(
    model, 
    tokenizer, 
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: int = 4096,
    use_cache: bool = True
):
    """
    构建文本生成器函数
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        temperature: 采样温度
        top_p: top-p采样参数
        max_gen_len: 最大生成长度
        use_cache: 是否使用缓存
    
    Returns:
        生成响应的函数
    """
    def response(prompt: str) -> str:
        # 对输入进行编码
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 创建文本流式输出器
        streamer = TextStreamer(tokenizer)
        
        # 生成回复
        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            streamer=streamer,
        )
        
        # 解码输出
        out = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 提取生成的回复部分
        out = out.split(prompt.lstrip("<s>"))[1].strip()
        return out

    return response

def main(args):
    """主函数"""
    # 如果启用Flash Attention
    if args.flash_attn:
        replace_llama_attn(inference=True)

    # 加载模型配置
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    # 设置位置编码扩展
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # 加载模型
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        # 4bit量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    )
    # 调整词表大小
    model.resize_token_embeddings(32001)

    # 加载分词器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    # 设置模型为评估模式
    model.eval()
    
    # 在支持的环境下编译模型
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    # 构建生成器
    respond = build_generator(
        model, 
        tokenizer, 
        temperature=args.temperature,
        top_p=args.top_p,
        max_gen_len=args.max_gen_len,
        use_cache=True
    )

    try:
        # 读取材料内容
        material = read_txt_file(args.material)
        
        # 构建提示
        prompt_template = PROMPT_DICT["prompt_llama2"]
        prompt = prompt_template.format_map({
            "instruction": f"{material}\n{args.question}"
        })

        # 生成回复
        output = respond(prompt=prompt)
        
    except Exception as e:
        print(f"生成过程出错: {str(e)}")

if __name__ == "__main__":
    # 解析命令行参数并运行主函数
    args = parse_config()
    main(args)
