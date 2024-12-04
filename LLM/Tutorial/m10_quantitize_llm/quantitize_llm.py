import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from torch.quantization import quantize_dynamic
import os
import torch.nn as nn
from collections import OrderedDict


def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Quantization with INT8 or INT4")
    parser.add_argument("--model_path", type=str, default='./model_load/llama3.2_1B/',
                        help="Path to pre-trained model")
    parser.add_argument("--quantization", type=str, default="int8",
                        choices=["int8", "int4"], help="Quantization method")
    parser.add_argument("--output_dir", type=str, default="./output/quantized_model/",
                        help="Output directory")
    return parser.parse_args()


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float32,
                                                 low_cpu_mem_usage=True)
    return model, tokenizer


class Int4Quantizer:
    @staticmethod
    def quantize_tensor(tensor, bits=4):
        # 计算缩放因子
        max_val = torch.max(torch.abs(tensor))
        scale = (2 ** (bits - 1) - 1) / max_val

        # 量化
        quantized = torch.round(tensor * scale)
        quantized = torch.clamp(quantized, -2 ** (bits - 1), 2 ** (bits - 1) - 1)

        # 反量化因子
        dequantize_scale = 1.0 / scale

        return quantized, dequantize_scale

    @staticmethod
    def dequantize_tensor(quantized_tensor, dequantize_scale):
        return quantized_tensor * dequantize_scale


def quantize_model(model, method):
    model.cpu()
    if method == "int8":
        # INT8量化
        model = quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )
    elif method == "int4":
        # INT4量化
        quantized_state_dict = OrderedDict()
        scales = OrderedDict()

        for name, param in model.state_dict().items():
            if param.dim() > 1:  # 只量化矩阵参数
                quantized_param, scale = Int4Quantizer.quantize_tensor(param)
                quantized_state_dict[name] = quantized_param
                scales[f"{name}_scale"] = scale
            else:
                quantized_state_dict[name] = param

        # 保存量化后的参数和缩放因子
        model.quantized_state_dict = quantized_state_dict
        model.scales = scales

    return model


def save_quantized_model(model, tokenizer, output_dir, method):
    os.makedirs(output_dir, exist_ok=True)

    if method == "int8":
        # 保存INT8量化模型
        model.save_pretrained(output_dir)
    elif method == "int4":
        # 保存INT4量化模型和缩放因子
        torch.save({
            'quantized_state_dict': model.quantized_state_dict,
            'scales': model.scales
        }, os.path.join(output_dir, 'quantized_model.pt'))

    # 保存tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized model and tokenizer saved to {output_dir}")


def verify_quantized_model(output_dir, method):
    if method == "int4":
        checkpoint = torch.load(os.path.join(output_dir, 'quantized_model.pt'))
        quantized_state_dict = checkpoint['quantized_state_dict']
        scales = checkpoint['scales']

        # 验证量化数据
        for name, param in quantized_state_dict.items():
            if param.dim() > 1:
                assert torch.max(torch.abs(param)) <= 8, f"Parameter {name} exceeds INT4 range"

        print("Quantization verification completed successfully")


def main():
    args = parse_arguments()

    print(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    print(f"Applying {args.quantization} quantization")
    quantized_model = quantize_model(model, args.quantization)

    print("Saving quantized model")
    save_quantized_model(quantized_model, tokenizer, args.output_dir, args.quantization)

    print("Verifying quantized model")
    verify_quantized_model(args.output_dir, args.quantization)

    print("Quantization process completed successfully")


if __name__ == "__main__":
    main()
