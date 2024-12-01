import os
from pathlib import Path
import json
from random import sample
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer
)

from dataset.babylm.babylm_dataset import BabylmDataset


# 配置参数
class Config:
    """模型训练的配置参数类"""
    # 训练超参数
    LR = 2.5e-4  # 学习率
    BATCH_SIZE = 8  # 批次大小
    SEQ_LENGTH = 128  # 序列长度
    TEMPERATURE = 2.0  # 知识蒸馏温度系数
    ALPHA = 0.5  # 知识蒸馏损失权重

    # 路径配置
    MODEL_PATH = Path("./models/")  # 模型存储根路径
    STUDENT_NAME = 'Llama-3-74M'  # 学生模型名称
    TEACHER_NAME = 'llama3.2_1B'  # 教师模型名称
    MODEL_OUTPUT = MODEL_PATH / STUDENT_NAME  # 模型输出路径
    DATA_PATH = Path("./data/")  # 数据集路径

    # 评估配置
    EVAL_SAMPLES = 8192  # 评估样本数量

    # WandB配置
    WANDB_LOG = False  # 是否启用WandB日志

    def __init__(self):
        # 创建输出目录
        if not os.path.exists(self.MODEL_OUTPUT):
            os.makedirs(self.MODEL_OUTPUT)


# 初始化配置
config = Config()


def load_models_and_tokenizer(config):
    """
    加载模型和分词器

    Args:
        config: 配置对象

    Returns:
        tuple: (tokenizer, student_model, teacher_model)
    """
    # 加载tokenizer
    teacher_dir = config.MODEL_PATH / config.TEACHER_NAME
    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    tokenizer.model_max_length = config.SEQ_LENGTH

    # 加载学生模型配置
    with open('./Llama-3-74M.json', 'r') as file:
        student_config = LlamaConfig.from_dict(json.load(file))

    # 初始化学生模型
    student = LlamaForCausalLM(student_config)

    # 加载教师模型
    teacher = LlamaForCausalLM.from_pretrained(
        teacher_dir,
        torch_dtype=torch.bfloat16  # 使用bfloat16精度以节省显存
    )

    return tokenizer, student, teacher


def prepare_datasets(config, tokenizer):
    """
    准备训练和评估数据集

    Args:
        config: 配置对象
        tokenizer: 分词器

    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    # 加载训练数据集
    train_dataset = BabylmDataset(
        config.DATA_PATH / "train",
        config.SEQ_LENGTH,
        tokenizer=tokenizer,
        random_chunk=True  # 随机分块以提升模型性能
    )

    # 加载完整评估数据集
    full_eval_dataset = BabylmDataset(
        config.DATA_PATH / "dev",
        config.SEQ_LENGTH,
        tokenizer=tokenizer,
        offset=0
    )

    # 随机采样评估数据集
    eval_indices = sample(range(len(full_eval_dataset)), config.EVAL_SAMPLES)
    eval_dataset = Subset(full_eval_dataset, eval_indices)

    return train_dataset, eval_dataset


class DistillationTrainingArguments(TrainingArguments):
    """知识蒸馏训练参数类"""

    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # 蒸馏损失权重
        self.temperature = temperature  # 温度系数


class DistillationTrainer(Trainer):
    """知识蒸馏训练器类"""

    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # 将教师模型移动到与学生模型相同的设备上
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()  # 设置教师模型为评估模式

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算知识蒸馏损失

        Args:
            model: 学生模型
            inputs: 输入数据
            return_outputs: 是否返回输出

        Returns:
            torch.Tensor: 损失值
        """
        # 计算学生模型输出
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # 计算教师模型输出
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
            teacher_logits = outputs_teacher.logits

        # 确保学生和教师输出维度一致
        assert outputs_student.logits.size() == teacher_logits.size()

        # 计算知识蒸馏损失
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
                loss_function(
                    F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                    F.softmax(teacher_logits / self.args.temperature, dim=-1),
                )
                * (self.args.temperature ** 2)
        )

        # 计算最终损失（学生损失和蒸馏损失的加权和）
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits

        return (loss, outputs_student) if return_outputs else loss


def main():
    """主函数"""
    # 加载模型和分词器
    tokenizer, student, teacher = load_models_and_tokenizer(config)

    # 准备数据集
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)

    # 创建数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 不使用掩码语言模型
    )

    # 打印模型参数量
    print(f'模型参数量: 学生模型 = {student.num_parameters()}')
    print(f'模型参数量: 教师模型 = {teacher.num_parameters()}')

    # 初始化WandB
    if config.WANDB_LOG:
        wandb.login()
        wandb.init(project='babylm', name=config.STUDENT_NAME)

    # 设置训练参数
    training_args = DistillationTrainingArguments(
        output_dir=config.MODEL_OUTPUT,
        overwrite_output_dir=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        num_train_epochs=6,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=config.BATCH_SIZE,
        save_total_limit=1,
        report_to="wandb" if config.WANDB_LOG else "none",
        warmup_steps=200,
        lr_scheduler_type="cosine",
        learning_rate=config.LR,
        logging_steps=20,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.1,
        alpha=config.ALPHA,
        temperature=config.TEMPERATURE,
    )

    # 初始化训练器
    trainer = DistillationTrainer(
        student,
        training_args,
        teacher_model=teacher,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 开始训练
    trainer.train()

    # 保存模型和分词器
    trainer.save_model(config.MODEL_OUTPUT)
    tokenizer.save_pretrained(config.MODEL_OUTPUT)


if __name__ == "__main__":
    main()
