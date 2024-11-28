"""
基于T5模型的智能客服问答系统
实现从头开始的T5模型，包括数据加载、训练过程、模型保存及推理

系统流程框图：
+------------------------+
|       数据加载         |
| (合成问答对处理与编码) |
+------------------------+
            ↓
+------------------------+
|       T5模型架构       |
| (Encoder & Decoder层)  |
+------------------------+
            ↓
+------------------------+
|       模型训练         |
| (前向传播与反向传播)   |
+------------------------+
            ↓
+------------------------+
|       模型保存         |
| (保存训练好的模型)    |
+------------------------+
            ↓
+------------------------+
|        推理服务        |
| (接收问题并生成答案)   |
+------------------------+
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
import random
from tqdm import tqdm
import copy
from dataclasses import dataclass
from typing import Optional, Union, Tuple


# 1. 配置类
@dataclass
class T5Config:
    """T5模型配置类"""
    vocab_size: int = 32128
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    num_layers: int = 6
    dropout_rate: float = 0.1
    max_length: int = 128
    pad_token_id: int = 0


# 2. 注意力机制
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, config: T5Config):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        assert self.d_model % self.num_heads == 0, "d_model必须能被num_heads整除"

        self.depth = self.d_model // self.num_heads

        self.wq = nn.Linear(config.d_model, config.d_model)
        self.wk = nn.Linear(config.d_model, config.d_model)
        self.wv = nn.Linear(config.d_model, config.d_model)

        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def split_heads(self, x, batch_size):
        """将最后一个维度分成(num_heads, depth)"""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)

        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.depth)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch_size, num_heads, seq_len_q, depth)
        out = out.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len_q, num_heads, depth)
        out = out.view(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)

        out = self.dense(out)  # (batch_size, seq_len_q, d_model)
        return out


# 3. 前馈网络
class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, config: T5Config):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# 4. 编码器层
class EncoderLayer(nn.Module):
    """编码器的单层"""

    def __init__(self, config: T5Config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# 5. 解码器层
class DecoderLayer(nn.Module):
    """解码器的单层"""

    def __init__(self, config: T5Config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(config.d_model, eps=1e-6)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        # 自注意力
        self_attn_output = self.self_attn(x, x, x, look_ahead_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        # 交叉注意力
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, padding_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# 6. 编码器
class Encoder(nn.Module):
    """T5 编码器"""

    def __init__(self, config: T5Config):
        super(Encoder, self).__init__()
        self.d_model = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = self.positional_encoding(config.max_length, config.d_model)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x += self.pos_encoding[:x.size(1), :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x

    @staticmethod
    def positional_encoding(max_len, d_model):
        """位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe


# 7. 解码器
class Decoder(nn.Module):
    """T5 解码器"""

    def __init__(self, config: T5Config):
        super(Decoder, self).__init__()
        self.d_model = config.d_model
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = self.positional_encoding(config.max_length, config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x += self.pos_encoding[:x.size(1), :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask)

        return x

    @staticmethod
    def positional_encoding(max_len, d_model):
        """位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe


# 8. T5模型
class T5Model(nn.Module):
    """完整的T5模型，包含编码器和解码器"""

    def __init__(self, config: T5Config):
        super(T5Model, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.final_linear = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids, decoder_input_ids, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None):
        enc_output = self.encoder(input_ids, enc_padding_mask)  # 编码器输出
        dec_output = self.decoder(decoder_input_ids, enc_output, look_ahead_mask, dec_padding_mask)  # 解码器输出
        logits = self.final_linear(dec_output)  # 生成词汇表大小的logits
        return logits


# 9. 数据集类
class CustomerServiceDataset(Dataset):
    """智能客服问答数据集"""

    def __init__(self, tokenizer, config: T5Config):
        """
        初始化数据集
        :param tokenizer: T5分词器
        :param config: 模型配置
        """
        self.tokenizer = tokenizer
        self.max_length = config.max_length
        self.data = self.generate_synthetic_data()

    def generate_synthetic_data(self):
        """生成合成的问答对数据"""
        qa_pairs = [
            {
                "question": "这个产品可以退货吗？",
                "answer": "是的，本店所有商品支持7天无理由退货，请保持商品完好。"
            },
            {
                "question": "发货需要多久？",
                "answer": "正常情况下24小时内发货，节假日可能会有延迟。"
            },
            {
                "question": "怎么查询物流信息？",
                "answer": "您可以在订单详情页面查看物流信息，或者直接输入运单号查询。"
            },
            {
                "question": "有什么优惠活动吗？",
                "answer": "目前正在进行满300减50的促销活动，详情请查看活动页面。"
            },
            {
                "question": "支持哪些支付方式？",
                "answer": "支持支付宝、微信支付、银行卡等多种支付方式。"
            },
            # 可根据需要添加更多合成问答对
        ]
        # 为了增加数据量，可以重复现有数据或生成更多变体
        synthetic_data = qa_pairs * 200  # 示例：生成1000条数据
        random.shuffle(synthetic_data)
        return synthetic_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qa_pair = self.data[idx]

        # 编码问题
        question_encoding = self.tokenizer(
            "问：" + qa_pair["question"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 编码答案
        answer_encoding = self.tokenizer(
            "答：" + qa_pair["answer"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = question_encoding["input_ids"].squeeze()
        attention_mask = question_encoding["attention_mask"].squeeze()
        labels = answer_encoding["input_ids"].squeeze()

        # 解码器输入通常是目标序列右移一位
        decoder_input_ids = self.shift_right(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        }

    @staticmethod
    def shift_right(input_ids):
        """将目标序列右移一位作为解码器的输入"""
        pad_token_id = 0  # 假设pad_token_id为0
        dec_input_ids = input_ids.new_zeros(input_ids.shape)
        dec_input_ids[:, 0] = pad_token_id
        dec_input_ids[:, 1:] = input_ids[:, :-1]
        return dec_input_ids


# 10. 训练器类
class CustomerServiceTrainer:
    """智能客服问答系统训练器"""

    def __init__(self, model: T5Model, tokenizer: T5Tokenizer, config: T5Config, device: torch.device):
        """
        初始化训练器
        :param model: T5模型
        :param tokenizer: T5分词器
        :param config: 模型配置
        :param device: 设备（CPU或GPU）
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def train(self, train_loader: DataLoader, num_epochs: int):
        """训练模型"""
        self.model.train()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            total_loss = 0
            progress_bar = tqdm(train_loader, desc="Training", leave=False)
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                decoder_input_ids = batch["decoder_input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    enc_padding_mask=attention_mask,
                    look_ahead_mask=None,
                    dec_padding_mask=None
                )

                # 计算损失
                loss = self.criterion(outputs.view(-1, self.config.vocab_size), labels.view(-1))

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1} 完成，平均损失: {avg_loss:.4f}")

    def save_model(self, path: str):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到 {path}")

    def load_model(self, path: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"模型已加载自 {path}")

    def inference(self, question: str) -> str:
        """模型推理"""
        self.model.eval()
        with torch.no_grad():
            # 编码问题
            input_encoding = self.tokenizer(
                "问：" + question,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            input_ids = input_encoding["input_ids"]
            attention_mask = input_encoding["attention_mask"]

            # 解码器输入初始化为pad_token_id
            decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]]).to(self.device)

            # 生成答案
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_length=self.config.max_length,
                num_beams=4,
                early_stopping=True
            )

            # 解码答案
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer


# 11. 主程序
def main():
    """智能客服问答系统示例"""
    # 初始化配置
    config = T5Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化分词器和模型
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5Model(config)

    # 准备数据
    dataset = CustomerServiceDataset(tokenizer, config)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 初始化训练器
    trainer = CustomerServiceTrainer(model, tokenizer, config, device)

    # 训练模型
    print("开始训练模型...")
    trainer.train(train_loader, num_epochs=5)

    # 保存模型
    trainer.save_model("customer_service_model.pth")

    # 加载模型并进行推理
    print("加载模型并进行推理...")
    trainer.load_model("customer_service_model.pth")

    # 测试问答
    test_questions = [
        "这个商品可以退货吗？",
        "快递大概需要多久？",
        "支持货到付款吗？"
    ]

    print("\n测试问答：")
    for question in test_questions:
        answer = trainer.inference(question)
        print(f"问题: {question}")
        print(f"答案: {answer}\n")


if __name__ == "__main__":
    main()
