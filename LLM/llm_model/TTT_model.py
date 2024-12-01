import math
from torch import nn, Tensor
import torch
from torch.optim import SGD
from typing import Callable, Self
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot  # 用于可视化计算图
from torch.nn import functional as F
import wandb  # 用于实验跟踪和可视化

# 数据参数
NUM_SAMPLES = 1000  # 样本数量
VOCAB_SIZE = 15      # 词汇表大小
SEQUENCE_LEN = 40    # 序列长度

# 训练参数
EPOCHS = 10          # 训练轮数
BATCH_SIZE = 50      # 批量大小

# 学习率参数
TTT_BASE_INNER_LEARNING_RATE = 1e-4
TTT_INNER_LEARNING_RATE_LEARNING_RATE = 1e-1
TTT_OUTER_LEARNING_RATE = 1e-3
EMBEDDING_DIM = 22   # 嵌入维度
LOW_PASS_FILTER_DIM = 10  # 低通滤波器维度
MINIBATCH_SIZE = 5   # 小批量大小
DROPOUT = 0.0        # dropout比率


class PositionalEncoding(nn.Module):
    """位置编码类，用于为输入序列添加位置信息。"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos
        self.register_buffer('pe', pe)  # 注册为缓冲区，不会被视为模型参数

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, 形状为 [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]  # 添加位置编码
        out: torch.Tensor = self.dropout(x)  # 应用dropout
        return out


class SequentialNumbers(Dataset):
    """生成序列数字数据集的类。"""

    def __init__(self: Self, num_samples: int, sequence_len: int) -> None:
        self.num_samples = num_samples
        self.sequence_len = sequence_len

    def __len__(self: Self) -> int:
        return self.num_samples

    def __getitem__(self: Self, idx: int) -> Tensor:
        """生成一个序列样本。"""
        sequence = torch.zeros(self.sequence_len, VOCAB_SIZE)

        for i in range(self.sequence_len):
            hot_index = (idx + i) % VOCAB_SIZE  # 计算热编码索引
            sequence[i, hot_index] = 1.0  # 设置热编码
        return sequence


class TTTInner(nn.Module):
    """TTTInner模块，进行在线推理和参数更新。"""

    def __init__(self: Self, mini_batch_size: int, filter_dim: int,
                 get_theta_k: Callable[[], torch.nn.Parameter],
                 get_theta_q: Callable[[], torch.nn.Parameter],
                 get_theta_v: Callable[[], torch.nn.Parameter],
                 get_inner_learning_rate: Callable[[Tensor], Tensor]) -> None:
        super(TTTInner, self).__init__()

        self.mini_batch_size = mini_batch_size
        self.filter_dim = filter_dim
        self.w = nn.Linear(filter_dim, filter_dim)  # 线性层
        torch.nn.init.kaiming_uniform_(self.w.weight)  # 权重初始化

        # 获取参数的函数
        self.get_theta_k = get_theta_k
        self.get_theta_q = get_theta_q
        self.get_theta_v = get_theta_v
        self.get_inner_learning_rate = get_inner_learning_rate

    def online_inference(self: Self, src: torch.Tensor) -> torch.Tensor:
        """进行在线推理并更新参数。"""
        _sequences, _batches, _features = src.shape
        src = torch.split(src, self.mini_batch_size)  # 按小批量分割

        outputs = []
        total_loss = 0
        for minibatch in src:
            minibatch_seq, minibatch_batch, minibatch_features = minibatch.shape

            # 计算视图
            train_view = minibatch @ self.get_theta_k()  # 训练视图
            label_view = minibatch @ self.get_theta_v()  # 标签视图
            test_view = minibatch @ self.get_theta_q()  # 测试视图

            # 重建损失
            reconstruction_target = label_view - train_view  # 计算重建目标
            w_train_view = self.w(train_view)  # 线性变换
            loss = nn.MSELoss()(w_train_view, reconstruction_target)  # 计算均方误差损失
            total_loss += loss

            # 计算梯度并手动更新
            gradients = grad(loss, list(self.w.parameters()), create_graph=True)
            assert gradients[0].shape == self.w.weight.shape

            # 记录梯度信息
            wandb.log({"w_grad": gradients[0].norm()})
            wandb.log({"w_bias_grad": gradients[1].norm()})

            # 计算每个参数的内学习率
            inner_learning_rate = self.get_inner_learning_rate(minibatch)
            inner_learning_rate = inner_learning_rate.reshape(-1, self.filter_dim ** 2)
            inner_learning_rate = inner_learning_rate.mean(dim=0)
            inner_learning_rate = inner_learning_rate.reshape(self.filter_dim, self.filter_dim)
            inner_learning_rate_bias = inner_learning_rate.mean(dim=1)

            # 记录内学习率信息
            wandb.log({"inner_learning_rate": inner_learning_rate.norm()})
            wandb.log({"inner_learning_rate_bias": inner_learning_rate_bias.norm()})
            wandb.log({"inner_learning_rate_specific_index": inner_learning_rate[0][0]})

            # 更新权重和偏置
            updated_weight = self.w.weight - inner_learning_rate * gradients[0]
            updated_bias = self.w.bias - inner_learning_rate_bias * gradients[1]

            # 计算输出
            z = torch.nn.functional.linear(test_view, updated_weight, updated_bias) + test_view
            outputs.append(z)

            # 停止梯度流动（优化）
            self.w.weight.requires_grad_(False)
            self.w.bias.requires_grad_(False)

            # 更新权重和偏置
            with torch.no_grad():
                self.w.weight = nn.Parameter(updated_weight, requires_grad=True)
                self.w.bias = nn.Parameter(updated_bias, requires_grad=True)

        average_loss = total_loss / len(src)  # 计算平均损失
        wandb.log({"inner_loss": average_loss})

        return torch.concat(outputs, dim=0)  # 返回输出


class TTTHead(nn.Module):
    """TTTHead模块，负责头部的参数和推理。"""

    def __init__(self: Self, mini_batch_size: int, input_dim: int, filter_dim: int,
                 ttt_base_inner_learning_rate: float) -> None:
        super(TTTHead, self).__init__()

        self.ttt_base_inner_learning_rate = ttt_base_inner_learning_rate

        # 初始化参数
        self.theta_k = nn.Parameter(torch.randn(input_dim, filter_dim))
        self.theta_v = nn.Parameter(torch.randn(input_dim, filter_dim))
        self.theta_q = nn.Parameter(torch.randn(input_dim, filter_dim))
        self.theta_o = nn.Parameter(torch.randn(filter_dim, input_dim))
        self.inner_learning_rate_params = nn.Linear(input_dim, filter_dim ** 2)

        # 初始化TTTInner模块
        self.inner = TTTInner(mini_batch_size, filter_dim=filter_dim,
                              get_theta_k=self.get_theta_k,
                              get_theta_q=self.get_theta_q, get_theta_v=self.get_theta_v,
                              get_inner_learning_rate=self.get_inner_learning_rate)

        # 权重初始化
        torch.nn.init.kaiming_uniform_(self.theta_k)
        torch.nn.init.kaiming_uniform_(self.theta_v)
        torch.nn.init.kaiming_uniform_(self.theta_q)
        torch.nn.init.kaiming_uniform_(self.theta_o)
        torch.nn.init.kaiming_uniform_(self.inner_learning_rate_params.weight)

        self.mini_batch_size = mini_batch_size
        self.input_dim = input_dim
        self.low_pass_filter_dim = filter_dim

    def train_head(self: Self, input: torch.Tensor) -> torch.Tensor:
        """训练头部并返回输出。"""
        sequences, batches, features = input.shape

        outputs = self.inner.online_inference(input)  # 在线推理
        assert outputs.shape == (sequences, batches, self.low_pass_filter_dim)

        outputs: Tensor = outputs @ self.theta_o  # 线性变换
        assert outputs.shape == (sequences, batches, self.input_dim)
        return outputs

    def get_theta_k(self: Self) -> torch.nn.Parameter:
        return self.theta_k

    def get_theta_q(self: Self) -> torch.nn.Parameter:
        return self.theta_q

    def get_theta_v(self: Self) -> torch.nn.Parameter:
        return self.theta_v

    def get_inner_learning_rate(self: Self, input: torch.Tensor) -> Tensor:
        """计算内学习率。"""
        pre_sigmoid = self.inner_learning_rate_params(input)
        wandb.log({"pre_sigmoid": pre_sigmoid.mean()})
        post_sigmoid = self.ttt_base_inner_learning_rate * F.sigmoid(pre_sigmoid)
        wandb.log({"post_sigmoid": post_sigmoid.mean()})
        return post_sigmoid


class TTTBlock(nn.Module):
    """TTTBlock模块，包含TTTHead。"""

    def __init__(self: Self, mini_batch_size: int, embedding_dim: int, filter_dim: int,
                 ttt_base_inner_learning_rate: float) -> None:
        super(TTTBlock, self).__init__()

        self.ttt_head = TTTHead(mini_batch_size=mini_batch_size, input_dim=embedding_dim,
                                filter_dim=filter_dim,
                                ttt_base_inner_learning_rate=ttt_base_inner_learning_rate)

    def train_block(self, input: torch.Tensor) -> torch.Tensor:
        """训练块并返回输出。"""
        sequences, batches, features = input.shape
        outputs = self.ttt_head.train_head(input)  # 训练头部
        return outputs

    def set_grad_fn(self, alter_grad_fn: Callable[[bool], None]) -> None:
        self.ttt_head.set_grad_fn(alter_grad_fn)

    def set_zero_grad_fn(self, zero_grad_fn: Callable[[], None]) -> None:
        self.ttt_head.set_zero_grad_fn(zero_grad_fn)


class TTTModel(nn.Module):
    """TTT模型，包含多个TTTBlock。"""

    def __init__(
            self: Self, mini_batch_size: int, embedding_dim: int, filter_dim: int, ttt_outer_learning_rate: float,
            ttt_base_inner_learning_rate: float, ttt_inner_learning_rate_learning_rate: float, vocab_size: int,
            num_layers: int, dropout: float) -> None:
        super(TTTModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=dropout)  # 位置编码
        self.encoder = nn.Embedding(vocab_size, embedding_dim)  # 嵌入层

        # 创建TTTBlock的模块列表
        self.ttt_blocks = nn.ModuleList([TTTBlock(mini_batch_size, embedding_dim, filter_dim,
                                                  ttt_base_inner_learning_rate)
                                        for _ in range(num_layers)])

        self.lm_head = nn.Linear(embedding_dim, vocab_size)  # 线性层用于输出

        # 初始化优化器
        params = []
        for block in self.ttt_blocks:
            params.extend([block.ttt_head.theta_k, block.ttt_head.theta_q,
                           block.ttt_head.theta_v, block.ttt_head.theta_o])
        params.extend(self.pos_encoder.parameters())
        params.extend(self.encoder.parameters())
        params.extend(self.lm_head.parameters())

        self.optim = SGD(params, lr=ttt_outer_learning_rate)  # 外部学习率优化器

        # 内部学习率优化器
        params = []
        for block in self.ttt_blocks:
            params.extend(block.ttt_head.inner_learning_rate_params.parameters())
        self.optim_inner_lr = SGD(
            params,
            lr=ttt_inner_learning_rate_learning_rate)

        self.criterion = nn.CrossEntropyLoss()  # 损失函数

    def forward(self: Self, src: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        assert src.shape == (SEQUENCE_LEN, BATCH_SIZE)
        src = self.encoder(src) * math.sqrt(self.embedding_dim)  # 嵌入并缩放
        src = self.pos_encoder(src)  # 添加位置编码

        assert src.shape == (SEQUENCE_LEN, BATCH_SIZE, EMBEDDING_DIM)

        output = src
        for block in self.ttt_blocks:
            output = block.train_block(output)  # 通过每个块进行训练

        output: Tensor = self.lm_head(output)  # 线性层输出
        assert output.shape == (SEQUENCE_LEN, BATCH_SIZE, self.vocab_size)

        return output

    def train_model(self: Self, src: torch.Tensor) -> torch.Tensor:
        """训练模型并返回损失。"""
        self.optim.zero_grad()  # 清空优化器梯度
        self.optim_inner_lr.zero_grad()  # 清空内部学习率优化器梯度

        assert src.shape == (SEQUENCE_LEN, BATCH_SIZE)
        shifted_labels = src[1:, :]  # 标签向右移动一位

        src = self.encoder(src) * math.sqrt(self.embedding_dim)  # 嵌入并缩放
        src = self.pos_encoder(src)  # 添加位置编码

        assert src.shape == (SEQUENCE_LEN, BATCH_SIZE, EMBEDDING_DIM)

        output = src
        for block in self.ttt_blocks:
            output = block.train_block(output)  # 通过每个块进行训练

        # 修剪最后一个预测的token（无法训练）
        output = output[:-1, :, :]

        # lm head
        output = self.lm_head(output)  # 线性层输出
        assert output.shape == (SEQUENCE_LEN - 1, BATCH_SIZE, self.vocab_size)

        # 损失重塑
        output = output.reshape(-1, self.vocab_size)  # 重塑为二维
        assert output.shape == ((SEQUENCE_LEN - 1) * BATCH_SIZE, self.vocab_size)
        shifted_labels = shifted_labels.reshape(-1)  # 标签重塑
        assert shifted_labels.shape == ((SEQUENCE_LEN - 1) * BATCH_SIZE,)

        loss = self.criterion(output, shifted_labels)  # 计算损失
        wandb.log({"outer_loss": loss.item()})  # 记录损失

        # 确保梯度为None
        assert list(self.encoder.parameters())[0].grad is None
        for block in self.ttt_blocks:
            assert block.ttt_head.inner.w.weight.grad is None
            assert block.ttt_head.theta_k.grad is None
            assert block.ttt_head.theta_q.grad is None
            assert block.ttt_head.theta_v.grad is None
            assert block.ttt_head.theta_o.grad is None
            assert block.ttt_head.inner_learning_rate_params.weight.grad is None
        assert self.lm_head.weight.grad is None

        loss.backward()  # 反向传播

        # 记录梯度信息
        wandb.log({"w_norm": self.ttt_blocks[0].ttt_head.inner.w.weight.norm()})
        wandb.log({"inner_lr_params_grad": self.ttt_blocks[0].ttt_head.inner_learning_rate_params.weight.grad.norm()})
        wandb.log({"inner_lr_params": self.ttt_blocks[0].ttt_head.inner_learning_rate_params.weight.norm()})

        # 更新参数
        assert list(self.encoder.parameters())[0].grad is not None
        for block in self.ttt_blocks:
            assert block.ttt_head.inner.w.weight.grad is None
            assert block.ttt_head.theta_k.grad is not None
            assert block.ttt_head.theta_q.grad is not None
            assert block.ttt_head.theta_v.grad is not None
            assert block.ttt_head.theta_o.grad is not None
            assert block.ttt_head.inner_learning_rate_params.weight.grad is not None
        assert self.lm_head.weight.grad is not None
        self.optim.step()  # 更新外部学习率
        self.optim_inner_lr.step()  # 更新内部学习率
        wandb.log({"inner_learning_rate_params_specific_weight": self.ttt_blocks[0].ttt_head.inner_learning_rate_params.weight[0][0]})

        return loss.item()  # 返回损失


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)  # 检测异常
    torch.manual_seed(1234)  # 设置随机种子

    # 初始化wandb
    wandb.init(
        project="ttt",
        config={
            "architecture": "Recurrent-FF",
            "dataset": "SequentialNumbers",
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # 选择设备

    # 初始化模型
    model = TTTModel(
        num_layers=3, mini_batch_size=MINIBATCH_SIZE, embedding_dim=EMBEDDING_DIM, filter_dim=LOW_PASS_FILTER_DIM,
        ttt_outer_learning_rate=TTT_OUTER_LEARNING_RATE, ttt_base_inner_learning_rate=TTT_BASE_INNER_LEARNING_RATE,
        ttt_inner_learning_rate_learning_rate=TTT_INNER_LEARNING_RATE_LEARNING_RATE, vocab_size=VOCAB_SIZE,
        dropout=DROPOUT)
    model = model.to(device)  # 移动模型到设备

    # 创建数据集和数据加载器
    dataset = SequentialNumbers(num_samples=NUM_SAMPLES, sequence_len=SEQUENCE_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 训练模型
    model.train()
    for epoch in range(0, EPOCHS):
        for i, data in enumerate(dataloader):
            print(f"Epoch: {epoch}, Batch: {i} / {len(dataloader)}")
            data = data.to(device)  # 移动数据到设备
            data = data.permute(1, 0, 2)  # 调整数据形状
            assert data.shape == (SEQUENCE_LEN, BATCH_SIZE, VOCAB_SIZE)

            # reshape to get argmax of feature dim
            data = data.argmax(dim=2)  # 获取热编码索引
            assert data.shape == (SEQUENCE_LEN, BATCH_SIZE)

            loss = model.train_model(data)  # 训练模型并获取损失

    # 在一个序列上进行评估
    model.eval()
    for i, data in enumerate(dataloader):
        data = data.to(device)  # 移动数据到设备
        data = data.permute(1, 0, 2)  # 调整数据形状
        assert data.shape == (SEQUENCE_LEN, BATCH_SIZE, VOCAB_SIZE)

        data = data.argmax(dim=2)  # 获取热编码索引
        assert data.shape == (SEQUENCE_LEN, BATCH_SIZE)

        # 批量大小为1
        data = data[:, 0:1]  # 选择第一个样本
        BATCH_SIZE = 1
        print(data)

        output = model(data)  # 获取模型输出

        # 打印预测结果
        output = output.argmax(dim=2)  # 获取预测的索引
        output = output.squeeze(1)  # 去掉多余的维度
        print(output)

        break  # 只评估一个批次
