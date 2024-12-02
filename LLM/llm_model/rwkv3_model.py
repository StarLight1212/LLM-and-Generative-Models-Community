# RWKV语言模型的基础配置
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import logging

# 模型相关常量
RWKV_K_CLAMP = 60  # 限制K值的最大范围，e^60 约等于 1e26
RWKV_K_EPS = 1e-8  # 避免除零的小常数
RWKV_HEAD_QK_DIM = 256  # 注意力头的维度

# CUDA相关配置
T_MAX = 1024          # 最大序列长度，如果需要处理更长序列需要增加这个值
B_GROUP_FORWARD = 4   # 前向传播的批处理组大小
B_GROUP_BACKWARD = 2  # 反向传播的批处理组大小


# TimeX CUDA核心部分
class TimeX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, k, B, C, T, eps):
        """
        前向传播函数
        参数:
        - w: 权重矩阵
        - k: key矩阵
        - B: batch size
        - C: 通道数
        - T: 序列长度
        - eps: epsilon值避免除零
        """
        ctx.B = B
        ctx.C = C
        ctx.T = T
        # 确保输入张量连续且满足尺寸要求
        w = w.contiguous()
        k = k.contiguous()
        ctx.save_for_backward(w, k)
        wk = torch.empty((B, C, T), device='cuda')
        timex_cuda.forward(w, k, wk, eps, B, C, T)
        return wk

  @staticmethod
    def backward(ctx, gwk):
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0
        w, k = ctx.saved_tensors
        gw = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        gk = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_cuda.backward(w, k, gwk.contiguous(), gw,
                            gk, ctx.B, ctx.C, ctx.T)
        return (gw.sum(dim=0), gk, None, None, None, None)


def RWKV_Init(module, config):  # fancy initialization of all lin & emb layer in the module
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters():  # find the name of the weight
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0
            scale = 1.0  # extra scale for gain

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == config.vocab_size and shape[1] == config.n_embd:  # token emb?
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == config.vocab_size and shape[1] == config.n_embd:  # final projection?
                    scale = 0.5

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            # print(str(shape[0]).ljust(5), str(shape[1]).ljust(5), f'{round(scale,2):g}'.ljust(4), name)

            gain *= scale
            if scale == -999:
                nn.init.eye_(m.weight)
            elif gain == 0:
                # zero init is great for some RWKV matrices
                nn.init.zeros_(m.weight)
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0.0, std=-scale)


# RWKV时间混合层
class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        attn_sz = config.n_embd

        with torch.no_grad():  # 初始化
            self.time_curve = torch.tensor([-(config.ctx_len - 2 - i) for i in range(config.ctx_len-1)]).unsqueeze(0)
            self.time_curve = self.time_curve.to('cuda')

            ratio_0_to_1 = (layer_id / (config.n_layer - 1))  # 0 到 1
            ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer))  # 1 到 ~0
            
            # 时间衰减
            decay_speed = torch.ones(attn_sz, 1)
            for h in range(attn_sz):
                decay_speed[h][0] = -5 + 8 * (h / (attn_sz-1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            # 时间首项
            zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(attn_sz)]) * 0.5).unsqueeze(1)
            self.time_first = nn.Parameter(torch.ones(attn_sz, 1) * math.log(0.3) + zigzag)
            
            # 时间混合
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 时间偏移

        # 定义线性层
        self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, config.n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()  # x = (Batch, Time, Channel)

        # 将 x 与前一时间步混合以生成 xk, xv, xr
        xx = self.time_shift(x)  # 时间偏移
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # 使用 xk, xv, xr 生成 k, v, r
        k = self.key(xk).transpose(-1, -2)
        v = self.value(xv).transpose(-1, -2)
        r = self.receptance(xr)

        # 限制 k 的值以避免溢出
        k = torch.clamp(k, max=RWKV_K_CLAMP)
        k = torch.exp(k)
        kv = k * v

        # 计算 W 曲线
        self.time_w = torch.cat(
            [torch.exp(self.time_decay) * self.time_curve, self.time_first], dim=-1)
        w = torch.exp(self.time_w)

        # 使用 W 混合 kv 和 k
        wkv = TimeX.apply(w, kv, B, C, T, 0)
        wk = TimeX.apply(w, k, B, C, T, RWKV_K_EPS)

        rwkv = torch.sigmoid(r) * (wkv / wk).transpose(-1, -2)
        rwkv = self.output(rwkv)
        return rwkv


class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 时间偏移

        with torch.no_grad():  # 初始化时间混合
            ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer))  # 1 到 ~0

            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * config.n_embd
        self.key = nn.Linear(config.n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, config.n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))  # 使用 ReLU 激活
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv


########################################################################################################
# 使用我们的模块构建 GPT 模型
########################################################################################################

class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)

class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)  # 第一层归一化
        self.ln2 = nn.LayerNorm(config.n_embd)  # 第二层归一化

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)  # 第零层归一化

        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(config, layer_id + 1000)  # 前馈网络
        else:
            self.att = RWKV_TimeMix(config, layer_id)  # 时间混合

        self.ffn = RWKV_ChannelMix(config, layer_id)  # 通道混合

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)        
        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            x = x + self.ffnPre(self.ln1(x))  # 在某些情况下更好
        else:
            x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)  # 嵌入层

        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])  # 堆叠多个块

        self.ln_out = nn.LayerNorm(config.n_embd)  # 输出层归一化
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出层

        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)  # 查询头
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)  # 键头
            self.head_k.scale_init = 0.1
            self.register_buffer("copy_mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))  # 复制掩码

        self.ctx_len = config.ctx_len

        RWKV_Init(self, config)  # 初始化模型

        logger.info("参数数量: %e", sum(p.numel() for p in self.parameters()))  # 打印参数数量

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)  # 正态初始化
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)  # 嵌入层初始化
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()  # 偏置初始化为零

    def configure_optimizers(self, train_config):
        # 将所有参数分为会和不会经历正则化权重衰减的参数
        decay = set()
        no_decay = set()

        for mn, m in self.named_modules():  # 禁用权重衰减
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # 完整参数名称
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "参数 %s 同时出现在 decay/no_decay 集合中!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "参数 %s 未分配到 decay/no_decay 集合中!" % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.Adam(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)

        return optimizer

    def forward(self, idx, targets=None):
        self.step += 1
        B, T = idx.size()
        assert T <= self.ctx_len, "无法前向传播，因为输入长度大于模型上下文长度."
        x = self.emb(idx)  # 嵌入输入

        x = self.blocks(x)  # 通过块

        x = self.ln_out(x)  # 输出层归一化

        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]  # 查询
            k = self.head_k(x)[:, :T, :]  # 键
            c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)  # 计算注意力
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)  # 应用掩码

            c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).float()  # 计算上下文
            x = self.head(x) + c  # 输出
        else:
            x = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))  # 计算损失

        return x, loss  # 返回输出和损失
