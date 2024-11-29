########################################################################################################
# RWKV v2-RNN语言模型 - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

# 导入必要的库
from torch.utils.cpp_extension import load  # 用于加载自定义的CUDA扩展
import math
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

# 设置日志记录
logger = logging.getLogger(__name__)

########################################################################################################
# CUDA内核
########################################################################################################

# 定义一些常量
T_MAX = 1024          # 如果上下文长度大于1024，增加此值
B_GROUP_FORWARD = 4   # 设置为8以获得最佳性能
B_GROUP_BACKWARD = 2  # 设置为2以获得最佳性能

# 加载自定义的CUDA内核
timex_cuda = load(name="timex", sources=["cuda/timex_op.cpp", "cuda/timex_cuda.cu"],
                  verbose=True, extra_cuda_cflags=['--use_fast_math', '--extra-device-vectorization', f'-DTmax={T_MAX}', f'-DBF={B_GROUP_FORWARD}', f'-DBB={B_GROUP_BACKWARD}'])

# 定义自定义的autograd函数
class TimeX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, k, B, C, T, eps):
        """
        前向传播函数
        Args:
            ctx: 上下文对象，用于保存信息
            w: 权重矩阵
            k: 键矩阵
            B: 批量大小
            C: 通道数
            T: 时间步长
            eps: 小常数，防止除零
        Returns:
            wk: 输出矩阵
        """
        ctx.B = B
        ctx.C = C
        ctx.T = T
        # 确保输入参数符合要求
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0
        w = w.contiguous()  # 确保w是连续的
        k = k.contiguous()  # 确保k是连续的
        ctx.save_for_backward(w, k)  # 保存w和k以供反向传播使用
        wk = torch.empty((B, C, T), device='cuda', memory_format=torch.contiguous_format)  # 创建输出矩阵
        timex_cuda.forward(w, k, wk, eps, B, C, T)  # 调用CUDA内核进行前向计算
        return wk

    @staticmethod
    def backward(ctx, gwk):
        """
        反向传播函数
        Args:
            ctx: 上下文对象
            gwk: 输出的梯度
        Returns:
            gw: 权重的梯度
            gk: 键的梯度
        """
        assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0
        w, k = ctx.saved_tensors  # 获取保存的w和k
        gw = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda', memory_format=torch.contiguous_format)  # 创建权重梯度矩阵
        gk = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda', memory_format=torch.contiguous_format)  # 创建键梯度矩阵
        timex_cuda.backward(w, k, gwk.contiguous(), gw, gk, ctx.B, ctx.C, ctx.T)  # 调用CUDA内核进行反向计算
        return (gw.sum(dim=0), gk, None, None, None, None)  # 返回梯度

########################################################################################################
# RWKV: RWKV时间混合 + RWKV通道混合
########################################################################################################

# 定义一些常量
RWKV_K_CLAMP = 60  # e^60 = 1e26
RWKV_K_EPS = 1e-16
RWKV_HEAD_QK_DIM = 256

def RWKV_Init(module, config):
    """
    初始化模型中的线性层和嵌入层
    Args:
        module: 模型模块
        config: 配置对象
    """
    for m in module.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        with torch.no_grad():
            name = '[unknown weight]'
            for name, parameter in module.named_parameters():  # 查找权重的名称
                if id(m.weight) == id(parameter):
                    break

            shape = m.weight.data.shape
            gain = 1.0
            scale = 1.0  # 用于增益的额外缩放

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == config.vocab_size and shape[1] == config.n_embd:  # 如果是token嵌入
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()  # 将偏置初始化为0
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == config.vocab_size and shape[1] == config.n_embd:  # 如果是最终投影
                    scale = 0.5

            if hasattr(m, 'scale_init'):
                scale = m.scale_init

            gain *= scale
            if scale == -999:
                nn.init.eye_(m.weight)  # 初始化为单位矩阵
            elif gain == 0:
                nn.init.zeros_(m.weight)  # 初始化为零
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)  # 正交初始化
            else:
                nn.init.normal_(m.weight, mean=0.0, std=-scale)  # 正态分布初始化

class RWKV_TimeMix(nn.Module):
    def __init__(self, config, layer_id):
        """
        初始化RWKV时间混合层
        Args:
            config: 配置对象
            layer_id: 层的ID
        """
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd

        attn_sz = config.n_embd

        ############# 初始化时间权重曲线 ###################################
        f1_begin = 3.0
        f1_end = 1.2
        f2_begin = 0.65
        f2_end = 0.4
        with torch.no_grad():  # 初始化时间权重曲线以提高收敛性
            decay_speed = torch.ones(attn_sz, 1)
            first_sa_layer_id = 1
            for h in range(attn_sz):
                f1 = f1_begin + (layer_id-first_sa_layer_id) / \
                    (config.n_layer-1-first_sa_layer_id) * (f1_end - f1_begin)
                f2 = f2_begin + (layer_id-first_sa_layer_id) / \
                    (config.n_layer-1-first_sa_layer_id) * (f2_end - f2_begin)
                if layer_id == first_sa_layer_id:
                    f1 += 0.5
                if layer_id == config.n_layer-2:
                    f2 = 0.4
                if layer_id == config.n_layer-1:
                    f2 = 0.37
                decay_speed[h][0] = math.pow(f2, h / (attn_sz-1) * 7) * f1
        self.time_decay = nn.Parameter(torch.log(decay_speed))  # 使用exp(self.time_decay)确保time_decay > 0
        self.time_curve = torch.tensor(
            [-(config.ctx_len - 2 - i) for i in range(config.ctx_len-1)]).unsqueeze(0)
        self.time_curve = self.time_curve.to('cuda')
        self.time_first = nn.Parameter(torch.ones(attn_sz, 1) * math.log(0.3))
        #############################################################################

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 零填充以实现时间偏移
        with torch.no_grad():  # 初始化为“偏移一半的通道”
            ww = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd // 2):
                ww[0, 0, i] = 0
        self.time_mix = nn.Parameter(ww)  # 时间混合参数

        # 定义线性层
        self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, config.n_embd, bias=False)

        # 初始化缩放因子
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入张量
        Returns:
            rwkv: 输出张量
        """
        B, T, C = x.size()  # 获取输入的批量大小、时间步长和通道数

        # 计算时间混合
        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)

        # 计算键、值和接收
        k = self.key(x).transpose(-1, -2)
        v = self.value(x).transpose(-1, -2)
        r = self.receptance(x)

        # 限制k的最大值
        k = torch.clamp(k, max=RWKV_K_CLAMP)
        k = torch.exp(k)  # 计算k的指数
        kv = k * v  # 计算k和v的乘积

        # 计算时间权重
        self.time_w = torch.cat(
            [torch.exp(self.time_decay) * self.time_curve, self.time_first], dim=-1)
        w = torch.exp(self.time_w)

        # 调用自定义的TimeX函数
        wkv = TimeX.apply(w, kv, B, C, T, 0)
        wk = TimeX.apply(w, k, B, C, T, RWKV_K_EPS)

        # 计算最终输出
        rwkv = torch.sigmoid(r) * (wkv / wk).transpose(-1, -2)
        rwkv = self.output(rwkv)  # 线性变换
        return rwkv


class RWKV_ChannelMix(nn.Module):
    def __init__(self, config, layer_id):
        """
        初始化RWKV通道混合层
        Args:
            config: 配置对象
            layer_id: 层的ID
        """
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 零填充以实现时间偏移

        with torch.no_grad():  # 初始化为“偏移一半的通道”
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd // 2):
                x[0, 0, i] = 0
        self.time_mix = nn.Parameter(x)  # 时间混合参数

        hidden_sz = 4 * config.n_embd  # 隐藏层大小
        self.key = nn.Linear(config.n_embd, hidden_sz, bias=False)  # 键线性层
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)  # 接收线性层
        self.value = nn.Linear(hidden_sz, config.n_embd, bias=False)  # 值线性层

        # 初始化缩放因子
        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入张量
        Returns:
            rkv: 输出张量
        """
        # 计算时间混合
        x = x * self.time_mix + self.time_shift(x) * (1 - self.time_mix)

        k = self.key(x)  # 计算键
        k = torch.square(torch.relu(k))  # 计算键的平方
        kv = self.value(k)  # 计算值

        rkv = torch.sigmoid(self.receptance(x)) * kv  # 计算最终输出
        return rkv

########################################################################################################
# 使用我们的模块构建GPT模型
########################################################################################################

class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        """
        GPT模型配置
        Args:
            vocab_size: 词汇表大小
            ctx_len: 上下文长度
            kwargs: 其他配置参数
        """
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)  # 设置其他配置参数

class Block(nn.Module):
    def __init__(self, config, layer_id):
        """
        初始化模型块
        Args:
            config: 配置对象
            layer_id: 层的ID
        """
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)  # 第一层归一化
        self.ln2 = nn.LayerNorm(config.n_embd)  # 第二层归一化

        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(config, layer_id + 1000)  # 前馈网络
        else:
            self.att = RWKV_TimeMix(config, layer_id)  # 时间混合层

        self.ffn = RWKV_ChannelMix(config, layer_id)  # 通道混合层

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入张量
        Returns:
            x: 输出张量
        """
        x = self.ln1(x)  # 第一层归一化
        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            x = x + self.ffnPre(x)  # 在某些情况下更好
        else:
            x = x + self.att(x)  # 添加注意力层的输出
        x = self.ln2(x)  # 第二层归一化
        x = x + self.ffn(x)  # 添加前馈网络的输出
        return x

class GPT(nn.Module):
    def __init__(self, config):
        """
        初始化GPT模型
        Args:
            config: 配置对象
        """
        super().__init__()
        self.step = 0  # 训练步数
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)  # 嵌入层

        # 创建多个块
        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)  # 输出层归一化
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出层

        # 定义查询和键的线性层
        self.head_q = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
        self.head_q.scale_init = 0
        self.head_k = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)
        self.head_k.scale_init = 0.1
        self.register_buffer("copy_mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))  # 复制掩码

        self.ctx_len = config.ctx_len  # 上下文长度

        RWKV_Init(self, config)  # 初始化权重

        logger.info("参数数量: %e", sum(p.numel() for p in self.parameters()))  # 打印参数数量

    def get_ctx_len(self):
        """获取上下文长度"""
        return self.ctx_len

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)  # 正态分布初始化
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)  # 嵌入层初始化
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()  # 偏置初始化为0

    def configure_optimizers(self, train_config):
        """配置优化器"""
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
        assert len(inter_params) == 0, "参数 %s 同时出现在衰减和不衰减集合中!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "参数 %s 未分配到衰减或不衰减集合中!" % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.Adam(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)

        return optimizer

    def forward(self, idx, targets=None):
        """
        前向传播函数
        Args:
            idx: 输入索引
            targets: 目标值
        Returns:
            x: 输出张量
            loss: 损失值
        """
        self.step += 1  # 增加步数
        B, T = idx.size()  # 获取批量大小和时间步长
        assert T <= self.ctx_len, "无法前向传播，因为输入长度大于模型上下文长度."
        x = self.emb(idx)  # 嵌入索引

        x = self.blocks(x)  # 通过块进行前向传播

        x = self.ln_out(x)  # 输出层归一化

        # 计算查询和键
        q = self.head_q(x)[:, :T, :]
        k = self.head_k(x)[:, :T, :]
        c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)  # 计算注意力得分
        c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)  # 应用掩码

        c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).float()  # 计算最终输出
        x = self.head(x) + c  # 线性变换并加上注意力得分

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))  # 计算交叉熵损失

        return x, loss  # 返回输出和损失

# model_run
import types
import copy
import torch
from torch.nn import functional as F

# 定义常量
RWKV_K_CLAMP = 60  # K的最大值
RWKV_K_EPS = 1e-16  # 防止除零的小常数
RWKV_HEAD_QK_DIM = 256  # 查询和键的维度

DEBUG_TIME = False   # 是否显示训练时间系数

class RWKV_RNN():
    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, n_layer, n_embd, ctx_len):
        """
        初始化RWKV RNN模型
        Args:
            MODEL_NAME: 模型文件名
            RUN_DEVICE: 运行设备（如'cuda'或'cpu'）
            model_type: 模型类型
            n_layer: 层数
            n_embd: 嵌入维度
            ctx_len: 上下文长度
        """
        self.RUN_DEVICE = RUN_DEVICE  # 设置运行设备
        self.model_type = model_type  # 设置模型类型
        self.n_layer = n_layer  # 设置层数
        self.n_embd = n_embd  # 设置嵌入维度
        self.ctx_len = ctx_len  # 设置上下文长度

        self.w = types.SimpleNamespace()  # 创建一个简单的命名空间用于存储权重

        # 加载模型权重
        w = torch.load(MODEL_NAME + '.pth', map_location=torch.device(RUN_DEVICE))
        for x in w.keys():
            # 处理时间相关的权重
            if '.time_' in x:
                w[x] = w[x].squeeze()  # 去掉多余的维度
            if '.time_decay' in x:
                w[x] = torch.exp(-torch.exp(w[x]))  # 计算时间衰减
            if '.time_first' in x:
                w[x] = torch.exp(w[x])  # 计算时间初始值
            if DEBUG_TIME and '.time_' in x:
                print(x, w[x].squeeze().cpu().numpy())  # 调试输出时间权重

            # 将权重存储到命名空间中
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()  # 创建新的命名空间
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])  # 设置权重
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})  # 创建字典
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())  # 创建新的命名空间
                    here = getattr(here, xx[i])

        self.clear()  # 清空状态

    def clear(self):
        """清空内部状态"""
        self.xx = {}  # 存储中间结果
        self.aa = {}  # 存储时间混合的中间结果
        self.bb = {}  # 存储时间混合的中间结果
        self.hk = None  # 存储历史键

    def save(self, target):
        """保存当前状态到目标对象"""
        target.xx = copy.deepcopy(self.xx)
        target.aa = copy.deepcopy(self.aa)
        target.bb = copy.deepcopy(self.bb)
        target.hk = copy.deepcopy(self.hk)

    def load(self, target):
        """从目标对象加载状态"""
        self.xx = copy.deepcopy(target.xx)
        self.aa = copy.deepcopy(target.aa)
        self.bb = copy.deepcopy(target.bb)
        self.hk = copy.deepcopy(target.hk)

    def LN(self, xx, w):
        """层归一化"""
        return F.layer_norm(xx, (self.n_embd,), weight=w.weight, bias=w.bias)

    def FF(self, xx, w, name):
        """前馈网络"""
        if name not in self.xx:
            self.xx[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)  # 初始化中间结果
        x = xx * w.time_mix + self.xx[name] * (1 - w.time_mix)  # 混合当前输入和上一个输入
        self.xx[name] = xx  # 更新中间结果

        r = torch.sigmoid(w.receptance.weight @ x)  # 计算接收权重
        k = torch.square(torch.relu(w.key.weight @ x))  # 计算键
        kv = w.value.weight @ k  # 计算值

        return r * kv  # 返回加权值

    def SA(self, xx, w, name):
        """自注意力机制"""
        if name not in self.xx:
            self.xx[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)  # 初始化中间结果
            self.aa[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)  # 初始化时间混合结果
            self.bb[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)  # 初始化时间混合结果
        x = xx * w.time_mix + self.xx[name] * (1 - w.time_mix)  # 混合当前输入和上一个输入
        self.xx[name] = xx  # 更新中间结果

        r = torch.sigmoid(w.receptance.weight @ x)  # 计算接收权重

        k = torch.exp(torch.clamp(w.key.weight @ x, max=RWKV_K_CLAMP))  # 计算键并限制最大值
        v = w.value.weight @ x  # 计算值
        kv = k * v  # 计算加权值

        # 更新时间混合结果
        a = self.aa[name] + w.time_first * kv
        b = self.bb[name] + w.time_first * k
        self.aa[name] = w.time_decay * self.aa[name] + kv
        self.bb[name] = w.time_decay * self.bb[name] + k

        rwkv = r * a / (b + RWKV_K_EPS)  # 计算RWKV输出

        return w.output.weight @ rwkv  # 返回最终输出

    def run(self, ctx):
        """运行模型"""
        w = self.w
        x = w.emb.weight[ctx[-1]]  # 获取最后一个上下文的嵌入

        for i in range(self.n_layer):
            x = self.LN(x, w.blocks[i].ln1)  # 第一层归一化
            if i == 0 and self.model_type == 'RWKV-ffnPre':
                x = x + self.FF(x, w.blocks[i].ffnPre, f'ffnPre.{i}')  # 前馈网络
            else:
                x = x + self.SA(x, w.blocks[i].att, f'att.{i}')  # 自注意力机制
            x = self.LN(x, w.blocks[i].ln2)  # 第二层归一化
            x = x + self.FF(x, w.blocks[i].ffn, f'ffn.{i}')  # 前馈网络

        x = self.LN(x, w.ln_out)  # 输出层归一化

        # 更新历史键
        if self.hk is None:
            self.hk = (w.head_k.weight @ x).unsqueeze(0)  # 初始化历史键
        else:
            self.hk = torch.cat([self.hk, (w.head_k.weight @ x).unsqueeze(0)], dim=0)  # 追加历史键
        if self.hk.shape[0] > self.ctx_len:
            self.hk = self.hk[-self.ctx_len:, :]  # 保持历史键的长度

        q = w.head_q.weight @ x  # 计算查询

        x = w.head.weight @ x  # 计算输出
        x = x.cpu().numpy().tolist()  # 转换为列表

        c = (self.hk @ q) / RWKV_HEAD_QK_DIM  # 计算上下文
        for i in range(len(c)):
            x[ctx[i]] += c[i]  # 更新输出

        return x  # 返回最终输出
