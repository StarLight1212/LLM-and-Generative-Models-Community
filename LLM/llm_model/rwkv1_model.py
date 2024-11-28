import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
logger = logging.getLogger(__name__)

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

def RWKV_Init(module, config):  # 初始化模块中的所有线性和嵌入层
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):  # 只处理线性层和嵌入层
            with torch.no_grad():
                # 获取权重的名称
                name = next((name for name, parameter in module.named_parameters() if id(m.weight) == id(parameter)), '[unknown weight]')
                
                shape = m.weight.data.shape  # 获取权重形状
                gain = 1.0  # 正值：正交增益，负值：正态分布标准差
                scale = 1.0  # 增益的额外缩放

                # 处理线性层
                if isinstance(m, nn.Linear):
                    if m.bias is not None:
                        m.bias.data.zero_()  # 将偏置初始化为零
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])  # 根据形状调整增益
                    if shape == (config.vocab_size, config.n_embd):  # 最终投影
                        scale = config.rwkv_emb_scale

                # 处理嵌入层
                elif isinstance(m, nn.Embedding):
                    gain = math.sqrt(max(shape))  # 根据最大形状调整增益
                    if shape == (config.vocab_size, config.n_embd):  # 词元嵌入
                        scale = config.rwkv_emb_scale

                # 如果有自定义缩放初始化
                scale = getattr(m, 'scale_init', scale)

                print(f"{shape[0]:<5} {shape[1]:<5} {round(scale, 2):<4} {name}")  # 打印形状和缩放信息

                # 初始化权重
                gain *= scale
                if gain == 0:
                    nn.init.zeros_(m.weight)  # 零初始化
                elif gain > 0:
                    nn.init.orthogonal_(m.weight, gain=gain)  # 正交初始化
                else:
                    nn.init.normal_(m.weight, mean=0, std=-gain)  # 正态初始化


class RWKV_TimeMix(nn.Module):
    """
    RWKV_TimeMix 模块实现了时间混合机制。
    该模块的架构包括：
    - 输入：x (B, T, C)
    - 线性层：key、value、receptance
    - 时间权重：time_w、time_alpha、time_beta、time_gamma
    - 输出：rwkv (B, T, n_embd)
    """
    
    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_attn % config.n_head == 0  # 确保注意力头数可以整除
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_head = config.n_head
        self.head_size = config.n_attn // config.n_head

        # 初始化时间权重
        with torch.no_grad():
            ww = torch.ones(config.n_head, config.ctx_len)
            curve = torch.tensor([-(config.ctx_len - 1 - i) for i in range(config.ctx_len)])  # 距离
            for h in range(config.n_head):
                decay_speed = math.pow(config.ctx_len, -(h + 1) / (config.n_head - 1)) if h < config.n_head - 1 else 0.0
                ww[h] = torch.exp(curve * decay_speed)  # 计算时间权重
        self.time_w = nn.Parameter(ww)

        # 初始化时间参数
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 时间偏移

        # 定义线性层
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)
        self.receptance = nn.Linear(config.n_embd, config.n_attn)
        self.output = nn.Linear(config.n_attn, config.n_embd)

        # 初始化缩放参数
        for layer in [self.key, self.receptance, self.output]:
            layer.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len
        
        # 处理时间权重
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])[:, :-TT].reshape(-1, TT, 2 * TT - 1)[:, :, TT - 1:]  # 形成循环矩阵
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        # 处理输入
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)

        # 计算 key、value 和 receptance
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        # 限制 k 的值范围
        k = torch.clamp(k, max=30, min=-60)
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        # 计算加权值
        kv = (k * v).view(B, T, self.n_head, self.head_size)
        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        # 计算 rwkv
        rwkv = torch.sigmoid(r) * wkv / sum_k
        rwkv = self.output(rwkv)

        return rwkv * self.time_gamma[:T, :]  # 返回加权结果


class RWKV_ChannelMix(nn.Module):
    """
    RWKV_ChannelMix类实现了一个通道混合模块，包含以下结构：
    - 输入通过时间偏移进行处理
    - 计算key、value和receptance
    - 通过加权计算得到rwkv输出
    """
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 时间偏移层
        
        # 定义隐藏层大小
        hidden_sz = 5 * config.n_ffn // 2  # 由于receptance门控，可以使用更小的隐藏层大小
        self.key = nn.Linear(config.n_embd, hidden_sz)  # key线性层
        self.value = nn.Linear(config.n_embd, hidden_sz)  # value线性层
        self.weight = nn.Linear(hidden_sz, config.n_embd)  # 权重线性层
        self.receptance = nn.Linear(config.n_embd, config.n_embd)  # receptance线性层

        # 初始化缩放参数
        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()  # 获取输入的批次大小、时间步长和通道数
        
        # 处理输入，进行时间偏移
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        k = self.key(x)  # 计算key
        v = self.value(x)  # 计算value
        r = self.receptance(x)  # 计算receptance
        
        # 计算加权值
        wkv = self.weight(F.mish(k) * v)  # 使用Mish激活函数

        # 计算rwkv输出
        rwkv = torch.sigmoid(r) * wkv

        return rwkv  # 返回rwkv输出


class RWKV_TinyAttn(nn.Module):
    """
    RWKV_TinyAttn类实现了一个小型注意力机制，包含以下结构：
    - 输入通过线性层计算查询、键、值
    - 计算注意力权重并应用于值
    - 输出经过线性层处理的结果
    """
    def __init__(self, config):
        super().__init__()
        self.d_attn = config.rwkv_tiny_attn  # 注意力维度
        self.n_head = config.rwkv_tiny_head   # 注意力头数
        self.head_size = self.d_attn // self.n_head  # 每个头的维度

        # 定义查询、键、值的线性变换
        self.qkv = nn.Linear(config.n_embd, self.d_attn * 3)
        self.out = nn.Linear(self.d_attn, config.n_embd)  # 输出线性层

    def forward(self, x, mask):
        B, T, C = x.size()  # 获取输入的批次大小、时间步长和通道数
        qkv = self.qkv(x)  # 计算查询、键、值
        q, k, v = qkv.chunk(3, dim=-1)  # 将qkv分割为q、k、v

        # 如果有多个头，则调整维度
        if self.n_head > 1:
            q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
            k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
            v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)

        # 计算注意力权重
        qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))  # (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        qk = qk.masked_fill(mask == 0, float('-inf'))  # 应用掩码
        qk = F.softmax(qk, dim=-1)  # 计算softmax
        qkv = qk @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)

        # 如果有多个头，则调整输出维度
        if self.n_head > 1:
            qkv = qkv.transpose(1, 2).contiguous().view(B, T, -1)  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        return self.out(qkv)  # 返回最终输出

########################################################################################################
# MHA_rotary: Multi-head Attention + Rotary Encoding + GeGLU FFN
########################################################################################################

class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding类实现了旋转嵌入，用于在多头注意力机制中引入位置编码。
    - 输入维度：dim
    - 基数：base，默认为10000
    - 计算频率的倒数并缓存余弦和正弦值
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        # 计算频率的倒数
        self.inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', self.inv_freq)  # 注册为缓冲区
        self.seq_len_cached = None  # 缓存序列长度
        self.cos_cached = None  # 缓存余弦值
        self.sin_cached = None  # 缓存正弦值

    def forward(self, x, seq_len=None):
        # 如果序列长度变化，重新计算余弦和正弦值
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device)  # 创建时间步长
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # 计算频率
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)  # 拼接频率
            self.cos_cached = emb.cos()  # 计算余弦
            self.sin_cached = emb.sin()  # 计算正弦
        return self.cos_cached, self.sin_cached  # 返回缓存的余弦和正弦值

def rotate_half(x):
    # 将输入张量分为两部分并旋转
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)  # 连接旋转后的部分

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    # 应用旋转位置编码
    cos, sin = cos[...,:q.shape[-2],:], sin[...,:q.shape[-2],:]  # 截取余弦和正弦值
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)  # 返回旋转编码后的q和k

class MHA_rotary(nn.Module):
    """
    MHA_rotary类实现了多头注意力机制，结合旋转位置编码和GeGLU前馈网络。
    - 输入：x (B, T, C)，其中B为批量大小，T为序列长度，C为特征维度。
    - 输出：经过注意力机制处理后的张量。
    """
    def __init__(self, config, layer_id, time_shift=False):
        super().__init__()
        self.layer_id = layer_id
        assert config.n_attn % config.n_head == 0  # 确保注意力头数可以整除总维度
        self.n_head = config.n_head  # 注意力头数
        self.ctx_len = config.ctx_len  # 上下文长度
        self.head_size = config.n_attn // config.n_head  # 每个头的维度

        # 如果启用时间偏移，初始化零填充层
        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # 定义线性层
        self.query = nn.Linear(config.n_embd, config.n_attn)
        self.key = nn.Linear(config.n_embd, config.n_attn)
        self.value = nn.Linear(config.n_embd, config.n_attn)

        # 注册下三角掩码
        self.register_buffer("mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))

        # 初始化旋转嵌入维度
        self.rotary_ndims = self.head_size // 2
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)  # 旋转嵌入实例

        self.output = nn.Linear(config.n_attn, config.n_embd)  # 输出层

    def forward(self, x):
        B, T, C = x.size()  # 获取输入的批量大小、序列长度和特征维度

        # 如果启用时间偏移，进行处理
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)

        # 计算查询、键、值
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)    # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)

        # 分离旋转嵌入部分
        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)  # 获取旋转嵌入
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # 应用旋转位置编码

        # 连接查询和键的剩余部分
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)

        # 计算自注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, hs) * (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))  # 应用因果掩码
        att = F.softmax(att, dim=-1)  # 计算softmax

        # 计算输出
        x = att @ v  # (B, nh, T, T) * (B, nh, T, hs) -> (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)  # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)

        x = self.output(x)  # 通过输出层
        return x  # 返回最终输出


class GeGLU(torch.nn.Module):
    """
    GeGLU模型架构：
    该模型使用Gated Linear Unit (GeLU) 激活函数，结合线性变换来处理输入数据。
    包含以下层：
    - key: 线性层，将输入映射到隐藏层大小的三倍
    - value: 线性层，将输入映射到隐藏层大小的三倍
    - weight: 线性层，将隐藏层大小的三倍映射回输入维度
    可选的时间偏移功能通过ZeroPad2d实现。
    """

    def __init__(self, config, layer_id, time_shift=False):
        super().__init__()
        self.layer_id = layer_id

        # 如果启用时间偏移，初始化相应的层
        if time_shift:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        hidden_sz = 3 * config.n_ffn  # 隐藏层大小为3倍的前馈网络大小
        self.key = nn.Linear(config.n_embd, hidden_sz)  # 查询线性层
        self.value = nn.Linear(config.n_embd, hidden_sz)  # 值线性层
        self.weight = nn.Linear(hidden_sz, config.n_embd)  # 输出线性层

    def forward(self, x):
        B, T, C = x.size()  # 获取输入的批量大小、序列长度和特征维度

        # 如果启用时间偏移，进行处理
        if hasattr(self, 'time_shift'):
            x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)

        k = self.key(x)  # 计算键
        v = self.value(x)  # 计算值
        y = self.weight(F.gelu(k) * v)  # 计算输出
        return y  # 返回最终输出

########################################################################################################
# MHA_pro: with more tricks
########################################################################################################

class MHA_pro(nn.Module):
    """
    MHA_pro模型架构：
    该模型实现了多头自注意力机制，包含以下层：
    - query、key、value: 线性层，将输入映射到注意力维度
    - time_w: 时间权重参数
    - time_alpha、time_beta、time_gamma: 用于时间加权的参数
    - mask: 生成的因果掩码
    - rotary_emb: 旋转嵌入，用于位置编码
    - head_mix: 用于混合头的卷积层
    - output: 输出线性层，将注意力输出映射回输入维度
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        assert config.n_attn % config.n_head == 0  # 确保注意力头数可以整除
        self.n_head = config.n_head
        self.ctx_len = config.ctx_len
        self.head_size = config.n_attn // config.n_head

        # 初始化时间相关参数
        self.time_w = nn.Parameter(torch.ones(self.n_head, config.ctx_len))
        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, config.ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, config.ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(config.ctx_len, 1))
        self.register_buffer("mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))  # 生成因果掩码

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 时间偏移
        self.query = nn.Linear(config.n_embd, config.n_attn)  # 查询线性层
        self.key = nn.Linear(config.n_embd, config.n_attn)    # 键线性层
        self.value = nn.Linear(config.n_embd, config.n_attn)  # 值线性层
        
        self.rotary_ndims = self.head_size // 2  # 旋转嵌入维度
        self.rotary_emb = RotaryEmbedding(self.rotary_ndims)  # 旋转嵌入实例

        self.head_mix = nn.Conv2d(self.n_head, self.n_head, kernel_size=1, bias=False)  # 混合头的卷积层
        self.output = nn.Linear(config.n_attn, config.n_embd)  # 输出线性层

    def forward(self, x):
        B, T, C = x.size()  # 获取输入的批量大小、序列长度和特征维度
        TT = self.ctx_len
        
        # 计算时间权重
        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])[:, :-TT].reshape(-1, TT, 2 * TT - 1)[:, :, TT-1:]  # 形成循环矩阵
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]  # 应用时间加权

        # 时间偏移混合
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        
        # 计算查询、键和值
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)    # (B, T, C) -> (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)

        # 旋转位置编码
        q, query_pass = q[..., :self.rotary_ndims], q[..., self.rotary_ndims:]
        k, key_pass = k[..., :self.rotary_ndims], k[..., self.rotary_ndims:]
        cos, sin = self.rotary_emb(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # 应用旋转编码
        q = torch.cat((q, query_pass), dim=-1)
        k = torch.cat((k, key_pass), dim=-1)  
        
        # 计算自注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))  # 应用因果掩码
        att = F.softmax(att, dim=-1)  # softmax
        att = att * w  # 应用时间加权
        att = self.head_mix(att)  # 混合头

        # 计算输出
        x = att @ v  # (B, nh, T, hs)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, nh, hs) -> (B, T, C)

        x = self.output(x) * self.time_gamma[:T, :]  # 通过输出层
        return x  # 返回最终输出

########################################################################################################
# The GPT Model with our blocks
########################################################################################################


class RMSNorm(nn.Module):
    """
    RMSNorm类实现了均方根归一化。
    该类的构造函数接受一个参数d，表示特征维度。
    forward方法对输入x进行归一化处理。
    """
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1. / 2)  # 计算归一化因子
        self.weight = nn.Parameter(torch.ones(d))  # 可学习的权重参数

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)  # 计算L2范数
        x_normed = x / (norm_x * self.dd + 1e-12)  # 归一化处理
        return self.weight * x_normed  # 返回加权后的归一化结果


class FixedNorm(nn.Module):
    """
    FixedNorm类实现了固定归一化。
    该类的构造函数接受一个参数d，表示特征维度。
    forward方法对输入x进行归一化处理。
    """
    def __init__(self, d):
        super().__init__()
        self.dd = d ** (-1. / 2)  # 计算归一化因子

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)  # 计算L2范数
        x_normed = x / (norm_x * self.dd + 1e-12)  # 归一化处理
        return x_normed  # 返回归一化结果


########################################################################################################

class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k,v in kwargs.items():
            setattr(self, k, v)

class Block(nn.Module):
    """
    Block类实现了模型的基本构建块。
    包含两个层归一化和一个注意力机制与前馈网络的组合。
    支持不同的模型类型：RWKV、MHA_rotary、MHA_shift和MHA_pro。
    """

    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config

        # 初始化层归一化
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        # 根据模型类型选择不同的注意力机制和前馈网络
        model_type_map = {
            'RWKV': (RWKV_TimeMix, RWKV_ChannelMix),
            'MHA_rotary': (MHA_rotary, GeGLU),
            'MHA_shift': (lambda c, l: MHA_rotary(c, l, time_shift=True), lambda c, l: GeGLU(c, l, time_shift=True)),
            'MHA_pro': (MHA_pro, RWKV_ChannelMix)
        }

        self.attn, self.mlp = model_type_map[config.model_type](config, layer_id)

    def forward(self, x):
        # 通过注意力机制和前馈网络进行前向传播
        x = x + self.attn(self.ln1(x))  # 添加注意力输出
        x = x + self.mlp(self.ln2(x))    # 添加前馈网络输出
        
        return x

class GPT(nn.Module):
    """
    GPT模型类，包含以下结构：
    1. 词嵌入层：将输入的token转换为向量表示。
    2. 多个Block层：每个Block包含注意力机制和前馈网络。
    3. 最终层归一化：对输出进行归一化处理。
    4. 输出层：将模型输出映射到词汇表大小。
    5. 额外的线性层用于计算q和k。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 词嵌入层
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # 多个Block层
        self.blocks = nn.Sequential(*[Block(config, i) for i in range(config.n_layer)])

        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.n_embd)
        # 用于减少早期token的置信度
        self.time_out = nn.Parameter(torch.ones(1, config.ctx_len, 1))
        # 输出层
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 额外的线性层
        self.head_q = nn.Linear(config.n_embd, 256)
        self.head_k = nn.Linear(config.n_embd, 256)
        self.register_buffer("copy_mask", torch.tril(torch.ones(config.ctx_len, config.ctx_len)))

        self.ctx_len = config.ctx_len

        # 初始化权重
        if self.config.model_type == 'RWKV':
            RWKV_Init(self, config)
        else:
            self.apply(self._init_weights)

        logger.info("参数总数: %e", sum(p.numel() for p in self.parameters()))

    def get_ctx_len(self):
        """获取上下文长度"""
        return self.ctx_len

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        """配置优化器，分离出需要和不需要正则化的参数"""
        decay, no_decay = set(), set()

        # 权重模块的白名单和黑名单
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (RMSNorm, nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn  # 完整参数名

                if pn.endswith('bias') or ('time' in fpn) or ('head' in fpn):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # 验证所有参数都已考虑
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"参数 {str(inter_params)} 同时出现在 decay 和 no_decay 集合中！"
        assert len(param_dict.keys() - union_params) == 0, f"参数 {str(param_dict.keys() - union_params)} 未分离到 decay/no_decay 集合中！"

        # 配置优化器
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)
        return optimizer

    def forward(self, idx, targets=None):
        """前向传播"""
        B, T = idx.size()
        assert T <= self.ctx_len, "输入长度超过模型上下文长度。"

        x = self.tok_emb(idx)  # 词嵌入

        x = self.blocks(x)  # 通过Block层

        x = self.ln_f(x)  # 最终层归一化

        # 计算q和k
        q = self.head_q(x)[:, :T, :]
        k = self.head_k(x)[:, :T, :]
        c = (q @ k.transpose(-2, -1)) * (1.0 / 256)  # 计算注意力得分
        c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)  # 应用掩码
        c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).float()  # 计算copy机制

        x = x * self.time_out[:, :T, :]  # 减少早期token的置信度
        x = self.head(x) + c  # 输出层

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))  # 计算损失

        return x, loss  # 返回输出和损失
