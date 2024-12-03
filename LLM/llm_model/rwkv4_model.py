########################################################################################################
# RWKV 语言模型 - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import math, os
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

# 尝试从 DeepSpeed 导入 FusedAdam 优化器
try:
    from deepspeed.ops.adam import FusedAdam
except:
    pass  # 一些 Windows 用户可能无法安装 DeepSpeed

# 设置日志记录器
logger = logging.getLogger(__name__)

RWKV_HEAD_QK_DIM = 0  # 定义一个常量，用于头部的 QK 维度
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')

# 定义一个自定义的 autograd 函数 L2Wrap
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)  # 保存 y 以便在反向传播中使用
        return loss  # 返回损失值

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]  # 获取保存的 y
        # 鼓励 logits 接近 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])  # 计算因子
        maxx, ids = torch.max(y, -1, keepdim=True)  # 找到 y 的最大值及其索引
        gy = torch.zeros_like(y)  # 创建与 y 相同形状的零张量
        gy.scatter_(-1, ids, maxx * factor)  # 在最大值位置填充因子
        return (grad_output, gy)  # 返回梯度

########################################################################################################
# CUDA 内核
########################################################################################################

T_MAX = 1024  # 最大时间步长，增加此值如果 ctx_len 较长 [注意：会占用大量显存！]
# 如果你切片 ctx 并在每个切片中传递隐藏状态，可以超出 CUDA 限制

from torch.utils.cpp_extension import load
# 加载 CUDA 扩展
wkv_cuda = load(name="wkv", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

# 定义自定义的 autograd 函数 WKV
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B  # 保存批次大小
        ctx.T = T  # 保存时间步长
        ctx.C = C  # 保存通道数
        assert T <= T_MAX  # 确保 T 不超过最大值
        assert B * C % min(C, 1024) == 0  # 确保 B 和 C 的乘积可以被 1024 整除

        # 根据环境变量选择数据类型
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            w = -torch.exp(w.contiguous())  # 计算 w 的负指数
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else:
            w = -torch.exp(w.float().contiguous())  # 转换为 float 类型并计算负指数
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()

        ctx.save_for_backward(w, u, k, v)  # 保存 w, u, k, v 以便反向传播使用
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)  # 创建输出张量
        wkv_cuda.forward(B, T, C, w, u, k, v, y)  # 调用 CUDA 内核进行前向计算

        # 根据环境变量返回不同的数据类型
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return y
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return y.half()  # 返回半精度
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return y.bfloat16()  # 返回 bfloat16 精度

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B  # 获取批次大小
        T = ctx.T  # 获取时间步长
        C = ctx.C  # 获取通道数
        assert T <= T_MAX  # 确保 T 不超过最大值
        assert B * C % min(C, 1024) == 0  # 确保 B 和 C 的乘积可以被 1024 整除

        w, u, k, v = ctx.saved_tensors  # 获取保存的张量
        gw = torch.zeros((B, C), device='cuda').contiguous()  # 创建梯度张量
        gu = torch.zeros((B, C), device='cuda').contiguous()  # 创建梯度张量
        gk = torch.zeros((B, T, C), device='cuda').contiguous()  # 创建梯度张量
        gv = torch.zeros((B, T, C), device='cuda').contiguous()  # 创建梯度张量

        # 根据环境变量选择数据类型
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        else:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)

        gw = torch.sum(gw, dim=0)  # 对 gw 进行求和
        gu = torch.sum(gu, dim=0)  # 对 gu 进行求和

        # 根据环境变量返回不同的数据类型
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return (None, None, None, gw, gu, gk, gv)
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())

# 定义运行 CUDA 的函数
def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())  # 调用 WKV 的 apply 方法

########################################################################################################
# RWKV: RWKV 时间混合 + RWKV 通道混合
########################################################################################################

# 初始化 RWKV 模型的函数
def RWKV_Init(model, args):  # 初始化模型中的所有线性和嵌入层
    print("\n[--> first run, init model params (very slow for large models) <--]")
    print("[so you shall only do it for 1 single GPU and save the checkpt and load it when using multiple GPU]\n")

    for mm in model.modules():  # 遍历模型中的所有模块
        if "RecursiveScriptModule" in str(type(mm)):
            if mm.original_name not in ["Linear"]:  # 只处理线性层
                continue
            ww = None
            for name, param in mm.named_parameters():  # 获取权重参数
                if name == "weight":
                    ww = param
        else:
            m = mm
            if not isinstance(m, (nn.Linear, nn.Embedding)):  # 只处理线性层和嵌入层
                continue
            ww = m.weight  # 获取权重

        with torch.no_grad():  # 不计算梯度
            name = "[unknown weight]"
            for name, parameter in model.named_parameters():  # 查找权重的名称
                if id(ww) == id(parameter):
                    break

            shape = ww.shape  # 获取权重的形状
            gain = 1.0
            scale = 1.0  # 增加增益的缩放

            if isinstance(m, nn.Embedding):  # 如果是嵌入层
                gain = math.sqrt(max(shape[0], shape[1]))  # 计算增益
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:  # 如果是词嵌入
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(m, nn.Linear):  # 如果是线性层
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])  # 计算增益
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:  # 如果是最终投影
                    scale = 0.5

            if hasattr(m, "scale_init"):  # 如果有 scale_init 属性
                scale = m.scale_init

            # print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {name}")

            gain *= scale  # 更新增益
            if scale == -999:
                nn.init.eye_(ww)  # 初始化为单位矩阵
            elif gain == 0:
                # 零初始化对某些 RWKV 矩阵非常有效
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)  # 正交初始化
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)  # 正态初始化

# 定义 RWKV 时间混合类
class RWKV_TimeMix(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id  # 层 ID
        self.ctx_len = config.ctx_len  # 上下文长度
        self.n_embd = config.n_embd  # 嵌入维度

        attn_sz = config.n_embd  # 注意力大小

        with torch.no_grad():  # 不计算梯度进行初始化
            ratio_0_to_1 = (layer_id / (config.n_layer - 1))  # 从 0 到 1 的比例
            ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer))  # 从 1 到接近 0 的比例
            
            # 精细的时间衰减
            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz-1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)  # 时间衰减参数

            # 精细的时间初始值
            zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(attn_sz)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)  # 时间初始值参数
            
            # 精细的时间混合
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))  # 时间混合参数 k
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)  # 时间混合参数 v
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))  # 时间混合参数 r
            
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 零填充
        """
        左填充：0，表示在宽度方向的左侧不填充。
        右填充：0，表示在宽度方向的右侧不填充。
        上填充：1，表示在高度方向的顶部填充 1 行零。
        下填充：-1，表示在高度方向的底部填充的数量为 -1，这意味着底部的填充量将根据输入的高度自动计算，以保持整体高度不变。
        实际效果
        上侧填充 1：在张量的顶部添加一行零。
        下侧填充 -1：根据输入的高度自动计算下侧的填充量，使得整体高度保持不变。
        import torch
        import torch.nn as nn
        
        # 创建一个示例张量
        x = torch.tensor([[[1, 2, 3, 4]]], dtype=torch.float32)  # 形状为 (1, 1, 4)
        
        # 应用零填充
        time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        padded_x = time_shift(x)
        
        print("原始张量形状:", x.shape)  # 输出: torch.Size([1, 1, 4])
        print("填充后的张量形状:", padded_x.shape)  # 输出: torch.Size([1, 1, 4])
        print("填充后的张量内容:\n", padded_x)  # 输出填充后的内容

        原始张量形状: torch.Size([1, 1, 4])
        填充后的张量形状: torch.Size([1, 1, 4])
        填充后的张量内容:
         tensor([[[0., 0., 0., 0.]]])
        """

        # 定义线性层
        self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(config.n_embd, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, config.n_embd, bias=False)  # 输出层

        # 初始化缩放参数
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    @torch.jit.script_method
    def jit_func(self, x):
        # 将 x 与前一个时间步混合以生成 xk, xv, xr
        xx = self.time_shift(x)  # 获取前一个时间步的输入
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 计算 k
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)  # 计算 v
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 计算 r

        # 使用 xk, xv, xr 生成 k, v, r
        k = self.key(xk)  # 计算 k
        v = self.value(xv)  # 计算 v
        r = self.receptance(xr)  # 计算 r
        sr = torch.sigmoid(r)  # 对 r 应用 sigmoid 激活函数

        return sr, k, v  # 返回 sr, k, v

    def forward(self, x):
        B, T, C = x.size()  # 获取输入的批次大小、时间步长和通道数

        sr, k, v = self.jit_func(x)  # 调用 jit_func 计算 sr, k, v

        rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)  # 计算 RWKV
        rwkv = self.output(rwkv)  # 通过输出层
        return rwkv  # 返回结果


# 定义 RWKV 通道混合类
class RWKV_ChannelMix(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id  # 层 ID

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 零填充

        with torch.no_grad():  # 不计算梯度进行初始化
            ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer))  # 从 1 到接近 0 的比例

            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))  # 时间混合参数 k
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))  # 时间混合参数 r

        hidden_sz = 4 * config.n_embd  # 隐藏层大小
        self.key = nn.Linear(config.n_embd, hidden_sz, bias=False)  # 线性层
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)  # 线性层
        self.value = nn.Linear(hidden_sz, config.n_embd, bias=False)  # 线性层

        # 初始化缩放参数
        self.value.scale_init = 0
        self.receptance.scale_init = 0

    @torch.jit.script_method
    def forward(self, x):
        xx = self.time_shift(x)  # 获取前一个时间步的输入
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 计算 k
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 计算 r

        k = self.key(xk)  # 计算 k
        k = torch.square(torch.relu(k))  # 对 k 应用 ReLU 激活函数并平方
        kv = self.value(k)  # 计算 v

        rkv = torch.sigmoid(self.receptance(xr)) * kv  # 计算 rkv
        return rkv  # 返回结果

########################################################################################################
# 使用我们的模块构建 GPT 模型
########################################################################################################

# 定义 GPT 配置类
class GPTConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size  # 词汇表大小
        self.ctx_len = ctx_len  # 上下文长度
        for k, v in kwargs.items():
            setattr(self, k, v)  # 设置其他参数


# 定义模型的基本块
class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config  # 配置
        self.layer_id = layer_id  # 层 ID

        self.ln1 = nn.LayerNorm(config.n_embd)  # 第一层归一化
        self.ln2 = nn.LayerNorm(config.n_embd)  # 第二层归一化

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)  # 第零层归一化

        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(config, 0)  # 前馈网络
        else:
            self.att = RWKV_TimeMix(config, layer_id)  # 时间混合

        self.ffn = RWKV_ChannelMix(config, layer_id)  # 通道混合

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)  # 对输入进行归一化
        if self.layer_id == 0 and self.config.model_type == 'RWKV-ffnPre':
            x = x + self.ffnPre(self.ln1(x))  # 在某些情况下更好
        else:
            x = x + self.att(self.ln1(x))  # 添加注意力层的输出
        x = x + self.ffn(self.ln2(x))  # 添加前馈网络的输出
        return x  # 返回结果


# 定义 GPT 模型
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.step = 0  # 步数
        self.config = config  # 配置

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)  # 嵌入层

        # 创建多个 Block
        self.blocks = nn.Sequential(*[Block(config, i)
                                    for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)  # 输出层归一化
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出层

        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)  # Q 线性层
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(config.n_embd, RWKV_HEAD_QK_DIM, bias=False)  # K 线性层
            self.head_k.scale_init = 0.1
            self.register_buffer("copy_mask", torch.tril(
                torch.ones(config.ctx_len, config.ctx_len)))  # 注册下三角矩阵作为掩码

        self.ctx_len = config.ctx_len  # 上下文长度

        try:
            if os.environ['RWKV_LOAD_MODEL'] == str(False):
                RWKV_Init(self, config)  # 初始化模型
        except:
            pass

        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))  # 打印参数数量

    def get_ctx_len(self):
        return self.ctx_len  # 获取上下文长度

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)  # 初始化线性层权重
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)  # 初始化嵌入层权重
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()  # 初始化偏置为零

    def configure_optimizers(self, train_config):
        no_decay = set()  # 不进行衰减的参数集合

        for mn, m in self.named_modules():  # 遍历所有模块
            for pn, p in m.named_parameters():  # 遍历所有参数
                fpn = '%s.%s' % (mn, pn) if mn else pn  # 完整参数名称
                no_decay.add(fpn)  # 添加到不衰减集合

        param_dict = {pn: p for pn, p in self.named_parameters()}  # 参数字典
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},  # 不衰减的参数组
        ]

        try:
            # 尝试使用 FusedAdam 优化器
            optimizer = FusedAdam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        except:
            print('\n\nDeepSpeed not found. Using torch optimizer instead (probably slower)\n\n')
            optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)  # 使用 PyTorch 的 Adam 优化器

        return optimizer  # 返回优化器

    def forward(self, idx, targets=None):
        idx = idx.to(self.emb.weight.device)  # 将输入索引移动到嵌入层的设备上

        self.step += 1  # 增加步数
        B, T = idx.size()  # 获取批次大小和时间步长
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."  # 确保输入长度不超过上下文长度

        x = self.emb(idx)  # 获取嵌入表示
        x = self.blocks(x)  # 通过多个 Block 进行前向传播
        x = self.ln_out(x)  # 进行输出层归一化

        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]  # 计算 Q
            k = self.head_k(x)[:, :T, :]  # 计算 K
            c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)  # 计算注意力得分
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)  # 应用掩码
            
            # 根据环境变量选择数据类型
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size)  # 计算输出
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).half()  # 半精度
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                c = c @ F.one_hot(idx, num_classes=self.config.vocab_size).bfloat16()  # bfloat16 精度

            x = self.head(x) + c  # 计算最终输出
        else:
            x = self.head(x)  # 计算最终输出

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.to(x.device).view(-1))  # 计算交叉熵损失

        return L2Wrap.apply(loss, x)  # 返回损失和输出


########################################################################################################
# RWKV 语言模型 - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types  # 导入 types 模块
import copy  # 导入 copy 模块
import torch  # 导入 PyTorch
import math, os  # 导入数学和操作系统模块
from torch.nn import functional as F  # 导入 PyTorch 的功能性模块
import torch.nn as nn  # 导入 PyTorch 的神经网络模块

RWKV_HEAD_QK_DIM = 0  # 定义一个常量，用于头部的 QK 维度
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM}\n')  # 打印 QK 维度

DEBUG_TIME = False  # 调试时间系数的开关

########################################################################################################
# CUDA 内核
########################################################################################################

# 检查运行设备是否为 CUDA
if os.environ['RWKV_RUN_DEVICE'] == 'cuda':
    T_MAX = 1024  # 最大时间步长，增加此值如果 ctx_len 较长 [注意：会占用大量显存！]
    # 如果你切片 ctx 并在每个切片中传递隐藏状态，可以超出 CUDA 限制

    from torch.utils.cpp_extension import load  # 从 PyTorch 导入 C++ 扩展加载功能
    # 加载 CUDA 扩展
    wkv_cuda = load(name="wkv", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"],
                    verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

    # 定义自定义的 autograd 函数 WKV
    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B  # 保存批次大小
            ctx.T = T  # 保存时间步长
            ctx.C = C  # 保存通道数
            assert T <= T_MAX  # 确保 T 不超过最大值
            assert B * C % min(C, 1024) == 0  # 确保 B 和 C 的乘积可以被 1024 整除

            # 根据环境变量选择数据类型
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                w = -torch.exp(w.contiguous())  # 计算 w 的负指数
                u = u.contiguous()  # 确保 u 是连续的
                k = k.contiguous()  # 确保 k 是连续的
                v = v.contiguous()  # 确保 v 是连续的
            else:
                w = -torch.exp(w.float().contiguous())  # 转换为 float 类型并计算负指数
                u = u.float().contiguous()  # 转换为 float 类型
                k = k.float().contiguous()  # 转换为 float 类型
                v = v.float().contiguous()  # 转换为 float 类型

            ctx.save_for_backward(w, u, k, v)  # 保存 w, u, k, v 以便反向传播使用
            y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)  # 创建输出张量
            wkv_cuda.forward(B, T, C, w, u, k, v, y)  # 调用 CUDA 内核进行前向计算

            # 根据环境变量返回不同的数据类型
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                return y
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                return y.half()  # 返回半精度
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                return y.bfloat16()  # 返回 bfloat16 精度

        @staticmethod
        def backward(ctx, gy):
            B = ctx.B  # 获取批次大小
            T = ctx.T  # 获取时间步长
            C = ctx.C  # 获取通道数
            assert T <= T_MAX  # 确保 T 不超过最大值
            assert B * C % min(C, 1024) == 0  # 确保 B 和 C 的乘积可以被 1024 整除

            w, u, k, v = ctx.saved_tensors  # 获取保存的张量
            gw = torch.zeros((B, C), device='cuda').contiguous()  # 创建梯度张量
            gu = torch.zeros((B, C), device='cuda').contiguous()  # 创建梯度张量
            gk = torch.zeros((B, T, C), device='cuda').contiguous()  # 创建梯度张量
            gv = torch.zeros((B, T, C), device='cuda').contiguous()  # 创建梯度张量

            # 根据环境变量选择数据类型
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
            else:
                wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)

            gw = torch.sum(gw, dim=0)  # 对 gw 进行求和
            gu = torch.sum(gu, dim=0)  # 对 gu 进行求和

            # 根据环境变量返回不同的数据类型
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                return (None, None, None, gw, gu, gk, gv)
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())

    # 定义运行 CUDA 的函数
    def RUN_CUDA(B, T, C, w, u, k, v):
        return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())  # 调用 WKV 的 apply 方法

############################################################################################################

RWKV_CFG = types.SimpleNamespace()  # 创建一个简单的命名空间，用于存储配置

# 定义通道混合类 RWKV_ChannelMix
class RWKV_ChannelMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()  # 调用父类构造函数
        self.layer_id = layer_id  # 保存层 ID

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 创建一个零填充层
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, RWKV_CFG.n_embd))  # 创建时间混合参数 k
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, RWKV_CFG.n_embd))  # 创建时间混合参数 r

        hidden_sz = 4 * RWKV_CFG.n_embd  # 隐藏层大小
        self.key = nn.Linear(RWKV_CFG.n_embd, hidden_sz, bias=False)  # 创建线性层 key
        self.receptance = nn.Linear(RWKV_CFG.n_embd, RWKV_CFG.n_embd, bias=False)  # 创建线性层 receptance
        self.value = nn.Linear(hidden_sz, RWKV_CFG.n_embd, bias=False)  # 创建线性层 value

    def forward(self, x):
        xx = self.time_shift(x)  # 对输入 x 进行时间移位
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 计算混合后的 k
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 计算混合后的 r

        k = self.key(xk)  # 计算 key
        k = torch.square(torch.relu(k))  # 对 key 进行 ReLU 激活和平方
        kv = self.value(k)  # 计算 value
        
        rkv = torch.sigmoid(self.receptance(xr)) * kv  # 计算 rkv
        return rkv  # 返回 rkv

# 定义时间混合类 RWKV_TimeMix
class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()  # 调用父类构造函数
        self.layer_id = layer_id  # 保存层 ID
        self.time_decay = nn.Parameter(torch.ones(RWKV_CFG.n_embd))  # 创建时间衰减参数
        self.time_first = nn.Parameter(torch.ones(RWKV_CFG.n_embd) * math.log(0.3))  # 创建时间初始参数
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 创建一个零填充层
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, RWKV_CFG.n_embd))  # 创建时间混合参数 k
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, RWKV_CFG.n_embd))  # 创建时间混合参数 v
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, RWKV_CFG.n_embd))  # 创建时间混合参数 r

        self.key = nn.Linear(RWKV_CFG.n_embd, RWKV_CFG.n_embd, bias=False)  # 创建线性层 key
        self.value = nn.Linear(RWKV_CFG.n_embd, RWKV_CFG.n_embd, bias=False)  # 创建线性层 value
        self.receptance = nn.Linear(RWKV_CFG.n_embd, RWKV_CFG.n_embd, bias=False)  # 创建线性层 receptance

        self.output = nn.Linear(RWKV_CFG.n_embd, RWKV_CFG.n_embd, bias=False)  # 创建线性层 output

    def forward(self, x):
        B, T, C = x.size()  # 获取输入的批次大小、时间步长和通道数

        xx = self.time_shift(x)  # 对输入 x 进行时间移位
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 计算混合后的 k
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)  # 计算混合后的 v
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 计算混合后的 r

        k = self.key(xk)  # 计算 key
        v = self.value(xv)  # 计算 value
        r = self.receptance(xr)  # 计算 receptance

        rwkv = torch.sigmoid(r) * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)  # 计算 rwkv
        
        rwkv = self.output(rwkv)  # 计算输出
        return rwkv  # 返回 rwkv

# 定义 Block 类
class Block(nn.Module):
    def __init__(self, layer_id):
        super().__init__()  # 调用父类构造函数
        self.layer_id = layer_id  # 保存层 ID

        self.ln1 = nn.LayerNorm(RWKV_CFG.n_embd)  # 创建层归一化层 ln1
        self.ln2 = nn.LayerNorm(RWKV_CFG.n_embd)  # 创建层归一化层 ln2
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(RWKV_CFG.n_embd)  # 创建层归一化层 ln0

        if self.layer_id == 0 and RWKV_CFG.model_type == 'RWKV-ffnPre':
            self.ffnPre = RWKV_ChannelMix(layer_id + 1000)  # 创建前馈网络
        else:
            self.att = RWKV_TimeMix(layer_id)  # 创建时间混合层

        self.ffn = RWKV_ChannelMix(layer_id)  # 创建通道混合层

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)  # 对输入进行层归一化
        if self.layer_id == 0 and RWKV_CFG.model_type == 'RWKV-ffnPre':
            x = x + self.ffnPre(self.ln1(x))  # 添加前馈网络的输出
        else:
            x = x + self.att(self.ln1(x))  # 添加时间混合层的输出
        x = x + self.ffn(self.ln2(x))  # 添加通道混合层的输出
        return x  # 返回输出

# 定义 RWKV_GPT 类
class RWKV_GPT(nn.Module):
    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, vocab_size, n_layer, n_embd, ctx_len):
        global RWKV_CFG  # 声明使用全局变量 RWKV_CFG
        super().__init__()  # 调用父类构造函数

        # 设置模型配置
        RWKV_CFG.RUN_DEVICE = RUN_DEVICE
        RWKV_CFG.model_type = model_type
        RWKV_CFG.vocab_size = vocab_size
        RWKV_CFG.n_layer = n_layer
        RWKV_CFG.n_embd = n_embd
        RWKV_CFG.ctx_len = ctx_len

        print('\nloading RWKV-GPT', MODEL_NAME)  # 打印加载模型信息

        self.emb = nn.Embedding(vocab_size, n_embd)  # 创建嵌入层

        self.blocks = nn.Sequential(*[Block(i) for i in range(n_layer)])  # 创建多个 Block

        self.ln_out = nn.LayerNorm(n_embd)  # 输出层归一化
        self.head = nn.Linear(n_embd, vocab_size, bias=False)  # 输出层

        if RWKV_HEAD_QK_DIM > 0:
            self.head_q = nn.Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False)  # Q 线性层
            self.head_q.scale_init = 0
            self.head_k = nn.Linear(n_embd, RWKV_HEAD_QK_DIM, bias=False)  # K 线性层
            self.head_k.scale_init = 0.1
            self.register_buffer("copy_mask", torch.tril(
                torch.ones(ctx_len, ctx_len)))  # 注册下三角矩阵作为掩码

        self.ctx_len = ctx_len  # 上下文长度
        self.eval()  # 设置为评估模式
        self.load_state_dict(torch.load(MODEL_NAME + '.pth'))  # 加载模型参数
        self.eval()  # 设置为评估模式

    def forward(self, idx):
        B, T = idx.size()  # 获取批次大小和时间步长
        assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."  # 确保输入长度不超过上下文长度
        
        x = self.emb(idx)  # 获取嵌入表示
        x = self.blocks(x)  # 通过多个 Block 进行前向传播
        x = self.ln_out(x)  # 进行输出层归一化

        if RWKV_HEAD_QK_DIM > 0:
            q = self.head_q(x)[:, :T, :]  # 计算 Q
            k = self.head_k(x)[:, :T, :]  # 计算 K
            c = (q @ k.transpose(-2, -1)) * (1.0 / RWKV_HEAD_QK_DIM)  # 计算注意力得分
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)  # 应用掩码

            # 根据环境变量选择数据类型
            if '32' in os.environ['RWKV_FLOAT_MODE']:
                c = c @ F.one_hot(idx, num_classes=RWKV_CFG.vocab_size)  # 计算输出
            elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
                c = c @ F.one_hot(idx, num_classes=RWKV_CFG.vocab_size).half()  # 半精度
            elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
                c = c @ F.one_hot(idx, num_classes=RWKV_CFG.vocab_size).bfloat16()  # bfloat16 精度

            x = self.head(x) + c  # 计算最终输出
        else:
            x = self.head(x)  # 计算最终输出        

        return x  # 返回输出

############################################################################################################

# 定义 RWKV_RNN 类
class RWKV_RNN():  # 目前在 FP32 下运行
    def __init__(self, MODEL_NAME, RUN_DEVICE, model_type, n_layer, n_embd, ctx_len):
        self.RUN_DEVICE = RUN_DEVICE  # 保存运行设备
        self.model_type = model_type  # 保存模型类型
        self.n_layer = n_layer  # 保存层数
        self.n_embd = n_embd  # 保存嵌入维度
        self.ctx_len = ctx_len  # 保存上下文长度

        self.w = types.SimpleNamespace()  # 创建一个简单的命名空间，用于存储权重

        # 加载模型参数
        w = torch.load(MODEL_NAME + '.pth',
                       map_location=torch.device(RUN_DEVICE))
        for x in w.keys():
            w[x] = w[x].float()  # 转换为 float 类型
            if '.time_' in x:
                w[x] = w[x].squeeze()  # 去除多余的维度
            if '.time_decay' in x:
                w[x] = -torch.exp(w[x])  # 计算时间衰减
            if DEBUG_TIME and '.time_' in x:
                print(x, w[x].squeeze().cpu().numpy())  # 打印时间信息

            xx = x.split('.')  # 按照 '.' 分割权重名称
            here = self.w  # 当前命名空间
            for i in range(len(xx)):
                if xx[i].isdigit():  # 如果是数字
                    ii = int(xx[i])  # 转换为整数
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()  # 创建新的命名空间
                    here = here[ii]  # 进入下一级命名空间
                else:
                    if i == len(xx) - 1:  # 如果是最后一个元素
                        setattr(here, xx[i], w[x])  # 设置属性
                    elif not hasattr(here, xx[i]):  # 如果没有该属性
                        if xx[i + 1].isdigit():  # 如果下一个是数字
                            setattr(here, xx[i], {})  # 设置为字典
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())  # 设置为命名空间
                    here = getattr(here, xx[i])  # 进入下一级命名空间

        self.clear()  # 清空状态

    def clear(self):
        self.xx = {}  # 清空 xx
        self.aa = {}  # 清空 aa
        self.bb = {}  # 清空 bb
        self.pp = {}  # 清空 pp
        self.hk = None  # 清空 hk

    def save(self, target):
        target.xx = copy.deepcopy(self.xx)  # 深拷贝 xx
        target.aa = copy.deepcopy(self.aa)  # 深拷贝 aa
        target.bb = copy.deepcopy(self.bb)  # 深拷贝 bb
        target.pp = copy.deepcopy(self.pp)  # 深拷贝 pp
        target.hk = copy.deepcopy(self.hk)  # 深拷贝 hk

    def load(self, target):
        self.xx = copy.deepcopy(target.xx)  # 深拷贝目标的 xx
        self.aa = copy.deepcopy(target.aa)  # 深拷贝目标的 aa
        self.bb = copy.deepcopy(target.bb)  # 深拷贝目标的 bb
        self.pp = copy.deepcopy(target.pp)  # 深拷贝目标的 pp
        self.hk = copy.deepcopy(target.hk)  # 深拷贝目标的 hk

    def LN(self, xx, w):
        return F.layer_norm(xx, (self.n_embd,), weight=w.weight, bias=w.bias)  # 进行层归一化

    def FF(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)  # 初始化 xx
        xk = xx * w.time_mix_k + self.xx[name] * (1 - w.time_mix_k)  # 计算混合后的 k
        xr = xx * w.time_mix_r + self.xx[name] * (1 - w.time_mix_r)  # 计算混合后的 r
        self.xx[name] = xx  # 更新 xx

        r = torch.sigmoid(w.receptance.weight @ xr)  # 计算 receptance
        k = torch.square(torch.relu(w.key.weight @ xk))  # 计算 key
        kv = w.value.weight @ k  # 计算 value

        return r * kv  # 返回 r 和 kv 的乘积

    def SA(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)  # 初始化 xx
            self.aa[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)  # 初始化 aa
            self.bb[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE)  # 初始化 bb
            self.pp[name] = torch.zeros(self.n_embd, device=self.RUN_DEVICE) - 1e30  # 初始化 pp

        xk = xx * w.time_mix_k + self.xx[name] * (1 - w.time_mix_k)  # 计算混合后的 k
        xv = xx * w.time_mix_v + self.xx[name] * (1 - w.time_mix_v)  # 计算混合后的 v
        xr = xx * w.time_mix_r + self.xx[name] * (1 - w.time_mix_r)  # 计算混合后的 r
        self.xx[name] = xx  # 更新 xx

        r = torch.sigmoid(w.receptance.weight @ xr)  # 计算 receptance

        k = w.key.weight @ xk  # 计算 key
        v = w.value.weight @ xv  # 计算 value

        pp = self.pp[name]  # 获取 pp
        aa = self.aa[name]  # 获取 aa
        bb = self.bb[name]  # 获取 bb
        ww = w.time_first + k  # 计算 ww
        p = torch.maximum(pp, ww)  # 计算 p
        e1 = torch.exp(pp - p)  # 计算 e1
        e2 = torch.exp(ww - p)  # 计算 e2
        a = e1 * aa + e2 * v  # 计算 a
        b = e1 * bb + e2  # 计算 b
        ww = pp + w.time_decay  # 更新 ww
        p = torch.maximum(ww, k)  # 计算 p
        e1 = torch.exp(ww - p)  # 计算 e1
        e2 = torch.exp(k - p)  # 计算 e2
        self.aa[name] = e1 * aa + e2 * v  # 更新 aa
        self.bb[name] = e1 * bb + e2  # 更新 bb
        self.pp[name] = p  # 更新 pp

        rwkv = r * a / b  # 计算 rwkv

        return w.output.weight @ rwkv  # 返回输出

    def run(self, ctx):
        w = self.w  # 获取权重
        x = w.emb.weight[ctx[-1]]  # 获取最后一个上下文的嵌入

        for i in range(self.n_layer):  # 遍历每一层
            if i == 0:
                x = self.LN(x, w.blocks[i].ln0)  # 对输入进行层归一化
            if i == 0 and self.model_type == 'RWKV-ffnPre':
                x = x + self.FF(self.LN(x, w.blocks[i].ln1), w.blocks[i].ffnPre, f'ffnPre.{i}')  # 添加前馈网络的输出
            else:
                x = x + self.SA(self.LN(x, w.blocks[i].ln1), w.blocks[i].att, f'att.{i}')  # 添加自注意力层的输出
            x = x + self.FF(self.LN(x, w.blocks[i].ln2), w.blocks[i].ffn, f'ffn.{i}')  # 添加前馈网络的输出

        x = self.LN(x, w.ln_out)  # 对输出进行层归一化

        if RWKV_HEAD_QK_DIM > 0:  # 如果 QK 维度大于 0
            if self.hk is None:
                self.hk = (w.head_k.weight @ x).unsqueeze(0)  # 初始化 hk
            else:
                self.hk = torch.cat(
                    [self.hk, (w.head_k.weight @ x).unsqueeze(0)], dim=0)  # 连接 hk
            if self.hk.shape[0] > self.ctx_len:  # 如果 hk 的长度超过上下文长度
                self.hk = self.hk[-self.ctx_len:, :]  # 截取 hk

            q = w.head_q.weight @ x  # 计算 Q

            x = w.head.weight @ x  # 计算输出
            x = x.cpu().numpy().tolist()  # 转换为列表

            c = (self.hk @ q) / RWKV_HEAD_QK_DIM  # 计算 c
            for i in range(len(c)):
                x[ctx[i]] += c[i]  # 更新输出
        else:
            x = w.head.weight @ x  # 计算输出
            x = x.cpu().numpy().tolist()  # 转换为列表

        return x  # 返回输出
