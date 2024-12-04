########################################################################################################
# RWKV 语言模型 - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import numpy as np  # 导入 NumPy 库
import os, math, gc  # 导入操作系统、数学和垃圾回收模块
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 的功能性模块
import torchvision as vision  # 导入 torchvision 库
import pytorch_lightning as pl  # 导入 PyTorch Lightning 库
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only  # 导入 PyTorch Lightning 的工具
from pytorch_lightning.strategies import DeepSpeedStrategy  # 导入 DeepSpeed 策略
import deepspeed  # 导入 DeepSpeed 库
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam  # 导入 DeepSpeed 的 Adam 优化器
# from pytorch_msssim import MS_SSIM  # 导入多尺度结构相似性（可选）

def __nop(ob):
    return ob  # 定义一个空操作函数，返回输入对象
MyModule = torch.jit.ScriptModule  # 定义 MyModule 为 TorchScript 模块
# MyFunction = __nop  # 将 MyFunction 定义为空操作
MyFunction = torch.jit.script_method  # 将 MyFunction 定义为 TorchScript 方法

import clip  # 导入 CLIP 模型
from transformers import CLIPModel  # 从 transformers 导入 CLIPModel

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()  # 初始化父类
        self.padding = (filter_size - 2) // 2  # 计算填充大小
        self.stride = stride  # 步幅
        self.channels = channels  # 通道数
        a = np.hanning(filter_size)[1:-1]  # 生成汉宁窗
        g = torch.Tensor(a[:, None] * a[None, :])  # 计算二维滤波器
        g = g / torch.sum(g)  # 归一化滤波器
        self.register_buffer(
            "filter", g[None, None, :, :].repeat((self.channels, 1, 1, 1))  # 注册滤波器为缓冲区
        )

    def forward(self, input):
        input = input**2  # 输入平方
        out = F.conv2d(
            input,
            self.filter,
            stride=self.stride,
            padding=self.padding,
            groups=input.shape[1],  # 按通道分组卷积
        )
        return (out + 1e-12).sqrt()  # 返回平方根结果，避免数值不稳定

class DISTS(torch.nn.Module):
    def __init__(self, load_weights=True):
        super(DISTS, self).__init__()  # 初始化父类
        vgg_pretrained_features = vision.models.vgg16(
            weights="VGG16_Weights.IMAGENET1K_V1"  # 加载预训练的 VGG16 特征提取器
        ).features
        # 定义多个阶段的卷积层
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])  # 添加前四层
        self.stage2.add_module(str(4), L2pooling(channels=64))  # 添加 L2 池化层
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])  # 添加后四层
        self.stage3.add_module(str(9), L2pooling(channels=128))  # 添加 L2 池化层
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])  # 添加后六层
        self.stage4.add_module(str(16), L2pooling(channels=256))  # 添加 L2 池化层
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])  # 添加后六层
        self.stage5.add_module(str(23), L2pooling(channels=512))  # 添加 L2 池化层
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])  # 添加后六层

        # 注册均值和标准差为缓冲区
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [3, 64, 128, 256, 512, 512]  # 定义通道数
        self.register_buffer(
            "alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))  # 注册 alpha 参数
        )
        self.register_buffer("beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))  # 注册 beta 参数
        self.alpha.data.normal_(0.1, 0.01)  # 初始化 alpha
        self.beta.data.normal_(0.1, 0.01)  # 初始化 beta
        weights = torch.load("test/DISTS_weights.pt")  # 加载权重
        self.alpha.data = weights["alpha"]  # 设置 alpha 权重
        self.beta.data = weights["beta"]  # 设置 beta 权重

        for param in self.parameters():
            param.requires_grad = False  # 冻结所有参数

    def forward_once(self, x):
        h = (x - self.mean) / self.std  # 标准化输入
        h = self.stage1(h)  # 通过第一阶段
        h_relu1_2 = h  # 保存第一阶段输出
        h = self.stage2(h)  # 通过第二阶段
        h_relu2_2 = h  # 保存第二阶段输出
        h = self.stage3(h)  # 通过第三阶段
        h_relu3_3 = h  # 保存第三阶段输出
        h = self.stage4(h)  # 通过第四阶段
        h_relu4_3 = h  # 保存第四阶段输出
        h = self.stage5(h)  # 通过第五阶段
        h_relu5_3 = h  # 保存第五阶段输出
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]  # 返回所有中间特征

    def forward(self, x, y, require_grad=False, batch_average=False):
        if require_grad:
            feats0 = self.forward_once(x)  # 计算第一个输入的特征
            feats1 = self.forward_once(y)  # 计算第二个输入的特征
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)  # 计算第一个输入的特征（不计算梯度）
                feats1 = self.forward_once(y)  # 计算第二个输入的特征（不计算梯度）
        dist1 = 0  # 初始化距离1
        dist2 = 0  # 初始化距离2
        c1 = 1e-6  # 常数1
        c2 = 1e-6  # 常数2
        w_sum = self.alpha.sum() + self.beta.sum()  # 计算权重和
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)  # 分割 alpha
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)  # 分割 beta

        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)  # 计算第一个输入的均值
            y_mean = feats1[k].mean([2, 3], keepdim=True)  # 计算第二个输入的均值
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)  # 计算 S1
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)  # 更新距离1

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)  # 计算第一个输入的方差
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)  # 计算第二个输入的方差
            xy_cov = (feats0[k] * feats1[k]).mean(
                [2, 3], keepdim=True
            ) - x_mean * y_mean  # 计算协方差
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)  # 计算 S2
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)  # 更新距离2

        score = 1 - (dist1 + dist2).squeeze()  # 计算最终得分

        if batch_average:
            return score.mean()  # 返回平均得分
        else:
            return score  # 返回得分

    class ToBinary(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):  # 前向传播
            return torch.floor(x + 0.5)  # 将输入四舍五入为整数

        @staticmethod
        def backward(ctx, grad_output):  # 反向传播
            return grad_output.clone()  # 返回梯度

########################################################################################################

class R_ENCODER(MyModule):
    def __init__(self, args):
        super().__init__()  # 初始化父类
        self.args = args  # 保存参数
        dd = 8  # 定义通道数的倍数
        self.Bxx = nn.BatchNorm2d(dd*64)  # 批归一化层

        # 定义卷积层
        self.CIN = nn.Conv2d(3, dd, kernel_size=3, padding=1)  # 输入卷积层
        self.Cx0 = nn.Conv2d(dd, 32, kernel_size=3, padding=1)  # 卷积层
        self.Cx1 = nn.Conv2d(32, dd, kernel_size=3, padding=1)  # 卷积层

        # 定义多个卷积层和批归一化层
        self.B00 = nn.BatchNorm2d(dd*4)
        self.C00 = nn.Conv2d(dd*4, 256, kernel_size=3, padding=1)
        self.C01 = nn.Conv2d(256, dd*4, kernel_size=3, padding=1)
        self.C02 = nn.Conv2d(dd*4, 256, kernel_size=3, padding=1)
        self.C03 = nn.Conv2d(256, dd*4, kernel_size=3, padding=1)

        self.B10 = nn.BatchNorm2d(dd*16)
        self.C10 = nn.Conv2d(dd*16, 256, kernel_size=3, padding=1)
        self.C11 = nn.Conv2d(256, dd*16, kernel_size=3, padding=1)
        self.C12 = nn.Conv2d(dd*16, 256, kernel_size=3, padding=1)
        self.C13 = nn.Conv2d(256, dd*16, kernel_size=3, padding=1)

        self.B20 = nn.BatchNorm2d(dd*64)
        self.C20 = nn.Conv2d(dd*64, 256, kernel_size=3, padding=1)
        self.C21 = nn.Conv2d(256, dd*64, kernel_size=3, padding=1)
        self.C22 = nn.Conv2d(dd*64, 256, kernel_size=3, padding=1)
        self.C23 = nn.Conv2d(256, dd*64, kernel_size=3, padding=1)

        self.COUT = nn.Conv2d(dd*64, args.my_img_bit, kernel_size=3, padding=1)  # 输出卷积层

    @MyFunction
    def forward(self, img):
        ACT = F.mish  # 激活函数

        # 编码过程
        x = self.CIN(img)  # 输入卷积
        xx = self.Bxx(F.pixel_unshuffle(x, 8))  # 像素反混淆
        x = x + self.Cx1(ACT(self.Cx0(x)))  # 残差连接

        # 多次像素反混淆和卷积
        x = F.pixel_unshuffle(x, 2)
        x = x + self.C01(ACT(self.C00(ACT(self.B00(x)))))  # 残差连接
        x = x + self.C03(ACT(self.C02(x)))  # 残差连接

        x = F.pixel_unshuffle(x, 2)
        x = x + self.C11(ACT(self.C10(ACT(self.B10(x)))))  # 残差连接
        x = x + self.C13(ACT(self.C12(x)))  # 残差连接

        x = F.pixel_unshuffle(x, 2)
        x = x + self.C21(ACT(self.C20(ACT(self.B20(x)))))  # 残差连接
        x = x + self.C23(ACT(self.C22(x)))  # 残差连接

        x = self.COUT(x + xx)  # 输出卷积
        return torch.sigmoid(x)  # 返回经过 Sigmoid 激活的输出

########################################################################################################

class R_DECODER(MyModule):
    def __init__(self, args):
        super().__init__()  # 初始化父类
        self.args = args  # 保存参数
        dd = 8  # 定义通道数的倍数
        self.CIN = nn.Conv2d(args.my_img_bit, dd*64, kernel_size=3, padding=1)  # 输入卷积层

        # 定义多个卷积层和批归一化层
        self.B00 = nn.BatchNorm2d(dd*64)
        self.C00 = nn.Conv2d(dd*64, 256, kernel_size=3, padding=1)
        self.C01 = nn.Conv2d(256, dd*64, kernel_size=3, padding=1)
        self.C02 = nn.Conv2d(dd*64, 256, kernel_size=3, padding=1)
        self.C03 = nn.Conv2d(256, dd*64, kernel_size=3, padding=1)

        self.B10 = nn.BatchNorm2d(dd*16)
        self.C10 = nn.Conv2d(dd*16, 256, kernel_size=3, padding=1)
        self.C11 = nn.Conv2d(256, dd*16, kernel_size=3, padding=1)
        self.C12 = nn.Conv2d(dd*16, 256, kernel_size=3, padding=1)
        self.C13 = nn.Conv2d(256, dd*16, kernel_size=3, padding=1)

        self.B20 = nn.BatchNorm2d(dd*4)
        self.C20 = nn.Conv2d(dd*4, 256, kernel_size=3, padding=1)
        self.C21 = nn.Conv2d(256, dd*4, kernel_size=3, padding=1)
        self.C22 = nn.Conv2d(dd*4, 256, kernel_size=3, padding=1)
        self.C23 = nn.Conv2d(256, dd*4, kernel_size=3, padding=1)

        self.Cx0 = nn.Conv2d(dd, 32, kernel_size=3, padding=1)  # 卷积层
        self.Cx1 = nn.Conv2d(32, dd, kernel_size=3, padding=1)  # 卷积层
        self.COUT = nn.Conv2d(dd, 3, kernel_size=3, padding=1)  # 输出卷积层

    @MyFunction
    def forward(self, code):
        ACT = F.mish  # 激活函数
        x = self.CIN(code)  # 输入卷积

        # 解码过程
        x = x + self.C01(ACT(self.C00(ACT(self.B00(x)))))  # 残差连接
        x = x + self.C03(ACT(self.C02(x)))  # 残差连接
        x = F.pixel_shuffle(x, 2)  # 像素反混淆

        x = x + self.C11(ACT(self.C10(ACT(self.B10(x)))))  # 残差连接
        x = x + self.C13(ACT(self.C12(x)))  # 残差连接
        x = F.pixel_shuffle(x, 2)  # 像素反混淆

        x = x + self.C21(ACT(self.C20(ACT(self.B20(x)))))  # 残差连接
        x = x + self.C23(ACT(self.C22(x)))  # 残差连接
        x = F.pixel_shuffle(x, 2)  # 像素反混淆

        x = x + self.Cx1(ACT(self.Cx0(x)))  # 残差连接
        x = self.COUT(x)  # 输出卷积
        
        return torch.sigmoid(x)  # 返回经过 Sigmoid 激活的输出

########################################################################################################

def cosine_loss(x, y):
    x = F.normalize(x, dim=-1)  # 对 x 进行归一化
    y = F.normalize(y, dim=-1)  # 对 y 进行归一化
    return 1 - torch.einsum('ij,ij->i',[x,y])  # 计算余弦损失

class RWKV_IMG(pl.LightningModule):
    def __init__(self, args):
        super().__init__()  # 初始化父类
        self.args = args  # 保存参数
            
        self.encoder = R_ENCODER(args)  # 初始化编码器
        self.decoder = R_DECODER(args)  # 初始化解码器

        self.clip_model = None  # 初始化 CLIP 模型
        clip_name = args.my_img_clip  # 获取 CLIP 模型名称
        if clip_name == 'B32':
            clip_name = 'ViT-B/32'  # 设置 CLIP 模型名称
        elif clip_name == 'B16':
            clip_name = 'ViT-B/16'  # 设置 CLIP 模型名称
        elif clip_name == 'L14':
            clip_name = 'ViT-L/14'  # 设置 CLIP 模型名称
        elif clip_name == 'OB32':
            clip_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"  # 设置 CLIP 模型名称
            self.clip_model = CLIPModel.from_pretrained(clip_name)  # 加载 CLIP 模型
            self.clip_model.encode_image = self.clip_model.get_image_features  # 设置图像特征提取方法
        if self.clip_model == None:
            self.clip_model, _ = clip.load(clip_name, jit = True)  # 加载 CLIP 模型
        self.register_buffer(
            "clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)  # 注册 CLIP 均值
        )
        self.register_buffer(
            "clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)  # 注册 CLIP 标准差
        )

        for n, p in self.named_parameters():  # 遍历模型参数
            if 'clip_model' in n:  # 如果参数属于 CLIP 模型
                p.requires_grad = False  # 冻结参数

        self.loss_dists = DISTS()  # 初始化 DISTS 损失
        # self.loss_ssim = MS_SSIM(data_range=1, size_average=True, channel=3)  # 初始化 SSIM 损失（可选）

    def configure_optimizers(self):
        args = self.args  # 获取参数
        optim_groups = [
            {"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0},  # 定义优化器参数组
        ]
        if self.deepspeed_offload:  # 如果使用 DeepSpeed
            return DeepSpeedCPUAdam(
                optim_groups,
                lr=self.args.lr_init,  # 学习率
                betas=self.args.betas,  # Adam 的 beta 参数
                eps=self.args.adam_eps,  # Adam 的 epsilon 参数
                bias_correction=True,  # 是否进行偏差修正
                adamw_mode=False,  # 是否使用 AdamW
                weight_decay=0,  # 权重衰减
                amsgrad=False,  # 是否使用 AMSGrad
            )
        return FusedAdam(
            optim_groups,
            lr=self.args.lr_init,  # 学习率
            betas=self.args.betas,  # Adam 的 beta 参数
            eps=self.args.adam_eps,  # Adam 的 epsilon 参数
            bias_correction=True,  # 是否进行偏差修正
            adam_w_mode=False,  # 是否使用 AdamW
            weight_decay=0,  # 权重衰减
            amsgrad=False,  # 是否使用 AMSGrad
        )
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy  # 获取训练策略
        if isinstance(strategy, DeepSpeedStrategy):  # 如果使用 DeepSpeed 策略
            config = strategy.config["zero_optimization"]  # 获取配置
            return config.get("offload_optimizer") or config.get("offload_param")  # 返回是否启用 offload
        return False

    def forward(self, img):
        z = self.encoder(img)  # 编码输入图像
        z = ToBinary.apply(z)  # 将编码结果转换为二进制
        out = self.decoder(z)  # 解码
        return out  # 返回解码结果

    def training_step(self, batch, batch_idx):
        args = self.args  # 获取参数
        img, txt = batch  # 获取输入图像和文本
        out = self(img)  # 前向传播

        if self.trainer.is_global_zero:  # 如果是全局进程
            if (self.trainer.global_step + 1) % (100 * int(args.devices)) == 0:  # 每100步保存一次图像
                img_dir = f"test/image_model/{args.run_name}"  # 图像保存目录
                if not os.path.exists(img_dir):  # 如果目录不存在
                    os.makedirs(img_dir)  # 创建目录
                vision.utils.save_image(
                    img[:4], f"{img_dir}/{self.trainer.global_step}-src.jpg"  # 保存源图像
                )
                vision.utils.save_image(
                    out[:4], f"{img_dir}/{self.trainer.global_step}-out.jpg"  # 保存输出图像
                )

        # loss_ssim = 1 - self.loss_ssim(out, img)  # 计算 SSIM 损失（可选）
        loss_dists = self.loss_dists(out, img, require_grad=True, batch_average=True)  # 计算 DISTS 损失

        iii = self.clip_model.encode_image((img - self.clip_mean) / self.clip_std)  # 编码源图像
        ooo = self.clip_model.encode_image((out - self.clip_mean) / self.clip_std)  # 编码输出图像
        loss_clip = torch.mean(cosine_loss(iii, ooo))  # 计算 CLIP 损失

        if args.my_img_l1_scale > 0:  # 如果 L1 损失权重大于0
            loss_l1 = F.l1_loss(out, img)  # 计算 L1 损失
            return loss_dists + loss_clip * args.my_img_clip_scale + loss_l1 * args.my_img_l1_scale  # 返回总损失
        else:
            return loss_dists + loss_clip * args.my_img_clip_scale  # 返回总损失

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)  # 收集所有批次的结果
        if self.trainer.is_global_zero:  # 如果是全局进程
            self.trainer.my_loss_all = all  # 保存所有损失

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# 初始化模型权重（大型模型较慢）...
#
############################################################################
"""
        )
        m = {}  # 初始化权重字典
        for n in self.state_dict():  # 遍历模型状态字典
            scale = 1  # 初始化缩放因子
            p = self.state_dict()[n]  # 获取参数
            shape = p.shape  # 获取参数形状
            ss = n.split('.')  # 分割参数名称

            # if ss[0] in ['encoder', 'decoder']:  # 如果参数属于编码器或解码器
            #     if ss[2] == 'bias':  # 如果是偏置
            #         scale = 0  # 不初始化偏置
            #     # elif n == 'encoder.CIN.weight':
            #     #     nn.init.dirac_(p)  # 使用 Dirac 初始化
            #     else:
            #         try:
            #             if ss[1][0] == 'C' and (int(ss[1][2]) % 2 == 1):  # 如果是奇数层
            #                 scale = 0  # 不初始化
            #         except:
            #             pass
            # m[n] = p * scale  # 应用缩放因子

            m[n] = p  # 保存参数

            m[n] = m[n].cpu()  # 将参数移到 CPU
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":  # 如果使用 fp16
                m[n] = m[n].half()  # 转换为 fp16
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":  # 如果使用 bf16
                m[n] = m[n].bfloat16()  # 转换为 bf16

        gc.collect()  # 垃圾回收
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        return m  # 返回初始化的权重


# model.py
########################################################################################################
# RWKV 语言模型 - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib  # 导入操作系统、数学、垃圾回收和动态导入模块
import torch  # 导入 PyTorch 库
# torch._C._jit_set_profiling_executor(True)  # 启用 JIT Profiling 执行器（可选）
# torch._C._jit_set_profiling_mode(True)  # 启用 JIT Profiling 模式（可选）
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torch.nn import functional as F  # 导入 PyTorch 的功能性模块
import pytorch_lightning as pl  # 导入 PyTorch Lightning 库
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only  # 导入 PyTorch Lightning 的工具
from pytorch_lightning.strategies import DeepSpeedStrategy  # 导入 DeepSpeed 策略
if importlib.util.find_spec('deepspeed'):  # 检查是否安装了 DeepSpeed
    import deepspeed  # 导入 DeepSpeed 库
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam  # 导入 DeepSpeed 的 Adam 优化器

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam  # 可选的 Adam 优化器

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])  # 打印环境变量 RWKV_MY_TESTING
except:
    os.environ["RWKV_MY_TESTING"] = ''  # 如果未设置，则初始化为空字符串

def __nop(ob):
    return ob  # 定义一个空操作函数，返回输入对象

MyModule = nn.Module  # 将 MyModule 定义为 nn.Module
MyFunction = __nop  # 将 MyFunction 定义为空操作
if os.environ["RWKV_JIT_ON"] == "1":  # 如果启用 JIT
    MyModule = torch.jit.ScriptModule  # 将 MyModule 定义为 TorchScript 模块
    MyFunction = torch.jit.script_method  # 将 MyFunction 定义为 TorchScript 方法

########################################################################################################
# CUDA 内核
########################################################################################################

T_MAX = int(os.environ["RWKV_T_MAX"])  # 最大时间步长，可能会占用大量显存
# 如果切片上下文并在每个切片中传递隐藏状态，则可以超出 CUDA 限制

from torch.utils.cpp_extension import load  # 导入 C++ 扩展加载器

if os.environ["RWKV_FLOAT_MODE"] == "bf16":  # 如果使用 bf16 浮点模式
    wkv_cuda = load(name=f"wkv_{T_MAX}_bf16", sources=["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"], verbose=True, extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):  # 定义 WKV 类
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):  # 前向传播
            ctx.B = B  # 保存批次大小
            ctx.T = T  # 保存时间步长
            ctx.C = C  # 保存通道数
            assert T <= T_MAX  # 确保时间步长不超过最大值
            assert B * C % min(C, 32) == 0  # 确保批次大小和通道数的乘积满足条件
            w = -torch.exp(w.float().contiguous())  # 计算权重
            u = u.contiguous()  # 确保 u 是连续的
            k = k.contiguous()  # 确保 k 是连续的
            v = v.contiguous()  # 确保 v 是连续的
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)  # 创建输出张量
            wkv_cuda.forward(B, T, C, w, u, k, v, y)  # 调用 CUDA 内核进行前向传播
            ctx.save_for_backward(w, u, k, v, y)  # 保存反向传播所需的张量
            return y  # 返回输出

        @staticmethod
        def backward(ctx, gy):  # 反向传播
            B = ctx.B  # 获取批次大小
            T = ctx.T  # 获取时间步长
            C = ctx.C  # 获取通道数
            assert T <= T_MAX  # 确保时间步长不超过最大值
            assert B * C % min(C, 32) == 0  # 确保批次大小和通道数的乘积满足条件
            w, u, k, v, y = ctx.saved_tensors  # 获取保存的张量
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)  # 创建梯度张量
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)  # 创建梯度张量
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)  # 创建梯度张量
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)  # 创建梯度张量
            wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)  # 调用 CUDA 内核进行反向传播
            gw = torch.sum(gw, dim=0)  # 对 gw 进行求和
            gu = torch.sum(gu, dim=0)  # 对 gu 进行求和
            return (None, None, None, gw, gu, gk, gv)  # 返回梯度

else:  # 如果不使用 bf16 浮点模式
    wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"], verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):  # 定义 WKV 类
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):  # 前向传播
            ctx.B = B  # 保存批次大小
            ctx.T = T  # 保存时间步长
            ctx.C = C  # 保存通道数
            assert T <= T_MAX  # 确保时间步长不超过最大值
            assert B * C % min(C, 32) == 0  # 确保批次大小和通道数的乘积满足条件
            if "32" in os.environ["RWKV_FLOAT_MODE"]:  # 如果使用 32 位浮点模式
                w = -torch.exp(w.contiguous())  # 计算权重
                u = u.contiguous()  # 确保 u 是连续的
                k = k.contiguous()  # 确保 k 是连续的
                v = v.contiguous()  # 确保 v 是连续的
            else:  # 如果使用其他浮点模式
                w = -torch.exp(w.float().contiguous())  # 计算权重
                u = u.float().contiguous()  # 确保 u 是连续的
                k = k.float().contiguous()  # 确保 k 是连续的
                v = v.float().contiguous()  # 确保 v 是连续的
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)  # 创建输出张量
            wkv_cuda.forward(B, T, C, w, u, k, v, y)  # 调用 CUDA 内核进行前向传播
            ctx.save_for_backward(w, u, k, v, y)  # 保存反向传播所需的张量
            if "32" in os.environ["RWKV_FLOAT_MODE"]:  # 如果使用 32 位浮点模式
                return y  # 返回输出
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":  # 如果使用 fp16 浮点模式
                return y.half()  # 返回 fp16 输出
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":  # 如果使用 bf16 浮点模式
                return y.bfloat16()  # 返回 bf16 输出

        @staticmethod
        def backward(ctx, gy):  # 反向传播
            B = ctx.B  # 获取批次大小
            T = ctx.T  # 获取时间步长
            C = ctx.C  # 获取通道数
            assert T <= T_MAX  # 确保时间步长不超过最大值
            assert B * C % min(C, 32) == 0  # 确保批次大小和通道数的乘积满足条件
            w, u, k, v, y = ctx.saved_tensors  # 获取保存的张量
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)  # 创建梯度张量
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)  # 创建梯度张量
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)  # 创建梯度张量
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)  # 创建梯度张量
            if "32" in os.environ["RWKV_FLOAT_MODE"]:  # 如果使用 32 位浮点模式
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)  # 调用 CUDA 内核进行反向传播
            else:  # 如果使用其他浮点模式
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.float().contiguous(), gw, gu, gk, gv)  # 调用 CUDA 内核进行反向传播
            gw = torch.sum(gw, dim=0)  # 对 gw 进行求和
            gu = torch.sum(gu, dim=0)  # 对 gu 进行求和
            if "32" in os.environ["RWKV_FLOAT_MODE"]:  # 如果使用 32 位浮点模式
                return (None, None, None, gw, gu, gk, gv)  # 返回梯度
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":  # 如果使用 fp16 浮点模式
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())  # 返回 fp16 梯度
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":  # 如果使用 bf16 浮点模式
                return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())  # 返回 bf16 梯度

def RUN_CUDA(B, T, C, w, u, k, v):  # 定义运行 CUDA 的函数
    return WKV.apply(B, T, C, w, u, k, v)  # 调用 WKV 的 apply 方法

########################################################################################################

class RWKV_TimeMix_RWKV5_Preview(MyModule):  # 定义 RWKV 时间混合类
    def __init__(self, args, layer_id):  # 初始化
        super().__init__()  # 调用父类构造函数
        self.args = args  # 保存参数
        self.layer_id = layer_id  # 保存层 ID

        self.head_size = 64  # 头的大小
        self.n_head = args.dim_att // self.head_size  # 计算头的数量
        assert args.dim_att % self.n_head == 0  # 确保维度可以被头的数量整除
        self.head_size_divisor = 8  # 头大小的除数

        self.chunk_len = 512  # 块长度
        assert args.ctx_len % self.chunk_len == 0  # 确保上下文长度可以被块长度整除

        with torch.no_grad():  # 不计算梯度
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 计算比例从 0 到 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 计算比例从 1 到接近 0
            ddd = torch.ones(1, 1, args.n_embd)  # 创建全 1 的张量
            for i in range(args.n_embd):  # 填充张量
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))  # 时间混合参数 k
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)  # 时间混合参数 v
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))  # 时间混合参数 r

            if 'r3' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 r3
                self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))  # 时间混合参数 g
                self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)  # 门控线性层

            # fancy time_decay
            decay_speed = torch.ones(self.n_head)  # 创建衰减速度张量
            for h in range(self.n_head):  # 填充衰减速度
                decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)  # 时间衰减参数
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            if 'r2' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 r2
                tmp = torch.zeros(self.n_head)  # 创建临时张量
                for h in range(self.n_head):  # 填充临时张量
                    tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
                self.time_faaaa = nn.Parameter(tmp)  # 时间参数
            else:
                self.time_first = nn.Parameter(torch.ones(self.n_head) * (-3.0))  # 时间参数

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 时间移位
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)  # 接收线性层
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)  # 键线性层
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)  # 值线性层
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)  # 输出线性层

        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)  # 组归一化

    if 'r3' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 r3
        @MyFunction
        def jit_func(self, x):  # JIT 函数
            B, TT, C = x.size()  # 获取输入的维度

            xx = self.time_shift(x)  # 将 x 与前一个时间步混合以生成 xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 计算 k
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)  # 计算 v
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 计算 r
            xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)  # 计算 g

            r = self.receptance(xr).view(B, TT, self.n_head, self.head_size).transpose(1, 2)  # BTC -> BHTS
            k = self.key(xk).view(B, TT, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1)  # BTC -> BHTS -> BHST
            v = self.value(xv).view(B, TT, self.n_head, -1).transpose(1, 2)  # BTC -> BHTS
            g = F.silu(self.gate(xg))  # 计算 g

            return r, k, v, g  # 返回 r, k, v, g

        @MyFunction
        def jit_func_2(self, r, k, v, g, w, wk, wb, ws):  # JIT 函数 2
            B, H, TT, S = r.size()  # 获取维度
            T = self.chunk_len  # 获取块长度

            s = torch.zeros(B, H, S, S, device=r.device, dtype=r.dtype)  # 状态
            x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype)  # 输出

            for i in range(TT // T):  # 遍历时间步
                rr = r[:, :, i*T:i*T+T, :]  # 获取 r
                kk = k[:, :, :, i*T:i*T+T]  # 获取 k
                vv = v[:, :, i*T:i*T+T, :]  # 获取 v

                x[:, :, i*T:i*T+T, :] = ((rr @ kk) * w) @ vv + (rr @ s) * wb  # 更新输出

                s = ws * s + (kk * wk) @ vv  # 更新状态
            
            x = x.transpose(1, 2).contiguous().view(B * TT, H*S)  # BHTS -> BTHS -> BTC
            x = self.ln_x(x / self.head_size_divisor).view(B, TT, H*S)  # 归一化
            return self.output(x)  # 返回输出

    else:
        @MyFunction
        def jit_func(self, x):  # JIT 函数
            B, TT, C = x.size()  # 获取输入的维度

            xx = self.time_shift(x)  # 将 x 与前一个时间步混合以生成 xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 计算 k
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)  # 计算 v
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 计算 r

            r = self.receptance(xr).view(B, TT, self.n_head, self.head_size).transpose(1, 2)  # BTC -> BHTS
            k = self.key(xk).view(B, TT, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1)  # BTC -> BHTS -> BHST
            v = self.value(xv).view(B, TT, self.n_head, self.head_size).transpose(1, 2)  # BTC -> BHTS

            return r, k, v  # 返回 r, k, v

        @MyFunction
        def jit_func_2(self, r, k, v, w, wk, wb, ws):  # JIT 函数 2
            B, H, TT, S = r.size()  # 获取维度
            T = self.chunk_len  # 获取块长度

            s = torch.zeros(B, H, S, S, device=r.device, dtype=r.dtype)  # 状态
            x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype)  # 输出

            for i in range(TT // T):  # 遍历时间步
                rr = r[:, :, i*T:i*T+T, :]  # 获取 r
                kk = k[:, :, :, i*T:i*T+T]  # 获取 k
                vv = v[:, :, i*T:i*T+T, :]  # 获取 v

                x[:, :, i*T:i*T+T, :] = ((rr @ kk) * w) @ vv + (rr @ s) * wb  # 更新输出

                s = ws * s + (kk * wk) @ vv  # 更新状态
            
            x = x.transpose(1, 2).contiguous().view(B * TT, H*S)  # BHTS -> BTHS -> BTC
            x = self.ln_x(x / self.head_size_divisor).view(B, TT, H*S)  # 归一化
            return self.output(x)  # 返回输出
    
    def forward(self, x):  # 前向传播
        H = self.n_head  # 获取头的数量
        T = self.chunk_len  # 获取块长度

        if 'r3' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 r3
            r, k, v, g = self.jit_func(x)  # 调用 JIT 函数
        else:
            r, k, v = self.jit_func(x)  # 调用 JIT 函数

        w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-1)  # 计算权重
        
        if 'r2' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 r2
            u = self.time_faaaa.float().unsqueeze(-1)  # 获取时间参数
        else:
            u = torch.exp(self.time_first.float()).unsqueeze(-1)  # 获取时间参数

################################################################################
########
        ws = w.pow(T).reshape(1, H, 1, 1)  # 计算状态

        ind = torch.arange(T-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)  # 创建索引
        w = w.repeat(1, T).pow(ind)  # 计算权重

        wk = w.reshape(1, H, 1, T)  # 计算权重
        wb = wk.transpose(-2, -1).flip(2)  # 计算权重

        w = torch.cat([w[:, 1:], u], dim=1)  # 连接权重
        w = F.pad(w, (0, T))  # 填充权重
        w = torch.tile(w, [T])  # 重复权重
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)  # 重塑权重
        w = w[:, :, T-1:].reshape(1, H, T, T)  # 重塑权重
########
################################################################################

        w = w.to(dtype=r.dtype)  # 转换权重数据类型
        wk = wk.to(dtype=r.dtype)  # 转换权重数据类型
        wb = wb.to(dtype=r.dtype)  # 转换权重数据类型
        ws = ws.to(dtype=r.dtype)  # 转换权重数据类型
        if 'r3' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 r3
            return self.jit_func_2(r, k, v, g, w, wk, wb, ws)  # 调用 JIT 函数 2
        else:
            return self.jit_func_2(r, k, v, w, wk, wb, ws)  # 调用 JIT 函数 2

########################################################################################################
# CUDA RWKV5 内核
########################################################################################################

if 'r4' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 r4
    HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])  # 获取头的大小
    wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])  # 加载 CUDA 内核
        
    class WKV_5(torch.autograd.Function):  # 定义 WKV_5 类
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):  # 前向传播
            with torch.no_grad():  # 不计算梯度
                assert r.dtype == torch.bfloat16  # 确保 r 的数据类型为 bfloat16
                assert k.dtype == torch.bfloat16  # 确保 k 的数据类型为 bfloat16
                assert v.dtype == torch.bfloat16  # 确保 v 的数据类型为 bfloat16
                assert w.dtype == torch.bfloat16  # 确保 w 的数据类型为 bfloat16
                assert u.dtype == torch.bfloat16  # 确保 u 的数据类型为 bfloat16
                assert HEAD_SIZE == C // H  # 确保头的大小与通道数匹配
                ctx.B = B  # 保存批次大小
                ctx.T = T  # 保存时间步长
                ctx.C = C  # 保存通道数
                ctx.H = H  # 保存头的数量
                assert r.is_contiguous()  # 确保 r 是连续的
                assert k.is_contiguous()  # 确保 k 是连续的
                assert v.is_contiguous()  # 确保 v 是连续的
                assert w.is_contiguous()  # 确保 w 是连续的
                assert u.is_contiguous()  # 确保 u 是连续的
                ew = (-torch.exp(w.float())).contiguous()  # 计算权重
                eew = (torch.exp(ew)).contiguous()  # 计算权重
                ctx.save_for_backward(r, k, v, eew, ew, u)  # 保存反向传播所需的张量
                y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # 创建输出张量
                wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)  # 调用 CUDA 内核进行前向传播
                return y  # 返回输出

        @staticmethod
        def backward(ctx, gy):  # 反向传播
            with torch.no_grad():  # 不计算梯度
                assert gy.dtype == torch.bfloat16  # 确保 gy 的数据类型为 bfloat16
                B = ctx.B  # 获取批次大小
                T = ctx.T  # 获取时间步长
                C = ctx.C  # 获取通道数
                H = ctx.H  # 获取头的数量
                assert gy.is_contiguous()  # 确保 gy 是连续的
                r, k, v, eew, ew, u = ctx.saved_tensors  # 获取保存的张量
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # 创建梯度张量
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # 创建梯度张量
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # 创建梯度张量
                gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # 创建梯度张量
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)  # 创建梯度张量
                wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)  # 调用 CUDA 内核进行反向传播
                gw = torch.sum(gw, 0).view(H, C//H)  # 对 gw 进行求和并重塑
                gu = torch.sum(gu, 0).view(H, C//H)  # 对 gu 进行求和并重塑
                return (None, None, None, None, gr, gk, gv, gw, gu)  # 返回梯度

    def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):  # 定义运行 CUDA RWKV5 的函数
        return WKV_5.apply(B, T, C, H, r, k, v, w, u)  # 调用 WKV_5 的 apply 方法

########################################################################################################

class RWKV_TimeMix_RWKV5(MyModule):  # 定义 RWKV 时间混合 RWKV5 类
    def __init__(self, args, layer_id):  # 初始化
        super().__init__()  # 调用父类构造函数
        self.args = args  # 保存参数
        self.layer_id = layer_id  # 保存层 ID

        self.head_size = args.head_size_a  # 头的大小
        assert HEAD_SIZE == self.head_size  # 确保 HEAD_SIZE 与参数匹配
        self.n_head = args.dim_att // self.head_size  # 计算头的数量
        assert args.dim_att % self.n_head == 0  # 确保维度可以被头的数量整除
        self.head_size_divisor = args.head_size_divisor  # 头大小的除数

        with torch.no_grad():  # 不计算梯度
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 计算比例从 0 到 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 计算比例从 1 到接近 0
            ddd = torch.ones(1, 1, args.n_embd)  # 创建全 1 的张量
            for i in range(args.n_embd):  # 填充张量
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)  # 创建衰减速度张量
            for n in range(args.dim_att):  # 填充衰减速度
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))  # 时间衰减参数
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(args.dim_att)]) * 0.5  # 创建锯齿波
            self.time_first = nn.Parameter(torch.ones(args.dim_att) * math.log(0.3) + zigzag)  # 时间参数

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))  # 时间混合参数 k
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)  # 时间混合参数 v
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))  # 时间混合参数 r

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 时间移位
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)  # 键线性层
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)  # 值线性层
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)  # 接收线性层
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)  # 输出线性层

        if 'a' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 a
            self.register_buffer("att_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))  # 注册注意力掩码
            d_qkv = args.n_embd // 16  # 计算 QKV 的维度
            self.qq = nn.Linear(args.n_embd, d_qkv, bias=False)  # Q 线性层
            self.kk = nn.Linear(args.n_embd, d_qkv, bias=False)  # K 线性层
            self.vv = nn.Linear(args.n_embd, d_qkv, bias=False)  # V 线性层
            self.oo = nn.Linear(d_qkv, args.n_embd, bias=False)  # 输出线性层
            with torch.no_grad():  # 不计算梯度
                self.time_mix_qq = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))  # 时间混合参数 qq
                self.time_mix_kk = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))  # 时间混合参数 kk
                self.time_mix_vv = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)  # 时间混合参数 vv

    if 'a' not in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式不包含 a
        @MyFunction
        def jit_func(self, x):  # JIT 函数
            xx = self.time_shift(x)  # 将 x 与前一个时间步混合以生成 xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 计算 k
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)  # 计算 v
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 计算 r
            k = self.key(xk)  # 计算 k
            v = self.value(xv)  # 计算 v
            r = self.receptance(xr)  # 计算 r
            sr = torch.sigmoid(r)  # 计算激活值
            return sr, k, v  # 返回激活值、k 和 v

        def forward(self, x):  # 前向传播
            B, T, C = x.size()  # 获取输入的维度
            sr, k, v = self.jit_func(x)  # 调用 JIT 函数
            rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)  # 计算 RWKV
            return self.output(rwkv)  # 返回输出

    if 'a' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 a
        @MyFunction
        def QKV(self, q, k, v):  # QKV 函数
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # 计算注意力
            att = att.masked_fill(self.att_mask == 0, float('-inf'))  # 应用掩码
            att = F.softmax(att, dim=-1)  # 计算 softmax
            x = att @ v  # 计算输出
            return x  # 返回输出

        @MyFunction
        def jit_funcQKV(self, x):  # JIT QKV 函数
            xx = self.time_shift(x)  # 将 x 与前一个时间步混合以生成 xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 计算 k
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)  # 计算 v
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 计算 r
            xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)  # 计算 qq
            xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)  # 计算 kk
            xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)  # 计算 vv
            k = self.key(xk)  # 计算 k
            v = self.value(xv)  # 计算 v
            r = self.receptance(xr)  # 计算 r
            sr = torch.sigmoid(r)  # 计算激活值
            qq = self.qq(xqq)  # 计算 qq
            kk = self.kk(xkk)  # 计算 kk
            vv = self.vv(xvv)  # 计算 vv
            return sr, k, v, qq, kk, vv  # 返回激活值、k、v、qq、kk 和 vv

        def forward(self, x):  # 前向传播
            B, T, C = x.size()  # 获取输入的维度
            sr, k, v, qq, kk, vv = self.jit_funcQKV(x)  # 调用 JIT QKV 函数
            rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)  # 计算 RWKV
            rwkv = self.output(rwkv) + self.oo(self.QKV(qq, kk, vv))  # 计算输出
            return rwkv  # 返回输出

########################################################################################################

class RWKV_ChannelMix(MyModule):  # 定义 RWKV 通道混合类
    def __init__(self, args, layer_id):  # 初始化
        super().__init__()  # 调用父类构造函数
        self.args = args  # 保存参数
        self.layer_id = layer_id  # 保存层 ID
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 时间移位

        with torch.no_grad():  # 不计算梯度
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 计算比例从 1 到接近 0
            ddd = torch.ones(1, 1, args.n_embd)  # 创建全 1 的张量
            for i in range(args.n_embd):  # 填充张量
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))  # 时间混合参数 k
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))  # 时间混合参数 r
        
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)  # 键线性层
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)  # 接收线性层
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)  # 值线性层

    @MyFunction
    def forward(self, x):  # 前向传播
        xx = self.time_shift(x)  # 将 x 与前一个时间步混合
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 计算 k
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 计算 r
        k = self.key(xk)  # 计算 k
        k = torch.relu(k) ** 2  # 计算 k 的激活值
        kv = self.value(k)  # 计算值
        return torch.sigmoid(self.receptance(xr)) * kv  # 返回加权输出

class MishGLU(MyModule):  # 定义 MishGLU 类
    def __init__(self, args, layer_id):  # 初始化
        super().__init__()  # 调用父类构造函数
        self.args = args  # 保存参数
        self.layer_id = layer_id  # 保存层 ID
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # 时间移位

        with torch.no_grad():  # 不计算梯度
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 计算比例从 1 到接近 0

            x = torch.ones(1, 1, args.n_embd)  # 创建全 1 的张量
            for i in range(args.n_embd):  # 填充张量
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))  # 时间混合参数 k
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))  # 时间混合参数 r
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)  # aa 线性层
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)  # bb 线性层
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)  # 值线性层

    @MyFunction
    def forward(self, x):  # 前向传播
        xx = self.time_shift(x)  # 将 x 与前一个时间步混合
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)  # 计算 xa
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)  # 计算 xb
        a = self.aa(xa)  # 计算 a
        b = self.bb(xb)  # 计算 b
        return self.value(a * F.mish(b))  # 返回加权输出

########################################################################################################
# RWKV 模型与我们的模块
########################################################################################################

class Block(nn.Module):  # 定义 Block 类
    def __init__(self, args, layer_id):  # 初始化
        super().__init__()  # 调用父类构造函数
        self.args = args  # 保存参数
        self.layer_id = layer_id  # 保存层 ID

        self.ln1 = nn.LayerNorm(args.n_embd)  # 第一层归一化
        self.ln2 = nn.LayerNorm(args.n_embd)  # 第二层归一化

        if self.layer_id == 0:  # 如果是第一层
            self.ln0 = nn.LayerNorm(args.n_embd)  # 第零层归一化
            if args.my_pos_emb > 0:  # 如果使用位置嵌入
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))  # 位置嵌入 x
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))  # 位置嵌入 y

        if self.layer_id == 0 and self.args.pre_ffn > 0:  # 如果是第一层并且使用前馈网络
            self.ffnPre = RWKV_ChannelMix(args, 0)  # 初始化前馈网络
        else:
            if 'r4' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 r4
                self.att = RWKV_TimeMix_RWKV5(args, layer_id)  # 初始化时间混合 RWKV5
            elif 'r' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 r
                self.att = RWKV_TimeMix_RWKV5_Preview(args, layer_id)  # 初始化时间混合 RWKV5 预览
            else:
                self.att = RWKV_TimeMix(args, layer_id)  # 初始化时间混合

        if 'g' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 g
            self.ffn = MishGLU(args, layer_id)  # 初始化 MishGLU
        else:
            self.ffn = RWKV_ChannelMix(args, layer_id)  # 初始化通道混合
        
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:  # 如果使用小型注意力
            self.tiny_ln = nn.LayerNorm(args.n_embd)  # 小型注意力归一化
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)  # 小型 Q 线性层
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)  # 小型 K 线性层
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)  # 小型 V 线性层
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))  # 注册小型掩码

        if args.dropout > 0:  # 如果使用 dropout
            self.drop0 = nn.Dropout(p = args.dropout)  # 第一个 dropout
            self.drop1 = nn.Dropout(p = args.dropout)  # 第二个 dropout
        
    def forward(self, x, x_emb=None):  # 前向传播
        args = self.args  # 获取参数
        B, T, C = x.size()  # 获取输入的维度
        if self.layer_id == 0:  # 如果是第一层
            x = self.ln0(x)  # 归一化
            if args.my_pos_emb > 0:  # 如果使用位置嵌入
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]  # 计算位置嵌入
                x = x + pos_emb  # 添加位置嵌入

        if self.args.dropout == 0:  # 如果不使用 dropout
            if self.layer_id == 0 and args.pre_ffn > 0:  # 如果是第一层并且使用前馈网络
                x = x + self.ffnPre(self.ln1(x))  # 添加前馈网络输出
            else:
                x = x + self.att(self.ln1(x))  # 添加注意力输出
            x = x + self.ffn(self.ln2(x))  # 添加前馈网络输出
        else:  # 如果使用 dropout
            if self.layer_id == 0 and args.pre_ffn > 0:  # 如果是第一层并且使用前馈网络
                x = self.drop0(x + self.ffnPre(self.ln1(x)))  # 添加前馈网络输出并应用 dropout
            else:
                x = self.drop0(x + self.att(self.ln1(x)))  # 添加注意力输出并应用 dropout
            x = self.drop1(x + self.ffn(self.ln2(x)))  # 添加前馈网络输出并应用 dropout

        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:  # 如果使用小型注意力
            xx = self.tiny_ln(x)  # 归一化
            q = self.tiny_q(xx)[:, :T, :]  # 计算 Q
            k = self.tiny_k(xx)[:, :T, :]  # 计算 K
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))  # 计算注意力
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)  # 应用掩码
            x = x + c @ self.tiny_v(x_emb)  # 添加小型注意力输出
        return x  # 返回输出

class L2Wrap(torch.autograd.Function):  # 定义 L2Wrap 类
    @staticmethod
    def forward(ctx, loss, y):  # 前向传播
        ctx.save_for_backward(y)  # 保存 y
        return loss  # 返回损失

    @staticmethod
    def backward(ctx, grad_output):  # 反向传播
        y = ctx.saved_tensors[0]  # 获取保存的张量
        # 鼓励 logits 接近 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])  # 计算因子
        maxx, ids = torch.max(y, -1, keepdim=True)  # 获取最大值和索引
        gy = torch.zeros_like(y)  # 创建梯度张量
        gy.scatter_(-1, ids, maxx * factor)  # 更新梯度
        return (grad_output, gy)  # 返回梯度

class RWKV(pl.LightningModule):  # 定义 RWKV 类
    def __init__(self, args):  # 初始化
        super().__init__()  # 调用父类构造函数
        self.args = args  # 保存参数
        if not hasattr(args, 'dim_att'):  # 如果未设置 dim_att
            args.dim_att = args.n_embd  # 设置为 n_embd
        if not hasattr(args, 'dim_ffn'):  # 如果未设置 dim_ffn
            args.dim_ffn = args.n_embd * 4  # 设置为 n_embd 的 4 倍
        if not hasattr(args, 'tiny_att_layer'):  # 如果未设置 tiny_att_layer
            args.tiny_att_layer = -1  # 设置为 -1
        if not hasattr(args, 'tiny_att_dim'):  # 如果未设置 tiny_att_dim
            args.tiny_att_dim = -1  # 设置为 -1
        assert args.n_embd % 32 == 0  # 确保 n_embd 可以被 32 整除
        assert args.dim_att % 32 == 0  # 确保 dim_att 可以被 32 整除
        assert args.dim_ffn % 32 == 0  # 确保 dim_ffn 可以被 32 整除

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)  # 嵌入层

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])  # 创建模块列表

        self.ln_out = nn.LayerNorm(args.n_embd)  # 输出层归一化
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)  # 输出线性层

        if args.head_qk > 0:  # 如果使用 QK 头
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)  # Q 线性层
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)  # K 线性层
            self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))  # 注册复制掩码
        if args.dropout > 0:  # 如果使用 dropout
            self.drop0 = nn.Dropout(p = args.dropout)  # 第一个 dropout

    def configure_optimizers(self):  # 配置优化器
        args = self.args  # 获取参数
        
        lr_decay = set()  # 学习率衰减集合
        lr_1x = set()  # 学习率 1x 集合
        lr_2x = set()  # 学习率 2x 集合
        lr_3x = set()  # 学习率 3x 集合
        for n, p in self.named_parameters():  # 遍历参数
            if ("time_mix" in n) and (args.layerwise_lr > 0):  # 如果是时间混合参数
                if args.my_pile_stage == 2:  # 如果是阶段 2
                    lr_2x.add(n)  # 添加到 2x 集合
                else:
                    lr_1x.add(n)  # 添加到 1x 集合
            elif ("time_decay" in n) and (args.layerwise_lr > 0):  # 如果是时间衰减参数
                if args.my_pile_stage == 2:  # 如果是阶段 2
                    lr_3x.add(n)  # 添加到 3x 集合
                else:
                    lr_2x.add(n)  # 添加到 2x 集合
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):  # 如果是时间参数
                if args.my_pile_stage == 2:  # 如果是阶段 2
                    lr_2x.add(n)  # 添加到 2x 集合
                else:
                    lr_1x.add(n)  # 添加到 1x 集合
            elif ("time_first" in n) and (args.layerwise_lr > 0):  # 如果是时间参数
                lr_3x.add(n)  # 添加到 3x 集合
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):  # 如果是权重衰减参数
                lr_decay.add(n)  # 添加到衰减集合
            else:
                lr_1x.add(n)  # 添加到 1x 集合

        lr_decay = sorted(list(lr_decay))  # 排序衰减集合
        lr_1x = sorted(list(lr_1x))  # 排序 1x 集合
        lr_2x = sorted(list(lr_2x))  # 排序 2x 集合
        lr_3x = sorted(list(lr_3x))  # 排序 3x 集合
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}  # 创建参数字典
        
        if args.layerwise_lr > 0:  # 如果使用分层学习率
            if args.my_pile_stage == 2:  # 如果是阶段 2
                optim_groups = [  # 优化器组
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},  # 测试: 2e-3 / args.lr_init
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},  # 测试: 3e-3 / args.lr_init
                ]
            else:
                optim_groups = [  # 优化器组
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]  # 默认优化器组

        if args.weight_decay > 0:  # 如果使用权重衰减
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]  # 添加衰减参数
            if self.deepspeed_offload:  # 如果使用 DeepSpeed
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)  # 返回 DeepSpeed 优化器
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)  # 返回 FusedAdam 优化器
        else:
            if self.deepspeed_offload:  # 如果使用 DeepSpeed
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)  # 返回 DeepSpeed 优化器
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)  # 返回 FusedAdam 优化器
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:  # 检查是否使用 DeepSpeed
        strategy = self.trainer.strategy  # 获取训练策略
        if isinstance(strategy, DeepSpeedStrategy):  # 如果是 DeepSpeed 策略
            cfg = strategy.config["zero_optimization"]  # 获取配置
            return cfg.get("offload_optimizer") or cfg.get("offload_param")  # 返回是否启用 offload
        return False

    def forward(self, idx):  # 前向传播
        args = self.args  # 获取参数
        B, T = idx.size()  # 获取输入的维度
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."  # 确保上下文长度不超过模型限制

        x = self.emb(idx)  # 嵌入输入
        x_emb = x  # 保存嵌入

        if args.dropout > 0:  # 如果使用 dropout
            x = self.drop0(x)  # 应用 dropout
        if args.tiny_att_dim > 0:  # 如果使用小型注意力
            for block in self.blocks:  # 遍历块
                if args.grad_cp == 1:  # 如果启用梯度检查点
                    x = deepspeed.checkpointing.checkpoint(block, x, x_emb)  # 应用检查点
                else:
                    x = block(x, x_emb)  # 直接调用块
        else:
            for block in self.blocks:  # 遍历块
                if args.grad_cp == 1:  # 如果启用梯度检查点
                    x = deepspeed.checkpointing.checkpoint(block, x)  # 应用检查点
                else:
                    x = block(x)  # 直接调用块

        x = self.ln_out(x)  # 输出层归一化

        if args.head_qk > 0:  # 如果使用 QK 头
            q = self.head_q(x)[:, :T, :]  # 计算 Q
            k = self.head_k(x)[:, :T, :]  # 计算 K
            c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)  # 计算注意力
            c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)  # 应用掩码

            if "32" in os.environ["RWKV_FLOAT_MODE"]:  # 如果使用 32 位浮点模式
                c = c @ F.one_hot(idx, num_classes=args.vocab_size)  # 计算输出
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":  # 如果使用 fp16 浮点模式
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()  # 计算输出
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":  # 如果使用 bf16 浮点模式
                c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()  # 计算输出

            x = self.head(x) + c  # 计算最终输出
        else:
            x = self.head(x)  # 计算最终输出

        return x  # 返回输出

    def training_step(self, batch, batch_idx):  # 训练步骤
        args = self.args  # 获取参数
        if args.my_qa_mask != 1:  # 如果不使用 QA 掩码
            idx, targets = batch  # 获取输入和目标
            logits = self(idx)  # 计算 logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # 计算交叉熵损失
            # if '0' in os.environ["RWKV_MY_TESTING"]:
            #     print('logits', logits)
            #     torch.set_printoptions(threshold=10000)
            #     print('idx', idx)
            #     exit(0)
        else:  # 如果使用 QA 掩码
            idx, targets, mask = batch  # 获取输入、目标和掩码
            mask = mask.view(-1)  # 重塑掩码
            sum_mask = torch.sum(mask).item()  # 计算掩码总和
            # if sum_mask == 0:
            #     return torch.tensor([0.0], requires_grad=True)

            logits = self(idx)  # 计算 logits
            if sum_mask == mask.shape[0]:  # 如果掩码总和等于掩码形状
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # 计算交叉熵损失
                # print('rank', self.global_rank, 'loss', loss.item())
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')  # 计算交叉熵损失
                # loss_raw = loss
                loss = torch.sum(loss * mask) / sum_mask  # 计算加权损失

                # torch.set_printoptions(threshold=10000)
                # if True: #self.global_rank == 1:
                #     tmp = ''
                #     sss = 0
                #     ccc = 0
                #     for i in range(mask.shape[0]):
                #         if mask[i] > 0:
                #             tmp += str(idx.view(-1)[i].item()) + ','
                #             sss += loss_raw.view(-1)[i].float().item()
                #             ccc += 1
                #     print('rank', self.global_rank, 'loss', loss.item(), 'lavg', sss / ccc)#, 'tmp', tmp, 'input', idx)

        return L2Wrap.apply(loss, logits)  # 返回损失和 logits

    def training_step_end(self, batch_parts):  # 训练步骤结束
        if pl.__version__[0]!='2':  # 如果不是 PyTorch Lightning 2.x
            all = self.all_gather(batch_parts)  # 收集所有部分
            if self.trainer.is_global_zero:  # 如果是全局零节点
                self.trainer.my_loss_all = all  # 保存损失

    def generate_init_weight(self):  # 生成初始化权重
        print(
            f"""
############################################################################
#
# 初始化模型权重（大型模型较慢）...
#
############################################################################
"""
        )
        m = {}  # 初始化权重字典
        for n in self.state_dict():  # 遍历模型状态字典
            p = self.state_dict()[n]  # 获取参数
            shape = p.shape  # 获取参数形状

            gain = 1.0  # 初始化增益
            scale = 1.0  # 初始化缩放因子
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:  # 如果是归一化或时间参数
                if 'ln_x.weight' in n:  # 如果是 ln_x 权重
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer  # 计算层缩放
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)  # 初始化为缩放值
                else:
                    m[n] = p  # 直接赋值
            else:
                if n == "emb.weight":  # 如果是嵌入权重
                    scale = -1 * self.args.lr_init  # 设置缩放因子
                else:
                    if shape[0] > shape[1]:  # 如果形状符合条件
                        gain = math.sqrt(shape[0] / shape[1])  # 计算增益
                    if 'r' in os.environ["RWKV_MY_TESTING"]:  # 如果测试模式包含 r
                        zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']  # 零初始化参数
                    else:
                        zero = [".att.key.", ".att.receptance.", ".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']  # 零初始化参数
                    for kk in zero:  # 遍历零初始化参数
                        if kk in n:  # 如果参数名称包含
                            scale = 0  # 设置缩放因子为 0
                    if n == "head.weight":  # 如果是输出权重
                        scale = 0.5  # 设置缩放因子
                    if "head_k." in n:  # 如果是 K 头权重
                        scale = 0.1  # 设置缩放因子
                    if "head_q." in n:  # 如果是 Q 头权重
                        scale = 0  # 设置缩放因子

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")  # 打印参数信息

                if self.args.accelerator.upper() == "GPU":  # 如果使用 GPU
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")  # 创建 CUDA 张量
                else:
                    m[n] = torch.empty((shape[0], shape[1]))  # 创建张量

                if scale == 0:  # 如果缩放因子为 0
                    nn.init.zeros_(m[n])  # 初始化为 0
                elif scale < 0:  # 如果缩放因子小于 0
                    nn.init.uniform_(m[n], a=scale, b=-scale)  # 均匀初始化
                else:  # 如果缩放因子大于 0
                    nn.init.orthogonal_(m[n], gain=gain * scale)  # 正交初始化

            m[n] = m[n].cpu()  # 将参数移到 CPU
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":  # 如果使用 fp16
                m[n] = m[n].half()  # 转换为 fp16
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":  # 如果使用 bf16
                m[n] = m[n].bfloat16()  # 转换为 bf16

            # if n == "emb.weight":
            #     print(m[n])

        gc.collect()  # 垃圾回收
        torch.cuda.empty_cache()  # 清空 CUDA 缓存
        return m  # 返回初始化的权重

# run_model.py
########################################################################################################
# RWKV 语言模型 - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types  # 导入 types 模块
import torch  # 导入 PyTorch 库
import math, os, gc  # 导入数学、操作系统和垃圾回收模块
from torch.nn import functional as F  # 导入 PyTorch 的功能性模块
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from typing import List, Dict  # 导入类型提示模块

MyModule = nn.Module  # 将 MyModule 定义为 nn.Module
def __nop(ob):  # 定义一个空操作函数
    return ob  # 返回输入对象
MyFunction = __nop  # 将 MyFunction 定义为空操作

# # 尝试使用 torchdynamo
# import torchdynamo
# MyFunction = torchdynamo.optimize(os.environ["RWKV_RUN_BACKEND"]) # !!!BUGGY!!! 输出错误

# 尝试使用 torch jit --> 对于 fp32 更快，对于 fp16 更慢（为什么？）
if os.environ["RWKV_JIT_ON"] == "1":  # 如果启用 JIT
    MyModule = torch.jit.ScriptModule  # 将 MyModule 定义为 TorchScript 模块
    MyFunction = torch.jit.script_method  # 将 MyFunction 定义为 TorchScript 方法

RWKV_HEAD_QK_DIM = 0  # 定义 RWKV_HEAD_QK_DIM
print(f'\nRWKV_HEAD_QK_DIM {RWKV_HEAD_QK_DIM} RWKV_JIT_ON {os.environ["RWKV_JIT_ON"]}\n')  # 打印头维度和 JIT 状态

DEBUG_TIME = False   # True False - 显示训练时间系数

RWKV_RESCALE_LAYER = 6  # 每 X 层将 x=x/2

############################################################################################################

class RWKV_RNN(MyModule):  # 定义 RWKV_RNN 类
    def __init__(self, args):  # 初始化方法
        super().__init__()  # 调用父类构造函数

        self.args = args  # 保存参数
        self.FLOAT_MODE = args.FLOAT_MODE  # 获取浮点模式
        self.RUN_DEVICE = args.RUN_DEVICE  # 获取运行设备

        with torch.no_grad():  # 不计算梯度
            w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')  # 加载模型权重
            # 精炼权重并发送到正确的设备
            keys = list(w.keys())  # 获取权重的键
            if 'pos_emb_x' in keys:  # 如果存在位置嵌入
                w['pos_emb'] = (w['pos_emb_x'] + w['pos_emb_y']).reshape(args.ctx_len+1, -1)[:-1,:]  # 合并位置嵌入
            keys = list(w.keys())  # 更新键列表
            print_need_newline = False  # 控制打印换行
            for x in keys:  # 遍历权重键
                block_id = 0  # 初始化块 ID
                if 'blocks.' in x:  # 如果是块权重
                    block_id = int(x.split('.')[1])  # 获取块 ID
                if 'att.output.weight' in x:  # 如果是注意力输出权重
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))  # 进行缩放
                if 'ffn.value.weight' in x:  # 如果是前馈网络值权重
                    w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))  # 进行缩放
                                
                if '.time_' in x:  # 如果是时间相关权重
                    w[x] = w[x].squeeze()  # 去除多余维度
                    if DEBUG_TIME:  # 如果调试时间
                        print(x, w[x].numpy())  # 打印权重
                if '.time_decay' in x:  # 如果是时间衰减权重
                    w[x] = w[x].float()  # 转换为浮点数
                    w[x] = -torch.exp(w[x])  # 计算负指数
                elif '.time_first' in x:  # 如果是时间首权重
                    w[x] = w[x].float()  # 转换为浮点数
                else:  # 其他权重
                    if self.FLOAT_MODE == "fp32":  # 如果使用 fp32
                        w[x] = w[x].float()  # 转换为浮点数
                    elif self.FLOAT_MODE == "bf16":  # 如果使用 bf16
                        w[x] = w[x].bfloat16()  # 转换为 bf16
                    elif self.FLOAT_MODE == "fp16":  # 如果使用 fp16
                        w[x] = w[x].half()  # 转换为 fp16

                w[x].requires_grad = False  # 不需要计算梯度
                if args.RUN_DEVICE == 'cuda' and x != 'emb.weight':  # 如果在 CUDA 上运行且不是嵌入权重
                    w[x] = w[x].cuda()  # 将权重移动到 GPU

                if ('blocks.' not in x) or ('blocks.0.' in x):  # 如果不是块权重或是第一个块
                    if print_need_newline:  # 如果需要换行
                        print('\n', end = '')  # 打印换行
                        print_need_newline = False  # 重置标志
                    print(x.ljust(40), str(w[x].dtype).replace('torch.', '').ljust(10), w[x].device)  # 打印权重信息
                else:  # 否则
                    print_need_newline = True  # 设置换行标志
                    print('.', end = '', flush = True)  # 打印点

        # 将权重存储在 self.w 中
        keys = list(w.keys())  # 获取权重的键
        self.w = types.SimpleNamespace()  # 创建简单命名空间
        for x in keys:  # 遍历权重键
            xx = x.split('.')  # 按点分割键
            here = self.w  # 当前命名空间
            for i in range(len(xx)):  # 遍历分割后的键
                if xx[i].isdigit():  # 如果是数字
                    ii = int(xx[i])  # 转换为整数
                    if ii not in here:  # 如果当前命名空间没有该键
                        here[ii] = types.SimpleNamespace()  # 创建新的命名空间
                    here = here[ii]  # 移动到下一级命名空间
                else:  # 如果不是数字
                    if i == len(xx) - 1:  # 如果是最后一个键
                        setattr(here, xx[i], w[x])  # 设置属性
                    elif not hasattr(here, xx[i]):  # 如果当前命名空间没有该键
                        if xx[i+1].isdigit():  # 如果下一个键是数字
                            setattr(here, xx[i], {})  # 设置为空字典
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())  # 设置为新的命名空间
                    here = getattr(here, xx[i])  # 移动到下一级命名空间

        self.eval()  # 设置为评估模式
        gc.collect()  # 垃圾回收
        torch.cuda.empty_cache()  # 清空 CUDA 缓存

    def LN(self, x, w):  # 定义层归一化方法
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)  # 返回层归一化结果

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @MyFunction  # 使用自定义函数装饰器
    def FF(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):  # 定义前馈网络方法
        if self.FLOAT_MODE == "bf16":  # 如果使用 bf16
            xk = x * time_mix_k + state[5*i+0].type(torch.bfloat16) * (1 - time_mix_k)  # 计算 k
            xr = x * time_mix_r + state[5*i+0].type(torch.bfloat16) * (1 - time_mix_r)  # 计算 r
            state[5*i+0] = x.float()  # 更新状态
        elif self.FLOAT_MODE == "fp16":  # 如果使用 fp16
            xk = x * time_mix_k + state[5*i+0].half() * (1 - time_mix_k)  # 计算 k
            xr = x * time_mix_r + state[5*i+0].half() * (1 - time_mix_r)  # 计算 r
            state[5*i+0] = x.float()  # 更新状态            
        else:  # 其他情况
            xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)  # 计算 k
            xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)  # 计算 r
            state[5*i+0] = x  # 更新状态

        r = torch.sigmoid(rw @ xr)  # 计算 r
        k = torch.square(torch.relu(kw @ xk))  # 计算 k
        kv = vw @ k  # 计算 kv

        return r * kv  # 返回结果

    @MyFunction  # 使用自定义函数装饰器
    def SA(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):  # 定义自注意力方法
        if self.FLOAT_MODE == "bf16":  # 如果使用 bf16
            xk = x * time_mix_k + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_k)  # 计算 k
            xv = x * time_mix_v + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_v)  # 计算 v
            xr = x * time_mix_r + state[5*i+1].type(torch.bfloat16) * (1 - time_mix_r)  # 计算 r
            state[5*i+1] = x.float()  # 更新状态
        elif self.FLOAT_MODE == "fp16":  # 如果使用 fp16
            xk = x * time_mix_k + state[5*i+1].half() * (1 - time_mix_k)  # 计算 k
            xv = x * time_mix_v + state[5*i+1].half() * (1 - time_mix_v)  # 计算 v
            xr = x * time_mix_r + state[5*i+1].half() * (1 - time_mix_r)  # 计算 r
            state[5*i+1] = x.float()  # 更新状态            
        else:  # 其他情况
            xk = x * time_mix_k + state[5*i+1] * (1 - time_mix_k)  # 计算 k
            xv = x * time_mix_v + state[5*i+1] * (1 - time_mix_v)  # 计算 v
            xr = x * time_mix_r + state[5*i+1] * (1 - time_mix_r)  # 计算 r
            state[5*i+1] = x  # 更新状态

        r = torch.sigmoid(rw @ xr)  # 计算 r
        k = kw @ xk  # 计算 k
        v = vw @ xv  # 计算 v

        if '16' in self.FLOAT_MODE:  # 如果使用 16 位浮点数
            kk = k.float()  # 转换为浮点数
            vv = v.float()  # 转换为浮点数
        else:  # 其他情况
            kk = k  # 保持 k
            vv = v  # 保持 v
        aa = state[5*i+2]  # 获取 aa
        bb = state[5*i+3]  # 获取 bb
        pp = state[5*i+4]  # 获取 pp
        ww = time_first + kk  # 计算 ww
        p = torch.maximum(pp, ww)  # 计算 p
        e1 = torch.exp(pp - p)  # 计算 e1
        e2 = torch.exp(ww - p)  # 计算 e2
        a = e1 * aa + e2 * vv  # 计算 a
        b = e1 * bb + e2  # 计算 b
        ww = pp + time_decay  # 更新 ww
        p = torch.maximum(ww, kk)  # 计算 p
        e1 = torch.exp(ww - p)  # 计算 e1
        e2 = torch.exp(kk - p)  # 计算 e2
        state[5*i+2] = e1 * aa + e2 * vv  # 更新状态
        state[5*i+3] = e1 * bb + e2  # 更新状态
        state[5*i+4] = p  # 更新状态
        if self.FLOAT_MODE == "bf16":  # 如果使用 bf16
            wkv = (a / b).type(torch.bfloat16)  # 计算 wkv
        elif self.FLOAT_MODE == "fp16":  # 如果使用 fp16
            wkv = (a / b).half()  # 计算 wkv
        else:  # 其他情况
            wkv = a / b  # 计算 wkv
        
        return ow @ (r * wkv)  # 返回结果

    def forward(self, ctx, state, preprocess_only = False):  # 定义前向传播方法
        with torch.no_grad():  # 不计算梯度
            w = self.w  # 获取权重
            args = self.args  # 获取参数

            x = w.emb.weight[ctx[-1]]  # 获取嵌入权重
            if self.RUN_DEVICE == 'cuda':  # 如果在 CUDA 上运行
                x = x.cuda()  # 将 x 移动到 GPU
            try:
                pos_emb = w.pos_emb[len(ctx)-1]  # 获取位置嵌入
                x = x + pos_emb  # 添加位置嵌入
            except:
                pass  # 忽略异常             

            if state == None:  # 如果状态为空
                state = torch.zeros(args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)  # 初始化状态
                for i in range(args.n_layer):  # 遍历层数
                    state[5*i+4] -= 1e30  # 设置状态

            for i in range(args.n_layer):  # 遍历层数
                if i == 0:  # 如果是第一层
                    x = self.LN(x, w.blocks[i].ln0)  # 进行层归一化
                
                ww = w.blocks[i].att  # 获取当前层的注意力模块
                x = x + self.SA(self.LN(x, w.blocks[i].ln1), state, i,  # 进行自注意力计算
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                
                ww = w.blocks[i].ffn  # 获取当前层的前馈网络模块
                x = x + self.FF(self.LN(x, w.blocks[i].ln2), state, i,  # 进行前馈网络计算
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)
                
                if (i+1) % RWKV_RESCALE_LAYER == 0:  # 如果达到缩放层
                    x = x / 2  # 进行缩放

            if preprocess_only:  # 如果只进行预处理
                return state  # 返回状态

            x = self.LN(x, w.ln_out)  # 进行层归一化
            x = w.head.weight @ x  # 计算输出

            return x.float(), state  # 返回输出和状态
