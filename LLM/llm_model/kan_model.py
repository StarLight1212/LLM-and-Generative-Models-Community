import torch
import torch.nn.functional as F
import math

# 定义 KANLinear 类，继承自 torch.nn.Module
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,  # 输入特征的数量
        out_features,  # 输出特征的数量
        grid_size=5,  # 网格大小
        spline_order=3,  # B-spline 的阶数
        scale_noise=0.1,  # 噪声的缩放因子
        scale_base=1.0,  # 基础权重的缩放因子
        scale_spline=1.0,  # B-spline 权重的缩放因子
        enable_standalone_scale_spline=True,  # 是否启用独立的 B-spline 缩放
        base_activation=torch.nn.SiLU,  # 基础激活函数
        grid_eps=0.02,  # 网格的 epsilon 值
        grid_range=[-1, 1],  # 网格的范围
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features  # 保存输入特征数量
        self.out_features = out_features  # 保存输出特征数量
        self.grid_size = grid_size  # 保存网格大小
        self.spline_order = spline_order  # 保存 B-spline 的阶数

        # 计算网格的步长
        h = (grid_range[1] - grid_range[0]) / grid_size
        # 创建网格
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 注册网格为缓冲区

        # 初始化基础权重和 B-spline 权重
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # 保存其他参数
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()  # 激活函数实例化
        self.grid_eps = grid_eps  # 网格 epsilon 值

        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        # 使用 Kaiming 初始化基础权重
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # 初始化 B-spline 权重
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的 B-spline 基函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: B-spline 基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)  # 增加一个维度
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # 计算基函数
        """
        import numpy as np

        # 定义输入张量 x 和网格 grid
        x = np.array([[0.5], [1.5], [2.5], [3.5]])  # 输入张量，形状为 (4, 1)
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # 网格，形状为 (5,)
        
        # 计算 bases
        # 1. 检查 x 是否大于等于 grid 的左边界
        left_check = (x >= grid[:-1])  # 形状为 (4, 4)
        # 2. 检查 x 是否小于 grid 的右边界
        right_check = (x < grid[1:])  # 形状为 (4, 4)
        # 3. 进行逻辑与操作
        bases = left_check & right_check  # 形状为 (4, 4)
        
        # 打印结果
        print("x:\n", x)
        print("grid:\n", grid)
        print("left_check:\n", left_check)
        print("right_check:\n", right_check)
        print("bases:\n", bases)
        """
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1
            ) + (
                (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )

        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算插值给定点的曲线系数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。

        返回:
            torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(A, B).solution  # 计算最小二乘解
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)  # 重塑输入张量

        # 计算基础输出和 B-spline 输出
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output  # 合并输出
        
        output = output.reshape(*original_shape[:-1], self.out_features)  # 恢复原始形状
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # 计算 B-spline
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # 计算未缩减的 B-spline 输出
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # (batch, in, out)

        # 对每个通道进行排序以收集数据分布
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device)
            .unsqueeze(1) * uniform_step + x_sorted[0] - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)  # 更新网格
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))  # 更新 B-spline 权重

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算正则化损失。

        这是对原始 L1 正则化的简单模拟，因为原始实现需要计算绝对值和熵。
        L1 正则化现在计算为 B-spline 权重的平均绝对值。
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


# 定义 KAN 类，继承自 torch.nn.Module
class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,  # 隐藏层的特征数量
        grid_size=5,  # 网格大小
        spline_order=3,  # B-spline 的阶数
        scale_noise=0.1,  # 噪声的缩放因子
        scale_base=1.0,  # 基础权重的缩放因子
        scale_spline=1.0,  # B-spline 权重的缩放因子
        base_activation=torch.nn.SiLU,  # 基础激活函数
        grid_eps=0.02,  # 网格的 epsilon 值
        grid_range=[-1, 1],  # 网格的范围
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size  # 保存网格大小
        self.spline_order = spline_order  # 保存 B-spline 的阶数

        self.layers = torch.nn.ModuleList()  # 存储 KANLinear 层
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)  # 更新网格
            x = layer(x)  # 前向传播
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
