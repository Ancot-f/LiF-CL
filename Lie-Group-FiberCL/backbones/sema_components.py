"""
SEMA 核心基础组件
=================
包含三个关键模块，构成 SEMA 自扩展适配器系统的基础：

1. Adapter — 功能适配器：插入 ViT MLP 层的瓶颈结构，用极少参数弥合预训练与下游任务差距
2. AE (AutoEncoder) — 表征描述器：轻量自编码器，通过重建误差检测特征分布偏移
3. Records — 运行统计缓冲区：维护重建误差的在线均值和标准差，用于计算 Z-score

这些组件被 AdapterModule (sema_block.py) 组合使用，
构成 SEMA 的自扩展检测 + 功能适配双系统。
"""

import torch
from torch import nn
from torch.nn import functional as F
import math


class Adapter(nn.Module):
    """功能适配器 —— 瓶颈 MLP（Bottleneck MLP）。

    结构：
        x → LayerNorm(可选) → Linear(d_model, bottleneck) → ReLU → Linear(bottleneck, d_model) → 输出

    参数效率：
        参数量 ≈ 2 × d_model × bottleneck
        ViT-B/16: 2 × 768 × 16 = 24,576 参数（仅占 ViT 主干 86M 的 0.03%）

    初始化策略（lora 模式）：
        - down_proj: Kaiming Uniform（保证梯度在初始时不消失）
        - up_proj: 全零初始化（适配器初始输出为零，不影响预训练特征）
    """

    def __init__(self,
                 config=None,       # 全局配置对象，含 d_model, attn_bn 等
                 adapter_id=None,   # 适配器标识符，如 "9.0"（第9层第0个适配器）
                 d_model=None,      # 输入/输出维度（覆盖 config.d_model）
                 bottleneck=None,   # 瓶颈维度（覆盖 config.attn_bn）
                 dropout=0.0,       # Dropout 率
                 init_option="bert",# 初始化方式："lora" | "bert"
                 adapter_scalar="1.0",        # 输出缩放因子
                 adapter_layernorm_option="in"):  # LayerNorm 位置："in" | "out" | "none"
        super().__init__()
        self.adapter_id = adapter_id
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option

        # 可选 LayerNorm（在瓶颈前或后）
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # 输出缩放：可学习标量或固定值
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # 瓶颈结构：降维 → 激活 → 升维
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        # LoRA 风格初始化：保证适配器初始输出为零
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)   # 关键：上投影权重为零
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        """前向传播：x → [LayerNorm] → down → ReLU → up → 输出"""
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        output = self.up_proj(down)
        return output


class AE(nn.Module):
    """表征描述器 (Representation Descriptor) —— 轻量自编码器。

    核心作用：
        - 训练时：学习重建当前任务在该层的特征分布
        - 推理时：重建误差大 → 当前输入偏离历史分布 → 触发扩展

    结构：
        x → encoder → 压缩表示 z (rd_dim) → decoder → 重建 x'
        损失：逐样本 MSE(x, x')

    为什么用自编码器？
        自编码器的重建误差天然反映输入与训练分布的匹配程度。
        相比直接比较特征向量，AE 可以捕获更细粒度的分布差异。
    """

    def __init__(self, config):
        super(AE, self).__init__()
        self.input_dim = config.d_model           # 输入维度（ViT-B/16: 768）
        self.config = config
        self.encoder = nn.Linear(self.input_dim, config.rd_dim)   # 768 → rd_dim
        self.decoder = nn.Linear(config.rd_dim, self.input_dim)   # rd_dim → 768
        self.weight_initialize()

    def forward(self, x):
        """编码 → 解码，返回重建后的特征。"""
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        return reconstruction

    def compute_reconstruction_loss(self, x):
        """计算逐样本重建损失。

        这是 SEMA 分布偏移检测的核心：
        1. 对每个样本的 token 序列取均值 → 得到 [B, d_model]
        2. 通过 AE 重建
        3. 计算每个样本独立的重建误差 → [B]

        返回的逐样本损失用于 Z-score 计算，
        判断每个样本是否偏离该 RD 训练时的分布。
        """
        x = x.mean(dim=1)                # [B, N, d_model] → [B, d_model]
        reconstruction = self.forward(x)  # 编码→解码
        reconstruction_losses = []
        B = x.shape[0]
        for i in range(B):               # 逐样本计算 MSE
            reconstruction_losses.append(self.reconstruction_loss(reconstruction[i], x[i]))
        reconstruction_losses = torch.stack(reconstruction_losses)  # [B]
        return reconstruction_losses

    def reconstruction_loss(self, reconstruction, x):
        """单个样本的 MSE 重建损失。"""
        reconstruction_loss = F.mse_loss(reconstruction, x)
        return reconstruction_loss

    def weight_initialize(self):
        """权重初始化：encoder/decoder 均使用 Kaiming Uniform + Zero Bias。"""
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
            nn.init.zeros_(self.encoder.bias)
            nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
            nn.init.zeros_(self.decoder.bias)


class Records:
    """运行统计缓冲区 —— 维护重建误差的在线均值和标准差。

    每个表征描述器 (AE) 有一个对应的 Records 实例，
    记录该 AE 在训练集上的重建误差分布。

    用于计算 Z-score：
        Z = |当前误差 - 历史均值| / 历史标准差
        Z > 阈值 → 分布偏移 → 触发扩展

    实现细节：
        - 固定容量缓冲区 (max_len=500)
        - 满后采用滑动窗口（FIFO）
        - 每次添加后重新计算均值/方差（Welford 风格在线更新）
    """

    def __init__(self, max_len=500) -> None:
        self._max_len = max_len        # 缓冲区最大容量
        self._curr_len = 0             # 当前已存储的样本数
        self.record = torch.zeros(self._max_len)  # 固定大小 tensor 缓冲区
        self._mean = 0                 # 当前均值
        self._var = 0                  # 当前方差
        self._powersumavg = 0
        self.updating = True           # 训练时为 True（收集统计），推理时为 False

    @property
    def length(self):
        """当前缓冲区中的样本数。"""
        return self._curr_len

    @property
    def mean(self):
        """当前均值。"""
        return self._mean

    @property
    def stddev(self):
        """当前标准差（方差开方）。"""
        return math.sqrt(self._var)

    def add_record(self, v):
        """添加新样本并更新运行统计。

        参数 v: 标量或 batch 张量（重建误差值），在 AdapterModule 中已 detach 并移至 CPU

        缓冲区管理：
        - 未满时：直接追加
        - 满后：滑动窗口（FIFO），丢弃最旧的样本
        """
        if not self.updating:
            return
        if self._curr_len < self._max_len:
            # 缓冲区未满：直接填充
            place_left = self._max_len - self._curr_len
            if place_left > len(v):
                self.record[self._curr_len:self._curr_len+len(v)] = v
                self._curr_len += len(v)
            else:
                self.record[self._curr_len:] = v[:place_left]
                self._curr_len = self._max_len
        else:
            # 缓冲区已满：滑动窗口
            self.record = torch.cat([self.record, v])
            self.record = self.record[len(v):]  # 丢弃最旧的 |v| 个样本
        # 重新计算均值和方差
        self._mean = torch.mean(self.record[:self._curr_len])
        self._var = torch.var(self.record[:self._curr_len])
