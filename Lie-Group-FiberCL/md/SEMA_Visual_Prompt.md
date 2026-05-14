# SEMA 模型架构与训练流程 — 图像生成 Prompt 文档

本文档面向 AI 图像生成工具（如 Midjourney、DALL-E、Stable Diffusion），以高度结构化的文字描述 SEMA 模型的所有组件和训练流程。生成图像时请保持统一的科技论文风格（白底、彩色模块、箭头连线、标注清晰），建议使用 **diagram / infographic / technical illustration** 风格。

---

## 一、项目总览

**论文**: Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning (CVPR 2025)
**方法名**: SEMA (Self-Expansion with Mixture of Adapters)
**任务**: 类增量持续学习 (Class-Incremental Learning)
**主干网络**: ViT-B/16 (Vision Transformer, 12层, 768维, 12头)
**核心思想**: 冻结预训练 ViT，在每个 Block 中插入可扩展的 Adapter 模块。新任务到来时自动检测分布偏移、自动添加新 Adapter、用软路由器动态组合多个 Adapter。

---

## 二、整体架构图 (Figure 1 级别 — 完整系统概览)

### 画面描述
一张横向大图，展示从输入图像到最终分类的完整数据流。从上到下或从左到右分为三个大区域：

### 区域 A: 输入
- 一张自然图像（如一只猫），顶部标注 "Input Image 224×224"
- 图像被切割成 14×14 = 196 个 patch，每个 patch 大小为 16×16 像素
- 展示一个紫色方块标注 "Patch Embedding: Conv2d(3→768, kernel=16, stride=16)"
- 输出形状标注: [1 + 196, 768]，其中 +1 是 CLS token（一个粉色小方块）
- 加上位置编码 (Position Embedding, 黄色波纹箭头)

### 区域 B: ViT Encoder (12 个 Block 堆叠)
- 展示 12 个 Block 垂直堆叠，用浅灰色方块表示
- 标注: "12 × Transformer Block, dim=768, heads=12, ALL FROZEN"
- 第 0-8 层用浅灰色，标注 "Layer 0-8: No Adapter (通用底层特征，不扩展)"
- 第 9-11 层用彩色边框高亮，标注 "Layer 9-11: SEMA Adapter Module 插入层（语义层，支持扩展）"
- 每个 Block 右下角有一个橙色小方块 "SEMA Module"

### 区域 C: 输出
- 最后一个 Block 输出后接 Norm → 取 CLS token → Linear Head (768 → 总类别数)
- 标注: "Classifier Head (trainable, expands with new tasks)"
- 右侧输出: "Cat (class 3)" 标注为绿色文字

### 关键标注
- 预训练权重区域用蓝色雪花标志标注 "Frozen (requires_grad=False)"
- Adapter 区域用橙色火焰标志标注 "Trainable (requires_grad=True)"
- 三条信息流箭头:
  1. 绿色箭头: "Forward Path (all tasks)"
  2. 红色虚线: "RD Loss (rd phase only)"
  3. 蓝色虚线: "Detection Signal (expansion detection)"

---

## 三、单个 Transformer Block 内部结构 (Figure 1 右侧放大)

### 画面描述
一个矩形方框 "Transformer Block Layer i"，内部展示如下组件和连接：

```
输入 [B, N, 768]
    │
    ├→ [Norm1: LayerNorm(768)] ────→ [MHSA: Multi-Head Self-Attention]
    │   紫色圆角矩形，标注 "LayerNorm"          蓝色矩形，标注 "Q/K/V Proj → Attention → Proj"
    │   形状: [B,N,768]→[B,N,768]             形状: [B,N,768]→[B,N,768]
    │                                                    │
    └─────────── identity shortcut (灰色细线, ⊕ 符号) ──→ + ─→ x'
                                                              │
    ┌─────────────────────────────────────────────────────────┘
    │
    x' [B, N, 768]
    │
    ├→ [Norm2: LayerNorm(768)] ──→ [MLP: fc1→GELU→fc2]
    │   紫色圆角矩形                  绿色矩形，标注 "768→3072→GELU→3072→768"
    │   形状不变                      形状不变
    │                                      │
    │                                      ├──→ [MLP output] ──→
    │                                      │
    ├→ [SEMA Adapter Module] ──────────────┤
    │   橙色矩形 (核心模块)                   │
    │   形状: [B,N,768]→[B,N,768]          │
    │                                      ↓
    │                                    + (MLP + Adapter)
    │                                      │
    └──── identity shortcut ────────────→ + ──→ 输出 [B, N, 768]
```

### 组件详解

#### Norm (LayerNorm) — 紫色圆角矩形
- 数学: (x - mean) / std * γ + β
- 作用: 稳定特征分布，防止梯度爆炸
- 输入: [B, N, 768]，输出: [B, N, 768]
- 标注: "LayerNorm, 可学习参数 γ, β (冻结)"

#### MHSA (Multi-Head Self-Attention) — 蓝色矩形
- 结构:
  1. q_proj: Linear(768→768) → 拆分 12 头 → 每头 64 维
  2. k_proj: Linear(768→768) → 拆分 12 头 → 每头 64 维
  3. v_proj: Linear(768→768) → 拆分 12 头 → 每头 64 维
  4. Attention: softmax(Q·K^T / √64) · V
  5. proj: Linear(768→768) + Dropout
- 输入: [B, N, 768]，输出: [B, N, 768]
- 标注: "12 Heads, each head 64 dim, 学习 token 间全局依赖"
- 可视化提示: 画出 197×197 的注意力热力图示意

#### MLP (Feed-Forward) — 绿色矩形
- 结构: fc1(768→3072) → GELU → Dropout → fc2(3072→768) → Dropout
- 输入: [B, N, 768]，输出: [B, N, 768]
- 标注: "mlp_ratio=4, GELU 激活, 逐 token 独立处理"

#### 残差连接 (⊕) — 灰色
- 两条 identity shortcut 用灰色细线表示
- ⊕ 符号处标注 "Residual Connection"

---

## 四、SEMA Adapter Module 内部结构 (Figure 2 级别 — 最核心)

### 画面描述
一个橙色的大矩形 "SEMA Module (per ViT Layer)"，内部包含以下子模块：

### 4.1 AdapterModule（单个适配器单元） — 橙色子框
- 每个此模块对应一个任务，包含三个子组件:

```
AdapterModule (Task ID: k)
├── Functional Adapter (瓶颈 MLP)
│   ├── LayerNorm(可选)
│   ├── down_proj: Linear(768 → 16)  ← 瓶颈压缩，16 是 ffn_num
│   ├── ReLU
│   ├── up_proj:   Linear(16 → 768)  ← 扩张恢复
│   └── 初始化: down=KaimingUniform, up=Zero (保证初始输出为0)
│
├── Representation Descriptor / AE (表征描述器 / 自编码器)
│   ├── encoder: Linear(768 → 128)   ← 128 是 rd_dim
│   ├── decoder: Linear(128 → 768)
│   └── 作用: 对特征做压缩再重建，重建误差用于检测分布偏移
│
└── Records (运行统计缓冲区)
    ├── 固定容量 500 的 FIFO 缓冲区
    ├── 存储训练时的重建误差值
    ├── 维护在线均值 μ 和标准差 σ
    └── Z-score = |当前误差 - μ| / σ
```

### 4.2 Router（软路由器） — 蓝色梯形
- 结构: Linear(768 → M)，M = 当前 adapter 数量
- 输入: x 对所有 token 做 mean pooling → [B, 768]
- 输出: logits [B, M] → softmax → mask [B, M]
- 加权组合: output = Σ mask[i] × Adapter_i(x)
- 新 adapter 被添加时:
  1. 创建 new_router = Linear(768 → 1)，权重初始化为 0
  2. 训练后 fix_router() 将新旧列拼接成 Linear(768 → M+1)

### 4.3 扩展检测逻辑 — 红色虚线框
- 条件判断流程图:
  1. 新数据 batch 输入
  2. 每个旧 adapter 的 AE 计算重建误差
  3. 计算 Z-score = |error - μ_history| / σ_history
  4. 检查: **所有**旧 adapter 的 Z-score > exp_threshold (如 2.0)?
     - YES → 添加新 AdapterModule + new_router，训练后冻结
     - NO  → 不扩展，直接用已有 router 组合
  5. 只在第 9-11 层检测，自顶向下扫描

---

## 五、训练流程全景图 (Figure 1 下半部分)

### 画面描述
一张横向时间线图，展示 SEMA 训练一个增量任务的完整流程。

### 阶段划分

```
┌─────────────────────────────────────────────────────────────┐
│                    一个增量任务的训练流程                        │
│                                                             │
│  新任务数据 (Task t, 10 个新类)                               │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────┐                                       │
│  │  Phase 0: 扩展检测  │ ← 红色区域                           │
│  │                    │                                      │
│  │  所有旧 AE 对当前   │                                      │
│  │  数据计算 Z-score   │                                      │
│  │        │            │                                      │
│  │   ┌────┴────┐      │                                      │
│  │   │ 全部超标? │      │                                      │
│  │   └────┬────┘      │                                      │
│  │    YES │     NO    │                                      │
│  │   添加新  │   不扩展  │                                      │
│  │  Adapter  │   ↓    │                                      │
│  └─────┬─────┘ 直接路由│                                      │
│        │         │    │                                      │
│        ▼         ▼    │                                      │
│  ┌──────────────────┐                                       │
│  │ Phase 1: func 阶段 │ ← 蓝色区域                             │
│  │                     │                                      │
│  │  Loss = CrossEntropy│                                     │
│  │  训练: functional +  │                                     │
│  │  router + fc        │                                     │
│  │  Epochs: 5          │                                      │
│  │  Optimizer: SGD/AdamW│                                    │
│  │  Scheduler: Cosine  │                                      │
│  └────────┬───────────┘                                      │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ Phase 2: rd 阶段   │ ← 绿色区域                             │
│  │                     │                                      │
│  │  Loss = MSE(特征,   │                                      │
│  │        AE重建(特征)) │                                      │
│  │  训练: AE encoder +  │                                     │
│  │  decoder             │                                     │
│  │  Epochs: 20         │                                      │
│  │  Optimizer: SGD/AdamW│                                    │
│  └────────┬───────────┘                                      │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ Phase 3: 冻结      │ ← 灰色区域                             │
│  │                     │                                      │
│  │  freeze_functional()│                                     │
│  │  freeze_rd()        │                                     │
│  │  fix_router()       │                                     │
│  │  统计缓冲区停止更新   │                                      │
│  └──────────────────┘                                       │
│                                                             │
│  输出: 任务评估 → 保存 Checkpoint → 上报 wandb                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 六、数据预处理管线

### 画面描述
一个横向带状流程图，展示图像从磁盘到模型输入的完整路径。

```
磁盘数据                          Transform                        模型输入
┌──────────┐    ┌────────────────────────────┐    ┌─────────────────────┐
│ CIFAR-100 │───→│ Training:                  │───→│ DummyDataset        │
│ (numpy)   │    │ RandomResizedCrop(224)     │    │ __getitem__:        │
│ 32×32     │    │   scale=(0.05, 1.0)       │    │   PIL/numpy →       │
│           │    │   ratio=(3/4, 4/3)        │    │   transform →       │
│ ImageNet-R│    │ RandomHorizontalFlip(0.5)  │    │   (idx, img_tensor, │
│ (folder)  │    │ ToTensor()                 │    │    label)            │
│           │    ├────────────────────────────┤    └─────────┬───────────┘
│ ImageNet-A│    │ Testing:                   │              │
│ (folder)  │    │ Resize(256, bicubic)      │              ▼
│           │    │ CenterCrop(224)            │    ┌─────────────────────┐
│ VTAB      │    │ ToTensor()                 │    │ DataLoader           │
│ (folder)  │    └────────────────────────────┘    │ batch_size=32        │
└──────────┘                                       │ shuffle=True(train)  │
                                                   │ num_workers=8        │
                                                   │ → (indices, imgs,    │
                                                   │    labels)            │
                                                   └─────────────────────┘
```

### 关键标注
- 输入尺寸: 原始 → 224×224×3 (Tensor 格式 [B, 3, 224, 224])
- **无归一化** (No ImageNet Normalization)，因为 ViT 预训练的 patch_embed 自带像素处理
- CIFAR 从 32×32 上采样到 224×224；ImageNet 系列原生 ≥224
- DummyDataset 支持两种模式: numpy 数组 → PIL Image → transform（CIFAR）；文件路径 → PIL loader → transform（ImageNet系列）

---

## 七、任务划分示意

### 画面描述
一个彩色表格或条形图，展示 100 类数据集如何被切成 10 个增量任务。

```
CIFAR-100 (100 类)

任务 0:  [██████████] 类 0-9    (init_cls=10)   ← 初始化 1 个 Adapter/层
任务 1:  [██████████] 类 10-19  (increment=10)  ← 检测→可能扩展
任务 2:  [██████████] 类 20-29
任务 3:  [██████████] 类 30-39
任务 4:  [██████████] 类 40-49
任务 5:  [██████████] 类 50-59
任务 6:  [██████████] 类 60-69
任务 7:  [██████████] 类 70-79
任务 8:  [██████████] 类 80-89
任务 9:  [██████████] 类 90-99

评估: 每个任务后测试所有已见过的类 (如任务 3 后测试 0-39 共 40 类)
```

### 关键标注
- 任务间不重叠 (Disjoint classes per task)
- `shuffle=True` 时类别顺序随机打乱
- 每个任务的 train_loader 只包含当前任务的新类
- test_loader 包含所有已见过的类

---

## 八、DataType 类型流转图

### 画面描述
维度/类型变化的追踪图，标注每一步的形状。

```
原始图像文件 / numpy 数组
      │
      ▼
PIL Image (data_manager.pil_loader 或 Image.fromarray)
      │
      ▼
transform (RandomResizedCrop / Resize + CenterCrop + ToTensor)
      │
      ▼
torch.Tensor [3, 224, 224]  范围 [0.0, 1.0]
      │
      ▼
DataLoader stack → [B, 3, 224, 224]
      │
      ▼
PatchEmbed → [B, 197, 768]  (1 CLS + 196 patches × 768 dim)
      │
      ▼
12 个 Transformer Block，每层 SEMAModule:
  ├── 内部 adapter 处理: [B, 197, 768] → [B, 197, 768]
  ├── Router 输入: mean_pool([B, 197, 768]) = [B, 768]
  ├── Router 输出: softmax(Linear(768→M)) = [B, M]
  └── AE 重建: 特征均值 [B, 768] → encode [B, 128] → decode [B, 768]
      │
      ▼
Norm + 取 CLS token → [B, 768]
      │
      ▼
Classifier fc(768 → total_classes) → logits [B, total_classes]
      │
      ▼
CrossEntropyLoss (func) 或 取 max 预测类别
```

---

## 九、参数状态总览表

### 画面描述
一个大表格，左侧是模块名，右侧用颜色标注 "Frozen" 或 "Trainable"。

| 模块 | 参数数量 (approx) | 状态 | 颜色建议 |
|------|------------------|------|---------|
| PatchEmbed | ~590K | Frozen (蓝色) | 🔵 |
| Position Embedding | ~151K | Frozen (蓝色) | 🔵 |
| CLS Token | 768 | Frozen (蓝色) | 🔵 |
| 12 × (Q/K/V Proj + Proj) | ~28M | Frozen (蓝色) | 🔵 |
| 12 × LayerNorm | ~37K | Frozen (蓝色) | 🔵 |
| 12 × MLP (fc1+fc2) | ~57M | Frozen (蓝色) | 🔵 |
| **SEMA Adapter (functional)** | ~24K/adapter/层 | **Trainable (橙色)** | 🟠 |
| **SEMA Router** | ~768×M/层 | **Trainable (橙色)** | 🟠 |
| **SEMA AE (encoder+decoder)** | ~200K/adapter/层 | **Trainable (橙色)** | 🟠 |
| **Classifier fc** | 768×总类数 | **Trainable (橙色)** | 🟠 |

### 核心数字
- ViT-B/16 总参数: ~86M
- 初始 adapter 参数 (1 adapter/层): ~3M (仅 3.5%)
- 10 个任务后 adapter 参数: ~3-12M (子线性增长)
- 每个 Adapter: 2×768×16 = 24,576 参数 (functional) + 768×128×2 = 196,608 参数 (AE)

---

## 十、关键设计哲学（用于生成宣传图/海报）

### 三个核心创新点对应的视觉隐喻

#### 创新 1: Representation Descriptor (表征描述器)
- **隐喻**: 指纹识别器 / 安检门
- **画面**: 数据流通过 AE 的编码器-解码器结构，重建误差用一个温度计或仪表盘显示。高误差 → 红色警报 → "陌生人!"
- **标注**: "AutoEncoder as Distribution Detector: learn to reconstruct → detect shift via Z-score"

#### 创新 2: Self-Expansion (自扩展)
- **隐喻**: 细胞分裂 / 树长新枝
- **画面**: 一个 adapter 检测到超出阈值 → 分裂/生长出新的 adapter。子线性增长曲线 (台阶状，越来越平)
- **标注**: "Self-Expanding: only when needed, at most 1 per task, sub-linear growth"

#### 创新 3: Mixture of Adapters (适配器混合)
- **隐喻**: 调音台 / 混合器
- **画面**: 多个彩色 adapter 输出通过一个推子控制台 (Router/Softmax) 混合成最终输出。各推子位置不同 → 知识软组合
- **标注**: "Soft Router: task-agnostic inference, knowledge sharing between similar tasks, no task-ID needed"

---

## 十一、标准配色方案

为保证多张图风格统一，建议使用以下配色：

| 用途 | 颜色 | HEX |
|------|------|-----|
| 冻结的预训练模块 | 浅蓝色 | #B3D9FF |
| 可训练的 Adapter | 橙色 | #FF8C00 |
| Router / 软路由 | 深蓝色 | #0066CC |
| AE / RD 模块 | 绿色 | #2ECC71 |
| 残差连接 | 灰色 | #999999 |
| 损失函数 | 红色 | #E74C3C |
| 分类器 | 紫色 | #9B59B6 |
| 数据/输入 | 黄色 | #F1C40F |
| 检测/警报 | 红色 | #E74C3C |
| 冻结/锁定标志 | 雪花图标 + 蓝色 | - |
| 可训练标志 | 火焰图标 + 橙色 | - |
| 扩展触发 | 闪电图标 + 红色 | - |

---

## 十二、生成建议

### 建议生成的图像列表

1. **主架构图** (横向大图, 16:9): 第二章的整体架构，展示输入→ViT→输出
2. **Block 内部结构图** (竖向或方形, 1:1): 第三章的单个 Block 详细结构
3. **SEMA Module 展开图** (竖向大图, 9:16): 第四章的 SEMAModule 内部，Adapter + Router + AE
4. **训练流程时间线** (横向, 16:9): 第五章的 Phase 0-3
5. **数据处理管线** (横向, 16:9): 第六章的数据预处理流程
6. **三维创新对比图** (三栏, 16:9): 第十章的三个创新 + 隐喻
7. **参数冻结总览图** (大表格, 4:3): 第九章的参数状态表
8. **扩展检测决策树** (竖向流程图, 9:16): 第四章的扩展检测逻辑

### 统一要求
- 所有图使用白色或极浅灰色背景
- 箭头清晰，标注英文（可附带中文翻译）
- 模块用圆角矩形，残差连接用细线
- 维度标注用小号灰色字体
- 所有 Linear 层标注输入→输出维度
- 风格: Technical diagram, clean, academic, professional
