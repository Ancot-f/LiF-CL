# SEMA-CL: 基于混合适配器自扩展的预训练模型持续学习

## 论文信息

- **标题**: Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning
- **作者**: Huiyi Wang, Haodong Lu, Lina Yao, Dong Gong
- **会议**: CVPR 2025
- **ArXiv**: https://arxiv.org/abs/2403.18886
- **代码基础**: 基于 [PILOT](https://github.com/sun-hailong/LAMDA-PILOT) 和 [AdaptFormer](https://github.com/ShoufaChen/AdaptFormer)

---

## 1. 项目目录结构

```
SEMA-CL-main/
├── main.py                          # 入口：加载配置文件，调度训练/评估
├── trainer.py                       # 训练编排（多随机种子、多任务循环）
├── eval.py                          # 从保存的检查点进行评估
├── sema_env.yaml                    # Conda 环境配置
├── README.md                        # 原始 README
├── LICENSE                          # 许可证
│
├── backbone/                        # Vision Transformer 骨干网络 & SEMA 模块
│   ├── vit_sema.py                  # 支持 SEMA 适配器的 ViT-B/16
│   ├── sema_block.py                # SEMA 核心：SEMAModules（适配器管理模块）、AdapterModule（单个适配器模块）
│   ├── sema_components.py           # 底层组件：Adapter（功能适配器）、AE（自编码器）、Records（统计缓冲区）
│   ├── linears.py                   # 线性分类头（SimpleLinear, CosineLinear 等）
│   ├── vit_adapter.py               # 标准适配器的 ViT（aper_adapter 基线用）
│   ├── vit_ssf.py                   # SSF（Scale-Shift-Feature）的 ViT
│   ├── vit_l2p.py                   # L2P（Learning to Prompt）的 ViT
│   ├── vit_dualprompt.py            # DualPrompt 的 ViT
│   ├── vit_memo.py                  # MEMO 的 ViT
│   ├── vit_coda_promtpt.py          # CODA-Prompt 的 ViT
│   ├── vpt.py                       # Visual Prompt Tuning (VPT) 支持
│   ├── prompt.py                    # CodaPrompt 实现
│   └── resnet.py                    # ResNet 骨干（非 ViT 方法用）
│
├── models/                          # 各类持续学习方法学习器
│   ├── base.py                      # BaseLearner：抽象 CL 基类
│   ├── sema.py                      # SEMA 学习器：核心训练逻辑
│   ├── finetune.py                  # 简单微调基线
│   ├── aper_adapter.py              # APER + 标准适配器
│   ├── aper_ssf.py                  # APER + SSF
│   ├── aper_vpt.py                  # APER + VPT
│   ├── aper_finetune.py             # APER + 微调
│   ├── l2p.py                       # Learning to Prompt
│   ├── dualprompt.py                # DualPrompt
│   ├── coda_prompt.py               # CODA-Prompt
│   ├── simplecil.py                 # SimpleCIL（无遗忘防护）
│   ├── icarl.py                     # iCaRL
│   ├── der.py                       # DER（动态表示扩展）
│   ├── coil.py                      # CoIL
│   ├── foster.py                    # FOSTER
│   └── memo.py                      # MEMO
│
├── utils/                           # 工具模块
│   ├── inc_net.py                   # 网络包装器（SEMAVitNet, BaseNet, IncrementalNet 等）
│   ├── factory.py                   # 模型工厂（get_model）
│   ├── data_manager.py              # DataManager：任务增量数据划分
│   ├── data.py                      # 数据集定义（CIFAR, ImageNet, VTAB 等）
│   └── toolkit.py                   # 工具函数（准确率、参数统计等）
│
├── exps/                            # 实验配置文件（JSON）
│   ├── sema_cifar.json              # CIFAR-100（10 任务 × 10 类）
│   ├── sema_inr_5task.json          # ImageNet-R（5 任务 × 40 类）
│   ├── sema_inr_10task.json         # ImageNet-R（10 任务 × 20 类）
│   ├── sema_inr_20tasks.json        # ImageNet-R（20 任务 × 10 类）
│   ├── sema_ina.json                # ImageNet-A（10 任务 × 10 类）
│   └── sema_vtab.json               # VTAB（5 任务 × 10 类）
│
└── images/                          # 论文插图
    ├── overview.png                 # 方法概览图
    └── expansion.png                # 自扩展过程示意图
```

---

## 2. 核心创新：SEMA 方法

### 2.1 问题设定

SEMA 解决的是**类别增量学习（Class-Incremental Learning, Class-IL）**问题，使用冻结的预训练 Vision Transformer（ViT-B/16）作为骨干网络。核心挑战是在**不存储旧任务样本**的前提下，平衡**稳定性**（保留旧知识）和**可塑性**（学习新任务）。

### 2.2 核心动机

现有基于预训练模型（PTM）的持续学习方法存在两难困境：
1. 对所有任务使用**固定数量**的适配器/提示词 → 可塑性受限，难以持续适应新任务
2. 为每个任务**定期增加**新模块 → 模型线性增长，知识复用能力差

**SEMA** 提出**自适应、按需自扩展**策略：通过检测不同表示层次上的分布偏移，自动决定是复用现有适配器还是新增适配器模块。

### 2.3 三大核心组件

#### 组件一：模块化适配器（AdapterModule）

每个 `AdapterModule`（定义于 `backbone/sema_block.py:11-53`）包含两个子模块：

1. **功能适配器（Functional Adapter）**（`backbone/sema_components.py:6-53`）：一个瓶颈 MLP（降维 → ReLU → 升维），插入到 ViT 块的 FFN 中。默认配置：`d_model=768`，`bottleneck=16`。负责实际的**特征适配**功能。

2. **表示描述符（Representation Descriptor, RD）**（`backbone/sema_components.py:56-89`）：一个轻量自编码器，将特征编码到低维空间（`rd_dim=128`）再重建。**重建损失（reconstruction loss）** 作为**分布偏移检测器**——当新任务数据相对于已有数据是分布外（out-of-distribution）时，RD 会产生高重建误差。

#### 组件二：自扩展机制（Self-Expansion）

定义于 `SEMAModules.forward()`（`backbone/sema_block.py:108-149`）：

扩展决策基于 RD 历史重建损失的 **z-score 判据**：

```
z_score = |rd_loss - mean| / stddev
```

其中 `mean` 和 `stddev` 由 `Records` 缓冲区（`backbone/sema_components.py:93-130`）维护，最多存储 `buffer_size=500` 个历史重建损失值。

**扩展触发条件**（必须全部满足）：

```python
扩展条件 = (
    z_scores.mean(dim=1).min() > config.exp_threshold  # (1) 最小 z-score 超过阈值
    and layer_id >= adapt_start_layer                    # (2) 当前层在扩展范围内
    and layer_id <= adapt_end_layer                     # (3) 当前层在扩展范围内
    and not added_for_task                              # (4) 当前任务尚未添加适配器
    and detecting_outlier                               # (5) 处于检测模式
)
```

关键设计决策：
- 仅在**深层**检测扩展（`adapt_start_layer=9`, `adapt_end_layer=11`）：浅层（0-8）捕获通用视觉特征，深层（9-11）捕获任务特定语义
- `exp_threshold`：困难数据集（ImageNet-A, VTAB）设为 1.0，标准基准（CIFAR-100, ImageNet-R）设为 2.0
- 一旦触发扩展添加新适配器，**前向传播提前终止**（`vit_sema.py:229-230` 中 `break`），因为新适配器需要先训练才能产生有效输出

#### 组件三：可扩展加权路由器（Expandable Weighting Router）

每个 `SEMAModules` 块有一个**路由器**（初始化时 `nn.Linear(768, 1)`），通过 softmax 加权学习组合多个适配器的输出：

```python
logits = self.router(x.mean(dim=1))           # [B, num_adapters]
mask = torch.softmax(logits, dim=1)            # 混合权重
func_out = (func_outs * mask.transpose(0,1).unsqueeze(-1).unsqueeze(-1)).sum(dim=0)  # 加权求和
```

路由器**扩展机制**：
1. 添加新适配器时，创建一个临时 `new_router`（仅映射到新适配器）
2. 在 `end_of_task_training()` 中，`fix_router()` 通过权重拼接合并新旧路由器（`backbone/sema_block.py:83-95`）

---

## 3. 训练流程

### 3.1 程序入口（`main.py`）

```
python3 main.py --config exps/sema_cifar.json
```

- 加载 JSON 配置 → 与命令行参数合并 → 调用 `train(args)` 或 `eval(args)`

### 3.2 训练编排（`trainer.py`）

```
for seed in seed_list:
    _train(args)                     # 内层训练循环
        ├── DataManager(args)        # 建立 CL 数据划分
        ├── factory.get_model("sema", args)  # 构建 SEMAVitNet + Learner
        └── for task in range(nb_tasks):
                ├── model.incremental_train(data_manager)
                ├── model.eval_task()
                └── model.after_task()
```

### 3.3 SEMA 学习器训练（`models/sema.py`）

#### 任务 0（首个任务 / 基会话）

```
_train_new():
    阶段 "func"（func_epoch 轮，lr=init_lr）:
        训练：功能适配器 + 路由器 + fc
        损失：CrossEntropy 分类损失
    
    阶段 "rd"（rd_epoch 轮，lr=rd_lr）:
        训练：仅表示描述符（RD）
        损失：自编码器重建损失（rd_loss）
```

#### 任务 t > 0（增量会话）

```
1. 检测阶段（detecting_outlier=True）:
   For each batch in detect_loader:
       前向传播
       检查 added_record（每层是否触发了适配器扩展的布尔标志）
       如果任意层触发扩展:
           → 为该层添加新适配器
           → 调用 _train_new() 训练新适配器
           → 冻结旧功能适配器、RD、路由器
           → 递归检测是否还需要更多扩展
       如果无扩展:
           → 直接跳到步骤 3

2. _init_train phase="func"（如果无扩展）:
   仅微调功能适配器 + 路由器 + fc

3. end_of_task_training():
   冻结所有功能适配器参数
   冻结所有 RD（停止统计更新）
   重置 newly_added 标志
   重置 added_for_task 标志
```

### 3.4 关键配置参数

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `func_epoch` | 5-20 | 功能适配器训练轮数 |
| `rd_epoch` | 20 | RD（自编码器）训练轮数 |
| `init_lr` | 0.005 | 功能适配器学习率 |
| `rd_lr` | 0.01 | RD 训练学习率 |
| `rd_dim` | 128 | RD 自编码器瓶颈维度 |
| `buffer_size` | 500 | RD 损失记录缓冲区大小（用于 z-score 计算） |
| `exp_threshold` | 1-2 | 触发扩展的 z-score 阈值 |
| `adapt_start_layer` | 9 | 允许扩展的起始 ViT 层 |
| `adapt_end_layer` | 11 | 允许扩展的结束 ViT 层 |
| `ffn_num` | 16 | 功能适配器瓶颈维度 |
| `detect_batch_size` | 128 | 异常检测的批大小 |
| `ffn_option` | "parallel" | 适配器在 FFN 中的位置（parallel / sequential） |

---

## 4. 组件架构详解

### 4.1 支持 SEMA 的 ViT 骨干网络（`backbone/vit_sema.py`）

```
VisionTransformer
├── patch_embed           # 图像分块嵌入（冻结）
├── cls_token / pos_embed # 位置嵌入（冻结）
├── blocks[0..11]         # 12 个 Transformer 块
│   └── Block
│       ├── norm1 + Attention  （冻结的 QKV）
│       ├── adapter_module: SEMAModules  ← SEMA 插入在此
│       └── norm2 + MLP（冻结）+ adapter（并行/顺序）
└── norm + head           # 最终 LayerNorm + 分类器（fc 可学习）
```

**初始化**：从 `timm` 加载预训练 ViT-B/16。将原始 `qkv` 权重拆分为独立的 `q_proj`、`k_proj`、`v_proj`。冻结所有预训练权重——仅适配器参数和 `fc` 可训练。

**前向传播**（`backbone/vit_sema.py:206-240`）：
```python
def forward_features(x):
    x = patch_embed(x) + pos_embed
    for each block:
        x = block(x)                # Attention + SEMA adapter + FFN
        if block 返回 added==True:   # 扩展时提前终止
            break
    return {features, rd_loss, added_record}
```

### 4.2 SEMA 块（`backbone/sema_block.py`）

#### AdapterModule（`backbone/sema_block.py:11-53`）

```python
class AdapterModule:
    functional: Adapter         # 瓶颈 MLP（768→16→768）
    rd: AE                      # 用于重建损失的自编码器
    rd_loss_record: Records     # 滚动统计缓冲区（最多 500 条记录）
    newly_added: bool           # 该适配器是否刚被添加

    def forward(x):
        func_out = functional(x)                 # 适配后的特征
        rd_loss = ae.compute_reconstruction_loss(x)  # 分布偏移信号
        z_score = get_z_score_deviation(rd_loss)      # 归一化偏差
        return func_out, rd_loss, z_score
```

#### SEMAModules（`backbone/sema_block.py:56-183`）

某一 ViT 层上所有适配器的容器：

```python
class SEMAModules:
    adapters: List[AdapterModule]       # 该层的所有适配器
    router: nn.Linear(768, num_adapters) # 适配器混合路由器
    new_router: Optional[nn.Linear(768, 1)]  # 新适配器的临时路由器
    detecting_outlier: bool             # 是否处于检测模式
    added_for_task: bool                # 当前任务是否已添加了适配器

    def forward(x):
        for each adapter:  # 遍历所有适配器
            func_out, rd_loss, z_score = adapter(x)
        if 满足扩展条件:
            add_adapter()               # 自扩展
            return zeros（占位符）
        else:
            router_weights = softmax(router(x.mean(dim=1)))
            func_out = weighted_sum(所有适配器输出, router_weights)
            return func_out, rd_loss, added=False
```

### 4.3 表示描述符组件（`backbone/sema_components.py`）

#### 功能适配器 Adapter（`backbone/sema_components.py:6-53`）

标准瓶颈适配器（受 AdaptFormer 启发）：
```
输入 (768维) → LayerNorm → Linear(768, 16) → ReLU → Linear(16, 768) → 输出
```
初始化方式：降维层用 Kaiming 均匀初始化，升维层用零初始化（LoRA 风格初始化）。

#### 表示描述符自编码器 AE（`backbone/sema_components.py:56-89`）

```
输入 (768维) → Linear(768, 128) → Linear(128, 768) → 重建
```
对均值池化后的 token 表示逐样本计算 MSE 重建损失。

#### 在线统计缓冲区 Records（`backbone/sema_components.py:93-130`）

- 维护固定大小的缓冲区（`max_len=500`）存储历史 RD 损失值
- 随新值到来**增量更新** `mean` 和 `variance`
- 用于计算异常检测的 z-score
- **每个任务结束后冻结**，防止统计量漂移

### 4.4 网络包装器（`utils/inc_net.py:550-563`）

```python
class SEMAVitNet(BaseNet):
    def forward(x):
        out = backbone(x)         # VisionTransformer.forward_features()
        features = out["features"]
        logits = fc(features)
        return {features, logits, rd_loss, added_record}
```

---

## 5. 训练与评估流程详解

### 5.1 首个任务训练（`_train_new`）

```
阶段 "func"（功能适配器训练）:
  - 优化器: SGD，lr=0.005，momentum=0.9，weight_decay=0.0005
  - 调度器: CosineAnnealingLR，降至 min_lr=0
  - 可训练参数: functional.*, router.*, fc.*  （RD 参数保持冻结）
  - 损失: CrossEntropy(logits, targets)
  - 当前任务类之外的 logits 被置为 -inf

阶段 "rd"（表示描述符训练）:
  - 优化器: SGD，lr=0.01，momentum=0.9，weight_decay=0.0005
  - 可训练参数: *.rd.*  （功能适配器和路由器保持冻结）
  - 损失: 自编码器重建损失（rd_loss）
  - 目的：使 AE 学会重建首个任务数据的特征
```

### 5.2 增量任务检测

```python
_detect_outlier(detect_loader, train_loader, test_loader, added=0):
    for batch in detect_loader:
        前向传播
        if 任意层的 added_record 为 True:
            → 该层触发了扩展
            → 训练新适配器: _train_new(train_loader, test_loader)
            → 冻结旧适配器和 RD
            → 重置 newly_added 标志
            → 递归检查是否还需要更多扩展
    return 总扩展次数
```

如果无扩展（`added == 0`），仅对功能适配器微调 `func_epoch` 轮。

### 5.3 评估

评估采用两种指标：
1. **CNN 准确率**：使用 `fc` 头的标准分类预测
2. **NME 准确率**：最近类均值分类器（仅适用于存储样本的方法）

SEMA **不存储样本**（`memory_size=0`），因此 NME 通常为 `None`。

---

## 6. 支持的基线方法

代码库实现了多种 PTM 类持续学习方法用于对比：

| 方法 | 模型文件 | 核心思路 |
|--------|-----------|----------|
| **SEMA** | `models/sema.py` | 基于 RD 检测的自扩展适配器混合 |
| **Finetune** | `models/finetune.py` | 普通微调（性能下界） |
| **SimpleCIL** | `models/simplecil.py` | 简单类别增量，无遗忘防护 |
| **APER-Adapter** | `models/aper_adapter.py` | 固定适配器 + 双分支架构 |
| **APER-SSF** | `models/aper_ssf.py` | Scale-Shift-Feature 适配 |
| **APER-VPT** | `models/aper_vpt.py` | APER 框架 + VPT |
| **L2P** | `models/l2p.py` | Learning to Prompt（提示词池选择） |
| **DualPrompt** | `models/dualprompt.py` | 通用 + 专家提示词 |
| **CODA-Prompt** | `models/coda_prompt.py` | 分解提示词学习 + 注意力 |
| **iCaRL** | `models/icarl.py` | 基于样本存储 + 知识蒸馏 |
| **DER** | `models/der.py` | 每任务新增骨干的动态扩展 |
| **FOSTER** | `models/foster.py` | 特征压缩 + 知识蒸馏 |
| **MEMO** | `models/memo.py` | 内存高效的扩展方法 |
| **CoIL** | `models/coil.py` | 坐标驱动的增量学习 |

---

## 7. 数据集与配置

| 数据集 | 配置文件 | 类别数 | 任务划分 | exp_threshold |
|---------|------------|---------|------------|---------------|
| CIFAR-100 | `sema_cifar.json` | 100 | 10 任务 × 10 类 | 2 |
| ImageNet-R | `sema_inr_5task.json` | 200 | 5 任务 × 40 类 | 2 |
| ImageNet-R | `sema_inr_10task.json` | 200 | 10 任务 × 20 类 | 2 |
| ImageNet-R | `sema_inr_20tasks.json` | 200 | 20 任务 × 10 类 | 2 |
| ImageNet-A | `sema_ina.json` | 200 | 10 任务 × 10 类 | 1 |
| VTAB | `sema_vtab.json` | 50 | 5 任务 × 10 类 | 1 |

所有数据集均使用 ImageNet-21K 预训练的 ViT-B/16 作为骨干网络。

---

## 8. 关键设计决策与原理

1. **为什么使用并行适配器？**（`ffn_option="parallel"`）：适配器输出与 FFN 输出相加而非顺序插入，在保留原始特征流的同时允许适配。

2. **为什么只在深层扩展？**（`adapt_start_layer=9, adapt_end_layer=11`）：第 0-8 层捕获跨任务共享的通用视觉特征；第 9-11 层捕获任务特定的语义信息。仅在深层扩展可防止不必要的增长并促进知识复用。

3. **为什么扩展时提前终止前向传播？**：当添加新适配器时，它尚未被训练，不能产生有意义的输出。需要先训练新适配器再进行后续前向传播。

4. **为什么任务结束后冻结 RD？**：RD 的统计缓冲区（`rd_loss_record`）必须在任务结束后停止更新，因为其统计量应仅反映**过往任务**的特征分布，才能在新任务到来时进行准确的异常检测。

5. **为什么能实现亚线性扩展？**：SEMA 仅在分布偏移超过现有适配器处理能力时才添加新适配器（通过 RD 检测）。方法实现**亚线性**模型增长，而非每个任务都添加一个适配器的线性增长。

6. **为什么不需要存储样本？**：SEMA 在**不存储**任何旧任务样本（`memory_size=0`）的条件下达到最先进性能，适用于隐私敏感的应用场景。

---

## 9. 损失函数

### 9.1 功能训练损失
```
L_func = CrossEntropy(logits[:, :total_classes], targets)
```
对于增量任务，旧类的 logits 被掩码为 `-inf`，使训练专注于新类。

### 9.2 表示描述符训练损失
```
L_rd = (1/B) * Σ MSE(reconstructed_i, original_i)
```
对均值池化 token 后的特征逐样本计算自编码器重建损失。

---

## 10. 检查点保存与加载

- **保存**（`models/sema.py:227-233`）：仅保存 `adapter` 和 `fc` 参数（不保存冻结的骨干网络）
- **加载**（`models/sema.py:235-236`）：使用 `strict=False` 允许加载到已扩展的网络中
- **评估加载**（`eval.py:61-65`）：先读取检查点确定每层有多少个适配器，重建网络结构后再加载权重
- **适配器模式检测**（`eval.py:71-82`）：通过正则表达式 `r'backbone\.blocks\.(\d+)\.adapter_module\.adapters\.(\d+)\.'` 解析检查点的 key 名来确定各层的适配器数量

---

## 11. 依赖与环境

主要依赖（来自 `sema_env.yaml`）：
- PyTorch（CUDA 版本）
- `timm`（PyTorch Image Models）—— 提供预训练 ViT 模型
- `tqdm` —— 进度条
- `numpy`, `scipy` —— 数值计算
- `easydict` —— 点号访问字典
- `torchvision` —— 数据集和数据增强

---

## 12. 完整代码调用流程

```
main.py
  └─> trainer.train(args)
        ├─> DataManager: 建立 CL 数据划分
        ├─> factory.get_model("sema", args):
        │     ├─> SEMAVitNet(args, True)
        │     │     └─> inc_net.get_backbone(args):
        │     │           └─> vit_sema.vit_base_patch16_224_sema()
        │     │                 └─> VisionTransformer（12 个 Block，
        │     │                       每 Block 含 1 个 SEMAModules，
        │     │                       每个 SEMAModules 含 1 个 AdapterModule）
        │     └─> Learner(args)  # sema.py 的 Learner
        └─> for task in range(nb_tasks):
              └─> Learner.incremental_train(data_manager)
                    ├─> 任务0: _train_new()
                    │     ├─> Phase func: 训练 functional + router + fc
                    │     └─> Phase rd: 训练表示描述符（RD）
                    └─> 任务 t>0:
                          ├─> _detect_outlier(): 检查各层 RD z-score
                          │     ├─> 若触发扩展: 添加适配器，_train_new，递归检测
                          │     └─> 若无扩展: 仅微调功能适配器
                          └─> end_of_task_training(): 全部冻结，重置标志
```
