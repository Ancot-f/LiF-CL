# Lie-Group-FiberCL 持续学习方法详解

## 目录

1. [SEMA — 自扩展适配器混合 (主方法)](#1-sema--自扩展适配器混合)
2. [Finetune — 简单微调基线](#2-finetune--简单微调基线)
3. [SimpleCIL — 简单类增量学习](#3-simplecil--简单类增量学习)
4. [L2P — 学习提示词](#4-l2p--learning-to-prompt)
5. [DualPrompt — 双重提示词](#5-dualprompt--双重提示词)
6. [CODA-Prompt — 分解注意力提示词](#6-coda-prompt--分解注意力提示词)
7. [MEMO — 内存高效模型](#7-memo--内存高效模型)
8. [iCaRL — 增量分类器与表征学习](#8-icarl--增量分类器与表征学习)
9. [DER — 动态扩展与表征](#9-der--动态扩展与表征)
10. [CoIL — 持续不变学习](#10-coil--持续不变学习)
11. [FOSTER — 特征增强与压缩](#11-foster--特征增强与压缩)
12. [APER 系列方法](#12-aper-系列方法)

---

## 前置知识

### 持续学习问题定义

持续学习 (Continual Learning, CL) 研究如何让模型**顺序学习多个任务**而不遗忘旧知识。

- **类增量学习 (Class-Incremental Learning, CIL)**：每个任务引入一批新的类别，模型需要在所有已见过的类别上进行分类
- **核心矛盾**：可塑性 (plasticity, 学习新知识) vs 稳定性 (stability, 保留旧知识)
- **灾难性遗忘 (Catastrophic Forgetting)**：当模型在新任务上训练时，在旧任务上的性能急剧下降

### 预训练模型 + 参数高效微调范式

本项目所有方法共享一个核心范式：

- 使用**冻结的预训练 ViT** (Vision Transformer) 作为特征提取器
- 通过轻量的**可学习模块** (Adapter / Prompt / SSF 等) 适应新任务
- 不同方法的核心区别在于：**如何管理这些轻量模块以平衡新旧知识**

### 项目架构

```
Lie-Group-FiberCL/
  main.py                  ← 训练入口, 循环执行所有任务
  models/
    base.py                ← BaseLearner: 所有方法的共享基类
    sema.py                ← SEMA Learner
    finetune.py / l2p.py / ... ← 各方法 Learner
  backbones/
    sema_vit.py            ← SEMA 专用 ViT (带 SEMAModules)
    vit_adapter.py         ← 普通 Adapter ViT
    vit_l2p.py             ← L2P ViT
    vit_ssf.py             ← SSF ViT
    sema_block.py          ← SEMAModules + AdapterModule (SEMA 核心)
    sema_components.py     ← Adapter, AE, Records (SEMA 基础组件)
  utils/
    inc_net.py             ← 网络工厂 + 所有网络包装器
    factory.py             ← 模型工厂, 根据 model_name 创建 Learner
    data_manager.py        ← 数据集管理 (任务划分)
```

---

## 1. SEMA — 自扩展适配器混合

> **论文**: Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning (CVPR 2025)
>
> **核心创新**: 模型根据数据分布偏移**自动决定**何时添加新适配器, 通过**软路由**组合多个适配器的输出, 实现**子线性参数增长**和**零遗忘**。

### 1.1 方法含义

SEMA 的核心思想可以用三句话概括：

1. **每个 ViT Block 中有一组 Adapter**：初始只有 1 个, 遇到新数据分布时自动增长
2. **自扩展检测**：用轻量自编码器 (AE) 检测数据分布是否偏离历史, 偏离则触发扩展
3. **软路由组合**：所有 Adapter 的输出通过一个可学习的路由器 (Router) 做 softmax 加权组合, 而非硬切换

#### 为什么叫 "自扩展"

传统方法要么固定 Adapter 数量 (限制可塑性), 要么每个任务加一个 Adapter (线性增长, 浪费参数)。SEMA 让模型**自行判断**是否需要新 Adapter：相似的任务共享 Adapter, 差异大的任务触发扩展。这实现了**按需增长**, 参数效率最高。

#### 为什么叫 "适配器混合"

SEMA 不使用 `if task_id == 0 then use adapter_0` 这种硬选择（容易选错）。而是对**所有** Adapter 输出做 softmax 加权：

```
output = Σ softmax(router(x_mean))_i × Adapter_i(x)
```

这允许跨任务的知识复用：新任务可以部分激活旧 Adapter, 同时主要依赖新 Adapter。

### 1.2 核心组件

#### Adapter (功能适配器)

```
结构: x → Linear(768, 16) → ReLU → Linear(16, 768) → 输出
参数量: 2 × 768 × 16 ≈ 24,576 (仅占 ViT 86M 的 0.03%)
初始化: down_proj 用 Kaiming Uniform, up_proj 用全零 (保证初始输出为零)
```

#### AE (表征描述器, Representation Descriptor)

```
结构: x → Linear(768, 128) → Linear(128, 768) → 重建 x'
损失: MSE(x, x')
作用: 学习当前任务的低秩特征分布, 重建误差用于分布偏移检测
```

线性 AE 等价于对特征矩阵做秩-128 的 SVD 近似。训练后, AE 编码器捕获了当前任务的 128 维特征主成分, 解码器尝试从这些主成分重建原始特征。

#### Records (运行统计缓冲区)

```
容量: 500 (固定大小滑动窗口)
维护: 重建误差的在线均值 μ 和标准差 σ
输出: Z-score = |重建误差 - μ| / σ
```

#### Router (软路由器)

```
结构: Linear(768, M)  (M = 当前 Adapter 数量)
输出: softmax 权重, 用于组合 M 个 Adapter 的输出
扩展: 每次添加新 Adapter 时, Router 从 M 列扩展到 M+1 列
```

### 1.3 训练流程

#### 完整训练循环 (main.py)

```
main()
  → _train_single(args)
      → DataManager(args)                           # 加载数据, 划分任务
      → model = factory.get_model("sema", args)      # 创建 SEMA Learner
      → WandbLogger(...)                             # 初始化 wandb
      → for task in range(nb_tasks):                 # 逐任务训练
          → model.incremental_train(data_manager)     # 训练当前任务
          → model.eval_task()                         # 评估所有已见类
          → model.after_task()                        # 更新 known_classes
          → wandb_logger.log_metrics(...)             # 上报指标
          → ckpt.save_task_state(...)                 # 保存检查点
```

#### 单任务训练 (models/sema.py: Learner)

`Learner` 继承 `BaseLearner`, 实现 SEMA 的增量训练逻辑。

**任务 0 (第一个任务)**:

```
incremental_train()
  → 初始化 fc 分类头 (768 → nb_classes)
  → _train()
      → _train_new(): 新增 Adapter 的两阶段训练
          → func 阶段 (5 epochs):
              → update_optimizer_and_scheduler(): 优化 functional + router + fc
              → _init_train(phase="func"):
                  对每个 batch:
                    outcome = self._network(inputs)
                    loss = CrossEntropy(logits[:, :total_classes], targets)
                    旧类 logits 设为 -inf (只让模型在新类上预测)
                    loss.backward() → optimizer.step()
          → rd 阶段 (20 epochs):
              → update_rd_optimizer_and_scheduler(): 优化 rd (AE)
              → _init_train(phase="rd"):
                  对每个 batch:
                    outcome = self._network(inputs)
                    loss = outcome["rd_loss"]  # 最新 Adapter 的 AE 重建误差
                    loss.backward() → optimizer.step()
      → SEMAModules.end_of_task_training(): 冻结所有功能模块和 AE
```

**任务 > 0 (后续任务)**:

```
incremental_train()
  → _train()
      → 设置所有 SEMAModules.detecting_outlier = True (进入检测模式)
      → _detect_outlier(): 自扩展检测循环
          → 对 detect_loader 中的每个 batch:
              outcome = self._network(inputs)
              added_record = outcome["added_record"]  # 哪些层触发了扩展
              if any(added_record):                   # 触发了扩展
                  → 关闭检测模式
                  → _train_new(): 训练新添加的 Adapter (func + rd)
                  → freeze_functional() + freeze_rd() + reset()
                  → 重新开启检测模式
                  → 递归调用 _detect_outlier() 继续检测
          → 返回 added (总共添加的 Adapter 数量)
      → 关闭所有检测模式
      → if added == 0:  # 没触发扩展, 只需要微调 Router
          → _init_train(phase="func"): 只训练 Router + fc
      → SEMAModules.end_of_task_training()
```

#### 自扩展检测机制 (backbones/sema_block.py: SEMAModules)

`SEMAModules` 是每层 ViT Block 中的适配器管理器, 负责扩展检测和路由。

```python
# SEMAModules.forward() 的核心逻辑
for adapter in self.adapters:
    func_out, rd_loss, z_score = adapter(x)   # 所有 Adapter 前向

# 扩展检测条件 (三个条件同时满足)
addition_criteria = (
    z_scores.mean(dim=1).min() > exp_threshold   # 条件1: 所有旧 AE 的 Z-score 都超阈值
    and layer_id in [adapt_start, adapt_end]      # 条件2: 在可扩展层范围内
    and not self.added_for_task                    # 条件3: 本任务未扩展过
    and self.detecting_outlier                     # 条件4: 处于检测模式
)

if addition_criteria:
    self.add_adapter()    # 触发扩展!
    return {"added": True, "func_out": zeros, ...}
else:
    # 软路由加权组合
    logits = router(x.mean(dim=1))        # [B, M]
    if new_router exists: logits = cat([logits, new_router(x)])
    mask = softmax(logits)                 # [B, M(+1)]
    func_out = Σ mask_i * func_out_i       # 加权求和
```

`AdapterModule` 是单个适配器单元, 计算 Z-score:

```python
# AdapterModule.forward()
func_out = functional(x)                           # 功能输出
rd_loss = AE.compute_reconstruction_loss(x)        # 逐样本重建误差
z_score = abs((rd_loss - running_mean) / running_std)  # Z-score

# AdapterModule.get_z_score_deviation()
mean, stddev = rd_loss_record.mean, rd_loss_record.stddev
if length < 2: return zeros
z_score = (rd_loss - mean) / stddev
return abs(z_score)
```

### 1.4 数学原理

#### 1.4.1 分布偏移检测的统计基础

设第 $k$ 个 Adapter 的表征描述器在历史数据上的重建误差服从分布 $\mathcal{N}(\mu_k, \sigma_k^2)$。

对新输入 $x$, 计算其重建误差 $e_k(x) = \|x - \hat{x}\|^2$ 和 Z-score:

$$
Z_k(x) = \left|\frac{e_k(x) - \mu_k}{\sigma_k}\right|
$$

**扩展决策**：如果**所有**现有 AE 的 Z-score 都超过阈值 $\tau$ (默认 2.0):

$$
\min_{k} \mathbb{E}_x[Z_k(x)] > \tau \quad \Rightarrow \text{触发扩展}
$$

这等价于假设检验：在 $\mathcal{N}(0,1)$ 下, $P(Z > 2) \approx 0.045$, 即只有 4.5% 的正常样本会被误判为异常。当**所有** AE 都认为输入异常时, 新数据分布的置信度极高。

#### 1.4.2 线性自编码器与 PCA 的等价性

线性 AE (无激活函数) 的目标是最小化重建误差:

$$
\min_{W_e, W_d} \|X - W_d W_e X\|_F^2
$$

其中 $W_e \in \mathbb{R}^{d \times r}$, $W_d \in \mathbb{R}^{r \times d}$, $r = 128$。

**定理** [Baldi & Hornik, 1989]: 上述优化问题的全局最优解对应的 $W_e$ 的列张成与 $X$ 的前 $r$ 个主成分 (PCA) 相同的子空间。即线性 AE 等价于对特征矩阵的秩-$r$ SVD 近似。

因此, AE 学到的低维表示 $z = W_e x$ 捕获了当前任务特征分布的主要变化方向。新任务的特征如果与这些方向差异大, 重建误差自然高。

#### 1.4.3 软路由的 MoE 解释

SEMA 的适配器混合可以视为一种简化的 Mixture of Experts (MoE):

$$
\text{AdapterMix}(x) = \sum_{i=1}^{M} \alpha_i(x) \cdot A_i(x)
$$

其中 $\alpha_i(x) = \text{softmax}(W_r \cdot \text{mean}(x))_i$ 是路由权重, $A_i$ 是第 $i$ 个 Adapter。

与标准 MoE 的区别：
- SEMA 使用**所有** Adapter 的加权组合 (dense MoE), 而非 top-k 选择 (sparse MoE)
- 路由器输入是 token 均值 (全局特征), 而非单个 token, 减少了计算量
- 旧 Adapter 被冻结, 只有新 Adapter + 新路由列被训练

#### 1.4.4 子线性参数增长的保证

假设共有 $T$ 个任务, 每层的可扩展范围为 $L$ 层 (默认 9-11, 共 3 层)。

- **最坏情况** (每任务都扩展): 参数增长 $O(T \cdot L)$, 线性
- **最好情况** (所有任务共享): 参数增长 $O(1)$, 常数
- **实际情况**: 相似任务共享 Adapter, 只有分布差异大的任务触发扩展, 增长是**子线性**的

SEMA 论文实验表明, 在 ImageNet-R 的 20 个任务中, 平均每层扩展约 4-5 次, 远少于 20 次。

### 1.5 关键超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `exp_threshold` | 2.0 | Z-score 阈值, 越高越难触发扩展 |
| `adapt_start_layer` | 9 | 开始检测扩展的层 (浅层不检测) |
| `adapt_end_layer` | 11 | 结束检测扩展的层 (共 12 层) |
| `ffn_num` | 16 | Adapter 瓶颈维度 |
| `rd_dim` | 128 | AE 压缩维度 |
| `buffer_size` | 500 | 重建误差缓冲区大小 |
| `func_epoch` | 5 | 功能阶段训练轮数 |
| `rd_epoch` | 20 | 表征描述器训练轮数 |
| `init_lr` | 0.005 | 功能阶段学习率 |
| `rd_lr` | 0.01 | RD 阶段学习率 |

---

## 2. Finetune — 简单微调基线

### 2.1 方法含义

最朴素的持续学习基线：对每个新任务直接微调整个模型的 Adapter 参数, 没有任何防遗忘机制。

### 2.2 训练流程

- `models/finetune.py:Learner`
  - `incremental_train()`: 初始化扩展的余弦分类器, 用 CrossEntropy 训练
  - `_train()`: 标准训练循环, 可选知识蒸馏
  - 使用余弦相似度分类器 (CosineLinear)

### 2.3 特点

- **优点**: 实现最简单, 新任务学习能力强 (可塑性好)
- **缺点**: 灾难性遗忘严重, 旧任务性能急剧下降

---

## 3. SimpleCIL — 简单类增量学习

### 3.1 方法含义

最简单的类增量学习基线：用预训练 ViT 提取特征, 用余弦相似度做最近类中心分类。

### 3.2 训练流程

- `models/simplecil.py:Learner`
  - 使用**冻结的**预训练 ViT 提取特征 (不训练任何参数)
  - 对新类别构建类中心 (class prototype), 分类用余弦相似度 NME (Nearest Mean of Exemplars)
  - 本质上是 zero-shot 分类, 质量完全取决于预训练 ViT 的特征判别能力

### 3.3 数学原理

NME 分类：对于 $K$ 个已见类别, 每个类维护一个特征均值向量 $\mu_k$:

$$
\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} \frac{f(x)}{\|f(x)\|}
$$

测试样本 $x$ 的分类: $\hat{y} = \arg\min_k \|\frac{f(x)}{\|f(x)\|} - \mu_k\|^2$

---

## 4. L2P — Learning to Prompt

> **论文**: Learning to Prompt for Continual Learning (CVPR 2022)

### 4.1 方法含义

L2P 维护一个**Prompt 池** (可学习的 token 集合), 训练时从池中选出最相关的 prompt 插入到 ViT 输入序列中。

### 4.2 核心组件

- `Prompt (backbones/prompt.py)`: Prompt 池 + key-query 相似度选择
- `VisionTransformer (backbones/vit_l2p.py)`: 带 prompt 插入的 ViT
- `Learner (models/l2p.py)`: L2P 训练逻辑

### 4.3 训练流程

```
incremental_train()
  → 初始化或扩展 PromptVitNet (backbone + prompt pool)
  → 冻结旧分类器头, 添加新分类器头 (SimpleContinualLinear)
  → _train():
      对每个 batch:
        x = backbone(x, task_id, cls_features, train=True)
        # backbone 内部: 从 prompt pool 选 top-k prompt → 插入 → forward
        loss = CrossEntropy(logits, targets) + pull_constraint * reduce_sim
        loss.backward() → optimizer.step()
  → after_task(): 更新已知类数
```

### 4.4 数学原理

**Prompt 选择机制**: 给定 Prompt 池 $P = \{p_1, ..., p_N\}$ 和对应的 key $K = \{k_1, ..., k_N\}$, 对输入 $x$:

1. 计算查询向量: $q = f(x) \in \mathbb{R}^d$ (CLS token 或特征均值)
2. 余弦相似度排序: $\text{sim}(q, k_i) = \frac{q^T k_i}{\|q\|\|k_i\|}$
3. 选 top-K: $\text{idx} = \arg\text{topK}(\text{sim})$
4. 构造 prompt: 拼接选中的 prompt token 到输入序列

---

## 5. DualPrompt — 双重提示词

> **论文**: DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning (ECCV 2022)

### 5.1 方法含义

DualPrompt 在 L2P 基础上引入**两种** Prompt：

- **G-Prompt (通用提示, General Prompt)**: 插入到浅层 (layer 1-2), 捕获任务无关的通用知识
- **E-Prompt (专家提示, Expert Prompt)**: 插入到深层 (layer 3-5), 类似 L2P 的 pool selection, 捕获任务特定知识

### 5.2 训练流程

- `models/dualprompt.py:Learner`
- `backbones/vit_dualprompt.py`: 支持 G-Prompt + E-Prompt 双机制的 ViT
  - G-Prompt: 在指定层的 Attention 前拼接固定的 prompt token
  - E-Prompt: 在指定层使用 prompt pool selection (类似 L2P)
  - 支持 Prompt Tuning 和 Prefix Tuning 两种模式

---

## 6. CODA-Prompt — 分解注意力提示词

> **论文**: CODA-Prompt: COntinual Decomposed Attention-based Prompting (CVPR 2023)

### 6.1 方法含义

CODA-Prompt 将 prompt 分解为**可组合的组件**, 通过注意力机制动态组合这些组件生成最终的 prompt。

### 6.2 核心特点

- `CodaPrompt (backbones/prompt.py)`: Gram-Schmidt 正交化 + 注意力加权组合
- 每个任务分配 pool 中的一部分 prompt 组件
- 训练时施加**正交性惩罚**, 确保不同任务的 prompt 组件正交
- 旧任务的 prompt 组件被冻结

### 6.3 训练流程

- `models/coda_prompt.py:Learner`
  - 每个任务调用 `prompt.process_task_count()` 分配新 prompt 组件
  - 训练: CrossEntropy + ortho_penalty (正交性损失)
  - 冻结旧 prompt, 冻结旧分类器参数

---

## 7. MEMO — 内存高效模型

> **论文**: MEMO: A Model-based Memory-efficient Approach for Continual Learning (NeurIPS 2022)

### 7.1 方法含义

MEMO 将 ViT 拆分为**共享部分** (前 N-1 层, Generalized) 和**任务特定部分** (最后一层, Specialized)。

### 7.2 核心架构

```
Generalized_Vit (前 11 层, 冻结共享)
      ↓ 输出中间特征
Specialized_Vit (最后 1 层 + 分类头, 每个任务独立)
      ↓
  logits
```

### 7.3 训练流程

- `models/memo.py:Learner`
  - 使用 AdaptiveNet 网络
  - `_train()`: 同时训练 TaskAgnosticExtractor 和 AdaptiveExtractor
  - 每个任务添加新的 Specialized_Vit, 旧模块被冻结

---

## 8. iCaRL — 增量分类器与表征学习

> **论文**: iCaRL: Incremental Classifier and Representation Learning (CVPR 2017)

### 8.1 方法含义

经典的持续学习方法, 结合**知识蒸馏** (KD) 和**示例回放** (Rehearsal)。

### 8.2 训练流程

- `models/icarl.py:Learner`
  - `incremental_train()`: 初始化余弦分类器 (SplitCosineLinear)
  - `_train()`:
      - 用旧模型做知识蒸馏: `kd_loss = KL(old_logits, new_logits)`
      - 分类损失: `ce_loss = CrossEntropy(logits, targets)`
      - 总损失: `loss = ce_loss + kd_loss`
  - `after_task()`:
      - `build_rehearsal_memory()`: 用 Herding 算法选择代表性样本存储
      - `_reduce_exemplar()`: 当内存满时均匀缩减旧类的样本数

### 8.3 数学原理

**Herding 算法**: 对每个类别, 贪心选择 $m$ 个样本使得它们的特征均值最接近真实的类中心:

$$
\forall k \in \{1, ..., m\}: i_k = \arg\min_i \left\| \mu_c - \frac{1}{k}\left(\sum_{j=1}^{k-1} f(x_{i_j}) + f(x_i)\right) \right\|
$$

其中 $\mu_c$ 是该类的真实特征均值。

---

## 9. DER — 动态扩展与表征

### 9.1 方法含义

DER 为每个任务**动态添加一个新的 backbone**, 所有 backbone 的输出拼接后分类。

### 9.2 训练流程

- `models/der.py:Learner`
  - 每个任务添加一个新的冻结 backbone 拷贝
  - 特征拼接: `features = cat([backbone_0(x), backbone_1(x), ...])`
  - 辅助分类器 aux_fc: 仅用最新 backbone 的特征做辅助分类
  - 总损失: CrossEntropy(main_logits, targets) + CrossEntropy(aux_logits, targets)

---

## 10. CoIL — 持续不变学习

### 10.1 方法含义

CoIL 通过**最优传输 (Optimal Transport, OT)** 来对齐新旧模型的特征分布, 减少遗忘。

### 10.2 训练流程

- `models/coil.py:Learner`
  - 两阶段训练:
    1. **OT 校准阶段**: 用 Sinkhorn 算法计算新旧特征之间的最优传输计划
    2. **特征训练阶段**: 用 OT loss 约束新旧特征分布对齐
  - 损失: CrossEntropy + λ * OT_distance(f_new(x), f_old(x))

### 10.3 数学原理

**最优传输**: 给定两个分布 $\mu$ (旧特征) 和 $\nu$ (新特征), 寻找一个传输计划 $\pi$ 使得总传输代价最小:

$$
\mathcal{W}(\mu, \nu) = \min_{\pi} \sum_{i,j} \pi_{ij} \cdot c(x_i, y_j)
$$

s.t. $\pi \mathbf{1} = \mu$, $\pi^T \mathbf{1} = \nu$, $\pi \geq 0$

CoIL 使用**熵正则化的 Sinkhorn 算法**高效求解, 然后作为蒸馏损失约束新模型。

---

## 11. FOSTER — 特征增强与压缩

### 11.1 方法含义

FOSTER 通过**特征增强** (Feature Boosting) 和**知识蒸馏**来持续学习。

### 11.2 训练流程

- `models/foster.py:Learner`
  - 每个任务添加新的 backbone 并拼接特征
  - **Boosting 阶段**: 训练新 backbone + 特征增强分类器
  - **Compression 阶段**: 用知识蒸馏将增强特征压缩回固定维度
  - 旧分类器 oldfc 用作蒸馏目标

---

## 12. APER 系列方法

> **论文**: Adaptive Parameter-Efficient Regularization for Continual Learning

APER 提供 4 种参数高效微调变体, 作为对比基线：

| 方法 | Backbone | 训练内容 | 文件 |
|------|----------|----------|------|
| **APER-Finetune** | 标准 ViT | 全微调 | `aper_finetune.py` |
| **APER-SSF** | SSF ViT | Scale + Shift 参数 | `aper_ssf.py` |
| **APER-VPT** | VPT ViT | Prompt Token | `aper_vpt.py` |
| **APER-Adapter** | Adapter ViT | Adapter MLP | `aper_adapter.py` |

### 12.1 共同训练流程

- 使用 MultiBranchCosineIncrementalNet: 双分支网络 (原始 ViT + 微调 ViT)
- 分类器: CosineLinear (余弦相似度)
- 损失: CrossEntropy + 可选知识蒸馏
- 所有方法共享相同的训练框架, 仅 backbone 不同

---

## 方法对比总结

| 方法 | 防遗忘机制 | 参数增长 | 是否需要回放 | 计算开销 |
|------|-----------|---------|-------------|---------|
| **SEMA** | 冻结旧 Adapter + 软路由 | 子线性 | 否 | 中等 |
| Finetune | 无 | 固定 | 否 | 低 |
| SimpleCIL | 冻结全部 | 固定 | 否 | 极低 |
| L2P | 冻结旧 Prompt | 线性 (每任务) | 否 | 低 |
| DualPrompt | G+E Prompt 冻结 | 线性 | 否 | 中 |
| CODA-Prompt | 正交化 + 冻结 | 线性 | 否 | 中 |
| MEMO | 冻结旧 Specialized Block | 线性 | 否 | 中 |
| iCaRL | KD + 回放 | 固定 | 是 | 中高 |
| DER | 冻结旧 Backbone | 线性 | 否 | 高 |
| CoIL | OT 特征对齐 | 固定 | 否 | 高 |
| FOSTER | Boosting + KD | 线性 | 否 | 高 |
| APER | 参数隔离 | 固定 | 否 | 低-中 |

---

## 共享基类 BaseLearner

所有 Learner 继承自 `models/base.py:BaseLearner`, 提供：

- **示例回放** (`build_rehearsal_memory`): Herding 算法选择代表性样本
- **NME 评估** (`_eval_nme`): 基于类中心的最近邻分类, 不使用分类头
- **CNN 评估** (`_eval_cnn`): 使用分类头的标准分类
- **分组准确率** (`accuracy` in `utils/toolkit.py`): 按任务组计算准确率, 分别统计旧类/新类
- **检查点保存/加载** (`save_checkpoint` / `load_checkpoint`)
- **t-SNE 可视化** (`tsne`)

每个 Learner 只需要实现 `incremental_train()` 和 `_train()` 两个方法 (SEMA 额外实现 `_train_new`, `_detect_outlier`, `_init_train`)。
