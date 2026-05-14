# SEMA 三大组件 — AI 绘图提示词

为 Midjourney / DALL-E / Stable Diffusion 准备，三个组件风格统一。

> **统一风格关键词**: technical diagram, scientific illustration, clean white background, rounded rectangles, gradient arrows, academic paper figure style, vector art, flat design, no text rendered as image, isometric or front-facing

---

## 1. Functional Adapter（功能适配器 / 瓶颈 MLP）

### Midjourney Prompt

```
A technical diagram of a bottleneck MLP adapter module for neural networks. The structure shows a wide input arrow entering a purple LayerNorm block, then flowing into a trapezoid-shaped "down_proj" block (768 dimensions narrowing to 16 dimensions, colored orange), passing through a ReLU activation block (small orange square), then flowing into an inverted-trapezoid "up_proj" block (16 dimensions expanding back to 768 dimensions, colored orange). The overall shape resembles an hourglass or bottleneck funnel. Below the diagram, annotate with dimension labels: "768 → 16 → 768". Style: clean scientific diagram, white background, flat design, rounded rectangles, minimal color palette of orange and purple, isometric 3D view, no text clutter --ar 16:9 --style raw
```

### DALL-E / Stable Diffusion Prompt

```
A clean scientific diagram of a "Bottleneck MLP Adapter" neural network module. The diagram shows data flowing from left to right through a bottleneck structure: Input (768 dimensions) → LayerNorm (purple rounded rectangle) → Down Projection Linear Layer (768 to 16, an orange trapezoid narrowing) → ReLU Activation (small orange square) → Up Projection Linear Layer (16 to 768, an orange trapezoid widening back) → Output (768 dimensions). The shape visually resembles an hourglass - wide, narrow, wide. Use minimal flat design style on pure white background. Only use orange, purple, and gray colors. Show dimension numbers along the flow. Isometric 3D perspective with subtle shadows. Professional academic paper figure style.
```

### 结构说明（供理解，不放入 prompt）
```
Input [B, N, 768]
    │
    ▼
[LayerNorm] (可选, 紫色)
    │
    ▼
[down_proj: Linear(768 → 16)] (橙色倒梯形, 收窄)
    │
    ▼
[ReLU] (橙色小方块)
    │
    ▼
[up_proj: Linear(16 → 768)] (橙色正梯形, 扩张)
    │
    ▼
Output [B, N, 768]

关键:
- down_proj 权重: Kaiming Uniform 初始化
- up_proj 权重: 全零初始化 (保证初始输出为 0，不影响预训练特征)
- 参数量: 2 × 768 × 16 = 24,576 (仅 ViT 86M 的 0.03%)
```

---

## 2. RD / AE（表征描述器 / 自编码器）

### Midjourney Prompt

```
A technical diagram of a lightweight Autoencoder used as a "Representation Descriptor" for distribution shift detection. The structure shows: a wide input vector (768 dimensions, gray) first passing through a "Mean Pooling" block (yellow rounded rectangle, aggregating all tokens into one vector), then flowing into a green trapezoid "Encoder" block compressing from 768 to 128 dimensions (narrowing, labeled "encoder: 768→128"), reaching a narrow bottleneck labeled "Latent Code z (128 dims)", then flowing into a green inverted-trapezoid "Decoder" block expanding from 128 back to 768 (widening, labeled "decoder: 128→768"), producing a "Reconstruction" output. A dotted red line connects the input and reconstruction with an "MSE Loss" label between them. Below, show a small gauge or meter icon labeled "Z-score = |error - μ| / σ". Style: clean scientific diagram, white background, flat design, green and yellow and gray color palette, isometric view --ar 16:9 --style raw
```

### DALL-E / Stable Diffusion Prompt

```
A clean scientific diagram of a "Representation Descriptor (RD) Autoencoder" for detecting distribution shifts in continual learning. Data flows left to right: Input Feature Vector [B, 768] (gray block) → "Mean Pool" operation (yellow rounded rectangle) → Encoder Linear Layer 768→128 (green trapezoid narrowing toward center) → Compressed Latent Code z (thin green bar, 128 dimensions) → Decoder Linear Layer 128→768 (green trapezoid widening back) → Reconstructed Feature (gray block). A red dashed arrow connects the original input to the reconstruction output with a label "MSE Loss". At the bottom, a small circular gauge dashboard shows "Z-score = |current_error - historical_mean| / historical_stddev" with a needle pointing to green/yellow/red zones. Minimal flat design, pure white background, only green, yellow, red, and gray colors. Isometric 3D perspective. Professional technical illustration.
```

### 结构说明（供理解，不放入 prompt）
```
Input Features [B, N, 768]
    │
    ▼
[Mean Pool over tokens] → [B, 768] (黄色)
    │
    ▼
[encoder: Linear(768 → 128)] (绿色梯形, 收窄)
    │
    ▼
Latent Code z [B, 128] (瓶颈)
    │
    ▼
[decoder: Linear(128 → 768)] (绿色倒梯形, 扩张)
    │
    ▼
Reconstruction x' [B, 768]
    │
    ▼ (与输入对比)
MSE(x, x') → reconstruction_error [B] (逐样本)

    │
    ▼
Records 缓冲区 (容量 500, FIFO)
  ├── mean μ (历史均值)
  └── std σ  (历史标准差)

Z-score = |当前误差 - μ| / σ
  ├── Z ≤ 2.0 → "熟悉" (绿色)
  └── Z > 2.0 → "陌生人!" (红色, 触发扩展)

关键:
- AE 只对特征取均值后的向量做压缩重建
- encoder/decoder 都是 Kaiming Uniform + Zero Bias 初始化
- 训练时收集误差到缓冲区; 推理时冻结, 统计不再更新
- rd_dim=128 (压缩比 6:1), 参数量: 768×128×2 ≈ 197K
```

---

## 3. Router（软路由器 / Mixture-of-Adapters Combiner）

### Midjourney Prompt

```
A technical diagram of a "Soft Router for Mixture of Adapters" module. The scene shows: at the left, a single input feature stream [B, 768] entering a "Mean Pool" block (yellow). This pooled vector then passes through a router block (deep blue rounded rectangle labeled "Router: Linear(768→M)") which splits into M parallel weighted paths. Each path has a different weight value displayed as a horizontal slider bar (like an audio mixing console fader, colored deep blue to light blue gradient). Below the router, M separate adapter modules (orange blocks labeled "Adapter 0", "Adapter 1", ..., "Adapter M-1") each process the same input in parallel. The M adapter outputs then merge at a "Weighted Sum Σ" node (blue circle with sigma symbol) where each adapter's output is multiplied by its softmax weight. The final combined output flows to the right. A "Softmax" label appears above the weight sliders. Style: clean scientific diagram, white background, flat design, color palette of deep blue (router), orange (adapters), yellow (pooling), isometric 3D view --ar 16:9 --style raw
```

### DALL-E / Stable Diffusion Prompt

```
A clean scientific diagram of a "Mixture of Adapters Router" for soft-combining multiple adapters in a neural network. The diagram shows: Left side - Input Features [B, N, 768] enter a yellow "Mean Pool" block producing [B, 768]. This pooled vector feeds into a deep blue "Router: Linear(768→M)" block which outputs M logits. The logits pass through a Softmax operation (blue rounded rectangle) generating M probability weights (w0, w1, w2...) shown as horizontal slider bars like an audio mixing console, with different fill levels for each weight. In parallel below, M orange "Adapter" blocks (Adapter 0, Adapter 1, Adapter 2...) each independently process the same input. All adapter outputs converge at a large blue Σ "Weighted Sum" node where each is multiplied by its weight and summed. The combined output flows to the right. A highlighted inset shows: when a new adapter is added, the router expands from M to M+1 columns, with the new column weight initially zero. Professional technical illustration, pure white background, flat design, orange and deep blue color scheme, isometric perspective.
```

### 结构说明（供理解，不放入 prompt）
```
Input Features [B, N, 768]
    │
    ├──────────────────────────┬──────────────────────┐
    │                          │                      │
    ▼                          ▼                      ▼
Adapter0(x)              Adapter1(x)           AdapterM-1(x)
[B,N,768]                [B,N,768]             [B,N,768]
    │                          │                      │
    │                          │                      │
    ▼                          ▼                      ▼
    └──────────┬───────────────┴──────────┬──────────┘
               │                          │
               ▼                          │
     x.mean(dim=1) → [B, 768]            │
               │                          │
               ▼                          │
     Router: Linear(768 → M)              │
               │                          │
               ▼                          │
     Softmax → mask [w₀, w₁, ..., w_M-1]  │
               │                          │
               ▼                          ▼
     ┌──────────────────────────────────────┐
     │  Weighted Sum Σ:                      │
     │  output = w₀·Adapter₀(x)             │
     │         + w₁·Adapter₁(x)             │
     │         + ...                        │
     │         + w_M-1·Adapter_M-1(x)       │
     └──────────────────────────────────────┘
               │
               ▼
     Combined Output [B, N, 768]

路由器扩展机制 (新增 adapter 时):
  旧: router = Linear(768 → M),    冻结, 不可训练
  新: new_router = Linear(768 → 1),  权重=0, 可训练
  合并: logits = concat([router(x), new_router(x)]), shape [B, M+1]
  训练后: fix_router() 将 new_router 拼入 router → Linear(768, M+1)

关键:
- Softmax 对所有列一起归一化 → 新旧 adapter 自动竞争权重
- 推理时不需要 task-ID, router 自动学会给相关 adapter 高权重
- 相似任务共享 adapter 权重, 不相似任务自然分离
```

---

## 三组件协作关系图 (Bonus)

### Midjourney Prompt (Overview)

```
A scientific overview diagram showing three components working together in a continual learning system. Left section: orange "Functional Adapter" bottleneck MLP with hourglass shape (768→16→768). Center section: green "Representation Descriptor (RD)" autoencoder (768→128→768) with a dashboard gauge showing Z-score. Right section: deep blue "Router" with multiple slider weights combining M adapter outputs through a softmax-weighted sum. The three components are connected by arrows: the RD's Z-score gauge triggers "If all Z > threshold → add new Adapter" (red lightning bolt), the new Adapter feeds into the Router's mixing console, and the Router's combined output goes to classification. Arrows show the detection→expansion→routing pipeline. Bottom annotation: "SEMA: Self-Expansion with Mixture of Adapters". Professional technical illustration, pure white background, flat design, unified color scheme (orange=adapter, green=RD, blue=router) --ar 16:9 --style raw
```

---

## 配色汇总

| 组件 | 主色 | HEX | 隐喻 |
|------|------|-----|------|
| Adapter (functional) | 橙色 | #FF8C00 | 可训练的"任务专家" |
| RD (AE) | 绿色 | #2ECC71 | "指纹识别器"，分布检测 |
| Router | 深蓝 | #0066CC | "调音台"，软混合控制 |
| 残差/池化/辅助 | 灰色 | #B0B0B0 | 辅助操作 |
| 损失/警报 | 红色 | #E74C3C | 错误信号 |

---

## 使用建议

1. **Midjourney**: 直接复制英文 prompt，建议先加 `--ar 16:9 --style raw --v 6.1` 参数。如果文字渲染不好，添加 `--no text --no letters` 排除文字，后期手动加标注。

2. **DALL-E 3**: 复制英文 prompt，天然支持文字渲染较好。如果形状不对，追加 "The shapes must clearly form a bottleneck/hourglass/funnel structure"。

3. **Stable Diffusion (SDXL)**: 使用英文 prompt，建议配合 ControlNet (Canny edge) 先手绘草稿再生成。

4. **生成顺序建议**: 先做 Adapter（最简单），做出满意风格后，将该图作为 style reference 给后续 RD 和 Router 的生成，保证三张图风格一致。
