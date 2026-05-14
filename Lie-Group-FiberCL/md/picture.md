# PPT 配图生成提示词（ChatGPT Image Generation 用）

每页一个 prompt，直接复制粘贴到 ChatGPT（GPT-4o with image generation）。每个 prompt 约 150-250 词，已针对 DALL-E / GPT-4o 图像生成优化。

---

## Page 1 — 封面图

```
A striking academic cover image for a presentation on machine learning and geometry. Split composition: left side shows a chalkboard with elegant mathematical formulas in white chalk fading into the background. Right side shows a beautiful iridescent 3D curved surface (a mathematical manifold) in deep blue and orange hues, with glowing golden geodesic arcs tracing across it. Between the two halves, a single luminous golden thread weaves through, symbolizing the fiber bundle connection. Dark gradient background from deep navy to near-black. Cinematic lighting, shallow depth of field on the manifold, editorial science magazine cover style. No text or labels. 16:9 aspect ratio.
```

---

## Page 2 — 经典统计学习理论三大支柱

```
A refined scientific illustration with three vertical columns on a clean light gray background. Left column (warm amber theme): a pair of balanced golden scales representing PAC learning, with formula fragments in elegant notation. Center column (deep teal theme): a shattered set diagram with 5 colored dots and several possible separating lines in teal, representing VC dimension. Right column (soft burgundy theme): nested concentric circles labeled S1, S2, S3 representing structural risk minimization. At the bottom, a single foundation block spans all three columns, labeled "i.i.d. Assumption" in subtle red glow, with dependency arrows from each column pointing down to it. Clean academic style with soft shadows and subtle gradients, like a Nature journal figure. No clipart, no human figures. 16:9.
```

---

## Page 3 — i.i.d. 假设的失效

```
A triptych visual narrative about distribution shift, dark navy background with subtle data grid lines. Story told in three horizontal bands with cinematic atmosphere.

Top band - "Temporal Drift": A stylized social media icon surrounded by blue word clouds on the left, morphing into red word clouds on the right, with a jagged time arrow between them. Data points scatter like statistical samples shifting distributions.

Middle band - "Weather Domain Shift": Split-screen of an autonomous vehicle's sensor view. Left half shows a sunny day with clean blue LiDAR point clouds. Right half shows heavy rain with distorted orange sensor readings. Rain streaks across the dividing line.

Bottom band - "Open World Recognition": A taxonomic tree growing new branches. A camera lens captures animal silhouettes — familiar ones clustered left with soft green checkmarks, unknown ones on the right with glowing amber question marks.

Clean, emotional, editorial data art meets science fiction concept art. 16:9.
```

---

## Page 4 — 类增量学习形式化定义

```
A modern isometric technical diagram on clean white background. A horizontal timeline shows 7 rounded rectangular task blocks (T0 through T6) arranged left to right. Each block is filled with small colored squares representing classes — T0 starts with 10 squares, each subsequent block adds 10 new colored squares while previous colors remain but appear slightly dimmed. The accumulation creates a beautiful expanding mosaic of colors across the timeline.

Below the timeline, three elegant constraint icons:
1. "Streaming Data": a single-file data arrow with a subtle lock symbol overlay
2. "Task-Agnostic": a question mark icon with a strikethrough over a task-ID badge
3. "Minimize Forgetting": a graph showing an ideal flat green line vs a declining red curve

Vector illustration style with subtle 3D depth and soft shadows. Premium software UI aesthetic. Colors: academic blues with warm amber for emphasis. 16:9.
```

---

## Page 5 — 现有方法对比与理论缺口

```
A polished comparison infographic, clean white background, like a premium consulting report visual. Three horizontal cards stacked vertically:

1. "Regularization" (ice blue theme): A neural network where certain connections have glowing blue shield locks protecting important weights. The locks accumulate, and delicate chains wrap around more connections over time, showing growing rigidity.

2. "Replay" (soft green theme): A memory bank of tiny thumbnail images in a grid. A red-cross overlay hovers over part of the stack with a privacy shield icon. A "Storage Limit" gauge shows the fill level near capacity.

3. "Dynamic Expansion" (warm purple theme): A network that grows new modules as branches. Each branch has identical fixed width — some too wide (wasted capacity, shown as hollow space), some too narrow (straining, shown with tension lines). A question mark floats over the expansion step size parameter.

A red dashed oval at the bottom encircles all three cards' shared weakness, pulsing with subtle red glow — the empty space inside the oval represents "missing principled expansion criterion."

16:9.
```

---

## Page 6 — 从欧氏空间到 Stiefel 流形

```
A dramatic split-comparison image showing the conceptual leap from Euclidean to manifold parameter space. 

LEFT half: A clean flat Cartesian coordinate plane with scattered points W1, W2, W3. A red dashed straight line connects W1 to W2 — Euclidean distance. The line passes through empty space. Clean white grid background.

RIGHT half: A stunning translucent iridescent curved surface — the Stiefel manifold — rendered like blown glass, shifting from deep blue to warm amber depending on viewing angle. Points W1, W2, W3 sit ON this surface. A glowing golden curve traces along every contour of the surface between W1 and W2 — the geodesic — never leaving the surface. Small fan-shaped principal angle arcs drawn near the points.

The contrast between the sterile flat space and the rich curved geometry tells the intellectual story. Dark gradient background on the right side. Photorealistic glass rendering, beautiful subsurface scattering on the manifold. 8K quality. 16:9.
```

---

## Page 7 — 李群的对称性与 Stiefel 流形的群论结构

```
A magnificent mathematical visualization in triangular composition on dark background.

TOP LEFT: A polished bronze 3D torus-like manifold representing the Lie group SO(d). The surface has a fine golden wireframe grid. Small flowing arrows trace smooth group multiplication paths along the surface. Dramatic rim lighting.

TOP RIGHT: The Stiefel manifold St(d,r) as a translucent blue-silver curved surface. The SO(d) manifold from the left appears above it, with internal coset "slices" highlighted in alternating translucent colors. A downward arrow shows the quotient map collapsing each coset to a point on the Stiefel manifold.

BOTTOM CENTER: Two glowing frames of orthogonal basis vectors (W1 and W2) sit on the Stiefel surface. Between corresponding vectors, colored arcs represent the r principal angles (θ1 through θr). The formula d_geo = ||arccos(σ(W1ᵀW2))||₂ floats nearby in elegant mathematical typography with soft golden glow.

Like a Quanta Magazine mathematical visualization. No text rendered as image. 16:9.
```

---

## Page 8 — 纤维丛：统一任务与参数空间

```
A breathtaking 3D scientific visualization of a Fiber Bundle — the conceptual centerpiece of the presentation. Architectural render meets scientific illustration.

BOTTOM LAYER: A gently undulating topographical surface (the Base Manifold B) in warm earth tones — terracotta, sand, amber. Small glowing dots mark task locations p1, p2, p3, p4 along a winding path. The surface has subtle elevation changes suggesting task distribution density.

MIDDLE LAYER: From each marked point, vertical translucent stalks rise upward. At the tip of each stalk floats a miniature iridescent Stiefel manifold (glass ellipsoid). Fibers above nearby tasks (p1 and p2) share similar color and orientation. The fiber above distant p4 is noticeably different — visually signaling "this task is unlike the others."

TOP LAYER: A luminous golden curve (the Section s) weaves through the fibers — passing smoothly through p1 and p2's fibers (sharing adapter), then branching upward at p3 (expansion), then continuing. 

INSET: A magnified circle showing local trivialization — U × F product structure clearly visible.

Volumetric lighting, dark edges fading to black. Cinematic 16:9. No text.
```

---

## Page 9 — 纤维丛的持续学习动力学（统计 vs 几何检测）

```
A dramatic before-and-after comparison infographic, split horizontally, clean light background. Like a WIRED magazine technology comparison spread.

TOP HALF — "Statistical Detection (SEMA)": An industrial-feeling pipeline flowing left to right. Components: data batch icon → hourglass-shaped autoencoder (768→128→768, in green) → reconstruction loss waveform → Z-score gauge showing a normal distribution with red tail region → threshold decision diamond → expansion trigger. A clock icon notes "20 extra epochs." The pipeline has many moving parts — buffer icons, gear icons — feeling engineered and complex. Color palette: cool grays and greens.

BOTTOM HALF — "Geometric Detection (Lie-SEMA)": A shorter, more elegant pipeline. Components: data batch icon → Stiefel manifold with a temporary adapter point appearing on it (beautiful blue glass surface with golden point) → SVD decomposition shown as a crystal prism splitting light → geodesic distance formula glowing in gold → geometric threshold diamond → expansion trigger. Fewer components, no buffers, no distribution curves. A golden checkmark notes "Direct computation, no extra training." Color palette: warm golds and deep blues.

A central arrow points downward between the halves: "Statistical Proxy → Geometric Ground Truth." 16:9.
```

---

## Page 10 — SEMA 架构总览

```
A luminous architectural technical diagram of SEMA. Clean white background with subtle grid.

LEFT: A natural photograph of a bird on a branch dissolves into a grid of 14×14 small colored patches. These feed into a translucent purple cube — the Patch Embedding.

CENTER: The ViT backbone rendered as 12 stacked glass floors of a luminous tower. Layers 0-8 are frosted gray glass with delicate ice-blue snowflake etchings — frozen. Layers 9-11 glow with warm amber light — alive with SEMA modules protruding like glowing orange balconies.

CALL OUT for one SEMA Module (magnified):
A rounded amber pod containing three beautiful sub-components:
1. Hourglass-shaped adapter (768→16→768) in orange gradient, subtly pulsing
2. Leaf-shaped autoencoder RD (768→128→768) in soft green, with a tiny Z-score gauge
3. Mixing console router in deep blue with M weighted vertical slider bars, converging at a sigma summation node

RIGHT: Classifier head as a purple crystal → output bar chart with top class highlighted.

Data flow arrows in warm gradient colors connecting all components. Sci-fi architecture meets academic figure. 16:9.
```

---

## Page 11 — Lie-SEMA：Stiefel 约束的 Adapter

```
A beautiful dual-panel comparison on clean split background.

LEFT PANEL (light background): "Standard Adapter" — A weight matrix visualized as a 2D grid of tiny colored squares (16 rows × 768 columns) floating freely in unbounded Euclidean space. Wispy drift trails behind extreme values (bright red outliers). The weights feel unanchored, messy, unconstrained. A ghostly hourglass shape faintly visible behind.

RIGHT PANEL (dark navy background): "Lie-SEMA Stiefel Adapter" — The weight matrix is now constrained. A beautiful translucent Stiefel manifold surface rendered in iridescent blue glass. 16 golden arrow vectors emerge from the surface — mutually orthogonal, unit-length, forming a perfect reference frame. A subtle geometric wireframe cage around them shows the constraint WᵀW = I₁₆ being elegantly satisfied.

A glowing transformation arrow between panels: "SVD Projection: W ← UVᵀ" with the SVD decomposition visualized as U·Σ·Vᵀ → dropping Σ → U·Vᵀ snapping the weights onto the manifold surface.

Below: the geodesic distance formula between two Stiefel frames in elegant golden mathematical typography.

The visual contrast between "messy freedom" and "structured geometry" tells the story. 16:9.
```

---

## Page 12 — 测地线距离 vs Z-score

```
An elegant editorial comparison visualization — like a New York Times Science section spread. Clean, airy, premium.

TITLE AREA (subtle, top): "How do we know when to expand?"

LEFT COLUMN — "Statistical Way": A stylized blue normal distribution bell curve with a needle gauge pointing to the red tail rejection region. Small icons indicate moving parts: a gear (AE training needed), a database (buffer maintenance), a bell curve with a question mark (normality assumption). The bell curve subtly wavers and distorts at its edges — showing sensitivity to outliers. Cool blue-gray palette.

CENTER — Relationship plot: A beautiful scatter plot on dark grid background, translucent glowing dots forming a roughly monotonic but non-linear point cloud. Golden trend line curves through it. Faint annotation "r ≈ 0.7". Like a NASA exoplanet data visualization. X-axis: geodesic distance. Y-axis: Z-score.

RIGHT COLUMN — "Geometric Way": The Stiefel manifold surface in translucent blue with two frames W_new and W_i marked as golden points. Principal angles drawn as elegant glowing arcs between corresponding basis vectors. The SVD computation shown as a crystal prism. Clean formula: d_geo = ||arccos(σ)||₂. No buffers, no training wheels, no assumptions. Warm gold and deep blue palette.

BOTTOM RIBBON: "Statistical detection learns distance from data. Geometric detection computes distance from structure."

16:9.
```

---

## Page 13 — Lie-SEMA 算法流程 + 实验结果

```
A polished publication-ready figure in two sections, clean white background.

TOP (60%): Horizontal flow pipeline of elegant glass capsule nodes connected by gradient arrows:

[New Task Data Arrives] → [Add Temporary Adapter (Haar init, Stiefel)] → [Train: func 5ep + rd 20ep] → [SVD Project All → Stiefel] → [Diamond: min d_geo > τ?] → two branches:
  YES branch (warm green glow) → [Confirm New Adapter] → [Fix Router] → [Freeze & Continue]
  NO branch (cool blue glow) → [Rollback Temporary] → [Fine-tune Router only] → [Continue]

The Stiefel manifold appears as a beautiful translucent surface in the SVD projection step. The decision diamond pulses amber.

BOTTOM (40%): Three elegant charts on slightly darker background:
- Left: Grouped bar chart "CIFAR-100 B50Inc5" comparing methods. Lie-SEMA bar highlighted with golden glow.
- Center: Line chart "Adapter Growth" showing sub-linear growth across tasks. Lie-SEMA line in gold, baselines in gray.
- Right: Ablation bar chart — Full Lie-SEMA (tallest, gold), minus constraints, minus geodesic, minus router.

The Economist data visualization style — clean grids, confident colors, no clutter. 16:9.
```

---

## Page 14 — 纤维丛理论对 CIL 的五大启示

```
A magnificent synthesis visualization — the intellectual climax. Dark gradient background (#0a1628) with a dreamlike semi-transparent fiber bundle diagram in the far background (watercolor style in muted blues and golds).

Five illuminated frosted-glass "revelation cards" arranged in a radial/roughly circular pattern. Each card has a unique accent color and is connected by thin golden light threads to the background fiber bundle:

CARD 1 (amber, top): Two nearby points on the base manifold with golden measuring tape between them. Theme: "Base Distance = Task Difference"

CARD 2 (teal, top-right): SO(d) manifold with Stiefel frames as points that slide along legal group orbits. Red X's mark points outside the manifold. Theme: "Structure Group = Boundary of Legal Transformations"

CARD 3 (burgundy, bottom-right): Close-up of two fibers with a parallel transport curve drawn between them — showing knowledge moving between task adapters via the connection. Theme: "Connection = Path of Knowledge Transfer"

CARD 4 (green, bottom-left): U×F product structure glowing in a local neighborhood, showing identical fiber structure within proximity. Theme: "Local Trivialization = Condition for Adapter Sharing"

CARD 5 (violet, left): Nested hypothesis circles (S1⊂S2⊂S3) projected onto the base manifold — when d_geo exceeds threshold, single hypothesis space cannot cover both tasks. Theme: "VC Dim + Fiber Bundle = Unified Capacity Theory"

Light rays connect each card to its geometric counterpart in the background. Like a Nature Reviews Physics key figure. 16:9.
```

---

## Page 15 — 从 PAC 到纤维丛的理论链条

```
A scholarly "theoretical genealogy" visualization — like a museum exhibit tracing the evolution of ideas. Three rows, three columns of glass-like cells forming a 3×3 matrix. Cream parchment background transitions to deep navy on the right side.

ROW 1 (PAC Learning):
- Left cell (cream): PAC formula P(|R−R_emp|>ε) ≤ 2exp(−2mε²), with prominent red "i.i.d. REQUIRED" stamp
- Center cell (transitional): Same formula cracking, with distribution shift arrows tearing through it
- Right cell (navy/gold): Base manifold B with a single glowing point p — "When B = {p}, PAC is a special case"

ROW 2 (VC Dimension):
- Left cell: VC formula with shattered set diagram
- Center cell: A single section on the fiber bundle fading as it tries to span distant task points
- Right cell: New golden section branches sprouting — "VC dim defines the covering radius of a section"

ROW 3 (SRM):
- Left cell: Nested hypothesis spaces S1⊂S2⊂S3 with risk bound
- Center cell: Red Φ(h/n) bar exploding when forced to cover dissimilar tasks
- Right cell: Golden balanced tradeoff — sharing (reuse adapter, lower Φ) vs branching (new adapter, lower R_emp)

Golden threads weave through all cells, connecting classical concepts to their geometric extensions. Illuminated manuscript meets modern data design. 16:9.
```

---

## Page 16 — 核心贡献总结

```
A powerful cinematic triptych — three vertical panels spanning a unified horizon with a dramatic sky. The definitive "take-home message" visual.

LEFT PANEL — "Statistics → Geometry": A gray bell curve and scatter plots on the left side dissolve into a beautiful golden Stiefel manifold with glowing geodesic on the right side. A luminous bridge of light connects them. Above the geometry side, subtle golden emblems suggesting determinism and intrinsic structure. Sky transitions from pale uncertain dawn to clear morning.

CENTER PANEL — "Manual → Adaptive": A human hand adjusting a mechanical knob dissolves into sparkling particles that reform as an elegant autonomous feedback loop — the geodesic distance self-triggering expansion. A subtle heartbeat waveform pulses at the base — alive, responsive, not rigid. Warm confident midday light.

RIGHT PANEL — "Engineering → Theory": Scattered gray industrial components (bolts, gears, heuristic knobs) dissolve and reassemble into a single unified crystalline geometric structure — the complete fiber bundle (B, π, F, G) glowing in amber, teal, burgundy, and violet. Brilliant midday sun.

Sparks of insight float upward through all three panels. Cinematic concept art meets premium keynote visual. The triptych format gives it gravitas and finality. 16:9 ultra-wide.
```

---

## Page 17 — 开放问题与未来方向

```
A contemplative, cinematic closing image — standing at the edge of known territory, looking toward unexplored horizons. Twilight atmosphere.

FOREGROUND (sharp, well-lit): A beautifully rendered fiber bundle structure sits on solid ground — undulating base manifold terrain, fibers rising like luminous crystal columns, a golden section weaving through. Complete, elegant, illuminated — "what we have built."

MIDGROUND (atmospheric mist): Four glowing crystalline monoliths rise from the fog, each an open question:
1. Ice blue monolith (left): Base manifold B partially obscured by fog, with data points suggesting "Can we learn B's topology from data?"
2. Emerald green monolith (center-left): Connection paths with a subtle neural network symbol overlay — "Can we learn the connection from data?"
3. Amber gold monolith (center-right): Fiber bundle curving, with Riemann curvature tensor symbols in orbit — "Does curvature explain catastrophic forgetting?"
4. Deep violet monolith (right): The fiber transforming — SO(n) morphing into SE(n), Sp(n) variants — "Can this generalize to other structure groups?"

BACKGROUND (distant horizon): Mist parts slightly to reveal hints of a larger theoretical continent — faint glowing geometric structures in the far distance, waiting to be explored.

HOPEFUL: A subtle path of stepping stones leads from the completed fiber bundle toward the question monoliths. Sky transitions from deep twilight blue to warm hopeful amber at the horizon. Christopher Nolan film final shot aesthetic. 8K cinematic. 16:9.
```

---

## 通用风格提示（粘贴到每个 prompt 末尾）

如果图片缺少风格统一感，可在任意 prompt 末尾追加：

```
Clean technical illustration style. No text rendered as an image unless specifically requested. Smooth gradients, soft shadows, premium quality. Suitable for an academic presentation.
```
