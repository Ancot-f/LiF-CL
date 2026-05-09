"""
Prompt 模块 — L2P / DualPrompt / CODA-Prompt 的 Prompt 池实现
==============================================================

包含三种 Prompt 机制：
  - Prompt: L2P (Learning to Prompt) 的 Prompt 池，通过 key-query 相似度选择 top-k prompt
  - EPrompt: DualPrompt 的 Expert Prompt 池，支持 prefix-tuning 风格的 G-Prompt + E-Prompt
  - CodaPrompt: CODA-Prompt 的分解注意力 Prompt，通过 Gram-Schmidt 正交化初始化

核心机制：可学习的 prompt token 池 + 基于余弦相似度的 top-k 选择
"""

import torch
import torch.nn as nn
import copy


# ===========================================================================
# 类: CodaPrompt
# 描述: CODA-Prompt 模块 -- 用于持续学习的基于正交注意力分解的 Prompt。
#
#   CodaPrompt 维护一个 prompt 池，使用 Gram-Schmidt 正交化来生成正交的
#   prompt 组件。每个任务分配池中的一部分 prompt 组件，并通过注意力机制
#   组合这些组件。在训练当前任务时，冻结历史任务的 prompt 组件以防止遗忘。
#
#   关键特性:
#     - Gram-Schmidt 正交初始化: 确保 prompt 组件之间的正交性
#     - 正交性惩罚: 训练时施加正交性正则化损失
#     - 任务级隔离: 通过 process_task_count() 管理任务级 prompt 组件分配
#     - 分步冻结: 训练时冻结历史任务对应的 prompt 组件
# ===========================================================================
class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768):
        """
        参数:
            emb_d: 嵌入维度（prompt 的向量维度）
            n_tasks: 总任务数（用于预分配 prompt 池大小）
            prompt_param: prompt 参数列表 [pool_size, prompt_length, ortho_mu]
                         - pool_size: prompt 池的总大小
                         - prompt_length: 每个 prompt 的 token 长度
                         - ortho_mu: 正交性惩罚的强度系数
            key_dim: key 向量的维度（用于计算 query-key 相似度）
        """
        super().__init__()
        self.task_count = 0  # 当前任务计数器（从 0 开始）
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)  # 解析 prompt 参数

        # 初始化 e-prompt 的各个组件（prompt 向量 P、key 向量 K、attention 向量 A）
        # 用于模型保存/加载的简便性，在此处初始化所有参数
        # 注意: 在持续学习的实践中，每个新任务开始时都会重新初始化新组件
        # 因为在任务序列开始时我们不知道会面临多少任务
        for e in self.e_layers:
            e_l = self.e_p_length
            # P: prompt 向量 [pool_size, prompt_length, emb_d]
            p = self.tensor_prompt(self.e_pool_size, e_l, emb_d)
            # K: key 向量 [pool_size, key_dim]
            k = self.tensor_prompt(self.e_pool_size, self.key_d)
            # A: attention 向量 [pool_size, key_dim]
            a = self.tensor_prompt(self.e_pool_size, self.key_d)
            # 使用 Gram-Schmidt 正交化初始化（确保 prompt 组件间正交）
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    def _init_smart(self, emb_d, prompt_param):
        """解析 prompt 参数并初始化基本设置。"""
        # prompt 池大小
        self.e_pool_size = int(prompt_param[0])
        # 每个 prompt 的 token 长度
        self.e_p_length = int(prompt_param[1])
        # 应用 e-prompt 的层索引列表（例如 [0,1,2,3,4] 表示在前5层使用）
        self.e_layers = [0, 1, 2, 3, 4]

        # 正交性惩罚的强度系数
        self.ortho_mu = prompt_param[2]

    def process_task_count(self):
        """处理任务计数增加：增加任务计数器并重新正交化 prompt 组件。

        在持续学习中，每次开始新任务时调用此函数。
        重新运行 Gram-Schmidt 正交化，确保新任务的 prompt 组件
        与已有组件正交。
        """
        self.task_count += 1

        # 在持续学习的实践中，我们使用 Gram-Schmidt 重新初始化新任务的组件
        # 这种修改更符合持续学习的精神，且对性能影响很小
        #
        # 此函数的代码参考自:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self, f'e_k_{e}')
            A = getattr(self, f'e_a_{e}')
            P = getattr(self, f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}', p)
            setattr(self, f'e_k_{e}', k)
            setattr(self, f'e_a_{e}', a)

    # 此函数的代码参考自:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):
        """对输入向量应用 Gram-Schmidt 正交化过程。

        Gram-Schmidt 过程将一组向量转换为正交向量组。
        在这里用于确保 prompt 池中分配给不同任务的 prompt 组件
        彼此正交，从而减少任务间的干扰。

        算法:
          对每个向量 v_k，减去其在所有已处理向量 u_j 上的投影:
            u_k = v_k - sum_{j<k} proj(u_j, v_k)
          然后归一化: u_k = u_k / ||u_k||
        """
        def projection(u, v):
            """计算向量 v 在向量 u 上的投影。"""
            denominator = (u * u).sum()
            if denominator < 1e-8:
                return None  # 避免除以零
            else:
                return (v * u).sum() / denominator * u

        # 检查是否为 3D 张量，如果是则展平最后两维
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0], -1)

        # 转置: 将向量按列排列（每列是一个待正交化的向量）
        vv = vv.T

        # 矩阵大小
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # 计算每个任务分配的 prompt 组件范围
        pt = int(self.e_pool_size / (self.n_tasks))  # 每个任务分配的组件数
        s = int(self.task_count * pt)  # 当前任务范围的起始索引
        f = int((self.task_count + 1) * pt)  # 当前任务范围的结束索引

        # 保留已有任务的正交化结果（已冻结部分）
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()

        # 对新任务范围内的向量进行正交化
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:, k]).to(vv.device)  # 随机初始化
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True  # 若投影无效，重新随机初始化
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo:
                    uu[:, k] = vk - uk  # 减去所有已有方向的投影

        # 归一化新生成的向量
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # 恢复原始行列方向
        uu = uu.T

        # 如果原始是 3D 张量，恢复形状
        if is_3d:
            uu = uu.view(shape_2d)

        return torch.nn.Parameter(uu)

    def forward(self, x_querry, l, x_block, train=False):
        """CodaPrompt 的前向传播。

        参数:
            x_querry: 查询向量 [B, C]，来自当前层的 CLS token 或其他特征
            l: 当前层索引
            x_block: 当前块的输入特征
            train: 是否处于训练模式（用于控制历史组件的冻结/可训练状态）

        返回:
            (p_return, loss, x_block):
              - p_return: [Ek, Ev] 格式的 prompt 输出，或 None
              - loss: 正交性惩罚损失
              - x_block: 原始输入块（不变）
        """
        # e-prompt 处理
        e_valid = False
        if l in self.e_layers:  # 只有指定层才使用 e-prompt
            e_valid = True
            B, C = x_querry.shape

            # 获取当前层的 P（prompt 向量）、K（key 向量）、A（attention 向量）
            K = getattr(self, f'e_k_{l}')
            A = getattr(self, f'e_a_{l}')
            p = getattr(self, f'e_p_{l}')

            # 计算当前任务对应的 prompt 组件索引范围
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)

            # 冻结/控制历史任务的 prompt 组件
            if train:
                if self.task_count > 0:
                    # 训练时: 分离历史任务（冻结）和当前任务（可训练）的组件
                    K = torch.cat((K[:s].detach().clone(), K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(), A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(), p[s:f]), dim=0)
                else:
                    # 第一个任务: 仅使用当前任务的组件
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                # 推理时: 使用所有已学习任务的组件
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]

            # 使用注意力机制和余弦相似度计算 prompt 组合权重
            # Query 与 Attention 向量计算: a_query = x_query @ A^T
            # 形状: (B x 1 x d) * soft([1 x k x d]) = (B x k x d) -> 注意力 = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)

            # 归一化 key 和 query，计算余弦相似度
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)  # [B, k]

            # 用注意力权重组合 prompt 组件
            # (B x 1 x k x 1) * [1 x plen x k x d] = (B x plen x d)
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # 将 prompt 拆分为 Key 和 Value 两部分
            # 前半部分作为 Key prompt (Ek)，后半部分作为 Value prompt (Ev)
            i = int(self.e_p_length / 2)
            Ek = P_[:, :i, :]  # Key prompt: [B, plen/2, d]
            Ev = P_[:, i:, :]  # Value prompt: [B, plen/2, d]

            # 正交性惩罚损失
            if train and self.ortho_mu > 0:
                # 对 K（key）、A（attention）和 P（prompt）三组向量分别计算正交性损失
                loss = self.ortho_penalty(K) * self.ortho_mu
                loss += self.ortho_penalty(A) * self.ortho_mu
                loss += self.ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # 组合 prompt 用于 prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        return p_return, loss, x_block

    def ortho_penalty(self, t):
        """计算正交性惩罚损失。

        公式: mean((t @ t^T - I)^2)
        这鼓励矩阵 t 的行向量彼此正交（即 t @ t^T 接近单位矩阵 I）。
        """
        return ((t @ t.T - torch.eye(t.shape[0])) ** 2).mean()

    def tensor_prompt(self, a, b, c=None, ortho=False):
        """创建可训练的 prompt 张量参数。

        参数:
            a: 第一维大小
            b: 第二维大小
            c: 可选的第三维大小（若提供则为 3D 张量）
            ortho: 是否使用正交初始化

        返回:
            初始化的 torch.nn.Parameter
        """
        if c is None:
            p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
        else:
            p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
        if ortho:
            nn.init.orthogonal_(p)
        else:
            nn.init.uniform_(p)
        return p


# ===========================================================================
# 类: EPrompt
# 描述: DualPrompt 的 E-Prompt 模块 -- 用于持续学习的 Expert Prompt。
#
#   EPrompt 通过维护一个可学习的 prompt 池，基于输入特征的相似度
#   选择 top-k 个最相关的 prompt。支持以下特性:
#     - Prompt Pool: 可学习的 prompt 向量池
#     - Key-Query 匹配: 使用余弦相似度匹配输入特征与 prompt key
#     - Batchwise Prompt: 批次级 prompt 选择（整个 batch 共享相同的 prompt）
#     - Prefix Tuning: 支持 multi-head prefix 格式（按注意力头拆分）
#     - Dual (Key+Value): 支持 Key-Value 分离的 prompt 结构
#
#   与 Prompt (L2P) 的主要区别:
#     - EPrompt 不直接拼接 prompt 到输入序列（由外部处理拼接逻辑）
#     - 支持多层 prefix 格式（num_layers > 1）
#     - 支持 Key/Value 分离的 prompt 结构
# ===========================================================================
class EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform',
                 prompt_pool=False, prompt_key=False, pool_size=None, top_k=None,
                 batchwise_prompt=False, prompt_key_init='uniform', num_layers=1,
                 use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False):
        """
        参数:
            length: 每个 prompt 的 token 长度
            embed_dim: 嵌入维度
            embedding_key: 计算查询 embedding 的方式
                          ('mean', 'max', 'mean_max', 'cls')
            prompt_init: prompt 初始化方式 ('zero', 'uniform')
            prompt_pool: 是否使用 prompt 池（True=从池中选择，False=全局 prompt）
            prompt_key: 是否使用可学习的 prompt key
            pool_size: prompt 池的大小
            top_k: 每次选择 top-k 个 prompt
            batchwise_prompt: 是否对整个 batch 使用相同的 prompt
            prompt_key_init: key 初始化方式
            num_layers: prompt 应用的层数
            use_prefix_tune_for_e_prompt: 是否使用 prefix-tuning 格式
            num_heads: 注意力头数（prefix 格式时需要）
            same_key_value: Key 和 Value prompt 是否共享
        """
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value

        if self.prompt_pool:
            # 使用 prompt 池模式
            if self.use_prefix_tune_for_e_prompt:
                # prefix-tuning 格式: prompt 形状包含注意力头维度
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    # Key 和 Value 共享: 1 组参数，复制为 2 组
                    prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length,
                                        self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    # 复制一份作为 Key/Value 对
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
                else:
                    # Key 和 Value 独立: 2 组独立的参数
                    prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length,
                                        self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
            else:
                # 标准 prompt 格式: 不按注意力头分
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length, embed_dim)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)

        # 如果使用可学习的 prompt key
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # 否则使用 prompt 的均值作为 key（仅兼容标准 prompt，不兼容 prefix 格式）
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """对向量或矩阵进行 L2 归一化。

        公式: x_normalized = x / sqrt(max(sum(x^2), epsilon))
        """
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        """EPrompt 的前向传播。

        参数:
            x_embed: 输入 embedding [B, N, C]（图像 patch 序列）
            prompt_mask: 可选的 prompt 索引掩码 [B, top_k]
            cls_features: 可选的 CLS 特征 [B, C]（用于 'cls' 模式的 key 计算）

        返回:
            out: 字典，包含:
              - 'batched_prompt': 选中的批量 prompt [num_layers, B, top_k*length, embed_dim]
              - 'prompt_idx': 选中的 prompt 索引 [B, top_k]
              - 'similarity': 相似度矩阵 [B, pool_size]
              - 'selected_key': 选中的 key 向量 [B, top_k, C]
              - 'reduce_sim': 用于 pull_constraint 损失计算的标量
              - 等等
        """
        out = dict()
        if self.prompt_pool:
            # === 步骤 1: 计算查询 embedding (query key) ===
            # 根据配置从输入序列中提取特征作为查询
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)  # [B, C]
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]  # [B, C]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)  # [B, C]
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # [B, C]
                else:
                    x_embed_mean = cls_features  # 使用外部提供的 CLS 特征
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            # === 步骤 2: 计算相似度 ===
            # 对 prompt key 和查询 embedding 做 L2 归一化，计算余弦相似度
            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1)  # [Pool_size, C]
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # [B, C]

            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t())  # [pool_size, B]
            similarity = similarity.t()  # [B, pool_size]

            # === 步骤 3: 选择 top-k 个最匹配的 prompt ===
            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1)  # [B, top_k]
            out['similarity'] = similarity

            # 批次级 prompt 选择: 整个 batch 使用相同的 top-k prompt
            if self.batchwise_prompt:
                prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                # 当唯一元素数量不足 pool_size 时，用最小值填充
                if prompt_id.shape[0] < self.pool_size:
                    prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                    id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                _, major_idx = torch.topk(id_counts, k=self.top_k)  # 选择出现频率最高的 top_k
                major_prompt_id = prompt_id[major_idx]  # [top_k]
                # 扩展到整个 batch
                idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous()  # [B, top_k]

            # 如果提供了 mask，使用它覆盖 idx
            if prompt_mask is not None:
                idx = prompt_mask  # [B, top_k]

            out['prompt_idx'] = idx

            # === 步骤 4: 组装选中的 prompt ===
            if self.use_prefix_tune_for_e_prompt:
                # Prefix 格式: [num_layers, B, top_k, length, num_heads, head_dim]
                batched_prompt_raw = self.prompt[:, :, idx]
                num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                # 合并 dual 和 top_k*length 维度: [num_layers, B, dual, top_k*length, num_heads, head_dim]
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
                )
            else:
                # 标准格式: [num_layers, B, top_k, length, embed_dim]
                batched_prompt_raw = self.prompt[:, idx]
                num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
                # 合并 top_k 和 length 维度: [num_layers, B, top_k*length, embed_dim]
                batched_prompt = batched_prompt_raw.reshape(
                    num_layers, batch_size, top_k * length, embed_dim
                )

            batched_key_norm = prompt_key_norm[idx]  # [B, top_k, C]

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            # === 步骤 5: 计算 pull_constraint 损失 ===
            # 鼓励查询 embedding 与选中的 prompt key 之间的相似度
            x_embed_norm = x_embed_norm.unsqueeze(1)  # [B, 1, C]
            sim = batched_key_norm * x_embed_norm  # [B, top_k, C]
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # 标量

            out['reduce_sim'] = reduce_sim
        else:
            # === 不使用 prompt 池: 全局统一 prompt ===
            # 所有样本使用相同的 prompt
            if self.use_prefix_tune_for_e_prompt:
                # Prefix 格式
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:
                    prompt_pool_shape = (self.num_layers, 1, self.length,
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
                else:
                    prompt_pool_shape = (self.num_layers, 2, self.length,
                                        self.num_heads, embed_dim // self.num_heads)
                    if self.prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif self.prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                        nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
            else:
                prompt_pool_shape = (self.num_layers, self.length, embed_dim)
                if self.prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif self.prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)
                batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)

        out['batched_prompt'] = batched_prompt

        return out


# ===========================================================================
# 类: Prompt
# 描述: L2P (Learning to Prompt) 的 Prompt 模块 -- 用于持续学习的基础 Prompt。
#
#   Prompt 是 L2P 方法的核心模块。它维护一个可学习的 prompt 池，
#   通过余弦相似度匹配从池中选择最相关的 prompt，将选中的 prompt
#   直接拼接到输入序列的前面。这与 VPT 类似，但 prompt 是从池中动态选择的。
#
#   与 EPrompt 的主要区别:
#     - Prompt (L2P) 直接将 prompt 拼接到输入序列 (prompted_embedding)
#     - Prompt 不支持多层结构（仅为单层）
#     - Prompt 不支持 prefix-tuning 格式（不按注意力头分）
#
#   参考: "Learning to Prompt for Continual Learning", Wang et al., CVPR 2022
# ===========================================================================
class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform',
                 prompt_pool=False, prompt_key=False, pool_size=None, top_k=None,
                 batchwise_prompt=False, prompt_key_init='uniform'):
        """
        参数:
            length: 每个 prompt 的 token 长度
            embed_dim: 嵌入维度
            embedding_key: 计算查询 embedding 的方式
            prompt_init: prompt 初始化方式 ('zero', 'uniform')
            prompt_pool: 是否使用 prompt 池
            prompt_key: 是否使用可学习的 prompt key
            pool_size: prompt 池大小
            top_k: 每次选 top-k 个 prompt
            batchwise_prompt: 是否对 batch 使用相同 prompt
            prompt_key_init: key 初始化方式
        """
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            # prompt 池形状: [pool_size, length, embed_dim]
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)

        # 如果使用可学习的 prompt key
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # 否则使用 prompt 的均值作为 key（对所有 token 长度维度取平均）
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """对向量进行 L2 归一化。"""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        """Prompt (L2P) 的前向传播。

        参数:
            x_embed: 输入 embedding [B, N, C]
            prompt_mask: 可选的 prompt 索引掩码
            cls_features: 可选的 CLS 特征

        返回:
            out: 字典，包含:
              - 'prompted_embedding': 拼接了 prompt 的完整输入 [B, prompt_len + N, C]
              - 'total_prompt_len': prompt 的总 token 数
              - 'prompt_idx': 选中的 prompt 索引
              - 'similarity': 相似度矩阵
              - 'reduce_sim': 用于损失计算的标量
        """
        out = dict()
        if self.prompt_pool:
            # === 步骤 1: 计算查询 embedding ===
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)  # [B, C]
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            # === 步骤 2: 计算相似度并选择 top-k prompt ===
            prompt_norm = self.l2_normalize(self.prompt_key, dim=1)  # [Pool_size, C]
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # [B, C]

            # 余弦相似度: matmul(BxC, CxPs) -> [B, Pool_size]
            similarity = torch.matmul(x_embed_norm, prompt_norm.t())

            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)  # [B, top_k]

                # 批次级 prompt 选择
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k)
                    major_prompt_id = prompt_id[major_idx]
                    idx = major_prompt_id.expand(x_embed.shape[0], -1)  # [B, top_k]
            else:
                idx = prompt_mask  # [B, top_k]

            # === 步骤 3: 获取选中的 prompt 并拼接到输入序列 ===
            batched_prompt_raw = self.prompt[idx]  # [B, top_k, length, C]
            batch_size, top_k, length, c = batched_prompt_raw.shape
            # 将 top_k 个 prompt 拼接成一个长序列: [B, top_k * length, C]
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)

            out['prompt_idx'] = idx
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # === 步骤 4: 计算 pull_constraint 损失 ===
            batched_key_norm = prompt_norm[idx]  # [B, top_k, C]
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # [B, 1, C]
            sim = batched_key_norm * x_embed_norm  # [B, top_k, C]
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # 标量

            out['reduce_sim'] = reduce_sim
        else:
            # === 不使用 prompt 池: 全局统一 prompt ===
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        # === 步骤 5: 拼接到输入序列 ===
        # 输入变为: [prompt, original_tokens] 即 [B, prompt_len + N, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out
