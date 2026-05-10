"""
李群-纤维丛工具函数 (Lie Group & Fiber Bundle Utilities)
=========================================================

核心数学结构:
  Stiefel 流形 St(d, r) = { W ∈ R^{d×r} | W^T W = I_r }

  几何性质:
    - 齐性空间: St(d, r) ≅ SO(d) / SO(d-r)
    - 黎曼度量: 由 SO(d) 的 Killing 形式诱导
    - 测地线: 闭合形式的测地线方程存在 [Edelman et al., 1998]
    - 切空间: T_W St(d,r) = { ξ | ξ^T W + W^T ξ = 0 }

相关论文:
  - Edelman, Arias, & Smith (1998) "The Geometry of Algorithms with Orthogonality Constraints"
  - Absil, Mahony, & Sepulchre (2008) "Optimization Algorithms on Matrix Manifolds"
"""

import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  Stiefel 流形投影
# ═══════════════════════════════════════════════════════════════

def stiefel_project(W):
    """将任意矩阵投影到 Stiefel 流形 (最近点投影)。

    给定 W ∈ R^{d×r} (unconstrained), 找最近的 W' ∈ St(d,r)。
    通过 SVD 将所有奇异值设为 1 实现:

        W = U Σ V^T  →  W' = U V^T

    Args:
        W: [d, r] 任意矩阵

    Returns:
        W': [d, r] Stiefel 矩阵, 满足 W'^T W' = I_r
    """
    U, _, Vt = torch.linalg.svd(W, full_matrices=False)
    return U @ Vt


def stiefel_project_(W):
    """原地投影到 Stiefel 流形 (修改原 tensor, 不返回新 tensor)。"""
    with torch.no_grad():
        proj = stiefel_project(W.data)
        W.data.copy_(proj)


# ═══════════════════════════════════════════════════════════════
#  切空间投影
# ═══════════════════════════════════════════════════════════════

def tangent_project(W, G):
    """将欧几里得梯度 G 投影到 Stiefel 流形在 W 处的切空间。

    切空间: T_W St(d,r) = { ξ ∈ R^{d×r} | ξ^T W + W^T ξ = 0 }

    正交投影公式 [Absil et al., 2008, §3.6]:
        Π_W(G) = G - W · sym(W^T G)

    其中 sym(A) = (A + A^T) / 2

    Args:
        W: [d, r] Stiefel 点
        G: [d, r] 欧几里得梯度

    Returns:
        ξ: [d, r] 切向量 (Riemannian gradient)
    """
    sym = 0.5 * (W.T @ G + G.T @ W)    # [r, r]
    xi = G - W @ sym                     # [d, r]
    return xi


# ═══════════════════════════════════════════════════════════════
#  回缩 (Retraction)
# ═══════════════════════════════════════════════════════════════

def stiefel_retraction(W, xi, step_size):
    """Stiefel 流形上的回缩: 沿切向量 xi 走一步后拉回流形。

    使用 QR 分解回缩:
        W_{new} = qf(W + η · ξ)  取 Q 因子

    这保证 W_{new} ∈ St(d,r) 且是二阶近似的 [Absil et al., §4.1.2]。

    Args:
        W: [d, r] 当前 Stiefel 点
        xi: [d, r] 切向量
        step_size: 步长 (学习率)

    Returns:
        W_new: [d, r] 更新后仍在 Stiefel 上的点
    """
    X = W + step_size * xi          # 沿切方向走一步
    Q, _ = torch.linalg.qr(X)       # QR 分解, Q ∈ St(d,r)
    # 保证 Q 在正确的半空间 (与 W 的符号一致)
    d = torch.diag(torch.sign(torch.diag(Q.T @ W)))
    return Q @ d


# ═══════════════════════════════════════════════════════════════
#  测地线距离
# ═══════════════════════════════════════════════════════════════

def stiefel_geodesic_distance(W1, W2):
    """计算 Stiefel 流形上两点之间的测地线距离。

    使用主角度 (principal angles):

        d(W1, W2) = ||Θ||_2

    其中 Θ = arccos(σ(W1^T W2)) 是主角度向量, σ 是奇异值。

    W1, W2 ∈ St(d,r), 则 W1^T W2 ∈ R^{r×r}, 奇异值在 [0,1]。

    Args:
        W1, W2: [d, r] Stiefel 点

    Returns:
        d: 标量, 测地线距离 (≥ 0)
    """
    cross = W1.T @ W2          # [r, r]
    s = torch.linalg.svdvals(cross)   # 奇异值, 均在 [0,1]
    s = torch.clamp(s, -1.0, 1.0)
    thetas = torch.acos(s)      # 主角度
    return torch.norm(thetas)    # L2 范数


def stiefel_min_geodesic_distance(W_new, W_list):
    """计算新权重 W_new 与旧权重列表中每个 W_old 的最小测地线距离。

    用于扩展检测: 如果 min_{i} d(W_new, W_i) > τ, 则新任务与所有旧任务"太远",
    需要添加新 Adapter。

    Args:
        W_new: [d, r] 新 Adapter 的 down_proj 权重
        W_list: list of [d, r] 旧 Adapter 的 down_proj 权重

    Returns:
        d_min: 标量, 最小测地线距离
    """
    if not W_list:
        return 0.0
    dists = [stiefel_geodesic_distance(W_new, W) for W in W_list]
    return min(dists)


# ═══════════════════════════════════════════════════════════════
#  Stiefel 初始化 (随机正交矩阵)
# ═══════════════════════════════════════════════════════════════

def stiefel_init(d, r):
    """在 Stiefel 流形上均匀采样一个初始点。

    通过 QR 分解高斯随机矩阵实现 (Haar 测度均匀采样)。

    Args:
        d: 行数 (特征维度)
        r: 列数 (低秩维度)

    Returns:
        W: [d, r] Stiefel 矩阵
    """
    X = torch.randn(d, r)
    Q, _ = torch.linalg.qr(X)
    return Q
