"""
Sparse Geometry-Aware MoE (Sparse Group Selection + Intra-Group Expert Routing)
===============================================================================

Core design:
  1. GroupBank: 4 fixed geometric groups (Identity, SO, LR, Affine)
  2. Sparse Group Selection: top-k groups selected via z-score corrected scores
     s_g = p(g|h) - beta * z_score_g  →  top-k  →  re-normalize
  3. Intra-group Expert Routing: within each selected group, choose experts
  4. Shared MambaFlow: shared semantic transport (not expanded)
  5. Group-aware AE/RD: per-group anomaly detection → z-score
  6. Intra-group Expert Expansion: only within the selected group

Key difference from sema_geometry_moe.py:
  - SPARSE group selection (top-k) instead of soft mixture of all 4 groups
  - z-score DIRECTLY fed back to group selection scores
  - Only selected groups participate in expert routing and output

Data flow:
  z = W_down LN(h)                         -- bottleneck projection
  s_g = group_logits - beta * z_score_g    -- z-score corrected scores
  G_selected = top-k(s_g)                  -- sparse group selection
  z^G = sum_{g in G_selected} w_g * T_g(z) -- selected-group MoE mix
  m = SharedMambaFlow(z^G)                 -- shared semantic flow
  a = W_up m                               -- output projection
  h_out = h + gamma * a                    -- residual connection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math
import logging


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Selective Scan — Mamba S6 core operator
# ═══════════════════════════════════════════════════════════════════════════════

def selective_scan(u, delta, A, B, C, D):
    """Selective SSM scan (sequential implementation, O(L) complexity).

    Implements S6 state space model recurrence:
      h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
      y_t = C_t * h_t + D * u_t
    """
    Bsz, L, D = u.shape
    N = A.shape[1]

    deltaA = torch.exp(delta.unsqueeze(-1) * A)
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(-2) * u.unsqueeze(-1)

    h = torch.zeros(Bsz, D, N, device=u.device, dtype=u.dtype)
    ys = []
    for i in range(L):
        h = deltaA[:, i] * h + deltaB_u[:, i]
        y_i = (h * C[:, i].unsqueeze(-2)).sum(dim=-1)
        ys.append(y_i)

    y = torch.stack(ys, dim=1) + u * D
    return y


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SharedMambaFlow — shared semantic flow operator (not expanded)
# ═══════════════════════════════════════════════════════════════════════════════

class SharedMambaFlow(nn.Module):
    """Shared Mamba semantic flow — stable transport across tasks.

    After sparse group mixing, models token-wise semantic state evolution.
    Shared across all tasks, not expanded.
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        inner_dim = dim * expand

        self.in_proj = nn.Linear(dim, inner_dim * 2)
        self.conv1d = nn.Conv1d(
            inner_dim, inner_dim, d_conv,
            groups=inner_dim, padding=d_conv - 1,
        )
        dt_rank = max(1, math.ceil(dim / 16))
        self.x_proj = nn.Linear(inner_dim, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Sequential(
            nn.Linear(dt_rank, inner_dim),
            nn.Softplus(),
        )

        A = torch.empty(inner_dim, d_state)
        for i in range(d_state):
            A[:, i] = -((i + 1) ** 0.5) * torch.ones(inner_dim)
        self.A_log = nn.Parameter(torch.log(-A))
        self.D = nn.Parameter(torch.ones(inner_dim))

        self.out_proj = nn.Linear(inner_dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.in_proj.bias)
        nn.init.kaiming_uniform_(self.x_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        B, L, D = x.shape

        x_and_res = self.in_proj(x)
        x_ssm, gate = x_and_res.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))
        x_conv = x_conv[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)

        ssm_params = self.x_proj(x_conv)
        dt_rank = self.x_proj.out_features - self.d_state * 2
        dt, B_ssm, C_ssm = ssm_params.split(
            [dt_rank, self.d_state, self.d_state], dim=-1)

        delta = self.dt_proj(dt)
        A = -torch.exp(self.A_log)

        y = selective_scan(x_conv, delta, A, B_ssm, C_ssm, self.D)
        y = y * F.silu(gate)

        out = self.out_proj(y)
        out = out + self.gamma * x
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SimpleAdapter — SEMA-style bottleneck MLP for shallow/mid layers
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleAdapter(nn.Module):
    """SEMA-identical bottleneck MLP: ReLU(down(x)) → up.

    Used in shallow/mid ViT layers (0-8) for baseline adaptation capacity.
    Follows EXACTLY sema_components.Adapter.forward(): no output scaling.
    """
    def __init__(self, d_model=768, bottleneck=16):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.up_proj = nn.Linear(bottleneck, d_model)

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        return self.up_proj(F.relu(self.down_proj(x)))


# ═══════════════════════════════════════════════════════════════════════════════
# 3.5 Stiefel / Grassmann utilities (Level 1+2: geometric fiber)
# ═══════════════════════════════════════════════════════════════════════════════

def stiefel_retraction(U):
    """Project U back onto Stiefel manifold St(d,D): U Σ V^T → U V^T."""
    _U, _, _Vh = torch.linalg.svd(U.float(), full_matrices=False)
    return (_U @ _Vh).to(U.dtype)

def stiefel_loss(down):
    """L_stiefel = ||down^T down - I||_F^2."""
    return torch.norm(down @ down.T - torch.eye(down.shape[0], device=down.device), p='fro') ** 2

def grassmann_geodesic_perturb(down, eps=0.01):
    """Perturb 'down' along a random Grassmann geodesic. eps relative to ||down||_F."""
    D, d = down.shape[1], down.shape[0]
    Δ = torch.randn_like(down)
    Δ = Δ - down @ (down.T @ Δ + Δ.T @ down) / 2  # project to tangent space
    scale = eps * torch.norm(down, p='fro') / (torch.norm(Δ, p='fro') + 1e-8)
    Δ = scale * Δ
    return stiefel_retraction(down + Δ)

def fiber_angles(down_a, down_b):
    """Canonical angles between two Stiefel fibers: θ_i = arccos(σ_i(U @ V^T)).
    Returns max and mean in radians. Large max → well-separated fibers."""
    U, V = down_a, down_b  # both [d, D] with orthonormal rows
    overlap = U @ V.T  # [d, d]
    _, sigma, _ = torch.linalg.svd(overlap.float(), full_matrices=False)
    angles = torch.acos(sigma.clamp(0, 1))
    return angles.max().item(), angles.mean().item()
    return angles.max().item(), angles.mean().item()

def parallel_transport_so(R_old, down_old, down_new):
    """Parallel transport SO(r) matrix from fiber(down_old) to fiber(down_new)."""
    O = down_new @ down_old.T  # [d, d]  fiber overlap
    U, _, Vh = torch.linalg.svd(O.float(), full_matrices=False)
    Q = (U @ Vh).to(R_old.dtype)  # optimal rotation between fibers
    return Q @ R_old @ Q.T  # conjugate by Q

def parallel_transport_lr(A_old, B_old, down_old, down_new):
    """Parallel transport LR factors via the fiber overlap."""
    O = down_new @ down_old.T
    A_new = O @ A_old
    B_new = B_old @ O.T
    return A_new, B_new

def parallel_transport_affine(W_old, b_old, down_old, down_new):
    """Parallel transport Affine parameters."""
    O = down_new @ down_old.T
    W_new = O @ W_old @ O.T
    b_new = O @ b_old
    return W_new, b_new

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Group Experts — geometric group-specific experts
# ═══════════════════════════════════════════════════════════════════════════════

class IdentityExpert(nn.Module):
    """Identity expert: T_ID(z) = z (pass-through, preserves input signal).

    Changed from zero-output to pass-through because:
      - z_G ≈ 0 kills gradient signal for up_proj.weight
      - pass-through gives up_proj meaningful token-dependent features
      - with router bias toward Identity at init, adapter ≈ simple bottleneck MLP
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, z):
        return z


class SOExpert(nn.Module):
    """SO(r) rotation expert: z_SO = z @ R, R encouraged toward SO(r)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.R = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)

    def forward(self, z):
        return z @ self.R

    def orthogonality_error(self):
        RTR = self.R.T @ self.R
        eye = torch.eye(self.dim, device=self.R.device)
        return torch.norm(RTR - eye, p='fro') ** 2


class LRExpert(nn.Module):
    """Low-rank expert: z_LR = z + z @ A @ B, A in R^{r×k}, B in R^{k×r}."""
    def __init__(self, dim, rank=None):
        super().__init__()
        self.dim = dim
        self.rank = rank or max(1, dim // 4)
        self.A = nn.Parameter(torch.randn(dim, self.rank) * 0.01 / math.sqrt(self.rank))
        self.B = nn.Parameter(torch.randn(self.rank, dim) * 0.01 / math.sqrt(self.rank))

    def forward(self, z):
        return z + z @ self.A @ self.B


class AffineExpert(nn.Module):
    """Affine expert: z_Affine = z @ W + b."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        return z @ self.W + self.b


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GroupBank — fixed group types + expandable group-specific experts
# ═══════════════════════════════════════════════════════════════════════════════

class GroupBank(nn.Module):
    """Fixed geometric group bank with expandable experts per group.

    Groups: Identity (non-expandable), SO (expandable), LR (expandable), Affine (expandable).
    """

    def __init__(self, dim, expandable_groups=('SO', 'LR', 'Affine')):
        super().__init__()
        self.dim = dim
        self.expandable_groups = expandable_groups

        self._expert_factory = {
            'Identity': lambda: IdentityExpert(dim),
            'SO':       lambda: SOExpert(dim),
            'LR':       lambda: LRExpert(dim),
            'Affine':   lambda: AffineExpert(dim),
        }

        self.groups: Dict[str, nn.ModuleList] = nn.ModuleDict()
        for group_name in ['Identity', 'SO', 'LR', 'Affine']:
            self.groups[group_name] = nn.ModuleList()
            self._add_expert_to_group(group_name)

    def _add_expert_to_group(self, group_name):
        expert = self._expert_factory[group_name]()
        existing = self.groups[group_name]
        if len(existing) > 0:
            target_device = next(existing[0].parameters()).device
            expert = expert.to(target_device)
            # Mean init + small noise: warm start from collective knowledge
            with torch.no_grad():
                for p_new, p_name in zip(expert.parameters(),
                                         expert.state_dict().keys()):
                    p_avg = torch.stack([
                        dict(e.named_parameters())[p_name].data
                        for e in existing
                    ]).mean(dim=0)
                    noise = torch.randn_like(p_avg) * 0.01 * p_avg.std()
                    p_new.copy_(p_avg + noise)
        existing.append(expert)
        return expert

    def add_expert(self, group_name):
        if group_name not in self.expandable_groups:
            logging.warning(f"Group '{group_name}' is not expandable. Skipping.")
            return False
        self._add_expert_to_group(group_name)
        logging.info(
            f"Group '{group_name}' added new expert "
            f"(now {len(self.groups[group_name])} experts)"
        )
        return True

    def get_expert(self, group_name, expert_idx):
        return self.groups[group_name][expert_idx]

    def num_experts(self, group_name):
        return len(self.groups[group_name])

    def forward_group(self, group_name, z, expert_weights=None):
        """Apply group transformation with optional intra-group expert mixing."""
        experts = self.groups[group_name]

        if expert_weights is None or len(experts) == 1:
            return experts[-1](z), None

        expert_outs = []
        for expert in experts:
            expert_outs.append(expert(z))

        stacked = torch.stack(expert_outs, dim=0)  # [E, B, N, dim]
        w = expert_weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [E, B, 1, 1]
        combined = (stacked * w).sum(dim=0)  # [B, N, dim]
        return combined, expert_outs

    def orthogonality_error(self):
        err = 0.0
        for expert in self.groups['SO']:
            err = err + expert.orthogonality_error()
        return err

    def deviation_penalty(self):
        """Penalize deviation from pass-through, weighted by param count.
        Prevents high-capacity groups (esp. Affine) from dominating z-score.
        """
        penalty = 0.0
        I = torch.eye(self.dim, device=next(self.parameters()).device)
        for expert in self.groups['SO']:
            penalty = penalty + torch.norm(expert.R - I, p='fro') ** 2
        for expert in self.groups['LR']:
            penalty = penalty + torch.norm(expert.A, p='fro') ** 2
            penalty = penalty + torch.norm(expert.B, p='fro') ** 2
        for expert in self.groups['Affine']:
            penalty = penalty + torch.norm(expert.W - I, p='fro') ** 2
            penalty = penalty + torch.norm(expert.b, p='fro') ** 2
        return penalty


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SparseGroupRouter — Sparse group selection + intra-group expert routing
# ═══════════════════════════════════════════════════════════════════════════════

class SparseGroupRouter(nn.Module):
    """Sparse group router with z-score feedback.

    Core innovation over HierarchicalRouter:
      1. Group scores: s_g = group_logits - beta * z_score_g  (z-score correction)
      2. Sparse selection: top-k groups by s_g (not soft mixture of all)
      3. Re-normalize weights among selected groups
      4. Intra-group expert routing only for selected groups

    Router input (simplified statistical features):
      router_input = concat(mean(tokens), z_scores, group_usage)
    """

    def __init__(self, dim, num_groups=4, beta=0.1, tau=1.0,
                 top_k=2, router_hidden=None):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        self.beta = beta
        self.tau = tau
        self.top_k = top_k  # number of groups to select

        router_input_dim = dim + num_groups * 2  # mean(D) + z_scores(G) + group_usage(G)
        router_hidden = router_hidden or max(dim // 16, 32)  # compact hidden

        self.group_router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_groups),
        )

        self.expert_routers = nn.ModuleDict()
        self._init_weights()

    def _init_weights(self):
        last = self.group_router[-1]
        nn.init.trunc_normal_(last.weight, std=0.02)
        # Bias Identity slightly: all groups start near pass-through, soft preference
        with torch.no_grad():
            last.bias.copy_(torch.tensor([0.5, 0.0, 0.0, 0.0]))

    def ensure_expert_router(self, group_name, num_experts):
        router_input_dim = self.group_router[0].in_features
        router_hidden = self.group_router[0].out_features

        if group_name not in self.expert_routers:
            router = nn.Sequential(
                nn.Linear(router_input_dim, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, num_experts),
            )
            self.expert_routers[group_name] = router
            nn.init.trunc_normal_(router[-1].weight, std=0.02)
            nn.init.zeros_(router[-1].bias)
        else:
            old_router = self.expert_routers[group_name]
            old_output = old_router[-1]
            old_num = old_output.out_features
            if num_experts > old_num:
                new_output = nn.Linear(
                    old_output.in_features, num_experts,
                    device=old_output.weight.device,
                )
                nn.init.trunc_normal_(new_output.weight, std=0.02)
                nn.init.zeros_(new_output.bias)
                with torch.no_grad():
                    new_output.weight.data[:old_num] = old_output.weight.data
                    new_output.bias.data[:old_num] = old_output.bias.data
                new_output.weight.requires_grad_(True)
                new_output.bias.requires_grad_(True)

                def _zero_old_grad(grad):
                    grad[:old_num] = 0
                    return grad
                new_output.weight.register_hook(_zero_old_grad)
                new_output.bias.register_hook(_zero_old_grad)

                old_router[-1] = new_output
                logging.info(
                    f"ExpertRouter '{group_name}': {old_num} -> {num_experts} experts"
                )

    def _build_router_input(self, x, z_scores=None, group_usage=None):
        B = x.shape[0]

        mean_tok = x.mean(dim=1)  # [B, D]

        parts = [mean_tok]

        if z_scores is not None:
            parts.append(z_scores)
        else:
            parts.append(torch.zeros(B, self.num_groups, device=x.device))

        if group_usage is not None:
            parts.append(group_usage)
        else:
            parts.append(torch.zeros(B, self.num_groups, device=x.device))

        return torch.cat(parts, dim=-1)

    def forward(self, x, z_scores=None, group_usage=None,
                group_expert_counts=None):
        """Sparse group routing with z-score feedback.

        1. Compute group scores: s_g = logits - beta * z_score_g
        2. Select top-k groups, mask out rest
        3. Re-normalize group weights among selected
        4. Intra-group expert routing for selected groups

        Returns:
            sparse_group_probs: [B, G] sparse re-normalized group probabilities
            expert_probs: {group_name: [B, num_experts]}
            selected_mask: [B, G] boolean mask of selected groups
        """
        router_input = self._build_router_input(x, z_scores, group_usage)

        # Step 1: Group logits with z-score correction
        group_logits = self.group_router(router_input)  # [B, G]

        if z_scores is not None:
            # s_g = logits_g - beta * z_score_g
            # High z-score → less likely to select this group
            group_logits = group_logits - self.beta * z_scores.detach()

        # Step 2: Sparse top-k selection
        # Get actual k (capped at num_groups)
        k = min(self.top_k, self.num_groups)
        top_k_vals, top_k_indices = torch.topk(group_logits, k, dim=-1)  # [B, k]

        # Create selection mask
        selected_mask = torch.zeros_like(group_logits)  # [B, G]
        selected_mask.scatter_(-1, top_k_indices, 1.0)

        # Step 3: Re-normalize probabilities among selected groups
        # Set non-selected logits to -inf before softmax
        masked_logits = group_logits.masked_fill(selected_mask == 0, float('-inf'))
        sparse_group_probs = F.softmax(masked_logits / self.tau, dim=-1)  # [B, G]

        # Step 4: Intra-group expert routing (only for selected groups)
        expert_probs = {}
        group_names_order = ['Identity', 'SO', 'LR', 'Affine']

        for gn in group_names_order:
            num_exp = (
                group_expert_counts.get(gn, 1)
                if group_expert_counts else 1
            )

            if gn in self.expert_routers and num_exp > 1:
                router = self.expert_routers[gn]
                expert_logits = router(router_input)  # [B, num_exp]
                expert_probs[gn] = F.softmax(expert_logits, dim=-1)
            else:
                expert_probs[gn] = torch.ones(
                    group_logits.shape[0], 1, device=x.device
                )

        return sparse_group_probs, expert_probs, selected_mask.bool()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. GroupAwareAE — per-group autoencoder for anomaly detection
# ═══════════════════════════════════════════════════════════════════════════════

class GroupAwareAE(nn.Module):
    """Per-group autoencoder for distribution shift detection.

    Each group maintains independent encoder-decoder pairs.
    Reconstruction error → z-score → triggers expansion when too high.
    """

    def __init__(self, dim, rd_dim=None, num_groups=4):
        super().__init__()
        self.dim = dim
        self.rd_dim = rd_dim or max(dim // 4, 4)
        self.num_groups = num_groups

        self.encoders = nn.ModuleList([
            nn.Linear(dim, self.rd_dim) for _ in range(num_groups)
        ])
        self.decoders = nn.ModuleList([
            nn.Linear(self.rd_dim, dim) for _ in range(num_groups)
        ])
        self._init_weights()

    def _init_weights(self):
        for enc, dec in zip(self.encoders, self.decoders):
            nn.init.kaiming_uniform_(enc.weight, a=math.sqrt(5))
            nn.init.zeros_(enc.bias)
            nn.init.kaiming_uniform_(dec.weight, a=math.sqrt(5))
            nn.init.zeros_(dec.bias)

    def forward(self, z, group_idx=None):
        if group_idx is not None:
            encoded = self.encoders[group_idx](z)
            return self.decoders[group_idx](encoded)
        else:
            reconstructions = []
            for g in range(self.num_groups):
                encoded = self.encoders[g](z)
                reconstructions.append(self.decoders[g](encoded))
            return torch.stack(reconstructions, dim=0)

    def compute_group_rd_loss(self, z, group_probs):
        """Compute group-weighted reconstruction loss.

        L_group_RD = sum_g w_g * MSE(AE_g(z), z)

        Only selected groups (non-zero weight) contribute meaningfully.
        """
        B, D = z.shape
        G = self.num_groups

        all_reconstructions = self.forward(z)  # [G, B, D]

        per_group_loss = torch.zeros(B, G, device=z.device)
        for g in range(G):
            per_group_loss[:, g] = F.mse_loss(
                all_reconstructions[g], z, reduction='none'
            ).mean(dim=-1)

        group_rd_loss = (per_group_loss * group_probs).sum(dim=-1)

        return group_rd_loss, per_group_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RunningRecords — per-group running statistics buffer
# ═══════════════════════════════════════════════════════════════════════════════

class RunningRecords:
    """Online running statistics — per-group RD error mean and stddev.

    Z-score = |current_error - historical_mean| / historical_stddev
    """

    def __init__(self, max_len=500):
        self._max_len = max_len
        self._curr_len = 0
        self.record = torch.zeros(max_len)
        self._mean = 0.0
        self._var = 0.0
        self.updating = True

    @property
    def length(self):
        return self._curr_len

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return math.sqrt(max(self._var, 1e-8))

    def add_record(self, v):
        if not self.updating:
            return
        v = v.detach().cpu()
        if self._curr_len < self._max_len:
            place_left = self._max_len - self._curr_len
            if place_left > len(v):
                self.record[self._curr_len:self._curr_len + len(v)] = v
                self._curr_len += len(v)
            else:
                self.record[self._curr_len:] = v[:place_left]
                self._curr_len = self._max_len
        else:
            self.record = torch.cat([self.record, v])
            self.record = self.record[len(v):]

        self._mean = float(torch.mean(self.record[:self._curr_len]))
        self._var = float(torch.var(self.record[:self._curr_len])) if self._curr_len >= 2 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SparseGroupMoEAdapter — Complete sparse group MoE adapter
# ═══════════════════════════════════════════════════════════════════════════════

class SparseGroupMoEAdapter(nn.Module):
    """Sparse Geometry-Aware MoE Adapter with top-k group selection.

    Full data flow:
      1. z = W_down LN(x)                        -- bottleneck projection
      2. RD z-scores from GroupAwareAE            -- anomaly signal
      3. s_g = logits_g - beta * z_score_g       -- corrected scores
      4. G_sel = top-k(s_g)                      -- sparse selection
      5. w_g = softmax(s_g | g in G_sel)          -- re-normalized weights
      6. z^G = sum_{g in G_sel} w_g * T_g(z)     -- selected-group MoE mix
      7. m = MambaFlow(z^G)                      -- shared semantic flow
      8. a = W_up(m)                             -- output projection
    """

    def __init__(self, config, layer_id, adapter_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.adapter_id = adapter_id

        d_model = config.d_model
        bottleneck = getattr(config, 'ffn_num', 16)
        num_groups = getattr(config, 'num_geo_groups', 4)

        # Stiefel fiber: down ∈ St(bottleneck, d_model), up = down^T
        self.down_proj = nn.Linear(d_model, bottleneck, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.mamba_beta = nn.Parameter(torch.tensor(0.01))

        # GroupBank: 4 groups (Identity, SO, LR, Affine)
        self.group_bank = GroupBank(bottleneck)

        # Shared MambaFlow
        mamba_cfg = {
            'd_state': getattr(config, 'mamba_d_state', 16),
            'd_conv': getattr(config, 'mamba_d_conv', 4),
            'expand': getattr(config, 'mamba_expand', 2),
        }
        self.mamba_flow = SharedMambaFlow(bottleneck, **mamba_cfg)

        self.group_names = ['Identity', 'SO', 'LR', 'Affine']
        self.group_name_to_idx = {n: i for i, n in enumerate(self.group_names)}
        self.idx_to_group_name = {i: n for n, i in self.group_name_to_idx.items()}

        # Sparse Group Router
        self.router = SparseGroupRouter(
            d_model, num_groups=num_groups,
            beta=getattr(config, 'router_beta', 0.1),
            tau=getattr(config, 'router_tau', 1.0),
            top_k=getattr(config, 'sparse_top_k', 2),
        )
        for gn in ['SO', 'LR', 'Affine']:
            self.router.ensure_expert_router(gn, 1)

        # Group-aware AE/RD (only in deep layers)
        self.not_addition_layer = (
            layer_id < config.adapt_start_layer
            or layer_id > config.adapt_end_layer
        )
        if not self.not_addition_layer:
            self.group_ae = GroupAwareAE(
                bottleneck,
                rd_dim=getattr(config, 'rd_dim', 128),
                num_groups=num_groups,
            )
            self.per_group_records: List[RunningRecords] = [
                RunningRecords(max_len=getattr(config, 'buffer_size', 500))
                for _ in range(num_groups)
            ]
        else:
            self.group_ae = None
            self.per_group_records = None

        self.newly_added = True
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        self.down_proj.weight.data = stiefel_retraction(self.down_proj.weight.data)

    def _up(self):
        """up = down^T (Stiefel: orthogonal projection)."""
        return self.down_proj.weight.T  # [bottleneck, d_model]

    def stiefel_project(self):
        """Project down back onto Stiefel after gradient step."""
        with torch.no_grad():
            self.down_proj.weight.data = stiefel_retraction(
                self.down_proj.weight.data)

    def stiefel_penalty(self):
        """L_stiefel = ||down @ down^T - I||²."""
        return stiefel_loss(self.down_proj.weight)

    def _get_group_expert_counts(self):
        return {
            gn: self.group_bank.num_experts(gn)
            for gn in ['SO', 'LR', 'Affine']
        }

    def _get_group_usage(self):
        usage = torch.zeros(len(self.group_name_to_idx))
        for gn in ['SO', 'LR', 'Affine']:
            idx = self.group_name_to_idx[gn]
            usage[idx] = float(self.group_bank.num_experts(gn))
        return usage

    def _compute_z_scores(self, per_group_loss, detach=True):
        """Batch-level z-score: stronger signal for distributional shift detection."""
        B, G = per_group_loss.shape
        batch_mean_loss = per_group_loss.mean(dim=0)  # [G] batch-average RD loss
        z_scores = torch.zeros(B, G, device=per_group_loss.device)
        for g in range(G):
            rec = self.per_group_records[g]
            if rec.length > 2:
                mean = rec.mean
                std = rec.stddev
                loss_g = batch_mean_loss[g].detach() if detach else batch_mean_loss[g]
                z_val = torch.abs((loss_g - mean) / std)
                z_scores[:, g] = z_val  # broadcast to all samples
        return z_scores

    def forward(self, x, compute_rd=True):
        """Sparse Group-MoE adapter forward pass.

        Args:
            x:          [B, N, d_model] ViT block output
            compute_rd: whether to compute RD (skip in func phase for speed)

        Returns:
            func_out:       [B, N, d_model] adapter output
            group_rd_loss:  scalar group-weighted RD loss
            z_scores:       [B, G] per-group z-score
            group_probs:    [B, G] sparse group probabilities
            expert_probs:   dict per-group expert probabilities
            selected_mask:  [B, G] which groups were selected
            added:          bool expansion triggered
        """
        B, N, D = x.shape

        # Step 1: Bottleneck projection
        z = self.down_proj(x)  # [B, N, r]
        z = F.relu(z)

        # Step 2: Routing preparation
        group_expert_counts = self._get_group_expert_counts()

        use_ae = (compute_rd and not self.not_addition_layer
                  and self.group_ae is not None)

        # Step 3: Pre-pass group transforms → AE → z-scores
        zd = z.detach()
        group_outputs_pre = []
        for i, gn in enumerate(self.group_names):
            g_out_pre, _ = self.group_bank.forward_group(gn, zd)
            group_outputs_pre.append(g_out_pre)

        if use_ae:
            G = len(self.group_names)
            per_group_rd = torch.zeros(B, G, device=z.device)
            uniform_probs = torch.ones(B, G, device=z.device) / G
            for i in range(G):
                g_latent = group_outputs_pre[i].detach().mean(dim=1)
                _, pg = self.group_ae.compute_group_rd_loss(
                    g_latent, uniform_probs
                )
                per_group_rd[:, i] = pg[:, i]
            z_scores = self._compute_z_scores(per_group_rd)
            group_usage = self._get_group_usage().to(z.device).unsqueeze(0).expand(B, -1)
        else:
            z_scores = None
            group_usage = None

        # Step 4: Router WITH z-score correction (s_g = logits - beta*z_score)
        group_probs, expert_probs, selected_mask = self.router(
            x, z_scores=z_scores, group_usage=group_usage,
            group_expert_counts=group_expert_counts,
        )

        # Step 5: Group-MoE mix with corrected routing
        group_outputs = []
        for i, gn in enumerate(self.group_names):
            g_out, _ = self.group_bank.forward_group(
                gn, z, expert_weights=expert_probs.get(gn)
            )
            group_outputs.append(g_out)

        stacked = torch.stack(group_outputs, dim=0)  # [G, B, N, r]
        w = group_probs.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [G, B, 1, 1]
        z_G = (stacked * w).sum(dim=0)  # [B, N, r]

        # Step 6: Shared MambaFlow with external residual
        m = z_G + self.mamba_beta * self.mamba_flow(z_G)

        # Step 7: Output projection with adapter gate
        a = self.gamma * F.linear(m, self._up())

        # Step 8: RD loss + records on POST-group latent
        added = False
        group_rd_loss = torch.tensor(0.0, device=x.device)
        z_scores_out = z_scores if z_scores is not None else \
                       torch.zeros(B, len(self.group_names), device=x.device)

        if use_ae and self.training:
            # RD loss: weighted sum over groups using group_probs
            group_rd_loss = (per_group_rd * group_probs).sum(dim=-1).mean()
            for g in range(len(self.group_names)):
                self.per_group_records[g].add_record(
                    per_group_rd[:, g].mean().unsqueeze(0))

        return {
            "func_out": a,
            "group_rd_loss": group_rd_loss,
            "z_scores": z_scores_out,
            "group_probs": group_probs,
            "expert_probs": expert_probs,
            "selected_mask": selected_mask,
            "z_G": z_G,  # [B, N, r] pre-MambaFlow bottleneck (for shared MambaFlow)
            "group_transforms": stacked,
            "added": added,
        }

    def init_from_adapter(self, other, eps=0.05):
        """Geodesic perturbation on down only. GroupBank stays at default init.

        - down_new = geodesic_perturb(down_old)   (Grassmann geodesic)
        - GroupBank: default near-pass-through     (fresh start, like Task 0)
        """
        with torch.no_grad():
            down_old = other.down_proj.weight.data
            self.down_proj.weight.data.copy_(
                grassmann_geodesic_perturb(down_old, eps=eps))

    def expand_fiber(self, k=8):
        """Expand fiber dimension: St(d,D) → St(d+k,D) with old as geodesic submanifold.

        New rows initialized orthogonal to old rows. GroupBank/GroupAwareAE expand accordingly.
        Old parameters frozen, new parameters trainable.
        """
        d_old, D = self.down_proj.weight.shape  # [d, D]
        d_new = d_old + k

        # 1. Expand down_proj: old rows frozen, new rows orthogonal
        old_down = self.down_proj.weight.data.clone().detach()
        for param in self.down_proj.parameters():
            param.requires_grad = False

        new_down_weight = nn.Parameter(torch.zeros(k, D))
        # QR to get orthogonal complement: find k vectors ⊥ old rows
        Q, _ = torch.linalg.qr(torch.randn(k, D).to(old_down.device))
        proj = Q @ old_down.T
        Q_orth = Q - proj @ old_down  # remove projection onto old span
        U, _, _ = torch.linalg.svd(Q_orth.float(), full_matrices=False)
        new_down_weight.data.copy_(U[:k].to(old_down.dtype) * 0.1)  # small init

        self.down_proj.weight = nn.Parameter(torch.cat([old_down, new_down_weight.data], dim=0))
        self.down_proj.out_features = d_new
        self.down_proj.in_features = D

        # 2. Expand GroupBank: block-diagonal [old, 0; 0, new_I]
        for gn in ['SO', 'LR', 'Affine']:
            for expert in self.group_bank.groups[gn]:
                self._expand_group_params(expert, d_old, k, gn)

        # 3. Expand GroupAwareAE: encoders/decoders to handle d_new dims
        if self.group_ae is not None:
            rd_dim = self.group_ae.rd_dim
            for i in range(self.group_ae.num_groups):
                old_enc = self.group_ae.encoders[i]
                old_dec = self.group_ae.decoders[i]
                # Expand: add k input/output dims, zero-init new weights
                new_enc = nn.Linear(d_new, rd_dim).to(old_enc.weight.device)
                new_dec = nn.Linear(rd_dim, d_new).to(old_dec.weight.device)
                with (torch.no_grad()):
                    new_enc.weight[:, :d_old] = old_enc.weight
                    new_enc.bias.copy_(old_enc.bias)
                    new_dec.weight[:d_old, :] = old_dec.weight
                    new_dec.bias[:d_old] = old_dec.bias
                for p in old_enc.parameters(): p.requires_grad = False
                for p in old_dec.parameters(): p.requires_grad = False
                self.group_ae.encoders[i] = new_enc
                self.group_ae.decoders[i] = new_dec

        # 4. Expand MambaFlow to handle d_new dims
        old_mamba = self.mamba_flow
        for param in old_mamba.parameters():
            param.requires_grad = False
        new_mamba = SharedMambaFlow(d_new, d_state=old_mamba.d_state,
                                    d_conv=getattr(self.config, 'mamba_d_conv', 4),
                                    expand=old_mamba.expand)
        new_mamba.to(old_down.device)
        with torch.no_grad():
            # Copy old weights into new (top-left block for matrices)
            for (no, po), (nn, pn) in zip(
                    old_mamba.named_parameters(), new_mamba.named_parameters()):
                if po.dim() >= 2 and po.shape == pn.shape[:po.dim()]:
                    # Same shape (e.g. bias-like) — copy directly
                    if po.shape == pn.shape:
                        pn.copy_(po)
        self.mamba_flow = new_mamba

        logging.info(
            f"Fiber expanded: St({d_old},{D}) → St({d_new},{D}) "
            f"(old {d_old} rows frozen, new {k} rows trainable)"
        )

    def _expand_group_params(self, expert, d_old, k, gn):
        """Block-diagonal expansion: old params in top-left, identity in bottom-right."""
        I_k = torch.eye(k, device=next(expert.parameters()).device)
        zero_ok = torch.zeros(k, d_old, device=next(expert.parameters()).device)
        for name, param in list(expert.named_parameters()):
            param.requires_grad = False
            old_val = param.data
            if 'R' in name:  # SO: R [d,d]
                new_val = torch.block_diag(old_val, I_k)
            elif 'A' in name:  # LR.A [d, r] — expand to [d+k, r+k//2]
                r = old_val.shape[1]
                rk = max(1, r * k // d_old)
                new_val = torch.zeros(d_old + k, r + rk, device=old_val.device)
                new_val[:d_old, :r] = old_val
                new_val[d_old:, r:] = I_k[:k, :rk] * 0.1
            elif 'B' in name:  # LR.B [r, d]
                r = old_val.shape[0]
                rk = max(1, r * k // d_old)
                new_val = torch.zeros(r + rk, d_old + k, device=old_val.device)
                new_val[:r, :d_old] = old_val
                new_val[r:, d_old:] = I_k[:rk, :k] * 0.1
            elif 'W' in name:  # Affine.W [d,d]
                new_val = torch.block_diag(old_val, I_k)
            elif 'b' in name:  # Affine.b [d]
                new_val = torch.cat([old_val, torch.zeros(k, device=old_val.device)])
            else:
                continue
            new_param = nn.Parameter(new_val)
            setattr(expert, name, new_param)

    def add_expert_to_group(self, group_name):
        success = self.group_bank.add_expert(group_name)
        if success:
            new_count = self.group_bank.num_experts(group_name)
            self.router.ensure_expert_router(group_name, new_count)
        return success

    def orthogonality_error(self):
        return self.group_bank.orthogonality_error()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SparseGroupMoEModules — Layer manager with sparse group expansion
# ═══════════════════════════════════════════════════════════════════════════════

class SparseGroupMoEModules(nn.Module):
    """Layer-level manager for Sparse Group-MoE adapters.

    Key differences from GeometrySEMAModules:
      - Uses SparseGroupMoEAdapter (top-k group selection)
      - Expansion is group-specific expert level
      - MambaFlow shared, not expanded
      - Multi-batch persistence detection for stable expansion
      - Only selected groups trigger expansion

    Expansion detection:
      1. Per-sample, compute per-group z-score
      2. Find the selected group with highest z-score
      3. If max_z > threshold persistently: add expert to that group
      4. Max 1 expansion per task per layer
    """

    def __init__(self, config, layer_id, writer=None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.writer = writer
        self.adapt_start_layer = config.adapt_start_layer
        self.adapt_end_layer = config.adapt_end_layer

        self.detecting_outlier = False
        self.added_for_task = False
        self.newly_added = True

        # Multi-batch persistence detection
        expansion_patience = getattr(config, 'expansion_patience', 3)
        self._z_score_accum = torch.zeros(4)
        self._z_score_count = 0
        self._expansion_patience = expansion_patience
        self._expansion_candidate = None

        # Adaptive threshold: EMA of historical max z-scores
        self._z_ema = 0.0      # EMA of per-task max z-scores
        self._z_var = 1.0      # EMA of squared diffs
        self._ema_decay = 0.05  # slow decay for long-term trend

        # Router entropy baseline (three-signal expansion detection)
        self._entropy_ema = 0.0
        self._entropy_decay = 0.01

        # FiberRouter: simple linear + learnable temperature (enough for fiber selection)
        self.fiber_router = nn.Linear(getattr(config, 'd_model', 768), 1)
        nn.init.zeros_(self.fiber_router.weight)
        nn.init.zeros_(self.fiber_router.bias)
        self.fiber_tau = nn.Parameter(torch.tensor(1.0))

        # Shared MambaFlow across fibers (semantic alignment after fiber mixing)
        bottleneck = getattr(config, 'ffn_num', 16)
        mamba_cfg = {'d_state': getattr(config, 'mamba_d_state', 16),
                     'd_conv': getattr(config, 'mamba_d_conv', 4),
                     'expand': getattr(config, 'mamba_expand', 2)}
        self.shared_mamba = SharedMambaFlow(bottleneck, **mamba_cfg)
        self.mamba_beta = nn.Parameter(torch.tensor(0.01))

        # Multi-fiber: each adapter is a fiber with Stiefel + GroupBank
        self.adapters: List[SparseGroupMoEAdapter] = nn.ModuleList()
        self.add_adapter(initialize=True)

        # Adaptive dual expansion: expert (stage 0) → fiber (stage 1)
        self._expand_stage = 0
        self.expansion_count = {'SO': 0, 'LR': 0, 'Affine': 0}

    @property
    def num_adapters(self):
        return len(self.adapters)

    def _device(self):
        if len(self.adapters) > 0:
            return next(self.adapters[0].parameters()).device
        return torch.device('cpu')

    def add_adapter(self, initialize=False):
        adapter_id = len(self.adapters)
        new_adapter = SparseGroupMoEAdapter(
            self.config, self.layer_id, adapter_id=adapter_id
        ).to(self._device())
        self.newly_added = True
        self.added_for_task = True
        self.adapters.append(new_adapter)
        self._resize_fiber_router()
        if not initialize:
            logging.info(f"Fiber {self.layer_id}.{adapter_id} added")

    def add_fiber(self, z_score=1.0):
        """Add new fiber. eps ∝ z_score: small shift for same-domain, large for cross."""
        if len(self.adapters) == 0:
            self.add_adapter(initialize=True)
            return
        self.add_adapter(initialize=False)
        new_fiber = self.adapters[-1]
        old_fiber = self.adapters[-2]
        eps = 0.03 * max(z_score, 1.0)  # z=1→3%, z=2→6%, z=5→15% of ||down||
        eps = min(eps, 0.15)
        new_fiber.init_from_adapter(old_fiber, eps=eps)
        logging.info(f"Fiber {self.layer_id}.{len(self.adapters)-1} "
                      f"eps={eps:.3f} (z={z_score:.2f})")
        logging.info(
            f"Fiber {self.layer_id}.{len(self.adapters)-1} "
            f"geodesic init from fiber {self.layer_id}.{len(self.adapters)-2}")

    def _resize_fiber_router(self):
        n = len(self.adapters)
        old_n = self.fiber_router.out_features
        if old_n < n:
            old = self.fiber_router
            new_r = nn.Linear(old.in_features, n, device=old.weight.device)
            with torch.no_grad():
                new_r.weight[:old_n] = old.weight
                new_r.bias[:old_n] = old.bias
            def _freeze_old_cols(grad):
                grad[:old_n] = 0.0
                return grad
            new_r.weight.register_hook(_freeze_old_cols)
            new_r.bias.register_hook(_freeze_old_cols)
            self.fiber_router = new_r

    def add_expert_to_group(self, group_name):
        """Add new expert to specified group in the latest adapter.

        After expansion, precisely unfreeze:
          - New expert params: trainable
          - Router new columns: trainable (old columns gradient-zeroed via hook)
          - This group's AE encoder/decoder: unfrozen
          - Other groups' AE: frozen
          - down_proj/up_proj/MambaFlow: frozen (shared components)
        """
        if self.adapters:
            adapter = self.adapters[-1]
            success = adapter.add_expert_to_group(group_name)
            if success:
                if adapter.group_ae is not None:
                    group_idx = adapter.group_name_to_idx.get(group_name)
                    if group_idx is not None and group_idx < len(adapter.group_ae.encoders):
                        for param in adapter.group_ae.encoders[group_idx].parameters():
                            param.requires_grad = True
                        for param in adapter.group_ae.decoders[group_idx].parameters():
                            param.requires_grad = True
                        logging.info(
                            f"Unfroze group_ae[{group_name}] for RD re-training"
                        )
            return success
        return False

    def forward(self, x, group_info=None):
        """Forward pass: sparse group MoE + expansion detection.

        Args:
            x:          [B, N, D] input token sequence
            group_info:  group position info (optional)

        Returns:
            dict with func_out, group_rd_loss, z_scores, group_probs,
                 expert_probs, selected_mask, added
        """
        zero = torch.tensor(0.0, device=x.device)
        not_addition_layer = (
            self.layer_id < self.adapt_start_layer
            or self.layer_id > self.adapt_end_layer
        )

        if not_addition_layer:
            adapter_out = self.adapters[-1](x, compute_rd=False)
            return {
                "func_out": adapter_out["func_out"],
                "group_rd_loss": zero,
                "z_scores": adapter_out.get("z_scores"),
                "group_probs": adapter_out.get("group_probs"),
                "expert_probs": adapter_out.get("expert_probs"),
                "selected_mask": adapter_out.get("selected_mask"),
                "added": False,
            }

        # Deep layers: multi-fiber mix + expansion detection
        compute_rd = self.detecting_outlier or getattr(self, '_training_rd', False)
        fiber_outs = [adapter(x, compute_rd=compute_rd)
                      for adapter in self.adapters]

        # Fiber softmax mixing with z-score feedback (symmetric to group routing)
        if len(self.adapters) > 1:
            x_pool = x.mean(dim=1)
            f_logits = self.fiber_router(x_pool)  # [B, F]
            # Fiber z-score: max over groups → penalize anomalous fibers
            f_z_list = []
            for fo in fiber_outs:
                zs = fo.get("z_scores")
                if zs is not None:
                    f_z_list.append(zs.max(dim=-1)[0])  # [B]
                else:
                    f_z_list.append(torch.zeros(x.shape[0], device=x.device))
            f_z = torch.stack(f_z_list, dim=-1)  # [B, F]
            f_logits = f_logits - 0.1 * f_z.detach()
            f_weights = F.softmax(f_logits / self.fiber_tau.abs(), dim=-1)
            w_f = f_weights.unsqueeze(-1).unsqueeze(-1)
            func_mixed = sum(w_f[:, i] * fiber_outs[i]["func_out"]
                             for i in range(len(self.adapters)))
            # Shared MambaFlow on fiber-mixed bottleneck (semantic alignment)
            z_mixed = sum(w_f[:, i] * fiber_outs[i]["z_G"]
                          for i in range(len(self.adapters)))
            m_shared = z_mixed + self.mamba_beta * self.shared_mamba(z_mixed)
            func_mixed = func_mixed + 0.2 * F.linear(m_shared, self.adapters[0]._up())
        else:
            func_mixed = fiber_outs[0]["func_out"]

        # Detection uses latest fiber only
        adapter = self.adapters[-1]
        adapter_out = fiber_outs[-1]

        # Track Router entropy baseline during RD phase (model frozen, router stable)
        gp = adapter_out.get("group_probs")
        if (gp is not None and not self.detecting_outlier
                and self.training and getattr(self, '_training_rd', False)
                and len(self.adapters) == 1 and self._entropy_ema < 0.01):
            batch_entropy = -(gp * (gp + 1e-8).log()).sum(dim=-1).mean().detach().item()
            self._entropy_ema = ((1 - self._entropy_decay) * self._entropy_ema
                                 + self._entropy_decay * batch_entropy)

        # Expansion detection with multi-batch persistence
        added = False
        if self.detecting_outlier and not self.added_for_task:
            z_scores = adapter_out.get("z_scores")
            group_probs = adapter_out.get("group_probs")
            selected_mask = adapter_out.get("selected_mask")

            if z_scores is not None and group_probs is not None:
                batch_z_mean = z_scores.mean(dim=0).detach().cpu()
                self._z_score_accum = (
                    self._z_score_accum * self._z_score_count
                    + batch_z_mean
                ) / (self._z_score_count + 1)
                self._z_score_count += 1

                # Only consider expandable groups: SO, LR, Affine
                expandable_groups = ['SO', 'LR', 'Affine']
                expandable_indices = [
                    adapter.group_name_to_idx[gn]
                    for gn in expandable_groups
                ]

                # Dual-signal expansion: p(g|h) high AND z-score high
                # Only expand a group the router actually selects
                batch_group_prob = group_probs.mean(dim=0).detach().cpu()
                best_idx = max(expandable_indices,
                             key=lambda i: (batch_group_prob[i].item()
                                            * self._z_score_accum[i].item()))
                max_z = self._z_score_accum[best_idx].item()
                max_p = batch_group_prob[best_idx].item()
                best_group = adapter.idx_to_group_name[best_idx]

                # Adaptive threshold: [base×0.5, base]. EMA only lowers it.
                base_threshold = self.config.exp_threshold
                raw_adaptive = self._z_ema + 3.0 * math.sqrt(max(self._z_var, 1e-6))
                adaptive_threshold = max(min(raw_adaptive, base_threshold),
                                        base_threshold * 0.5)

                # Compute current batch entropy for three-signal check
                batch_entropy = -(gp * (gp + 1e-8).log()).sum(dim=-1).mean().item()
                entropy_ratio = (batch_entropy / max(self._entropy_ema, 0.01))
                entropy_high = (self._entropy_ema > 0 and entropy_ratio > 1.2)

                # logging.info(  # disabled: per-batch detect state

                # Multi-signal trigger (OR):
                anomaly_z = (max_z > adaptive_threshold)
                anomaly_h = entropy_high
                if (max_p > 0.15 and (anomaly_z or anomaly_h)):
                    if (self._expansion_candidate == best_group
                            and self._z_score_count >= self._expansion_patience):
                        # Adaptive dual expansion: signal-driven
                        # Expert: one group dominates (p_max - p_second > 0.15) → direction right, capacity short
                        # Fiber:   probs spread out            → whole space insufficient
                        gp_sorted = batch_group_prob.sort(descending=True)[0]
                        prob_concentrated = (gp_sorted[0] - gp_sorted[1] > 0.15)

                        if prob_concentrated and max_z < 1.5:
                            self.add_expert_to_group(best_group)
                            self.expansion_count[best_group] += 1
                            fid = len(self.adapters) - 1
                            msg = (f"expert in '{best_group}' (fiber_{fid}, z={max_z:.3f})")
                        else:
                            self.add_fiber(z_score=max_z)
                            fid = len(self.adapters) - 1
                            msg = (f"fiber_{fid} (z={max_z:.3f}, {len(self.adapters)} total)")
                        self.added_for_task = True
                        added = True
                        logging.info(
                            f"Block {self.layer_id}: Added {msg} "
                            f"H={batch_entropy:.3f}/{self._entropy_ema:.3f}")
                        self._z_ema = (1 - self._ema_decay) * self._z_ema + self._ema_decay * max_z
                        self._z_var = (1 - self._ema_decay) * self._z_var + self._ema_decay * (max_z - self._z_ema) ** 2
                        self._z_score_accum.zero_()
                        self._z_score_count = 0
                        self._expansion_candidate = None
                    else:
                        self._expansion_candidate = best_group
                else:
                    self._expansion_candidate = None

        return {
            "func_out": func_mixed,
            "group_rd_loss": adapter_out["group_rd_loss"],
            "z_scores": adapter_out.get("z_scores"),
            "group_probs": adapter_out.get("group_probs"),
            "expert_probs": adapter_out.get("expert_probs"),
            "selected_mask": adapter_out.get("selected_mask"),
            "group_transforms": adapter_out.get("group_transforms"),
            "added": added,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Task-end management
    # ═══════════════════════════════════════════════════════════════════════

    def end_of_task_training(self):
        # Update adaptive threshold EMA with this task's observed max z-score
        task_max_z = self._z_score_accum.max().item()
        if task_max_z > 0:
            self._z_ema = (1 - self._ema_decay) * self._z_ema + self._ema_decay * task_max_z
            self._z_var = (1 - self._ema_decay) * self._z_var + self._ema_decay * (task_max_z - self._z_ema) ** 2

        self.freeze_functional()
        self.freeze_rd()
        self.reset_newly_added_status()
        self.added_for_task = False
        self._expand_stage = 0
        self._z_score_accum.zero_()
        self._z_score_count = 0
        self._expansion_candidate = None

    def reset_newly_added_status(self):
        self.newly_added = False
        for adapter in self.adapters:
            adapter.newly_added = False

    def freeze_functional(self):
        for param in self.fiber_router.parameters():
            param.requires_grad = False
        self.fiber_tau.requires_grad_(False)
        for param in self.shared_mamba.parameters():
            param.requires_grad = False
        self.mamba_beta.requires_grad_(False)
        for adapter in self.adapters:
            for param in adapter.down_proj.parameters():
                param.requires_grad = False
            adapter.gamma.requires_grad_(False)
            adapter.mamba_beta.requires_grad_(False)
            for gn, experts in adapter.group_bank.groups.items():
                for expert in experts:
                    for param in expert.parameters():
                        param.requires_grad = False
            for param in adapter.mamba_flow.parameters():
                param.requires_grad = False
            for param in adapter.router.parameters():
                param.requires_grad = False

    def freeze_rd(self):
        for adapter in self.adapters:
            if adapter.group_ae is not None:
                for param in adapter.group_ae.parameters():
                    param.requires_grad = False
                if adapter.per_group_records:
                    for rec in adapter.per_group_records:
                        rec.updating = False
