"""
Geometry-Aware SEMA with Group-MoE and Shared Mamba Flow
=========================================================

Core components implementing the geometry-aware continual learning architecture:
- Fixed GroupBank with expandable group-specific experts
- Group types: Identity, SO, LR, Affine, MambaFlow
- Shared MambaFlow: geometry-conditioned semantic transport (not expandable)
- Hierarchical Router: GroupRouter → ExpertRouter
- Group-Aware AE/RD: per-group reconstruction descriptor for expansion detection

Architecture reference: suggest.md sections 1-10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math
import copy
import logging


# ═══════════════════════════════════════════════════════════════════════════════
# Simplified Selective Scan for MambaFlow
# ═══════════════════════════════════════════════════════════════════════════════

def selective_scan(u, delta, A, B, C, D):
    """Selective SSM scan (sequential, O(L) — replace with parallel scan for speed).

    Implements: h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
                 y_t = C_t * h_t + D * u_t

    Args:
        u:  [B, L, D]  input sequence
        delta: [B, L, D]  step size per channel
        A:    [D, N]       diagonal state matrix (N = d_state)
        B:    [B, L, N]    input projection
        C:    [B, L, N]    output projection
        D:    [D]          skip connection

    Returns:
        y: [B, L, D]  output sequence
    """
    Bsz, L, D = u.shape
    N = A.shape[1]

    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [B, L, D, N]
    deltaB_u = delta.unsqueeze(-1) * B * u.unsqueeze(-1)  # [B, L, D, N]

    h = torch.zeros(Bsz, D, N, device=u.device, dtype=u.dtype)
    ys = []
    for i in range(L):
        h = deltaA[:, i] * h + deltaB_u[:, i]
        y_i = (h * C[:, i].unsqueeze(-2)).sum(dim=-1)  # [B, D]
        ys.append(y_i)

    y = torch.stack(ys, dim=1)  # [B, L, D]
    y = y + u * D
    return y


# ═══════════════════════════════════════════════════════════════════════════════
# Shared MambaFlow (State Space Model as semantic flow operator)
# ═══════════════════════════════════════════════════════════════════════════════

class SharedMambaFlow(nn.Module):
    """Shared geometry-conditioned Mamba flow operator.

    Operates in the bottleneck latent space (r dims) after group-conditioned
    mixing. NOT expandable — treated as a stable semantic transport operator.

    Architecture: simplified S6-style selective SSM with:
    - Fixed diagonal A (HiPPO-LegS initialization)
    - Input-dependent delta (step size) for selectivity
    - Learned B, C matrices
    - Gating with SiLU and residual connection
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        inner_dim = dim * expand

        # Gating projection
        self.in_proj = nn.Linear(dim, inner_dim * 2)
        # 1D depthwise conv for local context
        self.conv1d = nn.Conv1d(
            inner_dim, inner_dim, d_conv, groups=inner_dim,
            padding=d_conv - 1,
        )
        # SSM input projection: x → (dt, B, C)
        dt_rank = max(1, math.ceil(dim / 16))
        self.x_proj = nn.Linear(inner_dim, dt_rank + d_state * 2, bias=False)
        # Delta projection: dt_rank → inner_dim
        self.dt_proj = nn.Sequential(
            nn.Linear(dt_rank, inner_dim),
            nn.Softplus(),
        )
        # Diagonal state matrix A (HiPPO-LegS init, learnable)
        A = torch.empty(inner_dim, d_state)
        for i in range(d_state):
            A[:, i] = -((i + 1) ** 0.5) * torch.ones(inner_dim)
        self.A_log = nn.Parameter(torch.log(-A))
        # Skip connection D
        self.D = nn.Parameter(torch.ones(inner_dim))
        # Output projection
        self.out_proj = nn.Linear(inner_dim, dim)
        # Residual gate
        self.gamma = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.in_proj.bias)
        nn.init.kaiming_uniform_(self.x_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """Forward pass: x → gating → conv → SSM scan → gate → residual.

        Args:
            x: [B, L, dim]  geometry-conditioned latent tokens

        Returns:
            out: [B, L, dim]  Mamba-flow processed tokens
        """
        B, L, D = x.shape

        # Gating split
        x_and_res = self.in_proj(x)  # [B, L, inner_dim * 2]
        x_ssm, gate = x_and_res.chunk(2, dim=-1)  # each [B, L, inner_dim]

        # 1D conv for local context
        x_conv = self.conv1d(
            x_ssm.transpose(1, 2)
        )  # [B, inner_dim, L + padding]
        x_conv = x_conv[:, :, :L].transpose(1, 2)  # [B, L, inner_dim]
        x_conv = F.silu(x_conv)

        # SSM parameters
        ssm_params = self.x_proj(x_conv)  # [B, L, dt_rank + d_state*2]
        dt_rank = self.x_proj.out_features - self.d_state * 2
        dt, B_ssm, C_ssm = ssm_params.split(
            [dt_rank, self.d_state, self.d_state], dim=-1
        )
        # dt: [B, L, dt_rank] → delta: [B, L, inner_dim]
        delta = self.dt_proj(dt)
        # B_ssm, C_ssm: [B, L, d_state]
        # A: [inner_dim, d_state]
        A = -torch.exp(self.A_log)  # [inner_dim, d_state]

        # Selective scan
        y = selective_scan(
            x_conv, delta, A, B_ssm, C_ssm, self.D
        )  # [B, L, inner_dim]

        # Gate
        y = y * F.silu(gate)

        # Output projection + residual
        out = self.out_proj(y)  # [B, L, dim]
        out = out + self.gamma * x
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Group Experts
# ═══════════════════════════════════════════════════════════════════════════════

class IdentityExpert(nn.Module):
    """Identity expert: T_ID(z) = 0 (zero residual, main path already has u_l)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, z):
        return torch.zeros_like(z)


class SOExpert(nn.Module):
    """SO(r) expert: z_SO = z @ R where R is learned and encouraged toward SO(r).

    The orthogonality constraint is applied via loss L_geo = ||R^T R - I||^2.
    R is a learned r×r matrix, not strictly constrained to SO(r) during forward.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.R = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)

    def forward(self, z):
        return z @ self.R

    def orthogonality_error(self):
        """Compute ||R^T R - I||^2 for the SO constraint loss."""
        RTR = self.R.T @ self.R
        eye = torch.eye(self.dim, device=self.R.device)
        return torch.norm(RTR - eye, p='fro') ** 2


class LRExpert(nn.Module):
    """Low-Rank expert: z_LR = z + z @ A @ B.

    A ∈ R^{r×k}, B ∈ R^{k×r}, where k << r (low-rank bottleneck).
    This is a low-rank correction to the identity path.
    """

    def __init__(self, dim, rank=None):
        super().__init__()
        self.dim = dim
        self.rank = rank or max(1, dim // 4)
        self.A = nn.Parameter(torch.randn(dim, self.rank) * 0.01 / math.sqrt(self.rank))
        self.B = nn.Parameter(torch.randn(self.rank, dim) * 0.01 / math.sqrt(self.rank))

    def forward(self, z):
        return z + z @ self.A @ self.B


class AffineExpert(nn.Module):
    """Affine expert: z_Affine = z @ W + b.

    More flexible, less constrained. Use sparingly.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.W = nn.Parameter(torch.eye(dim) + torch.randn(dim, dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        return z @ self.W + self.b


# ═══════════════════════════════════════════════════════════════════════════════
# GroupBank: Fixed group types with expandable experts
# ═══════════════════════════════════════════════════════════════════════════════

class GroupBank(nn.Module):
    """Fixed bank of group candidates with expandable group-specific experts.

    Group types (not expandable in v1):
      - Identity: single zero-residual path
      - SO:       expandable SO(r) experts
      - LR:       expandable low-rank experts
      - Affine:   expandable affine experts
      - MambaFlow: shared, not expandable

    Group-specific expert expansion:
      SO_0, SO_1, SO_2, ...
      LR_0, LR_1, ...
      Affine_0, Affine_1, ...
    """

    def __init__(self, dim, expandable_groups=('SO', 'LR', 'Affine'),
                 mamba_config=None):
        super().__init__()
        self.dim = dim
        self.expandable_groups = expandable_groups
        self.mamba_config = mamba_config or {}

        # Expert factory
        self._expert_factory = {
            'Identity': lambda: IdentityExpert(dim),
            'SO': lambda: SOExpert(dim),
            'LR': lambda: LRExpert(dim),
            'Affine': lambda: AffineExpert(dim),
            'MambaFlow': lambda: SharedMambaFlow(dim, **self.mamba_config),
        }

        # Group → list of experts
        self.groups: Dict[str, nn.ModuleList] = nn.ModuleDict()
        # Initialize with one expert per group
        for group_name in ['Identity', 'SO', 'LR', 'Affine', 'MambaFlow']:
            self.groups[group_name] = nn.ModuleList()
            self._add_expert_to_group(group_name)

    def _add_expert_to_group(self, group_name):
        expert = self._expert_factory[group_name]()
        self.groups[group_name].append(expert)
        return expert

    def add_expert(self, group_name):
        """Add a new expert to an expandable group. Returns True if successful."""
        if group_name not in self.expandable_groups:
            logging.warning(
                f"Group '{group_name}' is not expandable. Skipping expansion."
            )
            return False
        self._add_expert_to_group(group_name)
        logging.info(f"Added new expert to group '{group_name}' "
                     f"(now {len(self.groups[group_name])} experts)")
        return True

    def get_expert(self, group_name, expert_idx):
        return self.groups[group_name][expert_idx]

    def num_experts(self, group_name):
        return len(self.groups[group_name])

    def forward_group(self, group_name, z, expert_weights=None):
        """Apply experts in a group with optional weighted combination.

        Args:
            group_name: name of the group
            z: [B, N, dim] latent tokens
            expert_weights: [B, num_experts] or None (use last expert only)

        Returns:
            out: [B, N, dim] group output
            expert_outputs: list of [B, N, dim] per-expert outputs
        """
        experts = self.groups[group_name]
        if expert_weights is None:
            # Use last expert only (for single-expert case or inference)
            return experts[-1](z), None

        # Weighted combination of experts
        expert_outs = []
        for expert in experts:
            expert_outs.append(expert(z))

        stacked = torch.stack(expert_outs, dim=0)  # [E, B, N, dim]
        # expert_weights: [B, E] → [E, B, 1, 1]
        w = expert_weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        combined = (stacked * w).sum(dim=0)  # [B, N, dim]
        return combined, expert_outs

    def orthogonality_error(self):
        """Sum of orthogonality errors across all SO experts."""
        err = 0.0
        for expert in self.groups['SO']:
            err = err + expert.orthogonality_error()
        return err


# ═══════════════════════════════════════════════════════════════════════════════
# Hierarchical Router (GroupRouter + ExpertRouter)
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalRouter(nn.Module):
    """Hierarchical geometric router: GroupRouter → ExpertRouter.

    Router input: concat(cls_token, mean(tokens), std(tokens),
                          group_wise_RD_z_scores, optional_group_usage)

    Group score: score_g = MLP(h)_g - beta * stopgrad(z_g)
    p(g|h) = softmax(score_g / tau)

    Final expert weight: w_{g,e} = p(g|h) * p(e|g,h)
    """

    def __init__(self, dim, num_groups=5, z_dim=5, beta=0.1,
                 tau=1.0, router_hidden=None):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        self.beta = beta
        self.tau = tau

        # Router input: cls_token(D) + mean(D) + std(D) + z_scores(num_groups) + usage(num_groups)
        router_input_dim = dim * 3 + num_groups * 2
        router_hidden = router_hidden or max(dim // 2, 64)

        # Group router
        self.group_router = nn.Sequential(
            nn.Linear(router_input_dim, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_groups),
        )

        # Expert routers (one per expandable group)
        self.expert_routers = nn.ModuleDict()

        self._init_weights()

    def _init_weights(self):
        last = self.group_router[-1]
        nn.init.trunc_normal_(last.weight, std=0.02)
        nn.init.zeros_(last.bias)

    def ensure_expert_router(self, group_name, num_experts):
        """Create or update expert router for a group."""
        router_input_dim = self.group_router[0].in_features
        router_hidden = self.group_router[0].out_features

        if group_name not in self.expert_routers:
            router = nn.Sequential(
                nn.Linear(router_input_dim, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, 1),  # 1 output, will be expanded
            )
            self.expert_routers[group_name] = router
            nn.init.trunc_normal_(router[-1].weight, std=0.02)
            nn.init.zeros_(router[-1].bias)

    def _build_router_input(self, x, z_scores=None, group_usage=None):
        """Build router input from features and RD statistics.

        Args:
            x: [B, N, D] token features
            z_scores: [B, num_groups] group-wise RD z-scores (or None)
            group_usage: [B, num_groups] group usage statistics (or None)

        Returns:
            router_input: [B, router_input_dim]
        """
        B, N, D = x.shape
        cls_token = x[:, 0]       # [B, D]
        mean_tok = x.mean(dim=1)  # [B, D]
        std_tok = x.std(dim=1)    # [B, D]

        parts = [cls_token, mean_tok, std_tok]

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
        """Forward: compute group probs and per-group expert probs.

        Args:
            x: [B, N, D] token features
            z_scores: [B, num_groups] group-wise RD z-scores
            group_usage: [B, num_groups] group usage stats
            group_expert_counts: dict[group_name → num_experts] for initialization

        Returns:
            group_probs: [B, num_groups]
            expert_probs: dict[group_name → [B, num_experts_in_group]]
        """
        router_input = self._build_router_input(x, z_scores, group_usage)

        # Group logits
        group_logits = self.group_router(router_input)  # [B, num_groups]

        # Z-score correction: score_g = logit_g - beta * stopgrad(z_g)
        if z_scores is not None:
            group_logits = group_logits - self.beta * z_scores.detach()

        group_probs = F.softmax(group_logits / self.tau, dim=-1)

        # Expert probs per group
        expert_probs = {}
        group_names = list(self.expert_routers.keys())
        for group_name in group_names:
            if group_name in self.expert_routers:
                router = self.expert_routers[group_name]
                num_exp = (
                    group_expert_counts.get(group_name, 1)
                    if group_expert_counts else 1
                )
                if num_exp <= 1:
                    expert_probs[group_name] = torch.ones(
                        group_probs.shape[0], 1, device=x.device
                    )
                else:
                    # Route to multiple experts
                    # Create expanded router: [in_dim, num_experts]
                    logits = router(router_input)  # [B, 1]
                    # Use separate routing tokens for multiple experts
                    all_logits = []
                    for e in range(num_exp):
                        all_logits.append(router(router_input))
                    expert_logits = torch.cat(all_logits, dim=-1)  # [B, E]
                    expert_probs[group_name] = F.softmax(expert_logits, dim=-1)

        # Ensure all expandable groups have entry
        for gn in ['SO', 'LR', 'Affine']:
            if gn not in expert_probs:
                cnt = (group_expert_counts.get(gn, 1)
                       if group_expert_counts else 1)
                expert_probs[gn] = torch.ones(
                    group_probs.shape[0], cnt, device=x.device
                ) / cnt

        # Identity and MambaFlow get uniform single-expert weight
        for gn in ['Identity', 'MambaFlow']:
            if gn not in expert_probs:
                expert_probs[gn] = torch.ones(
                    group_probs.shape[0], 1, device=x.device
                )

        return group_probs, expert_probs


# ═══════════════════════════════════════════════════════════════════════════════
# Group-Aware AE/RD (per-group reconstruction descriptor)
# ═══════════════════════════════════════════════════════════════════════════════

class GroupAwareAE(nn.Module):
    """Group-aware AutoEncoder for reconstruction-based distribution detection.

    Each group maintains its own AE: AE_g(z) → reconstruct z
    L_group_RD = sum_g p(g|h) * MSE(AE_g(z), z)

    Used to compute group-wise RD z-scores for expansion detection.
    """

    def __init__(self, dim, rd_dim=None, num_groups=5):
        super().__init__()
        self.dim = dim
        self.rd_dim = rd_dim or max(dim // 4, 4)
        self.num_groups = num_groups

        # Per-group encoder/decoder
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
        """Encode-decode for a specific group or all groups.

        Args:
            z: [B, dim] token mean (pooled)
            group_idx: int or None (None → return all)

        Returns:
            reconstruction: [B, dim] or [num_groups, B, dim]
        """
        if group_idx is not None:
            encoded = self.encoders[group_idx](z)
            return self.decoders[group_idx](encoded)
        else:
            reconstructions = []
            for g in range(self.num_groups):
                encoded = self.encoders[g](z)
                reconstructions.append(self.decoders[g](encoded))
            return torch.stack(reconstructions, dim=0)  # [G, B, dim]

    def compute_group_rd_loss(self, z, group_probs):
        """Compute group-weighted RD loss.

        L_group_RD = sum_g p(g|h) * MSE(AE_g(z), z)

        Args:
            z: [B, dim] token mean
            group_probs: [B, num_groups] group probabilities

        Returns:
            group_rd_loss: [B] per-sample loss
            per_group_loss: [B, num_groups] per-group per-sample loss
        """
        B, D = z.shape
        all_reconstructions = self.forward(z)  # [G, B, D]
        G = all_reconstructions.shape[0]

        # Per-group MSE
        per_group_loss = torch.zeros(B, G, device=z.device)
        for g in range(G):
            per_group_loss[:, g] = F.mse_loss(
                all_reconstructions[g], z, reduction='none'
            ).mean(dim=-1)

        # Weighted by group probs
        group_rd_loss = (per_group_loss * group_probs).sum(dim=-1)  # [B]
        return group_rd_loss, per_group_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Running Records (per-group RD statistics)
# ═══════════════════════════════════════════════════════════════════════════════

class RunningRecords:
    """Online running statistics for RD losses, per-group.

    Mirrors the original Records class but operates per-group.
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
        self._var = float(torch.var(self.record[:self._curr_len]))


# ═══════════════════════════════════════════════════════════════════════════════
# GroupMoEAdapter: combines GroupBank, Router, MambaFlow, GroupAwareAE
# ═══════════════════════════════════════════════════════════════════════════════

class GroupMoEAdapter(nn.Module):
    """Geometry-aware Group-MoE Adapter — replaces the original AdapterModule.

    Data flow:
      1. z = W_down LN(h)                                    (bottleneck projection)
      2. pi = GroupRouter(z, RD_stats)                        (group routing)
      3. z^G = sum_g pi_g * T_g(z)                           (group-MoE mixing)
      4. m = SharedMambaFlow(z^G)                             (semantic flow)
      5. a = W_up m                                           (output projection)
      6. h_out = h + gamma * a                               (residual)

    Group-aware AE/RD runs on bottleneck latent z for expansion detection.
    """

    def __init__(self, config, layer_id, adapter_id=0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.adapter_id = adapter_id

        d_model = config.d_model
        bottleneck = getattr(config, 'ffn_num', 16)
        num_groups = getattr(config, 'num_geo_groups', 5)

        # Bottleneck projections (shared)
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.up_proj = nn.Linear(bottleneck, d_model)

        # LayerNorm before bottleneck
        self.norm = nn.LayerNorm(d_model)

        # Residual gate (init zero for stable start)
        self.gamma = nn.Parameter(torch.zeros(1))

        # GroupBank with expandable experts
        mamba_cfg = {
            'd_state': getattr(config, 'mamba_d_state', 16),
            'd_conv': getattr(config, 'mamba_d_conv', 4),
            'expand': getattr(config, 'mamba_expand', 2),
        }
        self.group_bank = GroupBank(bottleneck, mamba_config=mamba_cfg)

        # Hierarchical router
        group_names = ['Identity', 'SO', 'LR', 'Affine', 'MambaFlow']
        self.group_name_to_idx = {n: i for i, n in enumerate(group_names)}
        self.idx_to_group_name = {i: n for n, i in self.group_name_to_idx.items()}

        self.router = HierarchicalRouter(
            d_model, num_groups=num_groups,
            beta=getattr(config, 'router_beta', 0.1),
            tau=getattr(config, 'router_tau', 1.0),
        )
        # Initialize expert routers for expandable groups
        for gn in ['SO', 'LR', 'Affine']:
            self.router.ensure_expert_router(gn, 1)

        # Group-aware AE/RD
        self.not_addition_layer = (
            layer_id < config.adapt_start_layer
            or layer_id > config.adapt_end_layer
        )
        if not self.not_addition_layer:
            self.group_ae = GroupAwareAE(bottleneck, num_groups=num_groups)
            self.rd_records: Dict[str, RunningRecords] = {}
            for gn in group_names:
                self.rd_records[gn] = RunningRecords(
                    max_len=getattr(config, 'buffer_size', 500)
                )
            self.per_group_records: List[RunningRecords] = [
                RunningRecords(max_len=getattr(config, 'buffer_size', 500))
                for _ in range(num_groups)
            ]
        else:
            self.group_ae = None
            self.rd_records = None
            self.per_group_records = None

        self.newly_added = True

        # Init weights
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.bias)

    def _get_group_expert_counts(self):
        return {
            gn: self.group_bank.num_experts(gn)
            for gn in ['SO', 'LR', 'Affine']
        }

    def _compute_z_scores(self, per_group_loss, detach=True):
        """Compute group-wise Z-scores from per-group RD losses.

        Args:
            per_group_loss: [B, num_groups]
            detach: if True, detach before computing (for router correction)

        Returns:
            z_scores: [B, num_groups]
        """
        B, G = per_group_loss.shape
        z_scores = torch.zeros(B, G, device=per_group_loss.device)
        for g in range(G):
            rec = self.per_group_records[g]
            if rec.length > 2:
                mean = rec.mean
                std = rec.stddev
                loss_g = per_group_loss[:, g].detach() if detach else per_group_loss[:, g]
                z_scores[:, g] = torch.abs((loss_g - mean) / std)
        return z_scores

    def _get_group_usage(self):
        """Return group usage statistics (proxy: num experts per group)."""
        usage = torch.zeros(len(self.group_name_to_idx))
        for gn in ['SO', 'LR', 'Affine']:
            idx = self.group_name_to_idx[gn]
            usage[idx] = float(self.group_bank.num_experts(gn))
        return usage

    def forward(self, x, group_info=None):
        """Forward pass: Group-MoE → MambaFlow → output.

        Args:
            x: [B, N, d_model] ViT block output (u_l)
            group_info: optional group positional info (not used directly here)

        Returns:
            dict with:
              func_out: [B, N, d_model] adapter output
              group_rd_loss: scalar
              z_scores: [B, num_groups] group-wise RD z-scores
              group_probs: [B, num_groups]
              expert_probs: dict
              added: bool (expansion triggered)
        """
        B, N, D = x.shape
        z = self.down_proj(self.norm(x))  # [B, N, r]

        # Router: compute group and expert probabilities
        group_expert_counts = self._get_group_expert_counts()

        if not self.not_addition_layer and self.group_ae is not None:
            # Compute group-aware RD loss for z-score
            z_pooled = z.mean(dim=1)  # [B, r]
            _, per_group_rd = self.group_ae.compute_group_rd_loss(
                z_pooled, torch.ones(B, len(self.group_name_to_idx),
                                     device=z.device) / len(self.group_name_to_idx)
            )
            z_scores = self._compute_z_scores(per_group_rd)  # [B, G]
            group_usage = self._get_group_usage().to(z.device).unsqueeze(0).expand(B, -1)
        else:
            z_scores = None
            group_usage = None

        group_probs, expert_probs = self.router(
            x, z_scores=z_scores, group_usage=group_usage,
            group_expert_counts=group_expert_counts,
        )

        # Group-MoE: mix outputs from different groups
        group_outputs = []
        group_names = ['Identity', 'SO', 'LR', 'Affine', 'MambaFlow']
        for i, gn in enumerate(group_names):
            g_out, _ = self.group_bank.forward_group(
                gn, z, expert_weights=expert_probs.get(gn)
            )
            group_outputs.append(g_out)

        # Weighted sum over groups
        # group_probs: [B, G], group_outputs: list of G x [B, N, r]
        stacked = torch.stack(group_outputs, dim=0)  # [G, B, N, r]
        w = group_probs[:, :, None, None]  # [B, G, 1, 1] — need to fix shape
        # group_probs: [B, G], we want [G, B, 1, 1]
        w = group_probs.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [G, B, 1, 1]
        z_G = (stacked * w).sum(dim=0)  # [B, N, r]

        # Shared MambaFlow (on geometry-conditioned latent)
        mamba_out = self.group_bank.groups['MambaFlow'][-1](z_G)  # [B, N, r]

        # Output projection
        a = self.up_proj(mamba_out)  # [B, N, D]

        # Group-aware RD
        added = False
        group_rd_loss = torch.tensor(0.0, device=x.device)
        if not self.not_addition_layer and self.group_ae is not None:
            z_pooled = z.mean(dim=1)
            group_rd_loss, per_group_rd = self.group_ae.compute_group_rd_loss(
                z_pooled, group_probs
            )
            group_rd_loss = group_rd_loss.mean()

            # Record per-group losses for statistics
            if self.training:
                for g in range(len(group_names)):
                    self.per_group_records[g].add_record(per_group_rd[:, g])

            # Compute z-scores for output
            z_scores = self._compute_z_scores(per_group_rd, detach=False)
        else:
            z_scores = torch.zeros(B, len(group_names), device=x.device)

        out = {
            "func_out": a,
            "group_rd_loss": group_rd_loss,
            "z_scores": z_scores,
            "group_probs": group_probs,
            "expert_probs": expert_probs,
            "added": added,
        }
        return out

    def add_expert_to_group(self, group_name):
        """Add a new expert to an expandable group."""
        success = self.group_bank.add_expert(group_name)
        if success:
            new_count = self.group_bank.num_experts(group_name)
            self.router.ensure_expert_router(group_name, new_count)
        return success

    def orthogonality_error(self):
        return self.group_bank.orthogonality_error()


# ═══════════════════════════════════════════════════════════════════════════════
# GeometrySEMAModules: Layer-level manager (replaces SEMAModules)
# ═══════════════════════════════════════════════════════════════════════════════

class GeometrySEMAModules(nn.Module):
    """Layer-level manager for Geometry-SEMA adapters.

    Replaces SEMAModules with:
    - Group-MoE Adapters instead of standard adapters
    - Group-specific expert expansion (not adapter-level expansion)
    - Geometric routing with RD z-score correction
    - Group-aware RD loss computation

    Key differences from SEMAModules:
    - Expansion is per-group-expert, not per-adapter
    - Router is hierarchical (GroupRouter + ExpertRouter)
    - MambaFlow is shared (not expanded)
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
        self.added_adapter = 0

        # Initialize with one GroupMoEAdapter
        self.adapters: List[GroupMoEAdapter] = nn.ModuleList()
        self.add_adapter(initialize=True)

        # Expansion counters per group (SO, LR, Affine)
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
        new_adapter = GroupMoEAdapter(
            self.config, self.layer_id, adapter_id=adapter_id
        ).to(self._device())
        self.newly_added = True
        self.added_for_task = True
        self.adapters.append(new_adapter)
        if not initialize:
            self.added_adapter += 1
        logging.info(
            f"GroupMoEAdapter {self.layer_id}.{adapter_id} added at block {self.layer_id}"
        )

    def add_expert_to_group(self, group_name):
        """Add a new expert to the specified group in the last adapter."""
        if self.adapters:
            return self.adapters[-1].add_expert_to_group(group_name)
        return False

    def forward(self, x, group_info=None):
        """Forward: Group-MoE adapter with expansion detection.

        Returns:
            dict with func_out, group_rd_loss, z_scores, group_probs, expert_probs, added
        """
        zero = torch.tensor(0.0, device=x.device)
        not_addition_layer = (
            self.layer_id < self.adapt_start_layer
            or self.layer_id > self.adapt_end_layer
        )

        if not_addition_layer:
            # Shallow/middle layer: simple adapter pass
            adapter_out = self.adapters[-1](x, group_info=group_info)
            out = {
                "func_out": adapter_out["func_out"],
                "group_rd_loss": zero,
                "z_scores": adapter_out.get("z_scores", None),
                "group_probs": adapter_out.get("group_probs", None),
                "expert_probs": adapter_out.get("expert_probs", None),
                "added": False,
            }
            return out

        # Deep layer: full Group-MoE with expansion detection
        adapter = self.adapters[-1]  # Use last adapter
        adapter_out = adapter(x, group_info=group_info)

        # Expansion detection
        added = False
        if self.detecting_outlier and not self.added_for_task:
            z_scores = adapter_out.get("z_scores")
            if z_scores is not None:
                # Check if all groups have high z-score
                group_probs = adapter_out.get("group_probs")
                # Find the group with highest probability
                if group_probs is not None:
                    # For each sample, find the expandable group with highest prob
                    expandable_indices = [
                        adapter.group_name_to_idx[gn]
                        for gn in ['SO', 'LR', 'Affine']
                    ]
                    # Mean z-score of most probable expandable group
                    z_mean = z_scores.mean(dim=0)  # [G]
                    max_z = max(
                        z_mean[idx].item() for idx in expandable_indices
                    )
                    if max_z > self.config.exp_threshold:
                        # Expand in the group with highest z-score
                        best_idx = expandable_indices[
                            max(
                                range(len(expandable_indices)),
                                key=lambda i: z_mean[expandable_indices[i]].item()
                            )
                        ]
                        best_group = adapter.idx_to_group_name[best_idx]
                        self.add_expert_to_group(best_group)
                        self.expansion_count[best_group] += 1
                        added = True
                        logging.info(
                            f"Block {self.layer_id}: added expert to group '{best_group}' "
                            f"(z={max_z:.3f} > threshold={self.config.exp_threshold})"
                        )

        out = {
            "func_out": adapter_out["func_out"],
            "group_rd_loss": adapter_out["group_rd_loss"],
            "z_scores": adapter_out.get("z_scores"),
            "group_probs": adapter_out.get("group_probs"),
            "expert_probs": adapter_out.get("expert_probs"),
            "added": added,
        }
        return out

    # ── Task-end freeze ──

    def end_of_task_training(self):
        self.freeze_functional()
        self.freeze_rd()
        self.reset_newly_added_status()
        self.added_for_task = False

    def reset_newly_added_status(self):
        self.newly_added = False
        for adapter in self.adapters:
            adapter.newly_added = False

    def freeze_functional(self):
        for adapter in self.adapters:
            # Freeze down/up projections
            for param in adapter.down_proj.parameters():
                param.requires_grad = False
            for param in adapter.up_proj.parameters():
                param.requires_grad = False
            # Freeze gamma
            adapter.gamma.requires_grad_(False)
            # Freeze all group experts
            for gn, experts in adapter.group_bank.groups.items():
                for expert in experts:
                    for param in expert.parameters():
                        param.requires_grad = False
            # Freeze router
            for param in adapter.router.parameters():
                param.requires_grad = False

    def freeze_rd(self):
        for adapter in self.adapters:
            if adapter.group_ae is not None:
                for param in adapter.group_ae.parameters():
                    param.requires_grad = False
                if adapter.rd_records:
                    for rec in adapter.rd_records.values():
                        rec.updating = False
                if adapter.per_group_records:
                    for rec in adapter.per_group_records:
                        rec.updating = False
