r"""
barrier.py - Geodesic Safety Barrier (GSB) with Exact Spherical-Cap Retraction
and Subspace Orthogonal Null-Space Steering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class GeodesicSafetyBarrier(nn.Module):
    r"""
    Geodesic Safety Barrier on the Intent Hypersphere S^{d_I-1}.
    Maintains K=19 boundary normal hyperplanes {w_k}_{k=1}^K \subset S^{d_I-1}
    with calibrated spherical-cap thresholds \gamma_k \in (0, 1).

    When a violation m_{k*}(z_intent) = <z_intent, w_{k*}> - \gamma_{k*} > 0 occurs:
    1. Computes the exact geodesic metric projection \pi_{K_{k*}}(z_intent) onto the
       safe spherical cap boundary <z^+, w_{k*}> = \gamma'_{k*} on S^{d_I-1}.
    2. Lifts the chord displacement \Delta z_intent = z^+ - z_intent into R^D.
    3. Projects the lifted perturbation onto the orthogonal complement P_{context}^\perp
       of the context subspace V_{context}, leaving benign inputs identically unperturbed
       (\Delta h \equiv 0 whenever \max_k m_k \le 0).
    """
    def __init__(
        self,
        intent_dim: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        num_harm_categories: int = 19,
        context_subspace_rank: int = 32,
        retraction_slack: float = 0.05
    ):
        super().__init__()
        self.intent_dim = intent_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim if context_dim is not None else (hidden_dim - intent_dim)
        self.num_harm_categories = num_harm_categories
        self.context_subspace_rank = min(context_subspace_rank, self.context_dim)
        self.retraction_slack = retraction_slack

        # Ethical constraint boundary normals: W in R^{K x d_I}
        self.harm_boundaries = nn.Parameter(torch.randn(num_harm_categories, intent_dim))
        nn.init.orthogonal_(self.harm_boundaries)

        # Calibrated spherical-cap thresholds \gamma_k \in (0, 1)
        self.severity_thresholds = nn.Parameter(torch.full((num_harm_categories,), 0.35))

        # Steering lift matrix W_steer: R^{d_I} -> R^D
        self.steering_proj = nn.Linear(intent_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.steering_proj.weight)

        # Context lifting matrix W_ctx: R^{d_C} -> R^D
        self.context_lift = nn.Linear(self.context_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.context_lift.weight)

        # Global context subspace basis B_ctx in R^{D x r_C}
        self.context_subspace = nn.Parameter(torch.randn(hidden_dim, self.context_subspace_rank) * 0.02)

    def evaluate_violation(self, z_intent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates spherical-cap boundary margins:
        m_k(z_intent) = <z_intent, w_k> - gamma_k
        """
        w_norm = F.normalize(self.harm_boundaries, p=2, dim=-1)
        thresholds = torch.clamp(self.severity_thresholds, min=0.10, max=0.85)
        projections = torch.matmul(z_intent, w_norm.T)
        margins = projections - thresholds
        return projections, margins

    def geodesic_cap_retraction(
        self,
        z_intent: torch.Tensor,
        w_active: torch.Tensor,
        gamma_active: torch.Tensor,
        active_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the exact metric projection \pi_{K_{k*}}(z) on S^{d_I-1} onto the
        safe spherical cap <z^+, w_{k*}> = \gamma'_{k*} = \gamma_{k*} - \delta_{slack}:
            u_\perp = (z - <z, w> w) / ||z - <z, w> w||_2
            z^+ = \gamma' w + \sqrt{1 - (\gamma')^2} u_\perp \in S^{d_I-1}
        Returns:
            z_retracted: Retracted unit intent vector on S^{d_I-1}
            delta_z: Chord displacement (z^+ - z) gated by active_mask (1[m_{k*} > 0])
        """
        gamma_target = torch.clamp(gamma_active - self.retraction_slack, min=-0.50, max=0.80)
        dot_zw = torch.sum(z_intent * w_active, dim=-1, keepdim=True)
        ortho_comp = z_intent - dot_zw * w_active
        u_perp = F.normalize(ortho_comp, p=2, dim=-1, eps=1e-7)

        sin_coeff = torch.sqrt(torch.clamp(1.0 - gamma_target ** 2, min=1e-7))
        z_cap = gamma_target * w_active + sin_coeff * u_perp
        z_cap = F.normalize(z_cap, p=2, dim=-1)

        delta_z = active_mask * (z_cap - z_intent)
        z_retracted = z_intent + delta_z
        return z_retracted, delta_z

    def project_context_nullspace(
        self,
        raw_delta_h: torch.Tensor,
        z_context: torch.Tensor,
        use_nullspace: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Projects raw_delta_h onto the orthogonal complement P_{context}^\perp of:
        1. The r_C-dimensional context subspace Q_C = orth(B_ctx) \in R^{D x r_C}, and
        2. The sample-conditioned lifted context vector \hat{v}_{ctx} = Norm(W_{ctx} z_{context}).
        """
        if not use_nullspace:
            zero_proj = torch.zeros_like(raw_delta_h)
            return raw_delta_h, zero_proj

        orig_dtype = raw_delta_h.dtype
        # Orthonormalize global context subspace basis Q_C in float32 for numerical stability
        q_c, _ = torch.linalg.qr(self.context_subspace.float(), mode="reduced")
        q_c = q_c.to(orig_dtype)  # [D, r_C]

        # Project out global context subspace Q_C Q_C^T
        coeff_subspace = torch.matmul(raw_delta_h, q_c)  # [..., r_C]
        proj_subspace = torch.matmul(coeff_subspace, q_c.T)  # [..., D]
        delta_h_step1 = raw_delta_h - proj_subspace

        # Project out sample-conditioned lifted context direction \hat{v}_{ctx}
        v_ctx = self.context_lift(z_context)
        v_ctx_norm = F.normalize(v_ctx, p=2, dim=-1, eps=1e-7)
        proj_sample = torch.sum(delta_h_step1 * v_ctx_norm, dim=-1, keepdim=True) * v_ctx_norm

        delta_h_perp = delta_h_step1 - proj_sample
        total_proj_removed = raw_delta_h - delta_h_perp
        return delta_h_perp, total_proj_removed

    def forward(
        self,
        h: torch.Tensor,
        z_intent: torch.Tensor,
        z_context: torch.Tensor,
        use_nullspace: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Performs Closed-Loop Geodesic Null-Space Intervention.
        Strictly satisfies \Delta h = 0 whenever \max_k m_k(z_intent) <= 0.
        """
        projections, margins = self.evaluate_violation(z_intent)

        # Identify worst-case violated category k* = argmax_k m_k(z_intent)
        max_margin, argmax_cat = torch.max(margins, dim=-1, keepdim=True)

        # Hard boundary indicator 1[m_{k*} > 0] (with straight-through gradient for training)
        hard_gate = (max_margin > 0.0).to(h.dtype)
        soft_gate = torch.sigmoid(max_margin / 0.05)
        active_mask = hard_gate.detach() - soft_gate.detach() + soft_gate if self.training else hard_gate

        w_norm = F.normalize(self.harm_boundaries, p=2, dim=-1)
        thresholds = torch.clamp(self.severity_thresholds, min=0.10, max=0.85)

        idx = argmax_cat.squeeze(-1)
        w_active = w_norm[idx]
        gamma_active = thresholds[idx].unsqueeze(-1)

        # 1. Exact spherical-cap geodesic retraction on S^{d_I-1}
        z_retracted, delta_z = self.geodesic_cap_retraction(
            z_intent, w_active, gamma_active, active_mask
        )

        # 2. Lift chord retraction into R^D scaled by radial norm ||h||_2
        r_h = torch.norm(h, p=2, dim=-1, keepdim=True)
        raw_delta_h = r_h * self.steering_proj(delta_z)

        # 3. Subspace orthogonal null-space projection P_{context}^\perp
        delta_h_perp, proj_ctx = self.project_context_nullspace(
            raw_delta_h, z_context, use_nullspace=use_nullspace
        )

        # 4. Apply gated null-space correction (identically 0 when max_margin <= 0 in eval)
        if not self.training:
            delta_h_perp = hard_gate * delta_h_perp

        h_steered = h + delta_h_perp

        info = {
            "violation_gate": hard_gate,
            "max_margin": max_margin,
            "argmax_category": argmax_cat,
            "projections": projections,
            "z_retracted": z_retracted,
            "delta_z_norm": torch.norm(delta_z, dim=-1).mean(),
            "delta_h_norm": torch.norm(delta_h_perp, dim=-1).mean(),
            "null_space_orthogonality": torch.abs(
                torch.sum(delta_h_perp * F.normalize(self.context_lift(z_context), p=2, dim=-1), dim=-1)
            ).max()
        }
        return h_steered, info

    def compute_barrier_loss(
        self,
        z_intent: torch.Tensor,
        is_harmful: torch.Tensor,
        category_id: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        z_context: Optional[torch.Tensor] = None,
        decomposer: Optional[nn.Module] = None,
        margin_target: float = 0.20
    ) -> Dict[str, torch.Tensor]:
        r"""
        Trains both the 19 spherical-cap harm boundaries AND the steering/null-space matrices:
        1. Harm category classification & positive margin enforcement on harmful queries.
        2. Negative margin enforcement on benign queries (preventing false refusals).
        3. Closed-loop steering retraction loss: ensures \phi_I(h + \Delta h) retracts
           strictly inside K_{safe} while \phi_C(h + \Delta h) preserves z_context.
        """
        projections, margins = self.evaluate_violation(z_intent)

        harm_mask = (is_harmful > 0.5)
        loss_cat = torch.tensor(0.0, device=z_intent.device)
        loss_harm_margin = torch.tensor(0.0, device=z_intent.device)
        loss_steer_safe = torch.tensor(0.0, device=z_intent.device)
        loss_ctx_preserve = torch.tensor(0.0, device=z_intent.device)

        if harm_mask.sum() > 0:
            target_cats = category_id[harm_mask]
            proj_harm = projections[harm_mask]
            loss_cat = F.cross_entropy(proj_harm / 0.1, target_cats)

            margins_harm = margins[harm_mask]
            gathered_margins = margins_harm.gather(1, target_cats.unsqueeze(1)).squeeze(1)
            loss_harm_margin = F.relu(margin_target - gathered_margins).mean()

            # Closed-loop steering training for W_steer, W_ctx, and B_ctx
            if h is not None and z_context is not None and decomposer is not None:
                h_harm = h[harm_mask]
                z_int_harm = z_intent[harm_mask]
                z_ctx_harm = z_context[harm_mask]

                h_steered_harm, _ = self.forward(h_harm, z_int_harm, z_ctx_harm, use_nullspace=True)
                z_int_post, z_ctx_post, _ = decomposer(h_steered_harm)

                # Post-steering intent must satisfy max_k m_k(z_int_post) <= -0.05
                _, margins_post = self.evaluate_violation(z_int_post)
                max_margin_post, _ = torch.max(margins_post, dim=-1)
                loss_steer_safe = F.relu(max_margin_post + 0.05).mean()

                # Post-steering context must remain invariant: d_g(z_ctx_post, z_ctx_harm) -> 0
                loss_ctx_preserve = decomposer.geodesic_distance(z_ctx_post, z_ctx_harm.detach()).mean()

        benign_mask = ~harm_mask
        loss_benign_margin = torch.tensor(0.0, device=z_intent.device)
        if benign_mask.sum() > 0:
            margins_benign = margins[benign_mask]
            loss_benign_margin = F.relu(margins_benign + 0.10).mean()

        total_loss = (
            loss_cat
            + 2.0 * loss_harm_margin
            + 3.0 * loss_benign_margin
            + 1.5 * loss_steer_safe
            + 1.0 * loss_ctx_preserve
        )
        return {
            "barrier_loss": total_loss,
            "loss_cat": float(loss_cat.item()),
            "loss_harm_margin": float(loss_harm_margin.item()),
            "loss_benign_margin": float(loss_benign_margin.item()),
            "loss_steer_safe": float(loss_steer_safe.item()),
            "loss_ctx_preserve": float(loss_ctx_preserve.item())
        }
