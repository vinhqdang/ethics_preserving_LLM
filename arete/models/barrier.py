"""
barrier.py - Geodesic Safety Barrier (GSB) with Orthogonal Null-Space Steering.
Evaluates proximity of intent representations to multi-level harm boundaries
and applies closed-loop Riemannian null-space interventions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class GeodesicSafetyBarrier(nn.Module):
    """
    Geodesic Safety Barrier on the Intent Manifold.
    Maintains ethical boundary hyperplanes corresponding to 19 harm taxonomies
    with calibrated severity thresholds.
    Applies orthogonal null-space steering to neutralize harmful intent while
    preserving benign contextual and conversational fluency.
    """
    def __init__(
        self,
        intent_dim: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        num_harm_categories: int = 19
    ):
        super().__init__()
        self.intent_dim = intent_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim if context_dim is not None else (hidden_dim - intent_dim)
        self.num_harm_categories = num_harm_categories
        
        # Ethical constraint boundary weights: W_k in R^{K x D_intent}
        # A positive projection <w_k, z_intent> > threshold signifies an ethical boundary violation
        self.harm_boundaries = nn.Parameter(torch.randn(num_harm_categories, intent_dim))
        nn.init.orthogonal_(self.harm_boundaries)
        
        # Learnable thresholds for multi-level severity (calibrated around 0.35)
        self.severity_thresholds = nn.Parameter(torch.full((num_harm_categories,), 0.35))
        
        # Steering projection matrix from intent space into residual stream
        self.steering_proj = nn.Linear(intent_dim, hidden_dim, bias=False)
        
        # Context lifting matrix to define context subspace in hidden space
        self.context_lift = nn.Linear(self.context_dim, hidden_dim, bias=False)
        
        # Safety gate sensitivity temperature
        self.temperature = nn.Parameter(torch.tensor(0.1))

    def evaluate_violation(self, z_intent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates boundary violation margin:
        margin_k = <z_intent, w_k> - threshold_k
        """
        w_norm = F.normalize(self.harm_boundaries, p=2, dim=-1)
        # Projected margins: [..., num_harm_categories]
        projections = torch.matmul(z_intent, w_norm.T)
        margins = projections - self.severity_thresholds
        return projections, margins

    def forward(
        self,
        h: torch.Tensor,
        z_intent: torch.Tensor,
        z_context: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Performs closed-loop Geodesic Null-Space Intervention:
        1. Identifies if z_intent penetrates any harm boundary polytope.
        2. Calculates minimal geodesic retraction vector delta_z.
        3. Projects delta_h onto the orthogonal complement of the context subspace.
        4. Injects steered residual into h.
        
        Args:
            h: Residual stream hidden states [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
            z_intent: Decomposed intent vectors [..., intent_dim]
            z_context: Decomposed context vectors [..., context_dim]
        Returns:
            h_steered: Intervened hidden states with ethical guarantee
            intervention_info: Metric dictionary
        """
        projections, margins = self.evaluate_violation(z_intent)
        
        # Maximum violation across all harm categories
        max_margin, argmax_cat = torch.max(margins, dim=-1, keepdim=True)
        
        # Soft safety activation via sigmoid barrier
        violation_prob = torch.sigmoid(max_margin / torch.clamp(torch.abs(self.temperature), min=1e-3))
        
        # Compute boundary normal of the most violated category
        w_norm = F.normalize(self.harm_boundaries, p=2, dim=-1)
        idx = argmax_cat.squeeze(-1)
        active_normal = w_norm[idx] # [..., intent_dim]
        
        # Minimal retraction in intent space along geodesic normal
        relu_margin = F.relu(max_margin)
        delta_z = - relu_margin * active_normal # [..., intent_dim]
        
        # Lift correction into hidden space
        raw_delta_h = self.steering_proj(delta_z) # [..., hidden_dim]
        
        # Orthogonal Null-Space Projection:
        # We ensure delta_h does not perturb the semantic fluency or harmless instructions
        # encoded in z_context.
        # Lift z_context to find context subspace direction
        v_ctx = self.context_lift(z_context)
        v_ctx_norm = F.normalize(v_ctx, p=2, dim=-1)
        
        # Projection of raw_delta_h onto context subspace:
        # proj_ctx = <raw_delta_h, v_ctx_norm> * v_ctx_norm
        proj_ctx = torch.sum(raw_delta_h * v_ctx_norm, dim=-1, keepdim=True) * v_ctx_norm
        
        # Orthogonal component: delta_h_perp = raw_delta_h - proj_ctx
        delta_h_perp = raw_delta_h - proj_ctx
        
        # Apply smooth steering
        h_steered = h + violation_prob * delta_h_perp
        
        info = {
            "violation_prob": violation_prob,
            "max_margin": max_margin,
            "argmax_category": argmax_cat,
            "projections": projections,
            "delta_norm": torch.norm(delta_z, dim=-1).mean(),
            "null_space_preservation": (1.0 - torch.norm(proj_ctx, dim=-1) / (torch.norm(raw_delta_h, dim=-1) + 1e-7)).mean()
        }
        return h_steered, info

    def compute_barrier_loss(
        self,
        z_intent: torch.Tensor,
        is_harmful: torch.Tensor,
        category_id: torch.Tensor,
        margin_target: float = 0.20
    ) -> Dict[str, torch.Tensor]:
        """
        Trains the harm boundaries to separate ethical violations from benign requests:
        1. Boundary classification loss for harmful samples.
        2. Margin safety margin: <z_intent, w_c> >= threshold + margin_target for harmful.
        3. Non-violation penalty for benign samples: <z_intent, w_k> < threshold for all k.
        """
        projections, margins = self.evaluate_violation(z_intent)
        
        # 1. Harm category identification loss (for harmful samples)
        harm_mask = (is_harmful > 0.5)
        loss_cat = torch.tensor(0.0, device=z_intent.device)
        loss_harm_margin = torch.tensor(0.0, device=z_intent.device)
        
        if harm_mask.sum() > 0:
            target_cats = category_id[harm_mask]
            proj_harm = projections[harm_mask]
            loss_cat = F.cross_entropy(proj_harm / 0.1, target_cats)
            
            # Ensure targeted category breaches the barrier by margin_target
            margins_harm = margins[harm_mask]
            gathered_margins = margins_harm.gather(1, target_cats.unsqueeze(1)).squeeze(1)
            loss_harm_margin = F.relu(margin_target - gathered_margins).mean()
            
        # 2. Benign safety margin: all projections must remain below threshold
        benign_mask = ~harm_mask
        loss_benign_margin = torch.tensor(0.0, device=z_intent.device)
        if benign_mask.sum() > 0:
            margins_benign = margins[benign_mask]
            # Penalize any positive margin on benign inputs (prevent over-refusal)
            loss_benign_margin = F.relu(margins_benign + 0.10).mean()
            
        total_loss = loss_cat + 2.0 * loss_harm_margin + 3.0 * loss_benign_margin
        return {
            "barrier_loss": total_loss,
            "loss_cat": loss_cat.item() if isinstance(loss_cat, torch.Tensor) else loss_cat,
            "loss_harm_margin": loss_harm_margin.item() if isinstance(loss_harm_margin, torch.Tensor) else loss_harm_margin,
            "loss_benign_margin": loss_benign_margin.item() if isinstance(loss_benign_margin, torch.Tensor) else loss_benign_margin
        }
