r"""
manifold.py - Invariant Intent-Context Manifold Decomposition for ARETE.
Implements Lie-Equivariant Intent-Context Decomposition (LE-ICD) and
geodesic metric projections on the latent intent manifold.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentContextDecomposer(nn.Module):
    r"""
    Decomposes intermediate hidden representations into orthogonal bundles:
    h = z_intent \oplus z_context
    such that z_intent is invariant under stylistic/adversarial transformations \tau,
    while z_context absorbs framing, syntax, and rhetorical wrappers.
    """
    def __init__(self, hidden_dim: int, intent_dim: int, context_dim: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intent_dim = intent_dim
        self.context_dim = context_dim if context_dim is not None else (hidden_dim - intent_dim)
        
        # Multi-head intent extraction projection
        self.intent_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, intent_dim, bias=False)
        )
        
        # Context extractor
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.context_dim, bias=False)
        )
        
        # Cross-bundle orthogonalizer
        self.reconstructor = nn.Linear(intent_dim + self.context_dim, hidden_dim, bias=False)
        
        # Normalization layer for hyperspherical projection
        self.intent_norm = nn.LayerNorm(intent_dim)
        self.context_norm = nn.LayerNorm(self.context_dim)

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: Hidden states [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        Returns:
            z_intent: Invariant intent representation [..., intent_dim]
            z_context: Contextual framing representation [..., context_dim]
            h_recon: Reconstructed hidden representation [..., hidden_dim]
        """
        raw_intent = self.intent_proj(h)
        z_intent = F.normalize(self.intent_norm(raw_intent), p=2, dim=-1)
        
        raw_context = self.context_proj(h)
        z_context = F.normalize(self.context_norm(raw_context), p=2, dim=-1)
        
        # Concatenate bundles for reconstruction
        z_combined = torch.cat([z_intent, z_context], dim=-1)
        h_recon = self.reconstructor(z_combined)
        
        return z_intent, z_context, h_recon

    def geodesic_distance(self, z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        r"""
        Computes geodesic distance on the hyperspherical manifold S^{d-1}:
        d(z1, z2) = arccos( \langle z1, z2 \rangle )
        """
        cos_sim = torch.sum(z1 * z2, dim=-1)
        cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
        return torch.acos(cos_sim)

    def compute_decomposition_loss(self, h_clean: torch.Tensor, h_adv: torch.Tensor):
        """
        Computes the LE-ICD loss:
        1. Invariance loss: z_intent(clean) should equal z_intent(adv)
        2. Orthogonality loss: <z_intent, z_context> -> 0
        3. Reconstruction loss: h_recon \approx h
        """
        z_int_c, z_ctx_c, h_rec_c = self.forward(h_clean)
        z_int_a, z_ctx_a, h_rec_a = self.forward(h_adv)
        
        # 1. Invariant geodesic divergence
        l_inv = torch.mean(self.geodesic_distance(z_int_c, z_int_a))
        
        # 2. Orthogonality between intent and context bundles
        # Flatten sequence dimension if 3D
        z_int_c_flat = z_int_c.reshape(-1, self.intent_dim)
        z_ctx_c_flat = z_ctx_c.reshape(-1, self.context_dim)
        cross_corr = torch.matmul(z_int_c_flat.T, z_ctx_c_flat) / z_int_c_flat.size(0)
        l_ortho = torch.norm(cross_corr, p='fro')
        
        # 3. Reconstruction loss to preserve language modeling capacity
        l_rec = F.mse_loss(h_rec_c, h_clean) + F.mse_loss(h_rec_a, h_adv)
        
        total_loss = l_inv + 0.1 * l_ortho + 0.05 * l_rec
        return {
            "total_loss": total_loss,
            "l_inv": l_inv.item(),
            "l_ortho": l_ortho.item(),
            "l_rec": l_rec.item()
        }
