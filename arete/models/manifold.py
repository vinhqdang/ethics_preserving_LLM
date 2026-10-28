r"""
manifold.py - Invariant Intent-Context Manifold Decomposition for ARETE.
Implements Bundle-Orthogonal Intent-Context Decomposition and
geodesic metric projections on the latent intent hypersphere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentContextDecomposer(nn.Module):
    r"""
    Decomposes intermediate hidden representations h \in R^D into directional
    coordinates on a product of unit Riemannian hyperspheres S^{d_I-1} x S^{d_C-1}
    together with radial norm r(h) = ||h||_2:
        z_intent = \phi_I(h) \in S^{d_I-1},  z_context = \phi_C(h) \in S^{d_C-1}
    such that z_intent is \epsilon-invariant under adversarial framing transformations,
    while z_context absorbs syntax, persona, and rhetorical wrappers.
    """
    def __init__(self, hidden_dim: int, intent_dim: int, context_dim: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intent_dim = intent_dim
        self.context_dim = context_dim if context_dim is not None else (hidden_dim - intent_dim)

        # Nonlinear intent encoder f_I: R^D -> R^{d_I}
        self.intent_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, intent_dim, bias=False)
        )

        # Nonlinear context encoder f_C: R^D -> R^{d_C}
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.context_dim, bias=False)
        )

        # Partitioned reconstruction mapping \Phi = [W_I, W_C]: R^{d_I + d_C} -> R^D
        self.reconstructor = nn.Linear(intent_dim + self.context_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.reconstructor.weight)

        # Layer normalization prior to spherical projection
        self.intent_norm = nn.LayerNorm(intent_dim)
        self.context_norm = nn.LayerNorm(self.context_dim)

    def encode_unnormalized(self, h: torch.Tensor):
        """Returns unnormalized pre-spherical embeddings for Euclidean ablation."""
        raw_intent = self.intent_proj(h)
        raw_context = self.context_proj(h)
        return raw_intent, raw_context

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: Hidden states [..., hidden_dim]
        Returns:
            z_intent: Unit intent direction [..., intent_dim] on S^{d_I-1}
            z_context: Unit context direction [..., context_dim] on S^{d_C-1}
            h_recon: Radial-scaled reconstructed state [..., hidden_dim]
        """
        raw_intent = self.intent_proj(h)
        z_intent = F.normalize(self.intent_norm(raw_intent), p=2, dim=-1)

        raw_context = self.context_proj(h)
        z_context = F.normalize(self.context_norm(raw_context), p=2, dim=-1)

        # Preserve radial norm r(h) = ||h||_2 during reconstruction
        r_h = torch.norm(h, p=2, dim=-1, keepdim=True)
        z_combined = torch.cat([z_intent, z_context], dim=-1)
        h_recon = r_h * self.reconstructor(z_combined)

        return z_intent, z_context, h_recon

    def get_subspace_weights(self):
        r"""Returns W_I in R^{D x d_I} and W_C in R^{D x d_C} from \Phi = [W_I, W_C]."""
        w_full = self.reconstructor.weight  # [hidden_dim, intent_dim + context_dim]
        w_i = w_full[:, :self.intent_dim]
        w_c = w_full[:, self.intent_dim:]
        return w_i, w_c

    def geodesic_distance(self, z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        r"""
        Computes exact geodesic distance on the unit hypersphere S^{d-1}:
        d_g(z1, z2) = arccos( clamp(<z1, z2>, -1, 1) )
        """
        cos_sim = torch.sum(z1 * z2, dim=-1)
        cos_sim = torch.clamp(cos_sim, -1.0 + eps, 1.0 - eps)
        return torch.acos(cos_sim)

    def compute_decomposition_loss(
        self,
        h_clean: torch.Tensor,
        h_adv: torch.Tensor,
        lambda_ortho: float = 0.1,
        lambda_rec: float = 0.05
    ):
        """
        Computes the Intent-Context Decomposition loss:
        1. Invariance loss: E[d_g(z_intent(h_clean), z_intent(h_adv))]
        2. Orthogonality loss: ||(1/B) Z_I^T Z_C||_F^2 + ||W_I^T W_C||_F^2
        3. Reconstruction loss: MSE(h_recon, h)
        """
        z_int_c, z_ctx_c, h_rec_c = self.forward(h_clean)
        z_int_a, z_ctx_a, h_rec_a = self.forward(h_adv)

        # 1. Geodesic intent invariance under adversarial transformation
        d_g_batch = self.geodesic_distance(z_int_c, z_int_a)
        l_inv = torch.mean(d_g_batch)

        # 2. Statistical cross-covariance orthogonality + geometric subspace orthogonality
        z_int_flat = torch.cat([z_int_c.reshape(-1, self.intent_dim), z_int_a.reshape(-1, self.intent_dim)], dim=0)
        z_ctx_flat = torch.cat([z_ctx_c.reshape(-1, self.context_dim), z_ctx_a.reshape(-1, self.context_dim)], dim=0)
        cross_cov = torch.matmul(z_int_flat.T, z_ctx_flat) / float(z_int_flat.size(0))
        l_cov_ortho = torch.sum(cross_cov ** 2)  # Squared Frobenius norm ||(1/B) Z_I^T Z_C||_F^2

        w_i, w_c = self.get_subspace_weights()
        w_i_norm = F.normalize(w_i, p=2, dim=0)
        w_c_norm = F.normalize(w_c, p=2, dim=0)
        subspace_overlap = torch.matmul(w_i_norm.T, w_c_norm)
        l_sub_ortho = torch.sum(subspace_overlap ** 2) / float(self.intent_dim)

        l_ortho = l_cov_ortho + l_sub_ortho

        # 3. Radial-preserving reconstruction fidelity
        l_rec = 0.5 * (F.mse_loss(h_rec_c, h_clean) + F.mse_loss(h_rec_a, h_adv))

        total_loss = l_inv + lambda_ortho * l_ortho + lambda_rec * l_rec
        return {
            "total_loss": total_loss,
            "l_inv": l_inv.item(),
            "l_ortho": l_ortho.item(),
            "l_rec": l_rec.item(),
            "d_g_max": d_g_batch.max().item(),
            "d_g_p95": torch.quantile(d_g_batch.float(), 0.95).item()
        }
