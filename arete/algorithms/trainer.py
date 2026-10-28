"""
trainer.py - End-to-end training pipeline for ARETE architecture.
Jointly optimizes the Bundle-Orthogonal Intent-Context Decomposer
and Geodesic Safety Barrier (GSB), including closed-loop steering matrices.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from arete.models.manifold import IntentContextDecomposer
from arete.models.barrier import GeodesicSafetyBarrier
from arete.data.dataset import get_dataloaders


class AreteTrainer:
    """
    Manages joint optimization of ARETE modules (Decomposer + Barrier + Null-Space Steerer).
    """
    def __init__(
        self,
        decomposer: IntentContextDecomposer,
        barrier: GeodesicSafetyBarrier,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.decomposer = decomposer.to(self.device)
        self.barrier = barrier.to(self.device)

        params = list(self.decomposer.parameters()) + list(self.barrier.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=15, eta_min=1e-5)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.decomposer.train()
        self.barrier.train()

        total_loss = 0.0
        total_inv = 0.0
        total_ortho = 0.0
        total_rec = 0.0
        total_barrier = 0.0
        total_steer_safe = 0.0
        total_ctx_preserve = 0.0
        n_batches = 0

        for batch in dataloader:
            h_clean = batch["h_clean"].to(self.device)
            h_adv = batch["h_adv"].to(self.device)
            category_id = batch["category_id"].to(self.device)
            is_harmful = batch["is_harmful"].to(self.device)

            self.optimizer.zero_grad()

            # 1. Decomposition forward & loss
            decomp_res = self.decomposer.compute_decomposition_loss(h_clean, h_adv)
            l_decomp = decomp_res["total_loss"]

            # 2. Extract intent & context for barrier and closed-loop steering evaluation
            z_int_c, z_ctx_c, _ = self.decomposer(h_clean)
            z_int_a, z_ctx_a, _ = self.decomposer(h_adv)

            # Barrier + Closed-Loop Steering loss on both clean and adversarial representations
            barrier_res_c = self.barrier.compute_barrier_loss(
                z_int_c, is_harmful, category_id,
                h=h_clean, z_context=z_ctx_c, decomposer=self.decomposer
            )
            barrier_res_a = self.barrier.compute_barrier_loss(
                z_int_a, is_harmful, category_id,
                h=h_adv, z_context=z_ctx_a, decomposer=self.decomposer
            )
            l_barrier = 0.5 * (barrier_res_c["barrier_loss"] + barrier_res_a["barrier_loss"])

            loss = l_decomp + 1.2 * l_barrier
            loss.backward()

            nn.utils.clip_grad_norm_(self.decomposer.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(self.barrier.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_inv += decomp_res["l_inv"]
            total_ortho += decomp_res["l_ortho"]
            total_rec += decomp_res["l_rec"]
            total_barrier += l_barrier.item()
            total_steer_safe += 0.5 * (barrier_res_c["loss_steer_safe"] + barrier_res_a["loss_steer_safe"])
            total_ctx_preserve += 0.5 * (barrier_res_c["loss_ctx_preserve"] + barrier_res_a["loss_ctx_preserve"])
            n_batches += 1

        self.scheduler.step()

        return {
            "loss": total_loss / n_batches,
            "loss_inv": total_inv / n_batches,
            "loss_ortho": total_ortho / n_batches,
            "loss_rec": total_rec / n_batches,
            "loss_barrier": total_barrier / n_batches,
            "loss_steer_safe": total_steer_safe / n_batches,
            "loss_ctx_preserve": total_ctx_preserve / n_batches
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.decomposer.eval()
        self.barrier.eval()

        correct_detection = 0
        post_steering_safe = 0
        total_harmful = 0
        false_refusals = 0
        total_benign = 0
        all_d_g = []
        ctx_preserve_sims = []

        for batch in dataloader:
            h_clean = batch["h_clean"].to(self.device)
            h_adv = batch["h_adv"].to(self.device)
            is_harmful = batch["is_harmful"].to(self.device)

            z_int_c, z_ctx_c, _ = self.decomposer(h_clean)
            z_int_a, z_ctx_a, _ = self.decomposer(h_adv)

            d_g = self.decomposer.geodesic_distance(z_int_c, z_int_a)
            all_d_g.append(d_g.cpu())

            # Evaluate barrier and closed-loop steering on adversarial representations
            h_steered, info = self.barrier(h_adv, z_int_a, z_ctx_a, use_nullspace=True)
            max_margins = info["max_margin"].squeeze(-1)
            predicted_violation = (max_margins > 0.0)

            harm_mask = (is_harmful > 0.5)
            benign_mask = ~harm_mask

            if harm_mask.sum() > 0:
                correct_detection += (predicted_violation[harm_mask] == True).sum().item()
                total_harmful += harm_mask.sum().item()

                # Check post-steering safety & context preservation on harmful samples
                z_int_post, z_ctx_post, _ = self.decomposer(h_steered[harm_mask])
                _, margins_post = self.barrier.evaluate_violation(z_int_post)
                max_margin_post, _ = torch.max(margins_post, dim=-1)
                post_steering_safe += (max_margin_post <= 0.0).sum().item()

                ctx_sim = F.cosine_similarity(z_ctx_a[harm_mask], z_ctx_post, dim=-1)
                ctx_preserve_sims.append(ctx_sim.cpu())

            if benign_mask.sum() > 0:
                false_refusals += (predicted_violation[benign_mask] == True).sum().item()
                total_benign += benign_mask.sum().item()

        all_d_g_tensor = torch.cat(all_d_g, dim=0).float()
        tpr = (correct_detection / total_harmful) if total_harmful > 0 else 0.0
        fpr = (false_refusals / total_benign) if total_benign > 0 else 0.0
        post_safe_rate = (post_steering_safe / total_harmful) if total_harmful > 0 else 0.0
        mean_ctx_sim = torch.cat(ctx_preserve_sims, dim=0).mean().item() if ctx_preserve_sims else 1.0

        return {
            "detection_rate_tpr": tpr,
            "false_refusal_fpr": fpr,
            "post_steering_safe_rate": post_safe_rate,
            "context_preservation_cossim": mean_ctx_sim,
            "mean_geodesic_invariance_dist": all_d_g_tensor.mean().item(),
            "p95_geodesic_invariance_dist": torch.quantile(all_d_g_tensor, 0.95).item(),
            "max_geodesic_invariance_dist": all_d_g_tensor.max().item(),
            "eval_samples": int(all_d_g_tensor.numel())
        }


def run_training_experiment(
    epochs: int = 14,
    batch_size: int = 64,
    hidden_dim: int = 1024,
    intent_dim: int = 256
) -> Dict[str, Any]:
    """Runs end-to-end training and evaluation of ARETE."""
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        num_samples=4000,
        hidden_dim=hidden_dim,
        intent_dim=intent_dim
    )
    decomposer = IntentContextDecomposer(hidden_dim=hidden_dim, intent_dim=intent_dim)
    barrier = GeodesicSafetyBarrier(intent_dim=intent_dim, hidden_dim=hidden_dim)
    trainer = AreteTrainer(decomposer, barrier)
    history = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t_metrics = trainer.train_epoch(train_loader)
        v_metrics = trainer.evaluate(val_loader)
        history.append({"epoch": epoch, **t_metrics, **v_metrics})
    return {
        "history": history,
        "final_metrics": history[-1],
        "training_time": time.time() - start_time,
        "decomposer": trainer.decomposer,
        "barrier": trainer.barrier
    }
