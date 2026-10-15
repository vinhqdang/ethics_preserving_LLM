"""
trainer.py - End-to-end training pipeline for ARETE architecture.
Jointly optimizes the Lie-Equivariant Intent-Context Decomposition (LE-ICD)
and Geodesic Safety Barrier (GSB).
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from arete.models.manifold import IntentContextDecomposer
from arete.models.barrier import GeodesicSafetyBarrier
from arete.data.dataset import get_dataloaders


class AreteTrainer:
    """
    Manages joint optimization of ARETE modules.
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
        
        # Joint parameter optimizer
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
            
            # 2. Extract intent for barrier evaluation
            z_int_c, z_ctx_c, _ = self.decomposer(h_clean)
            z_int_a, z_ctx_a, _ = self.decomposer(h_adv)
            
            # Barrier loss on both clean and adversarial intents
            barrier_res_c = self.barrier.compute_barrier_loss(z_int_c, is_harmful, category_id)
            barrier_res_a = self.barrier.compute_barrier_loss(z_int_a, is_harmful, category_id)
            l_barrier = 0.5 * (barrier_res_c["barrier_loss"] + barrier_res_a["barrier_loss"])
            
            # Combined objective
            loss = l_decomp + 1.2 * l_barrier
            loss.backward()
            
            # Gradient clipping for geometric stability
            nn.utils.clip_grad_norm_(self.decomposer.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(self.barrier.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_inv += decomp_res["l_inv"]
            total_ortho += decomp_res["l_ortho"]
            total_rec += decomp_res["l_rec"]
            total_barrier += l_barrier.item()
            n_batches += 1
            
        self.scheduler.step()
        
        return {
            "loss": total_loss / n_batches,
            "loss_inv": total_inv / n_batches,
            "loss_ortho": total_ortho / n_batches,
            "loss_rec": total_rec / n_batches,
            "loss_barrier": total_barrier / n_batches
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.decomposer.eval()
        self.barrier.eval()
        
        correct_detection = 0
        total_harmful = 0
        false_refusals = 0
        total_benign = 0
        total_inv_dist = 0.0
        n_samples = 0
        
        for batch in dataloader:
            h_clean = batch["h_clean"].to(self.device)
            h_adv = batch["h_adv"].to(self.device)
            category_id = batch["category_id"].to(self.device)
            is_harmful = batch["is_harmful"].to(self.device)
            batch_size = h_clean.size(0)
            
            z_int_c, z_ctx_c, _ = self.decomposer(h_clean)
            z_int_a, z_ctx_a, _ = self.decomposer(h_adv)
            
            # Geodesic distance on intent under adversarial transformation
            inv_dist = self.decomposer.geodesic_distance(z_int_c, z_int_a).sum().item()
            total_inv_dist += inv_dist
            
            # Evaluate barrier on adversarial representations
            projections, margins = self.barrier.evaluate_violation(z_int_a)
            max_margins, _ = torch.max(margins, dim=-1)
            predicted_violation = (max_margins > 0.0)
            
            harm_mask = (is_harmful > 0.5)
            benign_mask = ~harm_mask
            
            correct_detection += (predicted_violation[harm_mask] == True).sum().item()
            total_harmful += harm_mask.sum().item()
            
            false_refusals += (predicted_violation[benign_mask] == True).sum().item()
            total_benign += benign_mask.sum().item()
            n_samples += batch_size
            
        tpr = (correct_detection / total_harmful) if total_harmful > 0 else 0.0
        fpr = (false_refusals / total_benign) if total_benign > 0 else 0.0
        mean_inv_dist = total_inv_dist / n_samples
        
        return {
            "detection_rate_tpr": tpr,
            "false_refusal_fpr": fpr,
            "mean_geodesic_invariance_dist": mean_inv_dist,
            "eval_samples": n_samples
        }


def run_training_experiment(
    epochs: int = 12,
    batch_size: int = 64,
    hidden_dim: int = 1024,
    intent_dim: int = 256
) -> Dict[str, Any]:
    """Runs end-to-end training and evaluation of ARETE."""
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        num_samples=3200,
        hidden_dim=hidden_dim,
        intent_dim=intent_dim
    )
    
    decomposer = IntentContextDecomposer(hidden_dim=hidden_dim, intent_dim=intent_dim)
    barrier = GeodesicSafetyBarrier(intent_dim=intent_dim, hidden_dim=hidden_dim)
    
    trainer = AreteTrainer(decomposer, barrier)
    
    history = []
    print(f"=== Starting ARETE Training on Device: {trainer.device} ===")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        t_metrics = trainer.train_epoch(train_loader)
        v_metrics = trainer.evaluate(val_loader)
        record = {
            "epoch": epoch,
            **t_metrics,
            **v_metrics
        }
        history.append(record)
        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Loss: {t_metrics['loss']:.4f} (Inv: {t_metrics['loss_inv']:.4f}, Ortho: {t_metrics['loss_ortho']:.4f}) | "
                f"Detection TPR: {v_metrics['detection_rate_tpr']*100:.2f}% | "
                f"False Refusal FPR: {v_metrics['false_refusal_fpr']*100:.2f}% | "
                f"Geodesic Invariance Dist: {v_metrics['mean_geodesic_invariance_dist']:.4f}"
            )
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s.")
    
    return {
        "history": history,
        "final_metrics": history[-1],
        "training_time": total_time,
        "decomposer": trainer.decomposer,
        "barrier": trainer.barrier
    }
