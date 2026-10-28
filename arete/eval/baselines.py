"""
baselines.py - Baseline implementations for comparative safety alignment evaluation.
Implements:
1. BaseUnaligned
2. SafeDPOBaseline (Preference-tuned boundary on entangled representations)
3. LinearRefusalSteering (RepEng difference-in-means 1D steering, Zou et al. 2023; Arditi et al. 2024)
4. ConditionalActivationSteering (CAST 1D condition-gated steering, Lee et al. 2024)
5. ExternalGuardrailClassifier (Multi-layer guardrail classifier proxy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class BaseUnaligned:
    """Unconstrained base model baseline."""
    def __call__(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return h, {"refusal": torch.zeros(h.size(0), dtype=torch.bool, device=h.device), "score": 0.0}


class SafeDPOBaseline(nn.Module):
    """
    Safe-DPO representation baseline.
    Operates on entangled representations without intent-context bundle decomposition.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.safety_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.safety_head.weight, std=0.02)
        self.refusal_shift = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.threshold = 0.50

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        logit = self.safety_head(h).squeeze(-1)
        prob = torch.sigmoid(logit)
        refusal = (prob > self.threshold)
        v_ref = F.normalize(self.refusal_shift, p=2, dim=-1)
        r_h = torch.norm(h, p=2, dim=-1, keepdim=True)
        h_out = h + refusal.unsqueeze(-1).to(h.dtype) * (0.35 * r_h * v_ref)
        return h_out, {"refusal": refusal, "prob": prob}


class LinearRefusalSteering(nn.Module):
    """
    Representation Engineering baseline (Zou et al. 2023; Arditi et al. 2024).
    Computes a 1D refusal direction v_refusal and applies isotropic additive steering
    h_steered = h + alpha * v_refusal. Because it lacks null-space projection and
    operates continuously along a 1D axis, it perturbs benign representations and
    triggers elevated false refusals on borderline benign queries.
    """
    def __init__(self, hidden_dim: int, alpha: float = 0.35):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.refusal_direction = nn.Parameter(torch.randn(hidden_dim) * 0.05)
        self.detection_head = nn.Linear(hidden_dim, 1)

    def fit_difference_in_means(self, h_harm: torch.Tensor, h_benign: torch.Tensor):
        with torch.no_grad():
            diff = h_harm.mean(dim=0) - h_benign.mean(dim=0)
            self.refusal_direction.copy_(F.normalize(diff, p=2, dim=-1))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        v_norm = F.normalize(self.refusal_direction, p=2, dim=-1)
        r_h = torch.norm(h, p=2, dim=-1, keepdim=True)
        # Unconditional isotropic shift along 1D refusal vector (standard activation addition)
        h_steered = h + self.alpha * r_h * v_norm
        logit = self.detection_head(h).squeeze(-1) + 0.35 * torch.sum(F.normalize(h, p=2, dim=-1) * v_norm, dim=-1)
        prob = torch.sigmoid(logit)
        refusal = (prob > 0.44)
        return h_steered, {"refusal": refusal, "prob": prob}


class ConditionalActivationSteering(nn.Module):
    """
    Conditional Activation Steering (CAST, Lee et al. 2024).
    Gates a 1D refusal steering vector on a single linear condition vector c_cond
    in raw hidden space without intent-context disentanglement or null-space projection.
    """
    def __init__(self, hidden_dim: int, alpha: float = 0.45):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.condition_head = nn.Linear(hidden_dim, 1)
        self.refusal_direction = nn.Parameter(torch.randn(hidden_dim) * 0.05)
        self.threshold = 0.48

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        logit = self.condition_head(h).squeeze(-1)
        prob = torch.sigmoid(logit)
        gate = (prob > self.threshold)
        v_norm = F.normalize(self.refusal_direction, p=2, dim=-1)
        r_h = torch.norm(h, p=2, dim=-1, keepdim=True)
        h_steered = h + gate.unsqueeze(-1).to(h.dtype) * (self.alpha * r_h * v_norm)
        return h_steered, {"refusal": gate, "prob": prob}


class ExternalGuardrailClassifier(nn.Module):
    """
    External Guardrail baseline (proxy for input/output moderation classifiers).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        self.threshold = 0.50

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        logit = self.classifier(h).squeeze(-1)
        prob = torch.sigmoid(logit)
        refusal = (prob > self.threshold)
        return h, {"refusal": refusal, "prob": prob}
