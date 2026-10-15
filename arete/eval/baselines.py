"""
baselines.py - Baseline implementations for comparative safety alignment evaluation.
Implements:
1. Base Unaligned
2. Standard Safe-RLHF / DPO representation proxy
3. Linear Representation Engineering (Refusal Direction Steering, Zou et al. 2023)
4. External Input Guardrail Classifier (Llama-Guard / Beaver-Guard proxy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class BaseUnaligned:
    """Unconstrained base model baseline."""
    def __call__(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return h, {"refusal": False, "score": 0.0}


class SafeRLHFBaseline(nn.Module):
    """
    Standard Safe-RLHF / DPO representation baseline.
    Trained with a linear reward/safety discriminator on entangled representations.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.safety_head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.safety_head.weight, std=0.02)
        self.threshold = 0.5

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Scores the safety probability
        logit = self.safety_head(h).squeeze(-1)
        prob = torch.sigmoid(logit)
        refusal = (prob > self.threshold)
        return h, {"refusal": refusal, "prob": prob}


class LinearRefusalSteering(nn.Module):
    """
    Representation Engineering baseline (Zou et al. 2023, Arditi et al. 2024).
    Extracts a singular linear refusal direction v_refusal and adds it to the residual stream:
    h_steered = h + alpha * v_refusal
    """
    def __init__(self, hidden_dim: int, alpha: float = 1.8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.refusal_direction = nn.Parameter(torch.randn(hidden_dim))
        nn.init.normal_(self.refusal_direction, std=0.05)
        self.detection_head = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        v_norm = F.normalize(self.refusal_direction, p=2, dim=-1)
        logit = self.detection_head(h).squeeze(-1)
        prob = torch.sigmoid(logit)
        refusal = (prob > 0.45)
        
        # When triggered, injects scalar refusal vector
        steering = self.alpha * v_norm
        h_steered = h + refusal.unsqueeze(-1).float() * steering
        return h_steered, {"refusal": refusal, "prob": prob}


class ExternalGuardrailClassifier(nn.Module):
    """
    External Guardrail baseline (proxy for Llama Guard / Beaver-Guard).
    Evaluates input features before model execution.
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
