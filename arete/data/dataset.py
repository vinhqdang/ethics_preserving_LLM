"""
dataset.py - Dataset generation and representation pairing for ARETE.
Constructs paired latent representations of benign, direct violation,
and adversarially cloaked prompt injections across harm categories.
"""

import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from arete.data.harm_taxonomy import HARM_CATEGORIES, NUM_HARM_CATEGORIES, DEFAULT_THRESHOLDS


class InvariantRepresentationDataset(Dataset):
    """
    Dataset of paired representations for training Lie-Equivariant Intent-Context
    Decomposition (LE-ICD) and Geodesic Safety Barrier (GSB).
    
    Each sample contains:
    - h_clean: Latent representation of clean intent (benign or harmful baseline)
    - h_adv: Latent representation of the same intent cloaked under adversarial transformation
    - category_id: Ground truth harm taxonomy category (-1 for benign)
    - is_harmful: Binary label (1 if violating policy, 0 if benign)
    - severity: Severity level (0: negligible, 1: minor, 2: moderate, 3: severe)
    """
    def __init__(
        self,
        num_samples: int = 2400,
        hidden_dim: int = 1024,
        intent_dim: int = 256,
        noise_std: float = 0.05,
        seed: int = 42
    ):
        super().__init__()
        self.num_samples = num_samples
        self.hidden_dim = hidden_dim
        self.intent_dim = intent_dim
        self.context_dim = hidden_dim - intent_dim
        self.noise_std = noise_std
        
        torch.manual_seed(seed)
        random.seed(seed)
        
        # Ground-truth canonical intent prototypes for each harm category + benign
        # Shape: [NUM_HARM_CATEGORIES, intent_dim]
        self.harm_prototypes = torch.randn(NUM_HARM_CATEGORIES, intent_dim)
        self.harm_prototypes = torch.nn.functional.normalize(self.harm_prototypes, p=2, dim=-1)
        
        # Benign intent prototype subspace (32 orthogonal modes)
        self.benign_prototypes = torch.randn(32, intent_dim)
        self.benign_prototypes = torch.nn.functional.normalize(self.benign_prototypes, p=2, dim=-1)
        
        # Adversarial context transformation bases (simulating 5 attack paradigms:
        # 0: Direct, 1: Suffix optimization, 2: Roleplay/hypothetical framing, 3: Syntactic cloaking, 4: Indirect injection)
        self.attack_context_bases = torch.randn(5, self.context_dim)
        self.attack_context_bases = torch.nn.functional.normalize(self.attack_context_bases, p=2, dim=-1)
        
        # Benign context bases (e.g. conversational, academic, formal tone)
        self.benign_context_bases = torch.randn(8, self.context_dim)
        self.benign_context_bases = torch.nn.functional.normalize(self.benign_context_bases, p=2, dim=-1)
        
        # Random orthogonal reconstruction matrix to project (z_intent, z_context) -> h
        raw_mixing = torch.randn(hidden_dim, hidden_dim)
        q, _ = torch.linalg.qr(raw_mixing)
        self.mixing_matrix = q # Orthogonal matrix for faithful representation manifold
        
        # Pre-synthesize dataset tensors
        self.data = self._generate_data()

    def _generate_data(self) -> List[Dict[str, torch.Tensor]]:
        data = []
        half = self.num_samples // 2
        
        # 1. Generate Harmful / Adversarial paired samples (50%)
        for i in range(half):
            cat_id = i % NUM_HARM_CATEGORIES
            base_intent = self.harm_prototypes[cat_id] + torch.randn(self.intent_dim) * self.noise_std
            base_intent = torch.nn.functional.normalize(base_intent, p=2, dim=-1)
            
            # Clean context (neutral phrasing)
            clean_ctx = self.benign_context_bases[random.randint(0, 7)] + torch.randn(self.context_dim) * self.noise_std
            clean_ctx = torch.nn.functional.normalize(clean_ctx, p=2, dim=-1)
            
            # Adversarial cloaked context (e.g., GCG suffix, roleplay framing)
            attack_type = random.randint(0, 4)
            adv_ctx = self.attack_context_bases[attack_type] + torch.randn(self.context_dim) * (self.noise_std * 2.0)
            adv_ctx = torch.nn.functional.normalize(adv_ctx, p=2, dim=-1)
            
            # Invariant intent under transformation tau
            adv_intent = base_intent + torch.randn(self.intent_dim) * (self.noise_std * 0.5)
            adv_intent = torch.nn.functional.normalize(adv_intent, p=2, dim=-1)
            
            # Project to residual stream space h = mixing * [intent; context]
            z_clean = torch.cat([base_intent, clean_ctx], dim=-1)
            z_adv = torch.cat([adv_intent, adv_ctx], dim=-1)
            
            h_clean = torch.matmul(z_clean, self.mixing_matrix)
            h_adv = torch.matmul(z_adv, self.mixing_matrix)
            
            severity = HARM_CATEGORIES[cat_id].base_severity.value
            
            data.append({
                "h_clean": h_clean,
                "h_adv": h_adv,
                "category_id": torch.tensor(cat_id, dtype=torch.long),
                "is_harmful": torch.tensor(1.0, dtype=torch.float32),
                "severity": torch.tensor(severity, dtype=torch.long),
                "attack_type": torch.tensor(attack_type, dtype=torch.long)
            })
            
        # 2. Generate Benign paired samples (50%) (ensuring zero over-refusal)
        for i in range(half):
            b_idx = random.randint(0, 31)
            base_intent = self.benign_prototypes[b_idx] + torch.randn(self.intent_dim) * self.noise_std
            base_intent = torch.nn.functional.normalize(base_intent, p=2, dim=-1)
            
            # Clean context
            clean_ctx = self.benign_context_bases[random.randint(0, 7)] + torch.randn(self.context_dim) * self.noise_std
            clean_ctx = torch.nn.functional.normalize(clean_ctx, p=2, dim=-1)
            
            # Benign prompt with complex/adversarial-looking framing (e.g. security audit questions)
            var_ctx = self.benign_context_bases[random.randint(0, 7)] + torch.randn(self.context_dim) * (self.noise_std * 1.5)
            var_ctx = torch.nn.functional.normalize(var_ctx, p=2, dim=-1)
            
            var_intent = base_intent + torch.randn(self.intent_dim) * (self.noise_std * 0.5)
            var_intent = torch.nn.functional.normalize(var_intent, p=2, dim=-1)
            
            z_clean = torch.cat([base_intent, clean_ctx], dim=-1)
            z_adv = torch.cat([var_intent, var_ctx], dim=-1)
            
            h_clean = torch.matmul(z_clean, self.mixing_matrix)
            h_adv = torch.matmul(z_adv, self.mixing_matrix)
            
            data.append({
                "h_clean": h_clean,
                "h_adv": h_adv,
                "category_id": torch.tensor(-1, dtype=torch.long),
                "is_harmful": torch.tensor(0.0, dtype=torch.float32),
                "severity": torch.tensor(0, dtype=torch.long),
                "attack_type": torch.tensor(-1, dtype=torch.long)
            })
            
        random.shuffle(data)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


def get_dataloaders(
    batch_size: int = 64,
    train_ratio: float = 0.8,
    num_samples: int = 3000,
    hidden_dim: int = 1024,
    intent_dim: int = 256
) -> Tuple[DataLoader, DataLoader]:
    """Creates train and validation DataLoader instances."""
    dataset = InvariantRepresentationDataset(
        num_samples=num_samples,
        hidden_dim=hidden_dim,
        intent_dim=intent_dim
    )
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
