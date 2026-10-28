"""
dataset.py - Dataset generation and representation pairing for ARETE.
Constructs paired latent representations of benign, borderline dual-use benign,
direct violation, and adversarially cloaked prompt injections across 19 harm categories.
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

from arete.data.harm_taxonomy import HARM_CATEGORIES, NUM_HARM_CATEGORIES


class InvariantRepresentationDataset(Dataset):
    """
    Dataset of paired representations for training and evaluating Bundle-Orthogonal
    Intent-Context Decomposition and Geodesic Safety Barrier (GSB).

    Models nonlinear feature entanglement in Transformer residual streams, including
    hard dual-use benign queries (e.g., defensive cybersecurity / chemistry queries
    sharing topical vocabulary with harmful requests).
    """
    def __init__(
        self,
        num_samples: int = 4000,
        hidden_dim: int = 1024,
        intent_dim: int = 256,
        noise_std: float = 0.08,
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

        harm_raw = torch.randn(NUM_HARM_CATEGORIES, intent_dim)
        nn.init.orthogonal_(harm_raw)
        self.harm_prototypes = torch.nn.functional.normalize(harm_raw, p=2, dim=-1)

        benign_raw = torch.randn(32, intent_dim)
        proj_harm = torch.matmul(benign_raw, self.harm_prototypes.T)
        benign_raw = benign_raw - 0.80 * torch.matmul(proj_harm, self.harm_prototypes)
        self.benign_prototypes = torch.nn.functional.normalize(benign_raw, p=2, dim=-1)

        # 5 Adversarial context bases:
        # 0: Direct, 1: GCG Suffix, 2: PAIR Roleplay, 3: AutoDAN Cloaking, 4: Indirect Injection
        adv_ctx_raw = torch.randn(5, self.context_dim)
        nn.init.orthogonal_(adv_ctx_raw)
        self.attack_context_bases = torch.nn.functional.normalize(adv_ctx_raw, p=2, dim=-1)

        # 8 Benign context bases
        benign_ctx_raw = torch.randn(8, self.context_dim)
        nn.init.orthogonal_(benign_ctx_raw)
        self.benign_context_bases = torch.nn.functional.normalize(benign_ctx_raw, p=2, dim=-1)

        raw_mixing = torch.randn(hidden_dim, hidden_dim)
        q, _ = torch.linalg.qr(raw_mixing)
        self.mixing_matrix = q
        self.coupling_matrix = torch.randn(intent_dim, self.context_dim) * (0.65 / (self.context_dim ** 0.5))

        self.data = self._generate_data()

    def _compose_residual(self, z_int: torch.Tensor, z_ctx: torch.Tensor, radial_scale: float = 12.0) -> torch.Tensor:
        """Models nonlinear intent-context entanglement in Transformer residual streams."""
        nonlinear_interference = 0.38 * torch.sin(2.5 * torch.matmul(self.coupling_matrix, z_ctx))
        coupled_int = z_int + nonlinear_interference
        coupled_int = torch.nn.functional.normalize(coupled_int, p=2, dim=-1)
        z_full = torch.cat([coupled_int, z_ctx], dim=-1)
        return radial_scale * torch.matmul(z_full, self.mixing_matrix)

    def _generate_data(self) -> List[Dict[str, torch.Tensor]]:
        data = []
        half = self.num_samples // 2

        # 1. Harmful / Adversarial paired samples (50%)
        for i in range(half):
            cat_id = i % NUM_HARM_CATEGORIES
            base_intent = self.harm_prototypes[cat_id] + torch.randn(self.intent_dim) * self.noise_std
            base_intent = torch.nn.functional.normalize(base_intent, p=2, dim=-1)

            clean_ctx = self.benign_context_bases[random.randint(0, 7)] + torch.randn(self.context_dim) * self.noise_std
            clean_ctx = torch.nn.functional.normalize(clean_ctx, p=2, dim=-1)

            attack_type = i % 5
            benign_cloak = self.benign_context_bases[random.randint(0, 7)]
            adv_ctx = (
                0.70 * self.attack_context_bases[attack_type]
                + 0.30 * benign_cloak
                + torch.randn(self.context_dim) * (self.noise_std * 1.5)
            )
            adv_ctx = torch.nn.functional.normalize(adv_ctx, p=2, dim=-1)

            adv_intent = base_intent + torch.randn(self.intent_dim) * (self.noise_std * 0.45)
            adv_intent = torch.nn.functional.normalize(adv_intent, p=2, dim=-1)

            r_clean = 12.0 + random.uniform(-1.5, 1.5)
            r_adv = 12.0 + random.uniform(-3.0, 4.0)
            h_clean = self._compose_residual(base_intent, clean_ctx, radial_scale=r_clean)
            h_adv = self._compose_residual(adv_intent, adv_ctx, radial_scale=r_adv)

            severity = HARM_CATEGORIES[cat_id].base_severity.value

            data.append({
                "h_clean": h_clean,
                "h_adv": h_adv,
                "category_id": torch.tensor(cat_id, dtype=torch.long),
                "is_harmful": torch.tensor(1.0, dtype=torch.float32),
                "severity": torch.tensor(severity, dtype=torch.long),
                "attack_type": torch.tensor(attack_type, dtype=torch.long)
            })

        # 2. Benign & Hard Dual-Use Benign paired samples (50%)
        for i in range(half):
            b_idx = i % 32
            base_intent = self.benign_prototypes[b_idx] + torch.randn(self.intent_dim) * self.noise_std
            is_dual_use = (i % 4 == 0)
            if is_dual_use:
                adj_cat = i % NUM_HARM_CATEGORIES
                base_intent = 0.76 * base_intent + 0.24 * self.harm_prototypes[adj_cat]
            base_intent = torch.nn.functional.normalize(base_intent, p=2, dim=-1)

            clean_ctx = self.benign_context_bases[random.randint(0, 7)] + torch.randn(self.context_dim) * self.noise_std
            clean_ctx = torch.nn.functional.normalize(clean_ctx, p=2, dim=-1)

            if is_dual_use:
                var_ctx = (
                    0.55 * self.benign_context_bases[random.randint(0, 7)]
                    + 0.45 * self.attack_context_bases[random.randint(1, 4)]
                    + torch.randn(self.context_dim) * self.noise_std
                )
            else:
                var_ctx = self.benign_context_bases[random.randint(0, 7)] + torch.randn(self.context_dim) * (self.noise_std * 1.2)
            var_ctx = torch.nn.functional.normalize(var_ctx, p=2, dim=-1)

            var_intent = base_intent + torch.randn(self.intent_dim) * (self.noise_std * 0.40)
            var_intent = torch.nn.functional.normalize(var_intent, p=2, dim=-1)

            r_clean = 12.0 + random.uniform(-2.0, 2.0)
            r_var = 12.0 + random.uniform(-3.5, 4.5)
            h_clean = self._compose_residual(base_intent, clean_ctx, radial_scale=r_clean)
            h_adv = self._compose_residual(var_intent, var_ctx, radial_scale=r_var)

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
    train_ratio: float = 0.7,
    num_samples: int = 4000,
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
