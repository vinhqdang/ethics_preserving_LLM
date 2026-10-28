"""
hook.py - Transformer Residual Stream Hook for ARETE Inference-Time Steering.
Attaches to intermediate Transformer layers (e.g., layer L/2 in Qwen / Llama),
decomposes the residual activations, evaluates the Geodesic Safety Barrier,
and applies Orthogonal Null-Space Steering during autoregressive token generation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

from arete.models.manifold import IntentContextDecomposer
from arete.models.barrier import GeodesicSafetyBarrier


class TransformerActivationHook:
    """
    Inference-time hook that intercepts residual stream activations in HuggingFace models.
    Supports auto-regressive decoding, kv-caching, and multi-token prompt evaluation.
    """
    def __init__(
        self,
        model: nn.Module,
        decomposer: IntentContextDecomposer,
        barrier: GeodesicSafetyBarrier,
        layer_idx: int = 14,
        enabled: bool = True,
        use_nullspace: bool = True
    ):
        self.model = model
        self.decomposer = decomposer
        self.barrier = barrier
        self.layer_idx = layer_idx
        self.enabled = enabled
        self.use_nullspace = use_nullspace

        # Intercept metrics
        self.intervention_history = []
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self):
        target_layer = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            target_layer = self.model.model.layers[self.layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            target_layer = self.model.transformer.h[self.layer_idx]
        elif hasattr(self.model, "layers"):
            target_layer = self.model.layers[self.layer_idx]
        else:
            raise ValueError("Unsupported model architecture for residual hook registration.")

        def hook_fn(module, args, output):
            if not self.enabled:
                return output

            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output

            # Decompose intermediate hidden states
            z_intent, z_context, _ = self.decomposer(h)

            # Closed-loop Geodesic Null-Space Intervention
            h_steered, info = self.barrier(h, z_intent, z_context, use_nullspace=self.use_nullspace)

            self.intervention_history.append({
                "violation_gate": info["violation_gate"].detach().cpu(),
                "max_margin": info["max_margin"].detach().cpu(),
                "argmax_cat": info["argmax_category"].detach().cpu()
            })

            if is_tuple:
                return (h_steered, *output[1:])
            return h_steered

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def clear_history(self):
        self.intervention_history.clear()

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
