"""
arete.eval - Empirical evaluation, benchmarks, and baselines.
"""

from arete.eval.baselines import (
    BaseUnaligned,
    SafeDPOBaseline,
    LinearRefusalSteering,
    ConditionalActivationSteering,
    ExternalGuardrailClassifier
)
from arete.eval.evaluate import run_comprehensive_benchmark

SafeRLHFBaseline = SafeDPOBaseline

__all__ = [
    "BaseUnaligned",
    "SafeDPOBaseline",
    "SafeRLHFBaseline",
    "LinearRefusalSteering",
    "ConditionalActivationSteering",
    "ExternalGuardrailClassifier",
    "run_comprehensive_benchmark"
]
