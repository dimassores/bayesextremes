"""
BayesExtremes - Bayesian methods for extreme value analysis
"""

__version__ = "0.1.0"
__author__ = "Dimas Soares"
__email__ = "dimassoareslima@gmail.com"

from bayesextremes.models import (
    MixedGammaParetoModel,
    PoissonMixture,
    GEV
)

__all__ = [
    "MixedGammaParetoModel",
    "PoissonMixture",
    "GEV"
]
