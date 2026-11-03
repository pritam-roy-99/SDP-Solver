"""Finite-party Bell-NPA and BFF conditional-entropy research tools."""

from .npa import NPA
from .bff import BFFNPA, BFFResult, binary_correlator, parse_bell

__version__ = "0.1.0"
__all__ = ["NPA", "BFFNPA", "BFFResult", "binary_correlator", "parse_bell"]
