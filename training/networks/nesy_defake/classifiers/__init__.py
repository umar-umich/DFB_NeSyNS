"""
Classifier + head modules
"""

from .multitask_head import MultiTaskHead
from .sparse_autoencoder import DualBranchSparseAutoencoder
from .projection_heads import (
    ResidualProjection,
    BottleneckAdapter,
    HoulsbyAdapter,
    build_projection_head,
    make_standard_projection,
)
from .feature_conditioned_gate import FeatureConditionedGate

__all__ = [
    'MultiTaskHead',
    'DualBranchSparseAutoencoder',
    'ResidualProjection',
    'BottleneckAdapter',
    'HoulsbyAdapter',
    'build_projection_head',
    'make_standard_projection',
    'FeatureConditionedGate',
]
