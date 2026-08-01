"""
NeSyDeFake Neural Networks
Modular components for deepfake detection
"""

# Foundation Models
# NOTE: TemporalFeatureExtractor + FrequencyFeatureExtractor archived to attic/
# (dead code; spatial-only pipeline). Only the spatial extractor remains.
from .foundation_models import (
    SpatialFeatureExtractor,
)

# Fusion Modules
from .fusion import (
    MultiModalFusion,
    CrossModalAttention
)

# Causal Modules
from .causal import (
    CausalDiscoveryModule
)

# Classifiers
from .classifiers import (
    MultiTaskHead,
    DualBranchSparseAutoencoder
)

__all__ = [
    # Foundation models
    'SpatialFeatureExtractor',
    # Fusion
    'MultiModalFusion',
    'CrossModalAttention',
    # Causal
    'LatentVariableEncoder',
    'SemanticConceptExtractor',
    'StructuralCausalCircuits',
    'CausalReasoner',
    'CausalDiscoveryModule',
    # Classifiers
    'MultiTaskHead',
    'DualBranchSparseAutoencoder'
]
