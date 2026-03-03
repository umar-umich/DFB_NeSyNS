"""
NeSyDeFake Neural Networks
Modular components for deepfake detection
"""

# Foundation Models
from .foundation_models import (
    TemporalFeatureExtractor,
    SpatialFeatureExtractor,
    FrequencyFeatureExtractor
)

# Fusion Modules
from .fusion import (
    MultiModalFusion,
    CrossModalAttention
)

# Causal Modules
from .causal import (
    LatentVariableEncoder,
    SemanticConceptExtractor,
    StructuralCausalCircuits,
    CausalReasoner,
    CausalDiscoveryModule
)

# Classifiers
from .classifiers import (
    MultiTaskHead,
    DualBranchSparseAutoencoder
)

__all__ = [
    # Foundation models
    'TemporalFeatureExtractor',
    'SpatialFeatureExtractor',
    'FrequencyFeatureExtractor',
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
