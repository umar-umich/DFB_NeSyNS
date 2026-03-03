"""
Classifier Modules
"""

from .multitask_head import MultiTaskHead
from .sparse_autoencoder import DualBranchSparseAutoencoder

__all__ = [
    'MultiTaskHead',
    'DualBranchSparseAutoencoder'
]
