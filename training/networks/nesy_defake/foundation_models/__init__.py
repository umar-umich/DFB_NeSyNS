"""
Foundation Model Feature Extractors

NOTE: TemporalFeatureExtractor and FrequencyFeatureExtractor were ARCHIVED to
attic/training/networks/nesy_defake/foundation_models/ (dead code — the active
pipeline is spatial-only, active_branches=['spatial']). Only the spatial
extractor remains.
"""

from .spatial_extractor import SpatialFeatureExtractor

__all__ = [
    'SpatialFeatureExtractor',
]
