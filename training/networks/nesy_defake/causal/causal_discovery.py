"""
Causal Discovery Module - Main orchestrator for causal reasoning
"""

import logging
import torch
import torch.nn as nn
from typing import List

from .latent_encoder import LatentVariableEncoder
from .concept_extractor import SemanticConceptExtractor
from .causal_learner import StructuralCausalCircuits
from .causal_reasoner import CausalReasoner

logger = logging.getLogger(__name__)


class CausalDiscoveryModule(nn.Module):
    """
    Module for learning causal structure from features
    
    Supports:
    1. Implicit concepts (learned from latent variables)
    2. Explicit concepts (from DeepFace semantic grounding)
    3. Combined (both implicit and explicit)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config['causal_module']
        
        # Check if semantic grounding is enabled
        self.use_explicit_concepts = config.get('semantic_grounding', {}).get('enabled', False)
        
        # Determine input dimension
        if self.use_explicit_concepts:
            input_dim = config.get('semantic_grounding', {}).get('fuse_with_features', True)
            if input_dim:
                # Using fused features (implicit + explicit)
                input_dim = 256  # Output from SemanticGroundingModule
            else:
                # Using only explicit concepts
                input_dim = self._calculate_semantic_dim(config)
        else:
            # Using only implicit features
            input_dim = config['fusion']['projection_dim']
        
        # Latent variable encoder
        self.latent_encoder = LatentVariableEncoder(
            input_dim=input_dim,
            z_spatial_dim=self.config['latent_variables']['z_spatial_dim'],
            z_temporal_dim=self.config['latent_variables']['z_temporal_dim'],
            z_frequency_dim=self.config['latent_variables']['z_frequency_dim']
        )
        
        # Calculate dimensions
        total_latent_dim = self.config['latent_variables']['total_latent_dim']
        explicit_concept_dim = 0
        if self.use_explicit_concepts:
            explicit_concept_dim = self._calculate_semantic_dim(config)
        
        # Semantic concept extractor (for implicit concepts)
        num_implicit_concepts = self.config['semantic_concepts']['concept_dim']
        self.concept_extractor = SemanticConceptExtractor(
            latent_dim=total_latent_dim,
            num_concepts=num_implicit_concepts
        )
        
        # Total concepts for causal learning
        self.num_total_concepts = num_implicit_concepts + explicit_concept_dim
        
        # Causal structure learner
        algorithm = self.config['discovery']['algorithm']
        if algorithm == 'structural_causal_circuits':
            self.causal_learner = StructuralCausalCircuits(
                num_variables=self.num_total_concepts,
                hidden_dim=self.config['dag_learning']['hidden_dim'],
                num_layers=self.config['dag_learning']['num_layers']
            )
        else:
            raise NotImplementedError(f"Algorithm {algorithm} not implemented")
        
        # Causal reasoning module
        self.causal_reasoner = CausalReasoner(
            num_concepts=self.num_total_concepts
        )
        
        logger.info(
            f"Causal Discovery Module initialized: "
            f"implicit_concepts={num_implicit_concepts}, "
            f"explicit_concepts={explicit_concept_dim}, "
            f"total_concepts={self.num_total_concepts}"
        )
    
    def _calculate_semantic_dim(self, config):
        """Calculate dimension of explicit semantic concepts"""
        sg_config = config.get('semantic_grounding', {})
        dim = 0
        if sg_config.get('extract_age', False):
            dim += 1
        if sg_config.get('extract_gender', False):
            dim += 2
        if sg_config.get('extract_emotion', False):
            dim += 7
        if sg_config.get('extract_race', False):
            dim += 6
        return dim
    
    def forward(self, features, explicit_concepts=None, return_graph=False):
        """
        Args:
            features: (B, D) - grounded features
            explicit_concepts: (B, E) - explicit semantic concepts (optional)
            return_graph: whether to return the learned DAG
        Returns:
            violation_score: (B,) - causal violation scores
            dag: adjacency matrix (optional)
            all_concepts: (B, total_concepts) - all concepts
        """
        # Extract latent variables from features
        z_spatial, z_temporal, z_frequency = self.latent_encoder(features)
        z_all = torch.cat([z_spatial, z_temporal, z_frequency], dim=1)
        
        # Extract implicit semantic concepts
        implicit_concepts = self.concept_extractor(z_all)
        
        # Combine with explicit concepts if available
        if explicit_concepts is not None and self.use_explicit_concepts:
            all_concepts = torch.cat([implicit_concepts, explicit_concepts], dim=1)
        else:
            all_concepts = implicit_concepts
        
        # Learn causal structure
        dag = self.causal_learner(all_concepts)
        
        # Compute causal violation scores
        violation_scores = self.causal_reasoner(all_concepts, dag)
        
        if return_graph:
            return violation_scores, dag, all_concepts
        return violation_scores
    
    def get_concept_names(self, config) -> List[str]:
        """Get names of all concepts (implicit + explicit)"""
        # Implicit concept names
        implicit_names = config['causal_module']['semantic_concepts']['concepts']
        
        # Explicit concept names (if enabled)
        explicit_names = []
        if self.use_explicit_concepts:
            sg_config = config.get('semantic_grounding', {})
            if sg_config.get('extract_age'):
                explicit_names.append('age')
            if sg_config.get('extract_gender'):
                explicit_names.extend(['gender_man', 'gender_woman'])
            if sg_config.get('extract_emotion'):
                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                explicit_names.extend([f'emotion_{e}' for e in emotions])
            if sg_config.get('extract_race'):
                races = ['asian', 'indian', 'black', 'white', 'middle_eastern', 'latino_hispanic']
                explicit_names.extend([f'race_{r}' for r in races])
        
        return implicit_names + explicit_names
