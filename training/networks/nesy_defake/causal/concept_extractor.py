"""
Semantic Concept Extractor
"""

import torch
import torch.nn as nn


class SemanticConceptExtractor(nn.Module):
    """Extract interpretable semantic concepts from latent variables"""
    
    def __init__(self, latent_dim, num_concepts):
        super().__init__()
        
        self.num_concepts = num_concepts
        
        # Concept extraction network
        self.concept_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_concepts),
            nn.Sigmoid()  # Concepts as probabilities [0, 1]
        )
    
    def forward(self, z):
        """
        Args:
            z: (B, latent_dim) - concatenated latent variables
        Returns:
            concepts: (B, num_concepts) - semantic concept activations
        """
        concepts = self.concept_net(z)
        return concepts
    
    def get_top_concepts(self, concepts, k=5):
        """
        Get top-k activated concepts for interpretability
        
        Args:
            concepts: (B, num_concepts)
            k: number of top concepts to return
        Returns:
            top_values: (B, k)
            top_indices: (B, k)
        """
        top_values, top_indices = torch.topk(concepts, k, dim=1)
        return top_values, top_indices
