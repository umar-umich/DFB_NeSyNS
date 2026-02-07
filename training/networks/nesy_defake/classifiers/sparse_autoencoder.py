"""
Sparse Autoencoder for Monosemantic Features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for discovering interpretable features
    
    Uses L1 sparsity penalty to encourage learning of monosemantic
    (single-meaning) features that are easier to interpret.
    """
    
    def __init__(self, config):
        super().__init__()
        
        input_dim = config['sparse_features']['sparse_autoencoder']['input_dim']
        expansion_factor = config['sparse_features']['sparse_autoencoder']['expansion_factor']
        self.hidden_dim = input_dim * expansion_factor
        
        self.l1_coef = config['sparse_features']['sparse_autoencoder']['l1_coefficient']
        
        # Encoder: compress to sparse overcomplete representation
        self.encoder = nn.Linear(input_dim, self.hidden_dim)
        
        # Decoder: reconstruct original features
        self.decoder = nn.Linear(self.hidden_dim, input_dim)
        
        # Track feature importance
        self.register_buffer('feature_importance', torch.zeros(self.hidden_dim))
        self.register_buffer('feature_activations', torch.zeros(self.hidden_dim))
        
    def forward(self, x, update_stats=False):
        """
        Args:
            x: (B, input_dim) - input features
            update_stats: whether to update feature importance statistics
        Returns:
            hidden: (B, hidden_dim) - sparse features
            loss: scalar - reconstruction + sparsity loss
        """
        # Encode with ReLU activation for sparsity
        hidden = F.relu(self.encoder(x))
        
        # Decode
        reconstructed = self.decoder(hidden)
        
        # Compute losses
        reconstruction_loss = F.mse_loss(reconstructed, x)
        sparsity_loss = self.l1_coef * torch.abs(hidden).sum(dim=1).mean()
        
        total_loss = reconstruction_loss + sparsity_loss
        
        # Update feature statistics (for interpretability)
        if update_stats and self.training:
            with torch.no_grad():
                # Track which features are active
                active = (hidden > 0).float().mean(dim=0)
                self.feature_activations = 0.99 * self.feature_activations + 0.01 * active
                
                # Track feature importance (magnitude)
                importance = hidden.abs().mean(dim=0)
                self.feature_importance = 0.99 * self.feature_importance + 0.01 * importance
        
        return hidden, total_loss
    
    def get_top_features(self, k=10):
        """
        Get top-k most important features for interpretability
        
        Returns:
            top_indices: (k,) - indices of most important features
            top_scores: (k,) - importance scores
        """
        top_scores, top_indices = torch.topk(self.feature_importance, k)
        return top_indices, top_scores
    
    def get_feature_sparsity(self):
        """
        Compute average sparsity of learned features
        
        Returns:
            sparsity: percentage of features active on average
        """
        return self.feature_activations.mean().item()
