"""
Causal Reasoner
"""

import torch
import torch.nn as nn


class CausalReasoner(nn.Module):
    """Compute causal violation scores using interventions"""
    
    def __init__(self, num_concepts):
        super().__init__()
        self.num_concepts = num_concepts
    
    def forward(self, concepts, dag):
        """
        Compute causal violation scores
        
        Args:
            concepts: (B, num_concepts) - observed concept values
            dag: (num_concepts, num_concepts) - adjacency matrix
        Returns:
            violation_scores: (B,) - causal violation scores per sample
        """
        batch_size = concepts.size(0)
        
        # Compute expected values based on causal parents
        violation_scores = []
        
        for i in range(self.num_concepts):
            # Get parents of concept i (incoming edges)
            parents = dag[:, i]  # (num_concepts,)
            
            # Expected value based on parents
            # E[X_i | parents] ≈ weighted sum of parent values
            expected = torch.matmul(concepts, parents.unsqueeze(1))  # (B, 1)
            
            # Violation = |observed - expected|
            violation = torch.abs(concepts[:, i:i+1] - expected)
            violation_scores.append(violation)
        
        # Average violation across all concepts
        total_violation = torch.cat(violation_scores, dim=1).mean(dim=1)
        
        return total_violation
    
    def compute_intervention_effect(self, concepts, dag, node_idx, intervention_value):
        """
        Compute P(Y | do(X=x)) - interventional distribution
        
        Args:
            concepts: (B, num_concepts)
            dag: (num_concepts, num_concepts)
            node_idx: index of node to intervene on
            intervention_value: value to set the node to
        Returns:
            intervened_concepts: (B, num_concepts) - concepts after intervention
        """
        intervened_concepts = concepts.clone()
        intervened_concepts[:, node_idx] = intervention_value
        
        # Propagate intervention effect through causal graph
        # (Simplified version - in practice, use structural equations)
        children = dag[node_idx, :] > 0.5
        for child_idx in children.nonzero(as_tuple=True)[0]:
            # Update child based on new parent value
            parents = dag[:, child_idx]
            expected = torch.matmul(intervened_concepts, parents)
            intervened_concepts[:, child_idx] = expected
        
        return intervened_concepts
    
    def compute_counterfactual(self, concepts, dag, node_idx, counterfactual_value):
        """
        Compute counterfactual: What if X had been x?
        
        Args:
            concepts: (B, num_concepts) - observed values
            dag: (num_concepts, num_concepts)
            node_idx: index of node for counterfactual
            counterfactual_value: counterfactual value
        Returns:
            counterfactual_concepts: (B, num_concepts)
        """
        # Step 1: Abduction - infer exogenous variables U
        # (Simplified - assumes U = observed - expected)
        
        # Step 2: Action - intervene on X
        counterfactual_concepts = self.compute_intervention_effect(
            concepts, dag, node_idx, counterfactual_value
        )
        
        # Step 3: Prediction - compute resulting values
        return counterfactual_concepts
