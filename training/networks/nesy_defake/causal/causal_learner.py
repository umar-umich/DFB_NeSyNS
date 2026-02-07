"""
Causal Structure Learner
"""

import torch
import torch.nn as nn


class StructuralCausalCircuits(nn.Module):
    """Learn DAG structure using differentiable causal discovery"""
    
    def __init__(self, num_variables, hidden_dim, num_layers):
        super().__init__()
        
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Learnable adjacency matrix (will be constrained to be a DAG)
        self.adjacency_logits = nn.Parameter(
            torch.randn(num_variables, num_variables) * 0.1
        )
        
        # MLP for each variable conditioned on parents
        self.mlps = nn.ModuleList([
            self._build_mlp(num_variables, hidden_dim, num_layers)
            for _ in range(num_variables)
        ])
    
    def _build_mlp(self, input_dim, hidden_dim, num_layers):
        """Build MLP for structural equations"""
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, concepts):
        """
        Learn and return the adjacency matrix
        
        Args:
            concepts: (B, num_variables) - semantic concepts
        Returns:
            dag: (num_variables, num_variables) - learned adjacency matrix
        """
        # Get adjacency matrix with sigmoid for [0, 1] range
        dag = torch.sigmoid(self.adjacency_logits)
        
        # Enforce no self-loops
        dag = dag * (1 - torch.eye(self.num_variables, device=dag.device))
        
        return dag
    
    def compute_dag_penalty(self):
        """
        Compute acyclicity penalty: h(W) = tr(e^W) - d
        
        This constraint ensures the learned graph is a DAG (no cycles)
        """
        dag = torch.sigmoid(self.adjacency_logits)
        dag = dag * (1 - torch.eye(self.num_variables, device=dag.device))
        
        # Matrix exponential trace penalty
        # tr(e^W) = d iff W is acyclic
        expm = torch.matrix_exp(dag)
        penalty = torch.trace(expm) - self.num_variables
        
        return torch.abs(penalty)  # Make it positive
    
    def get_causal_parents(self, node_idx, threshold=0.5):
        """
        Get causal parents of a node
        
        Args:
            node_idx: index of the node
            threshold: edge weight threshold
        Returns:
            parent_indices: list of parent node indices
        """
        dag = torch.sigmoid(self.adjacency_logits)
        dag = dag * (1 - torch.eye(self.num_variables, device=dag.device))
        
        parents = dag[:, node_idx]
        parent_indices = (parents > threshold).nonzero(as_tuple=True)[0]
        
        return parent_indices.cpu().tolist()
