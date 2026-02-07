"""
Multi-Task Classification Head
"""

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """Multi-task classifier for final predictions"""
    
    def __init__(self, config):
        super().__init__()
        
        input_dim = config['classifier']['input_dim']
        hidden_dims = config['classifier']['hidden_dims']
        dropout = config['classifier']['dropout']
        
        # Shared backbone
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_backbone = nn.Sequential(*layers)
        
        # Task-specific heads
        self.tasks = config['classifier']['tasks']
        self.task_heads = nn.ModuleDict()
        
        for task in self.tasks:
            task_name = task['name']
            output_dim = task['output_dim']
            
            if task['type'] == 'binary':
                # Binary classification
                self.task_heads[task_name] = nn.Linear(prev_dim, output_dim)
            elif task['type'] == 'regression':
                # Regression task
                self.task_heads[task_name] = nn.Sequential(
                    nn.Linear(prev_dim, output_dim),
                    nn.Sigmoid()  # Bound regression outputs to [0, 1]
                )
            elif task['type'] == 'multi_class':
                # Multi-class classification
                self.task_heads[task_name] = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, input_dim) - input features
        Returns:
            outputs: dict - predictions for each task
        """
        shared_features = self.shared_backbone(x)
        
        outputs = {}
        for task in self.tasks:
            task_name = task['name']
            outputs[task_name] = self.task_heads[task_name](shared_features)
        
        return outputs
    
    def get_task_names(self):
        """Get list of task names"""
        return [task['name'] for task in self.tasks]
