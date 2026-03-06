"""
networks/nesy_defake/classifiers/multitask_head.py
===================================================
Module 5: Multi-Task Head with fully independent per-task MLPs.

Each task gets its own trunk (hidden layers) + output head. No shared
parameters between tasks — no gradient interference.

Why separate trunks matter:
  Classification and uncertainty estimation require different feature
  spaces. With a shared trunk, the classification gradient (cross-entropy
  on logits) and the uncertainty gradient (MSE on a [0,1] calibration
  target) compete for the same weights, causing the trunk to compromise
  between two conflicting objectives. Separate trunks let each task
  specialise independently.

Architecture (per task):
  x (B, input_dim)
    └─→ Linear(input_dim, H0) → LayerNorm(H0) → GELU → Dropout
    └─→ Linear(H0, H1)        → LayerNorm(H1) → GELU → Dropout
    └─→ output_head            → task output

Config (classifier section of YAML):
  input_dim:   1024
  hidden_dims: [512, 256]     # applied independently to every task trunk
  dropout:     0.4
  tasks:
    - name: classification
      type: binary             # → Linear(last_hidden, output_dim)
      output_dim: 2
    - name: uncertainty
      type: regression         # → Linear(...) → Sigmoid  (bounded [0,1])
      output_dim: 1
"""

import torch
import torch.nn as nn


def _build_trunk(input_dim: int, hidden_dims: list, dropout: float):
    """
    Build one task's hidden-layer trunk.

    Uses LayerNorm + GELU (not BatchNorm + ReLU) because:
      - LayerNorm is instance-independent → correct at batch size 1 (inference)
      - GELU is smooth → better gradient flow for the shallow depths here
    """
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        layers.append(nn.LayerNorm(h_dim))
        layers.append(nn.GELU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_dim = h_dim
    return nn.Sequential(*layers), prev_dim


class TaskMLP(nn.Module):
    """
    One fully independent task module: trunk + output head.

    Wrapped in its own nn.Module so each task's parameters are clearly
    namespaced (e.g. task_mlps.classification.trunk.0.weight).
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float,
                 output_dim: int, task_type: str):
        super().__init__()
        self.task_type = task_type

        trunk, last_dim = _build_trunk(input_dim, hidden_dims, dropout)
        self.trunk = trunk

        if task_type in ('binary', 'multi_class'):
            self.head = nn.Linear(last_dim, output_dim)
        elif task_type == 'regression':
            # Sigmoid bounds output to [0, 1] — correct for uncertainty/violation
            self.head = nn.Sequential(
                nn.Linear(last_dim, output_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown task type: '{task_type}'. "
                             f"Expected 'binary', 'multi_class', or 'regression'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class MultiTaskHead(nn.Module):
    """
    Multi-task classifier with one independent MLP per task.

    Each task in config['classifier']['tasks'] gets its own trunk + head.
    The trunks share the same hyperparameters (hidden_dims, dropout) but
    have completely separate parameters — no weight sharing, no gradient
    interference between tasks.
    """

    def __init__(self, config: dict):
        super().__init__()

        cls_cfg     = config['classifier']
        input_dim   = cls_cfg['input_dim']          # 1024
        hidden_dims = cls_cfg['hidden_dims']         # [512, 256]
        dropout     = cls_cfg['dropout']             # 0.4
        tasks_cfg   = cls_cfg['tasks']

        self.task_names = [t['name'] for t in tasks_cfg]

        # One independent TaskMLP per task — no shared parameters
        self.task_mlps = nn.ModuleDict()
        for task in tasks_cfg:
            self.task_mlps[task['name']] = TaskMLP(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
                output_dim=task['output_dim'],
                task_type=task['type'],
            )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, input_dim) fused features (+ causal residual projection)
        Returns:
            dict mapping task name → task output tensor
              'classification' → (B, 2)   logits (no softmax; use with CrossEntropy)
              'uncertainty'    → (B, 1)   in [0, 1] via Sigmoid
              'violation_score'→ (B, 1)   in [0, 1] via Sigmoid  (if configured)
        """
        return {name: self.task_mlps[name](x) for name in self.task_names}

    def get_task_names(self) -> list:
        return self.task_names
