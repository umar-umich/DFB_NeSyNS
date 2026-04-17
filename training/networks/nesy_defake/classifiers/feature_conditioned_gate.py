"""
FeatureConditionedGate — per-image evidence-branch gating.

gate_values = sigmoid(Linear(spatial_raw))  -> (B, num_gates)

Replaces static scalar gates when evidence_gate.conditioned=true — lets
the model trust the causal branch more on compressed images, the concept
branch more on clear frontal faces, etc. Weights zero-init, biases seeded
from config so gates start at the same sigmoid values as the scalar mode.
"""

import torch
import torch.nn as nn


class FeatureConditionedGate(nn.Module):
    def __init__(self, input_dim: int, num_gates: int, gate_inits=None):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_gates)
        with torch.no_grad():
            nn.init.zeros_(self.linear.weight)
            if gate_inits is not None:
                for i, val in enumerate(gate_inits):
                    self.linear.bias[i] = val
            else:
                nn.init.constant_(self.linear.bias, -1.0)

    def forward(self, spatial_raw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(spatial_raw))
