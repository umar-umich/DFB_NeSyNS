"""
Evidence fusion for NeSy-DeFake.

Encapsulates the three evidence-fusion strategies used in Ablations 2/3/4:

  1. NeSy-EDL CMEF     : ConfidenceModulatedEvidenceFusion — per-sample,
                         confidence-based branch weighting in EDL space.
                         (Functional successor of CausalViolationAttentionFusion.)
  2. Conditioned gates : FeatureConditionedGate — per-image sigmoid gates
                         conditioned on the spatial backbone features.
  3. Static gates      : scalar sigmoid(Parameter) gates, one per branch.

Also owns the gate parameters/modules so the detector doesn't have to.
`forward(...)` returns a dict with the fused evidence plus every
diagnostic scalar the detector used to inline into the prediction dict.
"""

from typing import Optional

import torch
import torch.nn as nn

from networks.nesy_defake.classifiers import FeatureConditionedGate


class EvidenceFusion(nn.Module):
    def __init__(
        self,
        use_concept: bool,
        use_causal: bool,
        use_nesy_edl: bool,
        conditioned: bool,
        gate_cfg: dict,
        backbone_dim: int,
        edl_num_classes: int = 2,
    ):
        super().__init__()
        self.use_concept = use_concept
        self.use_causal = use_causal
        self.use_nesy_edl = use_nesy_edl
        self.conditioned = conditioned and (use_concept or use_causal)

        concept_init = float(gate_cfg.get('concept_init', -0.5))
        causal_init = float(gate_cfg.get('causal_init', -0.8))

        # Static scalar gates — always built if the branch exists, used as
        # fallback when neither CMEF nor conditioned gating is active.
        if use_concept:
            self.concept_gate = nn.Parameter(torch.tensor(concept_init))
        if use_causal:
            self.causal_ev_gate = nn.Parameter(torch.tensor(causal_init))

        # NeSy-EDL CMEF
        if use_nesy_edl and (use_concept or use_causal):
            from networks.nesy_defake.losses.nesy_edl_loss import (
                ConfidenceModulatedEvidenceFusion)
            gate_inits = []
            if use_concept:
                gate_inits.append(concept_init)
            if use_causal:
                gate_inits.append(causal_init)
            self.cmef = ConfidenceModulatedEvidenceFusion(
                num_symbolic_branches=len(gate_inits),
                num_classes=edl_num_classes,
                gate_inits=gate_inits,
            )

        # Per-image conditioned gates
        if self.conditioned:
            gate_inits = []
            if use_concept:
                gate_inits.append(concept_init)
            if use_causal:
                gate_inits.append(causal_init)
            self.conditioned_gate = FeatureConditionedGate(
                input_dim=backbone_dim,
                num_gates=len(gate_inits),
                gate_inits=gate_inits,
            )

    def forward(
        self,
        spatial_evidence: torch.Tensor,
        spatial_raw: torch.Tensor,
        concept_out: Optional[dict],
        causal_out: Optional[dict],
    ) -> dict:
        """Fuse branch evidences. Strategy priority: CMEF > conditioned > static.

        Returns dict with:
          total_evidence   : (B, K) fused Dirichlet evidence
          branch_evidences : {'spatial', 'concept'?, 'causal'?}
          cmef_diag        : dict or None (NeSy-EDL diagnostics)
          concept_gate, causal_gate : scalar or (B,1) tensor — for logging
          concept_conf, causal_conf : scalar tensors from CMEF (if active)
        """
        branch_evidences = {'spatial': spatial_evidence}
        out = {'concept_gate': None, 'causal_gate': None,
               'concept_conf': None, 'causal_conf': None,
               'cmef_diag': None}

        if self.use_nesy_edl and hasattr(self, 'cmef'):
            symbolic_evs = []
            if concept_out is not None:
                symbolic_evs.append(concept_out['evidence'])
                branch_evidences['concept'] = concept_out['evidence']
            if causal_out is not None:
                symbolic_evs.append(causal_out['evidence'])
                branch_evidences['causal'] = causal_out['evidence']
            total_evidence, diag = self.cmef(spatial_evidence, symbolic_evs)
            out['cmef_diag'] = diag
            gi = 0
            if concept_out is not None:
                out['concept_gate'] = diag.get(f'branch_{gi}_gate')
                out['concept_conf'] = diag.get(f'branch_{gi}_conf_mean')
                gi += 1
            if causal_out is not None:
                out['causal_gate'] = diag.get(f'branch_{gi}_gate')
                out['causal_conf'] = diag.get(f'branch_{gi}_conf_mean')

        elif self.conditioned and hasattr(self, 'conditioned_gate'):
            gate_vals = self.conditioned_gate(spatial_raw)     # (B, num_gates)
            total_evidence = spatial_evidence
            gi = 0
            if concept_out is not None:
                g = gate_vals[:, gi].unsqueeze(1)
                total_evidence = total_evidence + g * concept_out['evidence']
                branch_evidences['concept'] = concept_out['evidence']
                out['concept_gate'] = g.mean().detach()
                gi += 1
            if causal_out is not None:
                g = gate_vals[:, gi].unsqueeze(1)
                total_evidence = total_evidence + g * causal_out['evidence']
                branch_evidences['causal'] = causal_out['evidence']
                out['causal_gate'] = g.mean().detach()

        else:
            total_evidence = spatial_evidence
            if concept_out is not None:
                g = torch.sigmoid(self.concept_gate)
                total_evidence = total_evidence + g * concept_out['evidence']
                branch_evidences['concept'] = concept_out['evidence']
                out['concept_gate'] = g.detach()
            if causal_out is not None:
                g = torch.sigmoid(self.causal_ev_gate)
                total_evidence = total_evidence + g * causal_out['evidence']
                branch_evidences['causal'] = causal_out['evidence']
                out['causal_gate'] = g.detach()

        out['total_evidence'] = total_evidence
        out['branch_evidences'] = branch_evidences
        return out
