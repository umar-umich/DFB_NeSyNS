"""
networks/nesy_defake/losses/unifalign.py
========================================
Label-aware Uniformity-Alignment losses (Wang & Isola, 2020).

https://arxiv.org/pdf/2005.10242

Applied as an auxiliary regularizer on L2-normalized embeddings to shape
class manifolds on the unit hypersphere:
  * alignment — pull same-class pairs together
  * uniformity — spread embeddings evenly over the sphere

Improves cross-dataset generalization by preventing feature collapse onto
a line (which overfits to training-distribution directions) and encouraging
a dense, uniform manifold where novel test-time features still have
same-class neighbors.

Both functions assume the input is already L2-normalized to the unit sphere.
"""

import torch


def alignment(embeddings: torch.Tensor,
              labels: torch.Tensor,
              alpha: float = 2.0) -> torch.Tensor:
    """Label-aware alignment over same-class pairs within a batch.

    Vectorized form (upper-triangle of the pairwise equal-label mask) —
    equivalent to the per-class loop but ~4x faster on large batches.

    Args:
        embeddings: (N, D) unit-normalized embeddings.
        labels:     (N,)   class labels.
        alpha:      power on the squared L2 distance (Wang & Isola: 2.0).

    Returns:
        Scalar loss. Zero if the batch has no positive pairs.
    """
    if embeddings.size(0) < 2:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    pos_mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    pos_idx = torch.nonzero(pos_mask, as_tuple=False)
    if pos_idx.numel() == 0:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    x = embeddings[pos_idx[:, 0]]
    y = embeddings[pos_idx[:, 1]]
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(embeddings: torch.Tensor,
               t: float = 2.0,
               clip_value: float = 1e-6) -> torch.Tensor:
    """Uniformity on the unit hypersphere (log-exp-mean-negative-sq-dist).

    Args:
        embeddings: (N, D) unit-normalized embeddings.
        t:          temperature (Wang & Isola: 2.0).
        clip_value: floor on the mean before log (numerical safety).
    """
    if embeddings.size(0) < 2:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    return (torch.pdist(embeddings, p=2)
            .pow(2).mul(-t).exp()
            .mean().clamp(min=clip_value).log())
