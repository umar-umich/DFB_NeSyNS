#!/usr/bin/env python3
"""
training/inspect_sample.py
===========================
Single-image forensic inspection for NeSy-DeFakeNet.

Usage (run from repo root or training/):
  python training/inspect_sample.py \\
    --config  training/config/detector/nesy_defake_ablation4_causal.yaml \\
    --weights path/to/best_model.pth \\
    --image   /data/.../frames/video_000/007.png \\
    --out_dir results/inspect/ \\
    [--label  0|1]              # 0=real, 1=fake  (enables GT annotation)
    [--out_dir results/myrun]   # default: results/<parent_folder>_<frame_stem>
    [--attrs  path.pt]          # override: explicit 58-d semantic tensor
    [--forensic path.pt]        # override: explicit 83-d forensic tensor
    [--attrs_subdir fast_semantic]    # subdir name used to find attrs .pt (default: fast_semantic)
    [--forensic_subdir forensic_features]  # subdir name for forensic .pt
    [--device cuda|cpu]

Feature auto-resolution
-----------------------
If --image points to a frame that lives inside a dataset directory tree
(i.e. the path contains a segment named 'frames' or 'frames_aug_*'):

    .../frames/VIDEO_NAME/FRAME.png

the script automatically finds:

    .../fast_semantic/VIDEO_NAME.pt        → precomputed_attrs  (58-d per frame)
    .../forensic_features/VIDEO_NAME.pt    → forensic_features  (83-d per frame)

These .pt files contain {'features': (N_frames, dim), 'frame_paths': [...]}
and the correct row is selected by matching FRAME.png — exactly as the
training dataset does.  Pass --attrs / --forensic to override with explicit
tensor files, or if the image is not in the standard directory tree.

Outputs written to out_dir/:
  sample_alignment.png         — r_diff bar chart: which subgraph is REAL/FAKE-aligned
  identity_broken_links.png    — causal graph: edges broken/created by fakes
  forensic_structural_broken_links.png
  forensic_noise_broken_links.png
  forensic_spectral_broken_links.png
  gradient_saliency.png        — pixel saliency  ∂ causal_signal / ∂ pixel
  rule_region_map.png          — top-firing consistency rules → face anatomy
  summary.json                 — all scalar predictions / evidence / uncertainty
"""

import argparse
import json
import os
import sys

# ── path setup ───────────────────────────────────────────────────────────────
# The repo is structured as DFB_NeSyNS/ (repo root) with training/ inside it.
# Several modules (fwa_blend, etc.) open files relative to the repo root at
# import time, so we must chdir there before any training imports happen.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # …/training
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)                   # …/DFB_NeSyNS
_ORIG_CWD    = os.getcwd()                                     # save before chdir
os.chdir(_REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

import torch
import yaml

# ── imports that require the training package on sys.path ────────────────────
from detectors import DETECTOR                                          # registry
from interpretability import visualization as viz
from interpretability.analyzers.scm_analysis import SCMAnalyzer


# ════════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════════

# CLIP normalization (OpenAI ViT, matches abstract_dataset.py defaults)
CLIP_MEAN = [0.481, 0.458, 0.408]
CLIP_STD  = [0.269, 0.261, 0.276]

# Subgraph names — must match ImprovedCausalBranch stack order
SUBGRAPH_NAMES = ['identity', 'forensic_structural', 'forensic_noise', 'forensic_spectral']

# ── Rule-to-face-anatomy mapping ─────────────────────────────────────────────
# Each entry: rule_name -> list of (label, color, (x0, y0, x1, y1)) in 0-1 coords
# Coords are (left, top, right, bottom) relative to image dimensions.
_BROW    = (0.10, 0.26, 0.90, 0.40)
_L_EYE   = (0.10, 0.38, 0.43, 0.52)
_R_EYE   = (0.57, 0.38, 0.90, 0.52)
_NOSE   = (0.32, 0.44, 0.68, 0.66)
_MOUTH  = (0.22, 0.63, 0.78, 0.82)
_L_CHEEK = (0.05, 0.46, 0.36, 0.72)
_R_CHEEK = (0.64, 0.46, 0.95, 0.72)
_CHEEKS  = (0.05, 0.46, 0.95, 0.72)
_CHIN    = (0.28, 0.78, 0.72, 0.97)
_JAW     = (0.05, 0.68, 0.95, 0.97)
_FOREHEAD = (0.15, 0.05, 0.85, 0.27)
_FULL    = (0.04, 0.04, 0.96, 0.96)

RULE_REGIONS: dict = {
    'cr_mutual_mouth':           [('mouth', '#e377c2', _MOUTH)],
    'cr_mutual_gender':          [('face', '#7f7f7f', _FULL)],
    'cr_mutual_smile_frown':     [('mouth', '#e377c2', _MOUTH),
                                  ('cheeks', '#17becf', _CHEEKS)],
    'cr_lighting_conflict':      [('face', '#bcbd22', _FULL)],
    'cr_happy_au6':              [('L-cheek', '#17becf', _L_CHEEK),
                                  ('R-cheek', '#17becf', _R_CHEEK)],
    'cr_happy_au12':             [('mouth', '#e377c2', _MOUTH)],
    'cr_surprise_au1au2':        [('brow', '#ff7f0e', _BROW),
                                  ('forehead', '#ffbb78', _FOREHEAD)],
    'cr_sad_au15':               [('mouth', '#e377c2', _MOUTH)],
    'cr_angry_au4':              [('brow', '#d62728', _BROW)],
    'cr_fear_au1au5':            [('brow', '#ff7f0e', _BROW),
                                  ('L-eye', '#aec7e8', _L_EYE),
                                  ('R-eye', '#aec7e8', _R_EYE)],
    'cr_disgust_au9':            [('nose', '#98df8a', _NOSE)],
    'cr_contempt_au14':          [('L-cheek', '#17becf', _L_CHEEK),
                                  ('R-cheek', '#17becf', _R_CHEEK)],
    'cr_neutral_any_au':         [('face', '#7f7f7f', _FULL)],
    'cr_double_chin_narrow_jaw': [('jaw', '#8c564b', _JAW),
                                  ('chin', '#c49c94', _CHIN)],
    'cr_square_face_narrow_jaw': [('jaw', '#8c564b', _JAW)],
    'cr_round_face_pointed_chin': [('chin', '#c49c94', _CHIN),
                                   ('jaw', '#8c564b', _JAW)],
    'cr_skin_tone_conflict':     [('face', '#bcbd22', _FULL)],
    'cr_skin_age_acne':          [('face', '#7f7f7f', _FULL)],
    'cr_symmetry_conflict':      [('L-face', '#1f77b4', (0.03, 0.10, 0.50, 0.92)),
                                  ('R-face', '#d62728', (0.50, 0.10, 0.97, 0.92))],
    'cr_image_quality':          [('face', '#9467bd', _FULL)],
}


# ════════════════════════════════════════════════════════════════════════════
#  Image preprocessing
# ════════════════════════════════════════════════════════════════════════════

def load_image_tensor(path: str, size: int = 224) -> torch.Tensor:
    """Return CHW float32 tensor in CLIP-normalized space, NOT batched."""
    img = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0           # (H, W, 3)
    mean = np.array(CLIP_MEAN, dtype=np.float32)
    std  = np.array(CLIP_STD,  dtype=np.float32)
    arr  = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1))         # (3, H, W)


def load_image_display(path: str, size: int = 224) -> np.ndarray:
    """Return (H, W, 3) uint8 array for overlay displays."""
    img = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


# ════════════════════════════════════════════════════════════════════════════
#  Auto-resolve precomputed features from dataset directory tree
# ════════════════════════════════════════════════════════════════════════════

def _resolve_feature_from_pt(frame_path: str, subdir: str,
                              expected_dim: int) -> torch.Tensor | None:
    """
    Replicate the dataset's feature-loading logic for a single frame.

    Layout assumed:
        {base_dir}/frames[_aug_*]/{video_name}/{frame_num}.png
        {base_dir}/{subdir}/{video_name}.pt
            → {'features': (N, dim), 'frame_paths': [...]}

    Returns the (dim,) tensor for this frame, or None if not found.
    """
    sep = '/' if '/' in frame_path else '\\'
    parts = frame_path.split(sep)

    frames_idx = None
    for pi, part in enumerate(parts):
        if part == 'frames' or part.startswith('frames_aug_'):
            frames_idx = pi
            break
    if frames_idx is None or frames_idx + 1 >= len(parts):
        return None

    video_name = parts[frames_idx + 1]
    base_dir   = sep.join(parts[:frames_idx])
    pt_path    = os.path.join(base_dir, subdir, f'{video_name}.pt')

    if not os.path.exists(pt_path):
        return None

    try:
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        features    = data['features']              # (N, dim)
        frame_paths = data.get('frame_paths', [])
        frame_filename = parts[-1]

        # Prefer exact filename match
        if frame_paths:
            for idx, fp in enumerate(frame_paths):
                if str(fp).endswith(frame_filename):
                    return features[idx]

        # Fallback: treat stem as 0-indexed frame number
        frame_num = int(os.path.splitext(frame_filename)[0])
        if frame_num < features.shape[0]:
            return features[frame_num]
    except Exception as exc:
        print(f'  [warn] Could not read {pt_path}: {exc}')

    return None


def auto_resolve_features(image_path: str,
                           attrs_subdir: str = 'fast_semantic',
                           forensic_subdir: str = 'forensic_features',
                           attrs_dim: int = 58,
                           forensic_dim: int = 83,
                           ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Try to load precomputed attrs + forensic tensors from the dataset tree.
    Returns (attrs_tensor | None, forensic_tensor | None).
    """
    attrs    = _resolve_feature_from_pt(image_path, attrs_subdir,    attrs_dim)
    forensic = _resolve_feature_from_pt(image_path, forensic_subdir, forensic_dim)
    return attrs, forensic


# ════════════════════════════════════════════════════════════════════════════
#  Model loading
# ════════════════════════════════════════════════════════════════════════════

def build_model(config_path: str, weights_path: str, device: torch.device):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model_name']
    model_class = DETECTOR[model_name]
    model = model_class(config).to(device)

    ckpt = torch.load(weights_path, map_location=device)
    # Handle wrapped checkpoints: dict with 'state_dict' or 'model' key
    if isinstance(ckpt, dict):
        sd = ckpt.get('state_dict') or ckpt.get('model') or ckpt
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════
#  Forward pass (with optional gradient tracking)
# ════════════════════════════════════════════════════════════════════════════

def run_forward(model, img_tensor: torch.Tensor, device: torch.device,
                label: int = 0, attrs=None, forensic=None,
                compute_grad: bool = False):
    """
    Run one forward pass.

    Returns (pred_dict, img_for_grad_or_None) where img_for_grad has
    .grad populated after the call when compute_grad=True.
    """
    img = img_tensor.unsqueeze(0).to(device)   # (1, 3, H, W)
    lbl = torch.tensor([label], dtype=torch.long, device=device)

    if compute_grad:
        img = img.detach().requires_grad_(True)

    data_dict: dict = {
        'spatial_frames': img,
        'label':          lbl,
    }
    if attrs is not None:
        data_dict['precomputed_attrs'] = attrs.unsqueeze(0).to(device)
    if forensic is not None:
        data_dict['forensic_features'] = forensic.unsqueeze(0).to(device)

    ctx = torch.enable_grad() if compute_grad else torch.no_grad()
    with ctx:
        pred = model.forward(data_dict)

        if compute_grad:
            # The SCM branch detaches spatial_raw before the compressor
            # (z = compressor(spatial_raw.detach())) so r_diff_g / causal_evidence
            # have no grad path back to the pixels.  Walk the candidate list and
            # pick the first tensor that actually has a grad_fn.
            target = None
            for key in ('prob', 'spatial_evidence', 'concept_evidence',
                        'causal_evidence', 'r_diff_g'):
                v = pred.get(key)
                if v is not None and isinstance(v, torch.Tensor) and v.grad_fn is not None:
                    target = v.sum()
                    print(f'  saliency target: {key}')
                    break
            if target is not None:
                target.backward()
            else:
                print('  [warn] no differentiable output found — saliency skipped')

    return pred, (img if compute_grad else None)


# ════════════════════════════════════════════════════════════════════════════
#  Gradient saliency
# ════════════════════════════════════════════════════════════════════════════

def save_gradient_saliency(img_grad: torch.Tensor, display_img: np.ndarray,
                            save_path: str) -> None:
    """
    Overlay pixel-level saliency from ∂(causal_signal)/∂pixel on the image.

    Works with ViT (no spatial conv layers needed — pure input-gradient approach).
    """
    # img_grad: (1, 3, H, W)
    sal = img_grad.detach().cpu().squeeze(0).abs()   # (3, H, W)
    sal = sal.mean(0).numpy()                         # (H, W)
    # Smooth slightly and normalise
    from scipy.ndimage import gaussian_filter
    sal = gaussian_filter(sal, sigma=3)
    vmax = np.percentile(sal, 99) + 1e-8
    sal_norm = np.clip(sal / vmax, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Original
    axes[0].imshow(display_img)
    axes[0].set_title('Input image', fontsize=11)
    axes[0].axis('off')

    # Saliency map (standalone, viridis)
    im = axes[1].imshow(sal_norm, cmap='hot', vmin=0.0, vmax=1.0)
    axes[1].set_title('Gradient saliency\n'
                       r'$|\partial\,\mathrm{causal\,signal}/\partial\,\mathrm{pixel}|$',
                       fontsize=10)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(display_img)
    axes[2].imshow(sal_norm, cmap='hot', alpha=0.55, vmin=0.0, vmax=1.0)
    axes[2].set_title('Overlay', fontsize=11)
    axes[2].axis('off')

    fig.suptitle('Input-gradient saliency  (regions most influential on causal branch)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  Rule → face-region overlay
# ════════════════════════════════════════════════════════════════════════════

def save_rule_region_map(violations: np.ndarray, rule_names: list,
                          display_img: np.ndarray, save_path: str,
                          top_k: int = 6) -> None:
    """
    Draw top-K firing consistency rules as coloured bounding boxes.

    violations: (K,) array of per-rule violation scores for this sample.
    rule_names: list of K rule name strings.
    """
    H, W = display_img.shape[:2]
    n = min(len(violations), len(rule_names))
    scores = violations[:n]

    # Rank by absolute score magnitude
    order = np.argsort(-np.abs(scores))[:top_k]
    top_rules = [(rule_names[i], float(scores[i])) for i in order if abs(scores[i]) > 1e-4]
    if not top_rules:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(display_img)

    legend_handles = []
    drawn = set()

    for rule, score in top_rules:
        regions = RULE_REGIONS.get(rule)
        if regions is None:
            # fallback: full face with a generic colour
            regions = [(rule, '#7f7f7f', _FULL)]

        alpha = min(0.35 + 0.45 * min(abs(score), 1.0), 0.80)
        for rlabel, color, (x0, y0, x1, y1) in regions:
            px = int(x0 * W);  py = int(y0 * H)
            pw = int((x1 - x0) * W);  ph = int((y1 - y0) * H)
            rect = mpatches.FancyBboxPatch(
                (px, py), pw, ph,
                boxstyle='round,pad=2',
                linewidth=2, edgecolor=color,
                facecolor=color, alpha=alpha,
            )
            ax.add_patch(rect)
            key = (rule, rlabel)
            if key not in drawn:
                legend_handles.append(
                    mpatches.Patch(facecolor=color, alpha=0.7,
                                   label=f'{rule} ({score:+.3f})'))
                drawn.add(key)

    ax.legend(handles=legend_handles, fontsize=7,
              loc='lower center', bbox_to_anchor=(0.5, -0.02),
              ncol=2, frameon=True, framealpha=0.85)
    ax.set_title('Top consistency-rule violations → face regions', fontsize=10)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  Causal graph visualizations
# ════════════════════════════════════════════════════════════════════════════

def save_causal_graphs(pred: dict, model, scm_analyzer: SCMAnalyzer,
                        out_dir: str, prob_fake: float | None = None) -> None:
    """
    Render per-subgraph causal graph figures:
      {sg}_broken_links.png   — single-panel edge-category overview (technical)
      {sg}_verdict_flow.png   — evidence-routing flow to REAL/FAKE terminals (paper)

    Adjacency matrices are sourced first from the forward-pass pred_dict
    (returned by ImprovedCausalBranch), then from the model parameters
    directly via SCMAnalyzer.collect_adjacencies().
    """
    # Collect adjacencies into scm_analyzer._adjacencies
    scm_analyzer.collect_adjacencies(model)
    for k, v in pred.items():
        if k.startswith('A_') and k not in scm_analyzer._adjacencies:
            if isinstance(v, torch.Tensor):
                scm_analyzer._adjacencies[k] = v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray):
                scm_analyzer._adjacencies[k] = v

    # Per-subgraph r_diff magnitudes (all non-negative)
    rdiff_sg: dict = {}
    if 'r_diff_g' in pred:
        rdiff_arr = pred['r_diff_g'].detach().cpu().numpy().flatten()
        for i, sg in enumerate(SUBGRAPH_NAMES):
            if i < len(rdiff_arr):
                rdiff_sg[sg] = float(rdiff_arr[i])

    for sg in SUBGRAPH_NAMES:
        A_real = scm_analyzer._adjacencies.get(f'A_{sg}_real')
        A_fake = scm_analyzer._adjacencies.get(f'A_{sg}_fake')
        if A_real is None or A_fake is None:
            continue
        names = scm_analyzer._node_names.get(sg, [])
        n = min(A_real.shape[0], len(names))

        # Per-node activation for this sample (squeeze batch dim)
        act = None
        inp_key = f'scm_input_{sg}'
        if inp_key in pred:
            v = pred[inp_key]
            if isinstance(v, torch.Tensor):
                act = v.detach().cpu().numpy().squeeze(0)
            elif isinstance(v, np.ndarray):
                act = v.squeeze(0) if v.ndim > 1 else v

        # Technical overview: single-panel broken/preserved/spurious
        bl_path = os.path.join(out_dir, f'{sg}_broken_links.png')
        viz.plot_scm_broken_links(
            A_real[:n, :n], A_fake[:n, :n], names[:n],
            title=f'Causal graph — {sg}',
            save_path=bl_path,
            n_latent=6, n_other=6,
        )
        print(f'  saved {bl_path}')

        # Paper figure: evidence-routing flow to REAL / FAKE verdict terminals
        vf_path = os.path.join(out_dir, f'{sg}_verdict_flow.png')
        viz.plot_scm_verdict_flow(
            A_real[:n, :n], A_fake[:n, :n], names[:n],
            title=f'Causal evidence routing — {sg}',
            save_path=vf_path,
            n_latent=6, n_other=6,
            act=act,
            r_diff_val=rdiff_sg.get(sg),
            prob_fake=prob_fake,
        )
        print(f'  saved {vf_path}')


# ════════════════════════════════════════════════════════════════════════════
#  Summary JSON
# ════════════════════════════════════════════════════════════════════════════

def _to_py(v):
    """Recursively convert tensors / numpy scalars to Python native types."""
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu()
        return v.item() if v.numel() == 1 else v.tolist()
    if isinstance(v, np.ndarray):
        return v.item() if v.size == 1 else v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def build_summary(pred: dict, label_arg: int | None) -> dict:
    summary: dict = {}
    if label_arg is not None:
        summary['ground_truth'] = 'real' if label_arg == 0 else 'fake'

    for key in ('prob', 'uncertainty'):
        if key in pred:
            summary[key] = _to_py(pred[key])

    # Per-branch evidence
    for key in ('spatial_evidence', 'concept_evidence', 'causal_evidence'):
        if key in pred:
            summary[key] = _to_py(pred[key])

    # Per-subgraph r_diff
    if 'r_diff_g' in pred:
        rdiff = _to_py(pred['r_diff_g'])
        if isinstance(rdiff, list) and len(rdiff) == 1:
            rdiff = rdiff[0]   # unwrap batch dim
        summary['r_diff_per_subgraph'] = {
            sg: float(rdiff[i]) for i, sg in enumerate(SUBGRAPH_NAMES)
            if i < len(rdiff)
        }
        summary['r_diff_total'] = float(sum(summary['r_diff_per_subgraph'].values()))
        verdict_causal = 'REAL-aligned' if summary['r_diff_total'] >= 0 else 'FAKE-aligned'
        summary['causal_verdict'] = verdict_causal

    # Verdict from probability
    if 'prob' in summary:
        p = summary['prob']
        p_scalar = p[1] if isinstance(p, list) and len(p) == 2 else (
            p[0] if isinstance(p, list) else p)
        summary['prob_fake'] = float(p_scalar)
        summary['prediction'] = 'fake' if p_scalar > 0.5 else 'real'
        summary['confidence'] = float(max(p_scalar, 1.0 - p_scalar))

    return summary


# ════════════════════════════════════════════════════════════════════════════
#  Rule names helper
# ════════════════════════════════════════════════════════════════════════════

def _load_rule_names(n_rules: int) -> list:
    for mod_path, attr in [
        ('networks.nesy_defake.semantic.consistency_rules',    'TRAINING_RULE_NAMES'),
        ('networks.nesy_defake.semantic.consistency_rules_v7', 'TRAINING_RULE_NAMES_V7'),
    ]:
        try:
            import importlib
            m = importlib.import_module(mod_path)
            names = list(getattr(m, attr))
            if len(names) >= n_rules:
                return names[:n_rules]
        except (ImportError, AttributeError):
            pass
    return [f'rule_{i}' for i in range(n_rules)]


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Single-image forensic inspection for NeSy-DeFakeNet',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--config',   required=True,  help='Detector YAML config path')
    p.add_argument('--weights',  required=True,  help='Model checkpoint .pth')
    p.add_argument('--image',    required=True,
                   help='Path to a preprocessed face frame\n'
                        '  e.g.  .../frames/video_000/007.png\n'
                        'Features are auto-resolved from the same dir tree.')
    p.add_argument('--out_dir',  default=None,
                   help='Output directory (default: results/<parent_folder>_<stem>)')
    p.add_argument('--label',    type=int, choices=[0, 1], default=None,
                   help='Ground-truth label: 0=real, 1=fake (optional)')
    p.add_argument('--attrs',    default=None,
                   help='Override: explicit path to a 58-d semantic attrs tensor (.pt)\n'
                        'Skip if image is in the standard dataset tree.')
    p.add_argument('--forensic', default=None,
                   help='Override: explicit path to an 83-d forensic features tensor (.pt)')
    p.add_argument('--attrs_subdir',    default='fast_semantic',
                   help='Subdir name used to look up attrs .pt  (default: fast_semantic)')
    p.add_argument('--forensic_subdir', default='forensic_features',
                   help='Subdir name for forensic .pt  (default: forensic_features)')
    p.add_argument('--device',   default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def _abspath(p: str) -> str:
    """Resolve a path that may be relative to the original invocation CWD."""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(_ORIG_CWD, p))


def main():
    args = parse_args()

    # Resolve all file-path args relative to where the user ran the command,
    # not relative to the repo root (which os.chdir set at import time).
    image_path    = _abspath(args.image)
    config_path   = _abspath(args.config)
    weights_path  = _abspath(args.weights)
    attrs_path    = _abspath(args.attrs)    if args.attrs    else None
    forensic_path = _abspath(args.forensic) if args.forensic else None

    if args.out_dir:
        out_dir = _abspath(args.out_dir)
    else:
        # Derive from image path: results/<parent_dir>_<stem>
        parent = os.path.basename(os.path.dirname(image_path))
        stem   = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(_ORIG_CWD, 'results', f'{parent}_{stem}')

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f'Building model from {config_path}')
    model = build_model(config_path, weights_path, device)
    print('Model ready.')

    # Preprocess
    img_size    = 224
    img_tensor  = load_image_tensor(image_path, img_size)
    display_img = load_image_display(image_path, img_size)

    # ── Feature resolution ────────────────────────────────────────────────
    # Explicit override takes priority; otherwise auto-resolve from tree.
    # Auto-resolve both at once (one pass through the directory tree)
    _auto_attrs, _auto_forensic = auto_resolve_features(
        image_path,
        attrs_subdir=args.attrs_subdir,
        forensic_subdir=args.forensic_subdir,
    )

    if attrs_path:
        attrs = torch.load(attrs_path, map_location='cpu')
        if isinstance(attrs, dict):
            attrs = attrs.get('features', attrs)
        print(f'  attrs loaded from --attrs override: shape {tuple(attrs.shape)}')
    else:
        attrs = _auto_attrs
        if attrs is not None:
            print(f'  attrs auto-resolved: shape {tuple(attrs.shape)}  '
                  f'(from {args.attrs_subdir}/)')
        else:
            print(f'  [info] attrs not found — concept/causal branch disabled')

    if forensic_path:
        forensic = torch.load(forensic_path, map_location='cpu')
        if isinstance(forensic, dict):
            forensic = forensic.get('features', forensic)
        print(f'  forensic loaded from --forensic override: shape {tuple(forensic.shape)}')
    else:
        forensic = _auto_forensic
        if forensic is not None:
            print(f'  forensic auto-resolved: shape {tuple(forensic.shape)}  '
                  f'(from {args.forensic_subdir}/)')
        else:
            print(f'  [info] forensic not found — causal branch disabled')

    label_arg = args.label if args.label is not None else 0   # default label for causal pass

    # ── Forward pass with gradient tracking (for saliency) ──────────────────
    print('Running forward pass (gradient-tracked) …')
    pred, img_with_grad = run_forward(
        model, img_tensor, device,
        label=label_arg, attrs=attrs, forensic=forensic,
        compute_grad=True,
    )

    # ── Decode probability once (used by alignment + verdict-flow) ──────────
    prob_val = pred.get('prob')
    p_fake: float | None = None
    pred_label: int | None = None
    if prob_val is not None:
        p = prob_val.detach().cpu().numpy().flatten()
        p_fake = float(p[1]) if len(p) == 2 else float(p[0])
        pred_label = 1 if p_fake > 0.5 else 0

    # ── 1. Sample alignment chart ────────────────────────────────────────────
    if 'r_diff_g' in pred:
        rdiff_arr = pred['r_diff_g'].detach().cpu().numpy().flatten()
        n_sg = min(len(rdiff_arr), len(SUBGRAPH_NAMES))

        unc_val = None
        if 'uncertainty' in pred:
            unc_val = float(pred['uncertainty'].detach().cpu().item())

        align_path = os.path.join(out_dir, 'sample_alignment.png')
        viz.plot_sample_graph_alignment(
            r_diff=rdiff_arr[:n_sg],
            subgraph_names=SUBGRAPH_NAMES[:n_sg],
            save_path=align_path,
            label=args.label,
            pred_label=pred_label,
            sample_id=os.path.basename(image_path),
            prob=p_fake,
            uncertainty=unc_val,
        )
        print(f'  saved {align_path}')

    # ── 2. Causal graphs (broken-links + verdict-flow) ───────────────────────
    scm_analyzer = SCMAnalyzer()
    print('Rendering causal graph panels …')
    save_causal_graphs(pred, model, scm_analyzer, out_dir, prob_fake=p_fake)

    # ── 3. Gradient saliency ─────────────────────────────────────────────────
    if img_with_grad is not None and img_with_grad.grad is not None:
        sal_path = os.path.join(out_dir, 'gradient_saliency.png')
        save_gradient_saliency(img_with_grad.grad, display_img, sal_path)
        print(f'  saved {sal_path}')
    else:
        print('  [warn] gradient not available — saliency skipped')

    # ── 4. Rule-region overlay ───────────────────────────────────────────────
    if 'violations' in pred:
        v_arr = pred['violations'].detach().cpu().numpy().flatten()
        n_rules = int(v_arr.shape[0])
        rule_names = _load_rule_names(n_rules)
        rule_path = os.path.join(out_dir, 'rule_region_map.png')
        save_rule_region_map(v_arr, rule_names, display_img, rule_path, top_k=6)
        print(f'  saved {rule_path}')
    else:
        print('  [info] no concept violations in prediction — rule map skipped '
              '(need --attrs)')

    # ── 5. Summary JSON ──────────────────────────────────────────────────────
    summary = build_summary(pred, args.label)
    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  saved {summary_path}')

    # ── Console report ───────────────────────────────────────────────────────
    print('\n' + '─' * 55)
    print(f'  Image   : {image_path}')
    if args.label is not None:
        gt_str = 'REAL' if args.label == 0 else 'FAKE'
        print(f'  GT label: {gt_str}')
    if 'prediction' in summary:
        print(f'  Prediction : {summary["prediction"].upper()}  '
              f'(P(fake)={summary.get("prob_fake", "?"):.3f})')
    if 'uncertainty' in summary:
        u_s = summary['uncertainty']
        if isinstance(u_s, list):
            u_s = u_s[0]
        print(f'  Uncertainty: {float(u_s):.4f}')
    if 'causal_verdict' in summary:
        print(f'  Causal verdict : {summary["causal_verdict"]}  '
              f'(Σr_diff={summary["r_diff_total"]:+.4f})')
    if 'r_diff_per_subgraph' in summary:
        for sg, val in summary['r_diff_per_subgraph'].items():
            direction = '▶ REAL' if val >= 0 else '▶ FAKE'
            print(f'    {sg:<28s} {val:+.4f}  {direction}')
    print('─' * 55)
    print(f'All outputs written to: {os.path.abspath(out_dir)}')


if __name__ == '__main__':
    main()
