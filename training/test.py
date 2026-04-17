"""
Standalone evaluation script — GenD-style clean outputs.

Output structure:
    runs/test/<run_name>/
    ├── metrics.csv                   # All metrics (frame + video level)
    ├── test_predictions.csv          # Per-frame predictions
    ├── hparams.yaml                  # Config snapshot
    └── test/
        ├── frame_metrics/
        │   ├── test_roc_frame.png
        │   ├── test_pr_curve.png
        │   ├── test_f1_curve.png
        │   ├── test_fpr_fnr_curve.png
        │   ├── test_confusion.png
        │   ├── test_confusion_norm.png
        │   └── test_probs_distribution.png
        └── video_metrics/
            ├── test_roc_video.png
            ├── test_pr_curve.png
            ├── test_confusion.png
            ├── test_confusion_norm.png
            └── test_probs_distribution.png
"""
import os
import argparse
import csv
import datetime
import random

import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import wasserstein_distance
from sklearn import metrics as M

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.nesy_defake_dataset import NeSyDeFakeDataset
from detectors import DETECTOR

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Evaluate a trained detector.')
parser.add_argument('--detector_path', type=str, required=True,
                    help='Path to detector YAML config file')
parser.add_argument('--test_dataset', nargs='+', default=None,
                    help='Override test datasets')
parser.add_argument('--weights_path', type=str, required=True,
                    help='Path to saved model weights (.pth)')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: logs/test/<auto>)')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Override test batch size')
# ── Interpretability / t-SNE ────────────────────────────────────────────
parser.add_argument('--tsne_mode', type=str, default=None,
                    choices=['best', 'worst', 'random', 'all'],
                    help='If set, generate a t-SNE plot per test dataset on '
                         'the top-K samples selected by this mode. '
                         '"best"=highest-confidence correct, '
                         '"worst"=highest-confidence wrong, '
                         '"random"=uniform sample, '
                         '"all"=produce all three.')
parser.add_argument('--tsne_top_k', type=int, default=500,
                    help='Sample budget for t-SNE (default: 500).')
parser.add_argument('--tsne_feature_key', type=str, default='feat',
                    help='Prediction-dict key used as embedding '
                         '(e.g. feat, l2_embeddings).')
parser.add_argument('--rule_error_analysis', action='store_true',
                    help='Collect per-frame consistency-rule violations '
                         'and write a slice report (TN/FP/TP/FN firing '
                         'means + error gaps) per test dataset.')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Metric computation (adapted from GenD)
# ---------------------------------------------------------------------------

def ovr_roc(labels, probs):
    num_classes = probs.shape[1]
    labels_oh = np.eye(num_classes)[labels]
    fprs, tprs, ths = [], [], []
    macro_auroc = M.roc_auc_score(labels_oh, probs, multi_class="ovr", average="macro")
    for i in range(num_classes):
        fpr, tpr, th = M.roc_curve(labels_oh[:, i], probs[:, i])
        th = np.nan_to_num(th, posinf=1.0)
        th = np.concatenate(([1], th, [0]))
        fpr = np.concatenate(([0], fpr, [1]))
        tpr = np.concatenate(([0], tpr, [1]))
        fprs.append(fpr); tprs.append(tpr); ths.append(th)
    return fprs, tprs, ths, macro_auroc


def ovr_prc(labels, probs):
    num_classes = probs.shape[1]
    labels_oh = np.eye(num_classes)[labels]
    precs, recs, ths = [], [], []
    macro_ap = M.average_precision_score(labels_oh, probs, average="macro")
    for i in range(num_classes):
        p, r, th = M.precision_recall_curve(labels_oh[:, i], probs[:, i])
        th = np.nan_to_num(th, posinf=1.0)
        th = np.concatenate(([1], th, [0]))
        p = np.concatenate(([0], p, [1]))
        r = np.concatenate(([1], r, [0]))
        precs.append(p); recs.append(r); ths.append(th)
    return precs, recs, ths, macro_ap


def calculate_eer(labels, probs):
    fpr, tpr, thresholds = M.roc_curve(labels, probs[:, 1], pos_label=1)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        eer_th = float(interp1d(fpr, thresholds)(eer))
    except ValueError:
        eer, eer_th = np.nan, np.nan
    return eer, eer_th


def calculate_tpr_at_fpr(labels, probs, fpr_targets=(0.001, 0.01, 0.05)):
    fpr, tpr, _ = M.roc_curve(labels, probs[:, 1], pos_label=1)
    results = []
    for t in fpr_targets:
        if t < fpr.min() or t > fpr.max():
            results.append(np.nan)
        else:
            results.append(float(np.interp(t, fpr, tpr)))
    return results


def compute_wasserstein1(probs, labels):
    is_real = labels == 0
    is_fake = labels == 1
    if not (is_real.any() and is_fake.any()):
        return {}
    return {
        'W1-sep-real': wasserstein_distance(probs[is_real, 0], probs[is_fake, 0]),
        'W1-sep-fake': wasserstein_distance(probs[is_real, 1], probs[is_fake, 1]),
        'W1-sep':      (wasserstein_distance(probs[is_real, 0], probs[is_fake, 0]) +
                         wasserstein_distance(probs[is_real, 1], probs[is_fake, 1])) / 2,
        'W1-conf-real': wasserstein_distance(probs[is_real, 0], probs[is_real, 1]),
        'W1-conf-fake': wasserstein_distance(probs[is_fake, 0], probs[is_fake, 1]),
        'W1-conf':     (wasserstein_distance(probs[is_real, 0], probs[is_real, 1]) +
                         wasserstein_distance(probs[is_fake, 0], probs[is_fake, 1])) / 2,
    }


def compute_video_predictions(img_names, probs, labels):
    """Aggregate frame predictions to video level by averaging.

    Returns video_probs, video_labels, video_ids, video_frames where the
    four arrays are aligned: entry i corresponds to video_ids[i], and
    video_frames[i] lists every frame path that was averaged into it.
    """
    video_dict = {}
    for name, prob, label in zip(img_names, probs, labels):
        parts = name.replace('\\', '/').split('/')
        video_id = parts[-2]
        if video_id not in video_dict:
            video_dict[video_id] = {'probs': [], 'label': int(label), 'frames': []}
        video_dict[video_id]['probs'].append(prob)
        video_dict[video_id]['frames'].append(name)

    video_ids    = list(video_dict.keys())
    video_probs  = np.array([np.mean(video_dict[v]['probs'], axis=0) for v in video_ids])
    video_labels = np.array([video_dict[v]['label'] for v in video_ids])
    video_frames = [video_dict[v]['frames'] for v in video_ids]
    return video_probs, video_labels, video_ids, video_frames


def compute_all_metrics(labels, probs, level):
    """Compute all metrics for a given level (frame/video). Returns dict."""
    metrics = {}
    fprs, tprs, ths, auroc = ovr_roc(labels, probs)
    precs, recs, pr_ths, mAP = ovr_prc(labels, probs)
    eer, eer_th = calculate_eer(labels, probs)

    metrics[f'auroc_{level}'] = auroc
    metrics[f'mAP_{level}'] = mAP
    metrics[f'eer_{level}'] = eer
    metrics[f'eer_th_{level}'] = eer_th

    # TPR @ FPR
    for target, tpr_val in zip([0.001, 0.01, 0.05],
                                calculate_tpr_at_fpr(labels, probs)):
        metrics[f'TPR@FPR={target}_{level}'] = tpr_val

    # Predictions using EER threshold
    preds = np.where(probs[:, 1] > eer_th, 1, 0) if not np.isnan(eer_th) else probs.argmax(1)
    metrics[f'acc_{level}'] = M.accuracy_score(labels, preds)
    metrics[f'balanced_acc_{level}'] = M.balanced_accuracy_score(labels, preds)
    metrics[f'f1_score_{level}'] = M.f1_score(labels, preds, average='macro')

    # Wasserstein
    w1 = compute_wasserstein1(probs, labels)
    for k, v in w1.items():
        metrics[f'{k}_{level}'] = v

    return metrics, fprs, tprs, ths, precs, recs, pr_ths, eer, preds


# ---------------------------------------------------------------------------
# Plotting (adapted from GenD)
# ---------------------------------------------------------------------------
CLASS_NAMES = {0: 'real', 1: 'fake'}


def _plot_curve(xs, ys, class_names=None, interpolate=200, mean=True):
    """Base curve plotter, returns (fig, ax)."""
    fig = plt.figure(figsize=(10, 8), tight_layout=True)
    gs = plt.GridSpec(1, 2, width_ratios=[4, 1])
    ax = plt.subplot(gs[0])
    ax_legend = plt.subplot(gs[1])
    palette = sns.husl_palette(len(xs))

    if interpolate > 0:
        x_new = np.linspace(0, 1, interpolate)
        ys = [interp1d(x, y)(x_new) for x, y in zip(xs, ys)]
        xs = [x_new] * len(xs)

    active = []
    for c, (x, y) in enumerate(zip(xs, ys)):
        auc = M.auc(x, y)
        name = f"{c}: {class_names[c]}" if class_names else str(c)
        label = f"{name} (AUC: {auc:.2f})"
        line = ax.plot(x, y, label=label, linewidth=1.5, color=palette[c])
        active.append((line[0], label))

    if mean and interpolate > 0:
        ys_mean = np.mean(ys, axis=0)
        xs_mean = np.mean(xs, axis=0)
        auc = M.auc(xs_mean, ys_mean)
        ax.plot(xs_mean, ys_mean, label=f"avg (AUC: {auc:.2f})",
                linewidth=1.5, color='black')
        active.append((ax.lines[-1], f"avg (AUC: {auc:.2f})"))

    ax.set_aspect('equal')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax_legend.axis('off')
    if active:
        lines, labels = zip(*active)
        ax_legend.legend(lines, labels, loc='center left', fontsize=10)
    return fig, ax


def plot_roc(fprs, tprs, title, path):
    fig, ax = _plot_curve(fprs, tprs, CLASS_NAMES)
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()


def plot_pr(precs, recs, title, path):
    fig, ax = _plot_curve(recs, precs, CLASS_NAMES)
    f_scores = np.linspace(0.1, 0.9, 9)
    for f in f_scores:
        r = np.linspace(0.001, 1, 100)
        p = f * r / (2 * r - f)
        mask = p > 0
        ax.plot(r[mask], p[mask], color='gray', alpha=0.2, linestyle='--')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()


def plot_f1(precs, recs, ths, title, path):
    f1s = []
    for p, r in zip(precs, recs):
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = np.where((p + r) == 0, 0, 2 * p * r / (p + r))
        f1s.append(f1[:-1])
    fig, ax = _plot_curve(ths, f1s, CLASS_NAMES)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()


def plot_fpr_fnr(fprs, tprs, ths, eer, title, path):
    if len(fprs) != 2:
        return
    fpr = fprs[1]
    fnr = 1 - tprs[1]
    fig, ax = _plot_curve([ths[1], ths[1]], [fpr, fnr],
                           {0: 'FPR', 1: 'FNR'}, mean=False)
    if eer is not None and not np.isnan(eer):
        ax.axhline(y=eer, color='black', linestyle='--')
        ax.text(0, eer + 0.02, f'EER: {eer:.4f}', fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('FPR / FNR', fontsize=12)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()


def plot_confusion(labels, preds, title, path, normalize=False):
    conf = M.confusion_matrix(labels, preds)
    fmt = 'd'
    if normalize:
        conf = conf / conf.sum(axis=1, keepdims=True) * 100
        conf[np.isnan(conf)] = 0
        fmt = '.2f'
    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    lbl = [f"{k}: {v}" for k, v in CLASS_NAMES.items()]
    sns.heatmap(conf, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=lbl, yticklabels=lbl, annot_kws={'fontsize': 12})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(title, fontsize=14)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()


def plot_probs_dist(probs, labels, title, path):
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_classes, 1, figsize=(10, 4 * n_classes))
    palette = sns.husl_palette(n_classes)
    for idx, (cls_id, cls_name) in enumerate(CLASS_NAMES.items()):
        ax = axes[idx]
        mask = labels == cls_id
        cls_probs = probs[mask]
        for pred_id, pred_name in CLASS_NAMES.items():
            sns.histplot(data=cls_probs[:, pred_id], label=f'y_hat={pred_name}',
                         color=palette[pred_id], alpha=0.2, bins=100,
                         stat='probability', kde=True, element='step', ax=ax)
        ax.set_xlabel('Scores')
        ax.set_ylabel('Probability')
        ax.set_title(f'p(y_hat | y={cls_name})', color=palette[idx])
        ax.set_xlim(-0.005, 1.005)
        ax.legend()
    plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()


def generate_all_plots(fprs, tprs, ths, precs, recs, pr_ths, eer,
                       probs, labels, preds, level, out_dir, dataset_name):
    """Generate all visualization plots for one level."""
    d = os.path.join(out_dir, 'test', f'{level}_metrics')
    tag = f'{dataset_name} ({level}-level)'

    plot_roc(fprs, tprs, f'Test ROC {tag}', f'{d}/test_roc_{level}.png')
    plot_pr(precs, recs, f'Test PR Curve {tag}', f'{d}/test_pr_curve.png')
    plot_f1(precs, recs, pr_ths, f'Test F1 Curve {tag}', f'{d}/test_f1_curve.png')
    plot_fpr_fnr(fprs, tprs, ths, eer, f'Test FPR vs FNR {tag}', f'{d}/test_fpr_fnr_curve.png')
    plot_confusion(labels, preds, f'Test Confusion {tag}', f'{d}/test_confusion.png')
    plot_confusion(labels, preds, f'Test Confusion {tag}', f'{d}/test_confusion_norm.png', normalize=True)
    plot_probs_dist(probs, labels, f'Test Score Distribution {tag}', f'{d}/test_probs_distribution.png')


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_testing_data(config):
    test_data_loaders = {}
    for test_name in config['test_dataset']:
        test_config = config.copy()
        test_config['test_dataset'] = test_name

        if (config.get('dataset_type') == 'nesydefake'
                or config['model_name'] == 'nesydefake_hybrid'):
            test_set = NeSyDeFakeDataset(test_config, mode='test')
        else:
            test_set = DeepfakeAbstractBaseDataset(config=test_config, mode='test')

        test_data_loaders[test_name] = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config['test_batchSize'],
            shuffle=False,
            num_workers=int(config['workers']),
            collate_fn=test_set.collate_fn,
            drop_last=False,
        )
    return test_data_loaders


@torch.no_grad()
def run_inference(model, data_loader, interp_engine=None):
    """Run inference on a single dataset, returns predictions, labels, file names.

    If *interp_engine* is provided, each batch's prediction dict is forwarded
    to the engine so analyzers (e.g. t-SNE) can accumulate their signals.
    """
    prediction_lists = []
    label_lists = []

    for data_dict in tqdm(data_loader, desc='  Inference'):
        if 'label_spe' in data_dict:
            data_dict.pop('label_spe')
        data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)
        for key in data_dict.keys():
            val = data_dict[key]
            if val is not None and isinstance(val, torch.Tensor):
                data_dict[key] = val.to(device)

        predictions = model(data_dict, inference=True)
        prob = predictions['prob'].cpu().numpy()  # shape: (B,)

        if interp_engine is not None:
            interp_engine.collect_batch(predictions, data_dict['label'])

        label_lists.append(data_dict['label'].cpu().numpy())
        prediction_lists.append(prob)

    preds_1d = np.concatenate(prediction_lists)   # p(fake)
    labels = np.concatenate(label_lists)
    # Convert to 2-class probabilities: [p(real), p(fake)]
    probs_2d = np.stack([1 - preds_1d, preds_1d], axis=1)

    dataset = data_loader.dataset
    img_names = getattr(dataset, 'image_list', None) or dataset.data_dict['image']
    return probs_2d, labels, img_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    test_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'config', 'test_config.yaml')
    if os.path.exists(test_config_path):
        with open(test_config_path, 'r') as f:
            config.update(yaml.safe_load(f))

    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if args.batch_size:
        config['test_batchSize'] = args.batch_size

    # Seed
    seed = config.get('manualSeed', 1024)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if config.get('cudnn', False):
        cudnn.benchmark = True

    # Output directory
    # Derive experiment name from detector config filename
    config_name = os.path.splitext(os.path.basename(args.detector_path))[0]
    if args.output_dir:
        out_dir = args.output_dir
    else:
        # Try to reuse the train experiment folder name from weights_path
        # Expected: logs/train/<config_name>_<datetime>/best_*.pth
        weights_parent = os.path.basename(os.path.dirname(args.weights_path))
        if weights_parent.startswith(config_name):
            experiment_folder = weights_parent
        else:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            experiment_folder = f'{config_name}_{timestamp}'
        out_dir = os.path.join('logs', 'test', experiment_folder)
    os.makedirs(out_dir, exist_ok=True)

    # Save config snapshot
    with open(os.path.join(out_dir, 'hparams.yaml'), 'w') as f:
        config_snapshot = {**config, 'weights_path': args.weights_path}
        yaml.dump(config_snapshot, f, default_flow_style=False)

    # Load model
    print(f'Loading model: {config["model_name"]}')
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    ckpt = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    print(f'Loaded weights from {args.weights_path}')

    # Prepare data
    test_data_loaders = prepare_testing_data(config)

    # Evaluate each dataset
    all_metrics = {}

    for dataset_name, data_loader in test_data_loaders.items():
        print(f'\n{"="*60}')
        print(f'Evaluating: {dataset_name}')
        print(f'{"="*60}')

        # Always create per-dataset subdirectory
        ds_out_dir = os.path.join(out_dir, dataset_name)
        os.makedirs(ds_out_dir, exist_ok=True)

        # Build an interpretability engine for this dataset if any
        # opt-in analyzer was requested via CLI. The engine is kept
        # minimal (only the requested analyzers are activated), since
        # other analyzers depend on training-time signals that may not
        # be present at test time.
        interp_engine = None
        want_tsne = args.tsne_mode is not None
        want_rule_errors = args.rule_error_analysis
        if want_tsne or want_rule_errors:
            try:
                from interpretability.engine import InterpretabilityEngine
                interp_cfg = {
                    'levels': {
                        'edl_uncertainty': False,
                        'branch_evidence': False,
                        'consistency_rules': want_rule_errors,
                        'ccv_analysis': False,
                        'scm_analysis': False,
                        'gate_analysis': False,
                        'disagreement': False,
                        'tsne': want_tsne,
                    },
                    'tsne': {
                        'enabled': want_tsne,
                        'mode': args.tsne_mode or 'worst',
                        'top_k': args.tsne_top_k,
                        'feature_key': args.tsne_feature_key,
                    },
                }
                engine_cfg = {**config, 'interpretability': interp_cfg}
                interp_engine = InterpretabilityEngine(
                    engine_cfg, causal_type=config.get('causal_branch_type', ''))
            except Exception as e:
                print(f'[warn] Interpretability engine init failed: {e}')
                interp_engine = None

        probs, labels, img_names = run_inference(
            model, data_loader, interp_engine=interp_engine)

        if interp_engine is not None:
            try:
                interp_engine.finalize(os.path.join(ds_out_dir, 'interpretability'))
            except Exception as e:
                print(f'[warn] Interpretability finalize failed: {e}')

        # --- Frame-level metrics ---
        frame_metrics, f_fprs, f_tprs, f_ths, f_precs, f_recs, f_pr_ths, f_eer, f_preds = \
            compute_all_metrics(labels, probs, 'frame')

        # --- Rule-error slice report (uses EER-thresholded frame preds) ---
        if interp_engine is not None and args.rule_error_analysis:
            rule_analyzer = interp_engine.analyzers.get('consistency_rules')
            if rule_analyzer is not None:
                try:
                    report = rule_analyzer.slice_report(
                        f_preds,
                        os.path.join(ds_out_dir, 'interpretability'),
                    )
                    if report:
                        print(f'  Rule slice report: counts={report["counts"]}')
                except Exception as e:
                    print(f'[warn] Rule slice report failed: {e}')

        # --- Video-level metrics ---
        video_probs, video_labels, video_ids, video_frames = \
            compute_video_predictions(img_names, probs, labels)
        video_metrics, v_fprs, v_tprs, v_ths, v_precs, v_recs, v_pr_ths, v_eer, v_preds = \
            compute_all_metrics(video_labels, video_probs, 'video')

        combined_metrics = {**frame_metrics, **video_metrics}
        all_metrics[dataset_name] = combined_metrics

        # --- Save metrics.csv ---
        metrics_path = os.path.join(ds_out_dir, 'metrics.csv')
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(combined_metrics.keys()))
            writer.writeheader()
            writer.writerow({k: f'{v:.6f}' if isinstance(v, float) else v
                             for k, v in sorted(combined_metrics.items())})

        # --- Save predictions.csv ---
        pred_path = os.path.join(ds_out_dir, 'test_predictions.csv')
        with open(pred_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['files', 'labels', 'prob_class_0', 'prob_class_1'])
            for name, label, p in zip(img_names, labels, probs):
                writer.writerow([name, int(label), f'{p[0]:.4f}', f'{p[1]:.4f}'])

        # --- Save misclassified entries (for error analysis) ---
        # Frame-level wrong predictions: one row per misclassified frame.
        # `f_preds` uses the EER threshold computed inside compute_all_metrics.
        frame_wrong_path = os.path.join(ds_out_dir, 'misclassified_frames.csv')
        with open(frame_wrong_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'video_id', 'label', 'pred', 'prob_fake', 'error_type'])
            for name, label, pred, p in zip(img_names, labels, f_preds, probs):
                if int(pred) != int(label):
                    video_id = name.replace('\\', '/').split('/')[-2]
                    err = 'FP' if int(label) == 0 else 'FN'  # real->fake / fake->real
                    writer.writerow([name, video_id, int(label), int(pred),
                                     f'{p[1]:.4f}', err])

        # Video-level wrong predictions: one row per misclassified video,
        # plus the list of frame paths so we can inspect them later.
        video_wrong_path = os.path.join(ds_out_dir, 'misclassified_videos.csv')
        with open(video_wrong_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'label', 'pred', 'mean_prob_fake',
                             'num_frames', 'frame_files', 'error_type'])
            for vid, vlabel, vpred, vp, vframes in zip(
                    video_ids, video_labels, v_preds, video_probs, video_frames):
                if int(vpred) != int(vlabel):
                    err = 'FP' if int(vlabel) == 0 else 'FN'
                    writer.writerow([
                        vid, int(vlabel), int(vpred),
                        f'{vp[1]:.4f}', len(vframes),
                        ';'.join(vframes), err,
                    ])

        num_frame_wrong = int((f_preds != labels).sum())
        num_video_wrong = int((v_preds != video_labels).sum())
        print(f'  Misclassified frames: {num_frame_wrong}/{len(labels)} '
              f'→ {frame_wrong_path}')
        print(f'  Misclassified videos: {num_video_wrong}/{len(video_labels)} '
              f'→ {video_wrong_path}')

        # --- Generate plots ---
        print('Generating frame-level plots...')
        generate_all_plots(f_fprs, f_tprs, f_ths, f_precs, f_recs, f_pr_ths, f_eer,
                           probs, labels, f_preds, 'frame', ds_out_dir, dataset_name)
        print('Generating video-level plots...')
        generate_all_plots(v_fprs, v_tprs, v_ths, v_precs, v_recs, v_pr_ths, v_eer,
                           video_probs, video_labels, v_preds, 'video', ds_out_dir, dataset_name)

        # --- Print summary ---
        print(f'\n--- {dataset_name} Results ---')
        print(f'  Frame AUROC:  {frame_metrics["auroc_frame"]:.4f}')
        print(f'  Frame EER:    {frame_metrics["eer_frame"]:.4f}')
        print(f'  Frame Acc:    {frame_metrics["acc_frame"]:.4f}')
        print(f'  Frame mAP:    {frame_metrics["mAP_frame"]:.4f}')
        print(f'  Video AUROC:  {video_metrics["auroc_video"]:.4f}')
        print(f'  Video EER:    {video_metrics["eer_video"]:.4f}')
        print(f'  Video Acc:    {video_metrics["acc_video"]:.4f}')
        print(f'  Video mAP:    {video_metrics["mAP_video"]:.4f}')
        for t in [0.001, 0.01, 0.05]:
            fk = f'TPR@FPR={t}_frame'
            vk = f'TPR@FPR={t}_video'
            print(f'  TPR@FPR={t}:  frame={frame_metrics[fk]:.4f}  video={video_metrics[vk]:.4f}')

    print(f'\nAll results saved to: {out_dir}')
    print('Done!')


if __name__ == '__main__':
    main()
