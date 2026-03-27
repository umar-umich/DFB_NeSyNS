"""
demo_plot_helpers.py
====================
Shared plotting and text report helpers for FaceBench semantic feature
visualization. Used by both SemanticPrecomputer.demo() and demo_semantic.py.
"""

import os

import numpy as np


CATEGORIES = [
    ('Hair', 0, 20), ('Forehead', 20, 23), ('Eyebrows', 23, 30),
    ('Eyes', 30, 45), ('Eyelashes', 45, 48), ('Nose', 48, 56),
    ('Mouth/Lips', 56, 66), ('Cheeks', 66, 70), ('Chin/Jaw', 70, 76),
    ('Face Shape', 76, 82), ('Ears', 82, 85), ('Skin', 85, 100),
    ('Facial Hair', 100, 106), ('Neck', 106, 108), ('Age', 108, 113),
    ('Other Appearance', 113, 116), ('Accessories', 116, 146),
    ('Makeup', 146, 159), ('Surrounding', 159, 171),
    ('Expression', 171, 179), ('Action Units', 179, 204),
    ('Identity', 204, 211),
]


def save_text_report(scores, attr_names, image_path, use_llm, txt_path,
                     raw_responses=None):
    with open(txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FaceBench Face-LLaVA Attribute Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Image:      {image_path}\n")
        f.write(f"Method:     {'Generative' if use_llm else 'Vision-only'}\n")
        f.write(f"Num attrs:  {len(scores)}\n")

        if use_llm:
            n_yes = (scores > 0.5).sum()
            n_no = (scores < 0.5).sum()
            n_unsure = (scores == 0.5).sum()
            f.write(f"Present (Yes): {n_yes} / {len(scores)}\n")
            f.write(f"Absent  (No):  {n_no} / {len(scores)}\n")
            if n_unsure > 0:
                f.write(f"Unsure:        {n_unsure} / {len(scores)}\n")
        else:
            f.write(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]\n")
            f.write(f"Score mean:  {scores.mean():.4f}\n")
            f.write(f"Score std:   {scores.std():.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("ALL ATTRIBUTES BY CATEGORY\n")
        f.write("=" * 70 + "\n")

        if use_llm and len(attr_names) == 211:
            for cat_name, start, end in CATEGORIES:
                cat_scores = scores[start:end]
                n_cat_yes = (cat_scores > 0.5).sum()
                f.write(f"\n--- {cat_name} (idx {start}-{end-1}, "
                        f"{n_cat_yes}/{end-start} present) ---\n")
                for i in range(start, end):
                    name = attr_names[i]
                    display_name = name.replace('_', ' ')
                    val = scores[i]
                    flag = ("  YES" if val > 0.5
                            else ("  NO" if val < 0.5 else "  UNSURE"))
                    raw = ""
                    if raw_responses and name in raw_responses:
                        raw = f"  (raw: {raw_responses[name]})"
                    f.write(f"  [{i:3d}] {display_name:<35s}{flag}{raw}\n")
        else:
            for i, (name, val) in enumerate(zip(attr_names, scores)):
                f.write(f"  [{i:3d}] {name:<35s} {val:.4f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("ALL ATTRIBUTES RANKED\n")
        f.write("=" * 70 + "\n")

        sorted_idx = np.argsort(scores)[::-1]
        for rank, i in enumerate(sorted_idx, 1):
            name = attr_names[i]
            display_name = name.replace('_', ' ')
            val = scores[i]
            flag = "YES" if val > 0.5 else ("NO" if val < 0.5 else "UNSURE")
            raw = ""
            if raw_responses and name in raw_responses:
                raw = f"  (raw: {raw_responses[name]})"
            f.write(f"  {rank:3d}. [{i:3d}] {display_name:<35s} {flag}{raw}\n")


def plot_description(img_pil, descriptions, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             gridspec_kw={'width_ratios': [1, 2]})
    axes[0].imshow(img_pil)
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].axis('off')
    text_content = ""
    for i, (prompt, response) in enumerate(descriptions):
        text_content += f"Q{i+1}: {prompt}\n{'─' * 60}\n"
        words = response.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > 80:
                text_content += line + "\n"
                line = w
            else:
                line = line + " " + w if line else w
        if line:
            text_content += line + "\n"
        text_content += "\n\n"

    axes[1].text(0.02, 0.98, text_content, transform=axes[1].transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    axes[1].set_title('Face-LLaVA Responses', fontsize=14, fontweight='bold')
    fig.suptitle('FaceBench Face-LLaVA — Free-Form Validation',
                 fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_llm_attributes(img_pil, scores, attr_names, top_k, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    sorted_idx = np.argsort(scores)[::-1]
    n_attrs = len(scores)

    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1.5, 1.5],
                           height_ratios=[1, 1], hspace=0.3, wspace=0.3)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img_pil)
    ax_img.set_title('Input Image', fontsize=14, fontweight='bold')
    ax_img.axis('off')

    ax_top = fig.add_subplot(gs[0, 1])
    top_n = min(top_k, n_attrs)
    top_idx = sorted_idx[:top_n]
    top_names = [attr_names[i].replace('_', ' ') for i in top_idx]
    top_scores = [scores[i] for i in top_idx]
    colors_top = ['#2ecc71' if s > 0.5 else '#e74c3c' for s in top_scores]
    y_pos = np.arange(top_n)
    ax_top.barh(y_pos, top_scores, color=colors_top, height=0.7)
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(top_names, fontsize=8)
    ax_top.invert_yaxis()
    ax_top.set_xlim(0, 1)
    ax_top.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_top.set_xlabel('P(present)')
    ax_top.set_title(f'Top-{top_n} Attributes', fontsize=14, fontweight='bold')

    ax_bot = fig.add_subplot(gs[0, 2])
    bot_n = min(top_k, n_attrs)
    bot_idx = sorted_idx[-bot_n:][::-1]
    bot_names = [attr_names[i].replace('_', ' ') for i in bot_idx]
    bot_scores = [scores[i] for i in bot_idx]
    colors_bot = ['#2ecc71' if s > 0.5 else '#e74c3c' for s in bot_scores]
    y_pos_b = np.arange(bot_n)
    ax_bot.barh(y_pos_b, bot_scores, color=colors_bot, height=0.7)
    ax_bot.set_yticks(y_pos_b)
    ax_bot.set_yticklabels(bot_names, fontsize=8)
    ax_bot.invert_yaxis()
    ax_bot.set_xlim(0, 1)
    ax_bot.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_bot.set_xlabel('P(present)')
    ax_bot.set_title(f'Bottom-{bot_n} Attributes',
                     fontsize=14, fontweight='bold')

    ax_cat = fig.add_subplot(gs[1, 0])
    cat_names = []
    cat_means = []
    cat_maxs = []
    for cat_name, start, end in CATEGORIES:
        cat_names.append(cat_name)
        cat_means.append(scores[start:end].mean())
        cat_maxs.append(scores[start:end].max())
    y_cat = np.arange(len(cat_names))
    ax_cat.barh(y_cat, cat_means, color='#3498db', height=0.6, label='Mean')
    ax_cat.barh(y_cat, cat_maxs, color='#3498db', height=0.6, alpha=0.3,
                label='Max')
    ax_cat.set_yticks(y_cat)
    ax_cat.set_yticklabels(cat_names, fontsize=8)
    ax_cat.invert_yaxis()
    ax_cat.set_xlim(0, 1)
    ax_cat.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax_cat.set_xlabel('Score')
    ax_cat.set_title('Category Summary', fontsize=14, fontweight='bold')
    ax_cat.legend(fontsize=8)

    ax_heat = fig.add_subplot(gs[1, 1:])
    heatmap_data = scores.reshape(1, -1)
    im = ax_heat.imshow(heatmap_data, aspect='auto', cmap='RdYlGn',
                        vmin=0, vmax=1, interpolation='nearest')
    ax_heat.set_yticks([])
    ax_heat.set_xlabel('Attribute Index')
    ax_heat.set_title('All 211 Attributes Heatmap',
                      fontsize=14, fontweight='bold')
    for cat_name, start, end in CATEGORIES:
        mid = (start + end) / 2
        ax_heat.axvline(x=start - 0.5, color='white', linewidth=0.5,
                        alpha=0.7)
        if (end - start) > 5:
            ax_heat.text(mid, 0.6, cat_name, ha='center', va='top',
                         fontsize=5, rotation=90, color='black', alpha=0.7)
    plt.colorbar(im, ax=ax_heat, label='P(present)', shrink=0.8)

    n_present = (scores > 0.5).sum()
    fig.suptitle(
        f'FaceBench — {n_present}/{len(scores)} attributes present (P>0.5)\n'
        f'Score range: [{scores.min():.3f}, {scores.max():.3f}], '
        f'mean: {scores.mean():.3f}, std: {scores.std():.3f}',
        fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def report_and_plot(img_pil, scores, attr_names, image_path,
                    save_path=None, top_k=30, suffix='demo',
                    use_llm=True, raw_responses=None):
    """Generate both plot and text report for scores."""
    if save_path is None:
        save_path = os.path.splitext(image_path)[0] + f'_facebench_{suffix}.png'
    txt_path = os.path.splitext(save_path)[0] + '.txt'
    plot_llm_attributes(img_pil, scores, attr_names, top_k, save_path)
    save_text_report(scores, attr_names, image_path, use_llm, txt_path,
                     raw_responses=raw_responses)
    print(f"\nPlot saved to:        {save_path}")
    print(f"Text report saved to: {txt_path}")
