"""
Meeting figures for the phase-1 gate and the phase-2 pilots. Reads the recorded JSON
results and renders clean PNGs to results/figures/. No model runs.

Palette, Okabe-Ito, colorblind-safe by construction. Categorical hues assigned in fixed
order. One axis per panel, recessive grid, value labels, takeaway titles.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/data/umar/Repos/DFB_NeSyNS"
OUT = f"{REPO}/results/figures"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito
BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
VERM, PURPLE, SKY, GREY = "#D55E00", "#CC79A7", "#56B4E9", "#8a8a8a"
INK, MUTED = "#222222", "#666666"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "white", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.edgecolor": "#cccccc",
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.labelsize": 11,
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#e8e8e8", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def barlabels(ax, bars, fmt="{:.3f}", dy=0.005):
    for b in bars:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, h + (dy if h >= 0 else -dy),
                fmt.format(h), ha="center", va=va, fontsize=8.5, color=INK)


# --------------------------------------------------------------- Fig 1: phase-1 gate
def fig_gate():
    dets = ["effort", "fsfm", "gend", "forada"]
    gt, matched, wrong = [], [], []
    for d in dets:
        s = json.load(open(f"{REPO}/results/pilot_rev3/{d}/deltas.json"))["summary"]
        gt.append(s["median_drop_gt"])
        matched.append(max(s["mean_drop_blur"], s["mean_drop_shift"]))
        wrong.append(s["median_drop_wrong"])
    x = np.arange(len(dets)); w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style(ax)
    b1 = ax.bar(x - w, gt, w, label="ground-truth repair", color=GREEN, zorder=3)
    b2 = ax.bar(x, matched, w, label="LPIPS-matched corruption", color=ORANGE, zorder=3)
    b3 = ax.bar(x + w, wrong, w, label="wrong-region control", color=GREY, zorder=3)
    barlabels(ax, b1); barlabels(ax, b2); barlabels(ax, b3)
    ax.axhline(0, color="#bbbbbb", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(dets)
    ax.set_ylabel("drop in p(fake)")
    ax.set_title("Phase-1 gate PASS: only restoring the true region drops p(fake)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(0.5, -0.16, "Restoring real pixels drops the score 0.45-0.84; a perceptually "
            "equal corruption and a wrong-region repair do nothing.",
            transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_phase1_gate.png", bbox_inches="tight")
    plt.close(fig); print("wrote fig1_phase1_gate.png")


# --------------------------------------------------------------- Fig 2: 1.5 + inpainting dead
def fig_repair_deadend():
    # recovery fraction and real-offset for gt / SD / LaMa / retrieval
    ops = ["ground-truth\nPoisson", "SD1.5\ninpaint", "LaMa\ninpaint", "retrieval\n(real pixels)"]
    frac = [1.00, -0.003, 0.007, -0.004]        # median fraction of gt effect recovered
    offset = [0.03, 0.574, 0.568, 0.619]        # real-repair offset (median), gt~small
    x = np.arange(len(ops))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes:
        style(ax)
    colors = [GREEN, VERM, VERM, ORANGE]
    b = axes[0].bar(x, frac, color=colors, zorder=3)
    barlabels(axes[0], b, fmt="{:.2f}")
    axes[0].axhline(0, color="#bbbbbb", lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(ops, fontsize=9)
    axes[0].set_ylabel("fraction of gt-repair effect recovered")
    axes[0].set_title("Only real original pixels verify")
    b2 = axes[1].bar(x, offset, color=colors, zorder=3)
    barlabels(axes[1], b2, fmt="{:.2f}")
    axes[1].set_xticks(x); axes[1].set_xticklabels(ops, fontsize=9)
    axes[1].set_ylabel("real-frame offset (p(fake) rise)")
    axes[1].set_title("Any replacement fakes a real frame equally")
    fig.suptitle("Pilot 1.5 STOP: retrieval fails like inpainting; the detector flags any "
                 "inner-face replacement", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_repair_deadend.png", bbox_inches="tight")
    plt.close(fig); print("wrote fig2_repair_deadend.png")


# --------------------------------------------------------------- Fig 3: Study 1A curve
def fig_1a():
    s = json.load(open(f"{REPO}/results/pilot1_5/study1a_effort.json"))
    levels = s["area_levels"]; curve = s["sd_offset_curve"]; lama = s["lama_offset_curve"]
    xs = [l * 100 for l in levels] + [40]         # full mask placed at 40% for display
    sd = [curve[str(l)]["median"] for l in levels] + [curve["full_mask"]["median"]]
    lx = [l * 100 for l in levels[:2]]
    ly = [lama[str(l)]["median"] for l in levels[:2]]
    thr = s["nuisance_threshold"]; adm = s["admissible_max_area"] * 100
    fig, ax = plt.subplots(figsize=(8, 4.8)); style(ax)
    ax.axvspan(0, adm, color=GREEN, alpha=0.08, zorder=0)
    ax.plot(xs, sd, "-o", color=BLUE, lw=2, ms=7, label="SD1.5 inpaint", zorder=3)
    ax.plot(lx, ly, "s", color=ORANGE, ms=8, label="LaMa inpaint", zorder=3)
    ax.axhline(thr, color=VERM, lw=1.5, ls="--", zorder=2)
    ax.text(41, thr + 0.01, f"nuisance threshold {thr:.3f}", color=VERM, fontsize=9, ha="right")
    ax.text(adm / 2, ax.get_ylim()[1] * 0.9 if False else 0.55,
            f"admissible\n<= {adm:.0f}% area", color=GREEN, fontsize=9, ha="center")
    for xv, yv in zip(xs, sd):
        ax.text(xv, yv + 0.02, f"{yv:.2f}", fontsize=8, ha="center", color=INK)
    ax.set_xlabel("inpainted region area (% of face crop)")
    ax.set_ylabel("real-frame offset (median p(fake) rise)")
    ax.set_title("Study 1A: small-region inpainting is admissible only below ~3% area")
    ax.set_xticks([1, 3, 8, 15, 30, 40]); ax.set_xticklabels(["1", "3", "8", "15", "30", "full"])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_study1a_offset_area.png", bbox_inches="tight")
    plt.close(fig); print("wrote fig3_study1a_offset_area.png")


# --------------------------------------------------------------- Fig 4: Study 1B contrast
def fig_1b():
    s = json.load(open(f"{REPO}/results/study1b/study1b_effort.json"))
    fams = ["GAN", "diffusion", "swap"]
    drops = [abs(s["spectral_drops"]["native"][f]["checker_med"]) for f in fams]
    auc_nat = [s["feature_auc"]["native"][f]["spectral"] for f in fams]
    auc_q50 = [s["feature_auc"]["q50"][f]["spectral"] for f in fams]
    x = np.arange(len(fams))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes:
        style(ax)
    b = axes[0].bar(x, drops, color=ORANGE, width=0.5, zorder=3)
    barlabels(axes[0], b, fmt="{:.3f}", dy=0.0008)
    axes[0].set_xticks(x); axes[0].set_xticklabels(fams)
    axes[0].set_ylim(0, 0.1)
    axes[0].set_ylabel("|spectral intervention drop|")
    axes[0].set_title("Detector ignores spectral cues", fontsize=11.5)
    w = 0.38
    bn = axes[1].bar(x - w / 2, auc_nat, w, label="native", color=BLUE, zorder=3)
    bq = axes[1].bar(x + w / 2, auc_q50, w, label="JPEG q50", color=SKY, zorder=3)
    barlabels(axes[1], bn, fmt="{:.2f}", dy=0.004); barlabels(axes[1], bq, fmt="{:.2f}", dy=0.004)
    axes[1].axhline(0.5, color=GREY, lw=1, ls=":")
    axes[1].text(2.4, 0.51, "chance", color=GREY, fontsize=8, ha="right")
    axes[1].set_xticks(x); axes[1].set_xticklabels(fams); axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("real-vs-fake separability (AUC)")
    axes[1].set_title("Fingerprint strong, survives compression", fontsize=11.5)
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Study 1B: spectral evidence exists (AUC 0.94-0.99) but the CLIP detector "
                 "does not use it", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.subplots_adjust(wspace=0.28)
    fig.savefig(f"{OUT}/fig4_study1b_spectral.png", bbox_inches="tight")
    plt.close(fig); print("wrote fig4_study1b_spectral.png")


# --------------------------------------------------------------- Fig 5: 2R attribution
def fig_2r():
    a = json.load(open(f"{REPO}/results/pilot2r/attribution2r.json"))
    r = a["results"]; chance = a["chance"]
    order = ["A_clip", "B_dct", "C_count", "D_enriched", "E_knn"]
    labels = ["CLIP", "DCT", "pred-count", "enriched\npredicates", "retrieval\nkNN"]
    seen = [r[k]["seen_mean"] for k in order]
    held = [r[k]["held_mean"] for k in order]
    x = np.arange(len(order)); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    style(axes[0]); style(axes[1])
    b1 = axes[0].bar(x - w / 2, seen, w, label="seen generators", color=BLUE, zorder=3)
    b2 = axes[0].bar(x + w / 2, held, w, label="held-out generators", color=ORANGE, zorder=3)
    barlabels(axes[0], b1, fmt="{:.2f}", dy=0.006); barlabels(axes[0], b2, fmt="{:.2f}", dy=0.006)
    axes[0].axhline(chance, color=VERM, lw=1.5, ls="--")
    axes[0].text(4.4, chance + 0.01, f"chance {chance:.2f}", color=VERM, fontsize=8, ha="right")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(0, 0.9); axes[0].set_ylabel("family attribution accuracy")
    axes[0].set_title("Predicates carry signal but never beat CLIP")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")

    v = json.load(open(f"{REPO}/results/pilot2r/vlm_ceiling.json"))["results"]
    vord = ["VLM_only", "D_det", "D_full"]
    vlab = ["VLM only", "deterministic", "+ VLM semantic"]
    vs = [v[k]["seen"] for k in vord]; vh = [v[k]["held"] for k in vord]
    xv = np.arange(len(vord))
    c1 = axes[1].bar(xv - w / 2, vs, w, label="seen", color=BLUE, zorder=3)
    c2 = axes[1].bar(xv + w / 2, vh, w, label="held-out", color=ORANGE, zorder=3)
    barlabels(axes[1], c1, fmt="{:.2f}", dy=0.006); barlabels(axes[1], c2, fmt="{:.2f}", dy=0.006)
    axes[1].axhline(0.4, color=VERM, lw=1.5, ls="--")
    axes[1].set_xticks(xv); axes[1].set_xticklabels(vlab, fontsize=9)
    axes[1].set_ylim(0, 0.9); axes[1].set_ylabel("family attribution accuracy")
    axes[1].set_title("Adding VLM semantics does not help")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Pilot 2R: attribution is SOFT; enriched predicates do not beat a frozen "
                 "CLIP probe, VLM adds nothing", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_2r_attribution.png", bbox_inches="tight")
    plt.close(fig); print("wrote fig5_2r_attribution.png")


if __name__ == "__main__":
    fig_gate(); fig_repair_deadend(); fig_1a(); fig_1b(); fig_2r()
    print("all figures in", OUT)
