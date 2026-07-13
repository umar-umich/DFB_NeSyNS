"""
Pilot 1.5 mechanism panels for the meeting, analogous to the phase-1 viz_sample.png.
Reconstructs image variants from disk using the recorded per-sample metadata and annotates
them with the stored calibrated p(fake) values (no re-scoring needed).

Each row: fake -> true-region mask -> ground-truth repair (p drops) -> retrieval repair
(p stays high) -> retrieved FFHQ face -> real source. Shows the STOP result, real
retrieved pixels do not verify because any inner-face replacement is flagged.
"""
import os, sys, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/data/umar/Repos/DFB_NeSyNS"
FFHQ = "/data/umar/Datasets/ffhq256_subset"
sys.path.insert(0, f"{REPO}/scripts")
import intervention_pilot as P
from pilot1_rev3 import poisson_paste


def load_variants(rec):
    """Reload fake/real/mask from disk for a given (method, vid, frame) and build the
    gt-repair and retrieval-repair variants. Returns dict of RGB images or None."""
    method, vid, nnn = rec["method"], rec["vid"], rec["frame"]
    target = vid.split("_")[0]
    base = f"{P.PP}/manipulated_sequences/{method}/c23"
    fake = cv2.imread(f"{base}/frames/{vid}/{nnn}.png")
    mraw = cv2.imread(f"{base}/masks/{vid}/{nnn}.png", 0)
    lp = f"{base}/landmarks/{vid}/{nnn}.npy"
    if fake is None or mraw is None or not os.path.exists(lp):
        return None
    mask = mraw > 127
    lm = np.load(lp)
    raw = P.read_raw_frame(f"{P.RAW}/original_sequences/youtube/c23/videos/{target}.mp4", int(nnn))
    if raw is None:
        return None
    real, _ = P.align_face(raw, lm)
    if real is None:
        return None
    ret = cv2.imread(f"{FFHQ}/{rec['retrieved']}")
    if ret is None:
        return None
    gt = poisson_paste(fake, real, mask)
    rr = poisson_paste(fake, ret, mask)
    def rgb(b): return cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    over = rgb(fake).copy(); over[mask] = (0.45 * over[mask] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
    return dict(fake=rgb(fake), mask=over, gt=rgb(gt), retr=rgb(rr),
                retrieved=rgb(ret), real=rgb(real))


def main():
    samples = json.load(open(f"{REPO}/results/pilot1_5/effort/retrieval.json"))["samples"]
    recs = [s[0] for s in samples if isinstance(s, list) and isinstance(s[0], dict)]
    # clearest illustrations: large gt drop, near-zero retrieval drop
    recs = [r for r in recs if r["drop_gt"] > 0.5 and abs(r["drop_retr"]) < 0.1]
    recs = sorted(recs, key=lambda r: -r["drop_gt"])[:3]

    cols = ["fake", "mask", "gt", "retr", "retrieved", "real"]
    titles = ["fake\np(fake)={po:.2f}", "true region\n(GT mask)",
              "GT repair (real px)\np(fake)={pg:.2f}", "retrieval repair\np(fake)={pr:.2f}",
              "retrieved real face\n(FFHQ)", "real source"]
    rows = []
    for r in recs:
        v = load_variants(r)
        if v is not None:
            rows.append((r, v))
    n = len(rows)
    fig, axes = plt.subplots(n, 6, figsize=(17, 3.0 * n))
    if n == 1:
        axes = axes[None, :]
    for i, (r, v) in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i][j]
            ax.imshow(v[c]); ax.axis("off")
            if i == 0:
                t = titles[j].format(po=r["p_orig"], pg=r["p_gt"], pr=r["p_retr"])
                ax.set_title(t, fontsize=10, fontweight="bold")
            else:
                if j == 0: ax.set_title(f"p(fake)={r['p_orig']:.2f}", fontsize=9)
                if j == 2: ax.set_title(f"p(fake)={r['p_gt']:.2f}", fontsize=9, color="#009E73")
                if j == 3: ax.set_title(f"p(fake)={r['p_retr']:.2f}", fontsize=9, color="#D55E00")
        axes[i][0].text(-0.12, 0.5, f"{r['method']}\n{r['vid']}/{r['frame']}",
                        transform=axes[i][0].transAxes, rotation=90, va="center",
                        ha="center", fontsize=8, color="#666666")
    fig.suptitle("Pilot 1.5: restoring REAL original pixels drops p(fake) (green); pasting a "
                 "retrieved real face does NOT (orange) -- any inner-face replacement is flagged",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0.01, 0, 1, 0.97])
    out = f"{REPO}/results/figures/fig6_pilot1_5_samples.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, "with", n, "sample rows")


if __name__ == "__main__":
    main()
