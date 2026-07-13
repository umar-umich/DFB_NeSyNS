"""
Pilot 1 rev3, diffusion-repair arm (CLAUDE_CODE_PILOTS Pilot 1 step 5, and the PARTIAL
outcome in PLANNING section 8.1).

The deployable intervention. A frozen diffusion inpainter reconstructs the cited region
instead of pasting the paired-real region. We measure how much of the gold-standard
ground-truth Poisson repair effect the diffusion repair recovers, per frame. We also
apply diffusion repair to the paired real frame, this offset is expected to be nonzero
because the inpainter synthesizes pixels, and it is the "the inpainter is itself a
generator" caveat made measurable.

Gate reading. GO-with-diffusion if diffusion repair recovers a substantial fraction of
the gt-repair drop. PARTIAL if gt-repair passed (it did, all four detectors) but
diffusion repair does not, then the test is sound and the inpainter is the weak link,
the pivot is a better repair operator, not abandoning the framework.

Inpainter: runwayml/stable-diffusion-inpainting (SD1.5, frozen). Native 512, our crops
are 256 so we upscale, inpaint the region, downscale back and composite only inside the
mask. (stabilityai/stable-diffusion-2-inpainting is no longer available on the hub.)

Run:
  HF_HOME=~/.cache/huggingface \
  /data/umar/miniconda3/envs/GenD/bin/python scripts/pilot1_diffusion.py --detector effort \
      --det-gpu 0 --sd-gpu 1
"""
import os, sys, json, random, argparse
import numpy as np
import cv2

REPO = "/data/umar/Repos/DFB_NeSyNS"
sys.path.insert(0, f"{REPO}/scripts")
import intervention_pilot as P
import pilot_detectors as DET
from pilot1_rev3 import poisson_paste

METHODS = P.METHODS
SEED = 0


class LaMaInpaint:
    """LaMa, a non-generative (Fourier-convolution) inpainter. Unlike a diffusion model it
    does not synthesize new high-frequency content from noise, so it should inject far
    fewer detector-visible artifacts, the recommended pivot when diffusion repair fails."""
    name = "lama"

    def __init__(self, gpu):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
        from simple_lama_inpainting import SimpleLama
        self.model = SimpleLama()

    def repair(self, bgr, region_mask):
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = (region_mask.astype(np.uint8) * 255)
        m = cv2.dilate(m, np.ones((5, 5), np.uint8), 1)
        out = self.model(Image.fromarray(rgb), Image.fromarray(m).convert("L"))
        out = np.array(out.convert("RGB").resize((bgr.shape[1], bgr.shape[0]), Image.BICUBIC))
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        comp = bgr.copy()
        comp[region_mask] = out_bgr[region_mask]
        return comp


class SDInpaint:
    name = "sd2-inpainting"
    def __init__(self, gpu):
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        self.torch = torch
        self.dev = f"cuda:{gpu}"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16,
            safety_checker=None).to(self.dev)
        self.pipe.set_progress_bar_config(disable=True)
        self.gen = torch.Generator(device=self.dev).manual_seed(SEED)

    def repair(self, bgr, region_mask):
        """Inpaint region_mask (bool) in a BGR crop, return a BGR crop with only the
        region replaced by the inpainted content (background untouched)."""
        from PIL import Image
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((512, 512), Image.BICUBIC)
        # dilate the mask slightly so the inpainter blends the seam
        m = (region_mask.astype(np.uint8) * 255)
        m = cv2.dilate(m, np.ones((5, 5), np.uint8), 1)
        mimg = Image.fromarray(m).resize((512, 512), Image.NEAREST)
        out = self.pipe(prompt="a natural human face, photograph",
                        image=img, mask_image=mimg, num_inference_steps=30,
                        guidance_scale=7.5, generator=self.gen).images[0]
        out = np.array(out.resize((W, H), Image.BICUBIC))            # RGB
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        comp = bgr.copy()
        comp[region_mask] = out_bgr[region_mask]                     # region only
        return comp


def build(method, vid, nnn):
    target = vid.split("_")[0]
    base = f"{P.PP}/manipulated_sequences/{method}/c23"
    fake = cv2.imread(f"{base}/frames/{vid}/{nnn}.png")
    mraw = cv2.imread(f"{base}/masks/{vid}/{nnn}.png", 0)
    lp = f"{base}/landmarks/{vid}/{nnn}.npy"
    if fake is None or mraw is None or not os.path.exists(lp):
        return None
    mask = mraw > 127
    if not (P.FG_MIN <= mask.mean() <= P.FG_MAX):
        return None
    lm = np.load(lp)
    raw = P.read_raw_frame(f"{P.RAW}/original_sequences/youtube/c23/videos/{target}.mp4", int(nnn))
    if raw is None:
        return None
    real, _ = P.align_face(raw, lm)
    if real is None:
        return None
    if float(((fake.astype(float) - real.astype(float))[~mask] ** 2).mean()) > P.QC_MSE:
        return None
    return dict(method=method, vid=vid, frame=nnn), fake, real, mask


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="effort", choices=DET.available())
    ap.add_argument("--det-gpu", type=int, default=0)
    ap.add_argument("--sd-gpu", type=int, default=1)
    ap.add_argument("--n-per", type=int, default=50)
    ap.add_argument("--inpainter", choices=["sd", "lama"], default="sd")
    args = ap.parse_args()
    outd = f"{REPO}/results/pilot_rev3/{args.detector}"
    os.makedirs(outd, exist_ok=True)
    ddev = torch.device(f"cuda:{args.det_gpu}")
    rng = random.Random(SEED)

    print(f"[load] detector + {args.inpainter} inpainter ...")
    model, transform = DET.load_detector(args.detector, ddev)
    sd = LaMaInpaint(args.sd_gpu) if args.inpainter == "lama" else SDInpaint(args.sd_gpu)
    tag = sd.name

    cand = P.gather_samples(rng)
    samples, counts = [], {m: 0 for m in METHODS}
    for (method, vid, nnn) in cand:
        if counts[method] >= args.n_per:
            continue
        b = build(method, vid, nnn)
        if b is None:
            continue
        samples.append(b); counts[method] += 1
        if all(counts[m] >= args.n_per for m in METHODS):
            break
    print(f"[sampling] valid: {counts}; inpainting {len(samples)} frames ...")

    imgs, keys = [], []
    for i, (rec, fake, real, mask) in enumerate(samples):
        gt = poisson_paste(fake, real, mask)              # gold-standard upper bound
        dfx = sd.repair(fake, mask)                        # diffusion repair on fake
        dreal = sd.repair(real, mask)                      # diffusion repair on real (offset)
        for k, im in [("orig", fake), ("gt", gt), ("diff", dfx),
                      ("real_orig", real), ("real_diff", dreal)]:
            imgs.append(im); keys.append((i, k))
        if (i + 1) % 20 == 0:
            print(f"  inpainted {i+1}/{len(samples)}")
    probs = DET.pfake_batch(model, transform, imgs, ddev)
    pmap = {}
    for (i, k), p in zip(keys, probs):
        pmap.setdefault(i, {})[k] = float(p)

    recs, fracs = [], []
    d_gt, d_diff, real_off = [], [], []
    for i, (rec, *_ ) in enumerate(samples):
        d = pmap[i]
        dgt = d["orig"] - d["gt"]; ddi = d["orig"] - d["diff"]
        ro = abs(d["real_diff"] - d["real_orig"])
        frac = ddi / dgt if dgt > 0.05 else float("nan")   # fraction of gt effect recovered
        rec.update(p_orig=d["orig"], p_gt=d["gt"], p_diff=d["diff"],
                   drop_gt=dgt, drop_diff=ddi, frac_recovered=frac, real_diff_offset=ro)
        recs.append(rec)
        d_gt.append(dgt); d_diff.append(ddi); real_off.append(ro)
        if not np.isnan(frac):
            fracs.append(np.clip(frac, -1, 2))

    def m(x): return float(np.mean(x)) if len(x) else float("nan")
    def md(x): return float(np.median(x)) if len(x) else float("nan")
    S = dict(
        detector=args.detector, inpainter=tag, n=len(recs), counts=counts,
        mean_drop_gt=m(d_gt), mean_drop_diff=m(d_diff),
        median_drop_gt=md(d_gt), median_drop_diff=md(d_diff),
        mean_frac_recovered=m(fracs), median_frac_recovered=md(fracs),
        mean_real_diff_offset=m(real_off), median_real_diff_offset=md(real_off),
        n_frac=len(fracs),
    )
    if S["median_frac_recovered"] >= 0.5:
        verdict = "diffusion repair recovers a substantial fraction of gt-repair -- deployable arm OK."
    elif S["median_frac_recovered"] >= 0.25:
        verdict = "diffusion repair partial -- usable but weaker than gt; report as fraction."
    else:
        verdict = "PARTIAL -- gt-repair gate passed but diffusion repair is the weak link; better inpainter needed."
    S["verdict"] = verdict

    suffix = "" if args.inpainter == "sd" else "_lama"
    json.dump({"summary": S, "samples": recs}, open(f"{outd}/diffusion{suffix}.json", "w"), indent=2)
    lines = [
        f"Pilot 1 rev3 -- diffusion-repair arm [{args.detector}] ({tag})",
        "=" * 68,
        f"samples: {S['n']} {S['counts']}",
        "",
        f"  gt-repair drop      mean/med : {S['mean_drop_gt']:.4f} / {S['median_drop_gt']:.4f}",
        f"  diffusion-rep drop  mean/med : {S['mean_drop_diff']:.4f} / {S['median_drop_diff']:.4f}",
        f"  fraction recovered  mean/med : {S['mean_frac_recovered']:.3f} / {S['median_frac_recovered']:.3f}  (n={S['n_frac']})",
        f"  real diffusion offset mean/med: {S['mean_real_diff_offset']:.4f} / {S['median_real_diff_offset']:.4f}",
        "",
        f"VERDICT: {verdict}",
    ]
    open(f"{outd}/diffusion{suffix}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {outd}/diffusion{suffix}.txt")


if __name__ == "__main__":
    main()
