"""
Pilot 1.5, retrieval-based repair, the replacement deployable verifier (PILOTS_2.md).

Phase 1 proved only real pixels lower p(fake), synthesized pixels are flagged by the
detector regardless of inpainter type (SD1.5 and LaMa both recovered ~0% and pushed reals
to ~0.57). So the deployable counterfactual should RETRIEVE real pixels, not generate
them. This pilot builds a frozen FAISS index over real FFHQ faces, retrieves the nearest
real face for a cited region, and Poisson-blends its real pixels into the query.

Runs on the same 100 FF++ frames, same calibrated detectors, same seeds as phase 1.
Compares retrieval repair against the phase-1 ground-truth Poisson repair upper bound and
against the known-dead inpainting arm (~0 recovery, ~0.57 real offset).

Corpus and feature definition are pre-registered in docs/PHASE2_PREREGISTRATION.md.

Run:
  HF_HOME=~/.cache/huggingface \
  /data/umar/miniconda3/envs/GenD/bin/python scripts/pilot1_5_retrieval.py \
      --detector effort --det-gpu 1 --clip-gpu 2
"""
import os, sys, json, random, argparse, glob
import numpy as np
import cv2

REPO = "/data/umar/Repos/DFB_NeSyNS"
FFHQ = "/data/umar/Datasets/ffhq256_subset"
CACHE = f"{REPO}/results/pilot1_5/ffhq_index.npz"
sys.path.insert(0, f"{REPO}/scripts")
import intervention_pilot as P
import pilot_detectors as DET
from pilot1_rev3 import poisson_paste

METHODS = P.METHODS
SEED = 0
TOPK = 10          # retrieve top-K by CLIP, then pick best by luminance match (lighting tie-break)


class Clip:
    def __init__(self, gpu):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.dev = f"cuda:{gpu}"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(self.dev)
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def embed_bgr(self, bgr_list, bs=128):
        import torch
        from PIL import Image
        out = []
        with torch.no_grad():
            for i in range(0, len(bgr_list), bs):
                pil = [Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)) for x in bgr_list[i:i+bs]]
                inp = self.proc(images=pil, return_tensors="pt").to(self.dev)
                f = self.model.get_image_features(**inp)
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def build_index(clip):
    """Embed the FFHQ corpus, return (embeddings NxD normalized, paths, luminance)."""
    import faiss
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return d["emb"], list(d["paths"]), d["lum"]
    paths = sorted(glob.glob(f"{FFHQ}/*.png"))
    print(f"[index] embedding {len(paths)} FFHQ faces ...")
    imgs, lum = [], []
    for p in paths:
        im = cv2.imread(p)
        imgs.append(im)
        lum.append(float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean()))
    emb = clip.embed_bgr(imgs)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, emb=emb, paths=np.array(paths), lum=np.array(lum, np.float32))
    print(f"[index] cached {CACHE}")
    return emb, paths, np.array(lum, np.float32)


def build_query(method, vid, nnn):
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
    ctrl = P.make_control_region(mask)
    if ctrl is None:
        return None
    return dict(method=method, vid=vid, frame=nnn), fake, real, mask, ctrl


def main():
    import torch, faiss
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="effort", choices=DET.available())
    ap.add_argument("--det-gpu", type=int, default=1)
    ap.add_argument("--clip-gpu", type=int, default=2)
    ap.add_argument("--n-per", type=int, default=50)
    args = ap.parse_args()
    outd = f"{REPO}/results/pilot1_5/{args.detector}"
    os.makedirs(outd, exist_ok=True)
    ddev = torch.device(f"cuda:{args.det_gpu}")
    rng = random.Random(SEED)

    print("[load] CLIP + detector ...")
    clip = Clip(args.clip_gpu)
    emb, paths, lum = build_index(clip)
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
    model, transform = DET.load_detector(args.detector, ddev)

    # gather the same 100 FF++ frames
    cand = P.gather_samples(rng)
    queries, counts = [], {m: 0 for m in METHODS}
    for (method, vid, nnn) in cand:
        if counts[method] >= args.n_per:
            continue
        q = build_query(method, vid, nnn)
        if q is None:
            continue
        queries.append(q); counts[method] += 1
        if all(counts[m] >= args.n_per for m in METHODS):
            break
    print(f"[sampling] valid: {counts}")

    # embed queries, retrieve
    q_imgs = [q[1] for q in queries]
    q_emb = clip.embed_bgr(q_imgs)
    D, I = index.search(q_emb, TOPK)          # cosine sim (higher better), top-K

    imgs, keys = [], []
    for i, (rec, fake, real, mask, ctrl) in enumerate(queries):
        # among top-K, pick the retrieved face whose mean luminance best matches the query
        qlum = float(cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY).mean())
        cands = I[i]
        best = min(cands, key=lambda j: abs(lum[j] - qlum))
        ret = cv2.imread(paths[best])
        rec["retrieved"] = os.path.basename(paths[best])
        rec["clip_sim"] = float(D[i][list(cands).index(best)])
        variants = dict(
            orig=fake,
            gt=poisson_paste(fake, real, mask),            # phase-1 upper bound
            retr=poisson_paste(fake, ret, mask),           # retrieval repair on fake
            retr_wrong=poisson_paste(fake, ret, ctrl),     # wrong-region control
            real_orig=real,
            real_retr=poisson_paste(real, ret, mask),      # real-repair offset for retrieval
        )
        for k, im in variants.items():
            imgs.append(im); keys.append((i, k))

    print(f"[detector] scoring {len(imgs)} images ...")
    probs = DET.pfake_batch(model, transform, imgs, ddev)
    pmap = {}
    for (i, k), p in zip(keys, probs):
        pmap.setdefault(i, {})[k] = float(p)

    d_gt, d_retr, d_wrong, real_off, fracs, idmis = [], [], [], [], [], []
    for i, (rec, *_) in enumerate(queries):
        d = pmap[i]
        dgt = d["orig"] - d["gt"]; dr = d["orig"] - d["retr"]
        ro = abs(d["real_retr"] - d["real_orig"])
        frac = dr / dgt if dgt > 0.05 else float("nan")
        rec.update(p_orig=d["orig"], p_gt=d["gt"], p_retr=d["retr"], p_wrong=d["retr_wrong"],
                   drop_gt=dgt, drop_retr=dr, drop_wrong=d["orig"] - d["retr_wrong"],
                   real_retr_offset=ro, frac_recovered=frac,
                   id_mismatch=1.0 - rec["clip_sim"])
        d_gt.append(dgt); d_retr.append(dr); d_wrong.append(d["orig"] - d["retr_wrong"])
        real_off.append(ro); idmis.append(1.0 - rec["clip_sim"])
        if not np.isnan(frac):
            fracs.append(np.clip(frac, -1, 2))

    def m(x): return float(np.mean(x)) if len(x) else float("nan")
    def md(x): return float(np.median(x)) if len(x) else float("nan")
    # id-mismatch vs offset correlation (step 7)
    if len(real_off) >= 3 and np.std(idmis) > 0 and np.std(real_off) > 0:
        corr = float(np.corrcoef(idmis, real_off)[0, 1])
    else:
        corr = float("nan")
    S = dict(
        detector=args.detector, corpus="FFHQ-256 (8750)", n=len(queries), counts=counts,
        mean_drop_gt=m(d_gt), median_drop_gt=md(d_gt),
        mean_drop_retr=m(d_retr), median_drop_retr=md(d_retr),
        mean_frac_recovered=m(fracs), median_frac_recovered=md(fracs), n_frac=len(fracs),
        mean_real_offset=m(real_off), median_real_offset=md(real_off),
        inpainting_real_offset_ref=0.574,        # phase-1 SD1.5 dead reference
        mean_drop_wrong=m(d_wrong), median_drop_wrong=md(d_wrong),
        idmismatch_vs_offset_corr=corr,
    )
    # gate (PILOTS_2 Pilot 1.5)
    if S["median_frac_recovered"] >= 0.4 and S["median_real_offset"] < 0.30 \
            and S["median_drop_wrong"] <= 0.10:
        verdict = "GO -- retrieval recovers substantial gt fraction, offset well below inpainting, wrong-region inert."
    elif S["median_drop_retr"] >= 0.10 and corr == corr and corr >= 0.3:
        verdict = "PARTIAL -- retrieval recovers real signal but offset is identity-mismatch driven; reweight descriptor toward identity, one retry."
    elif S["median_frac_recovered"] < 0.1 or S["median_real_offset"] >= 0.5:
        verdict = "STOP -- retrieval behaves like inpainting; no deployable pixel-space verifier, scope deployable claim to spectral+small-region."
    else:
        verdict = "INSPECT -- between thresholds; see distributions."
    S["verdict"] = verdict

    json.dump({"summary": S, "samples": queries}, open(f"{outd}/retrieval.json", "w"),
              indent=2, default=str)
    lines = [
        f"Pilot 1.5 -- retrieval-based repair [{args.detector}] (FFHQ-256 corpus, n=8750)",
        "=" * 74,
        f"samples: {S['n']} {S['counts']}",
        "",
        f"  gt-repair drop (upper bound)  mean/med : {S['mean_drop_gt']:.4f} / {S['median_drop_gt']:.4f}",
        f"  RETRIEVAL repair drop         mean/med : {S['mean_drop_retr']:.4f} / {S['median_drop_retr']:.4f}",
        f"  fraction of gt recovered      mean/med : {S['mean_frac_recovered']:.3f} / {S['median_frac_recovered']:.3f}  (n={S['n_frac']})",
        f"  retrieval real-repair offset  mean/med : {S['mean_real_offset']:.4f} / {S['median_real_offset']:.4f}",
        f"     vs inpainting offset (phase-1 dead) : {S['inpainting_real_offset_ref']:.3f}",
        f"  wrong-region control drop     mean/med : {S['mean_drop_wrong']:.4f} / {S['median_drop_wrong']:.4f}",
        f"  id-mismatch vs offset corr             : {S['idmismatch_vs_offset_corr']:.3f}",
        "",
        f"VERDICT: {verdict}",
    ]
    open(f"{outd}/retrieval.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {outd}/retrieval.txt")


if __name__ == "__main__":
    main()
