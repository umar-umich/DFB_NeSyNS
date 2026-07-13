"""
Pilot 1 rev3, InternVL proposal recall + localization check
(CLAUDE_CODE_PILOTS Pilot 1 steps 10 and 11).

Step 10, proposal recall. The frozen InternVL3-8B proposer looks at each fake face crop
and names, from the fixed codebook vocabulary, which regions show which artifact. We map
its named regions to canonical landmark boxes on the aligned crop and measure recall and
precision against the ground-truth manipulation mask. A grounding model is NOT on the
critical path, the region boxes are deterministic, InternVL only proposes region ids.

Step 11, localization check. For each proposed region we run the necessity test, Poisson
repair of that region box with the paired real content, and label the proposal PASS if
the detector p(fake) drop clears a threshold. We then compare ground-truth mask overlap
of PASS versus FAIL proposals. This is the headline of the framework in miniature,
intervention-confirmed predicates should localize on the true mask better than rejected
ones.

InternVL on cuda:vlm-gpu, the frozen detector on cuda:det-gpu.

Run:
  HF_HOME=~/.cache/huggingface \
  /data/umar/miniconda3/envs/GenD/bin/python scripts/pilot1_vlm.py --detector effort \
      --det-gpu 0 --vlm-gpu 1
"""
import os, sys, json, re, random, argparse
import numpy as np
import cv2

REPO = "/data/umar/Repos/DFB_NeSyNS"
sys.path.insert(0, f"{REPO}/scripts")
import intervention_pilot as P
import pilot_detectors as DET
from pilot1_rev3 import poisson_paste

METHODS = P.METHODS
SEED = 0
NECESSITY_DROP = 0.10          # a proposed region PASSES if gt-repair drops p(fake) by this

# canonical region boxes on the 256x256 aligned crop (GenD 5pt template, scale 1.3)
# derived landmarks: LE(96,120) RE(160,120) NOSE(128,156) LM(102,191) RM(154,191)
REGION_BOXES = {
    "left_eye":     (72, 100, 116, 138),
    "right_eye":    (140, 100, 184, 138),
    "inter_ocular": (112, 104, 144, 140),
    "nose":         (104, 132, 152, 180),
    "mouth":        (96, 174, 160, 210),
    "nasolabial":   (92, 164, 164, 196),
    "left_cheek":   (60, 138, 104, 192),
    "right_cheek":  (152, 138, 196, 192),
    "forehead":     (80, 56, 176, 104),
    "hairline":     (72, 34, 184, 64),
    "chin":         (100, 206, 156, 244),
    "jawline":      (56, 178, 200, 248),
}
PREDICATES = ["boundary_artifact", "blend_seam_visible", "frequency_anomaly",
              "noise_inconsistency", "geometry_inconsistency", "identity_drift",
              "texture_anomaly", "physiology_violation", "lighting_inconsistency"]

# The VLM's role in the framework is PROPOSAL, not detection. The detector already
# flagged the frame, so the proposer is told it is manipulated and asked to LOCALIZE.
VLM_PROMPT = (
    "<image>\n"
    "This 256x256 aligned face crop has been flagged by a forensic detector as a likely "
    "deepfake or face swap. Your job is to localize the manipulation. Identify the facial "
    "regions most likely to contain the manipulation artifacts and name the artifact type "
    "for each.\n"
    f"Allowed regions: {', '.join(REGION_BOXES)}.\n"
    f"Allowed artifact types: {', '.join(PREDICATES)}.\n"
    "List every region you find suspicious, from most to least. Respond with ONLY a JSON "
    'array of objects like [{"region":"nose","artifact":"blend_seam_visible"}]. '
    "Use only allowed region and artifact names."
)


def box_mask(box, shape):
    m = np.zeros(shape[:2], bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


def overlap_frac(box, gt_mask):
    """Fraction of the region box covered by the GT manipulation mask."""
    bm = box_mask(box, gt_mask.shape)
    a = bm.sum()
    return float((bm & gt_mask).sum() / a) if a else 0.0


def true_regions(gt_mask, thr=0.25):
    return {r for r, b in REGION_BOXES.items() if overlap_frac(b, gt_mask) >= thr}


def parse_vlm(txt):
    """Extract the JSON array and return the set of valid region names proposed."""
    m = re.search(r"\[.*\]", txt, re.S)
    regions = set()
    if not m:
        return regions
    try:
        arr = json.loads(m.group(0))
    except Exception:
        # salvage region names by keyword match
        for r in REGION_BOXES:
            if r in txt:
                regions.add(r)
        return regions
    for o in arr:
        if isinstance(o, dict):
            r = str(o.get("region", "")).strip()
            if r in REGION_BOXES:
                regions.add(r)
    return regions


class InternVL:
    def __init__(self, gpu):
        import torch
        from transformers import AutoModel, AutoTokenizer
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        self.torch = torch
        self.dev = f"cuda:{gpu}"
        mid = "OpenGVLab/InternVL3-8B"
        self.tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            mid, dtype=torch.bfloat16, trust_remote_code=True,
            low_cpu_mem_usage=True).eval().to(self.dev)
        self.tf = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def propose(self, bgr):
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pv = self.tf(img).unsqueeze(0).to(self.torch.bfloat16).to(self.dev)
        with self.torch.no_grad():
            out = self.model.chat(self.tok, pv, VLM_PROMPT,
                                  dict(max_new_tokens=128, do_sample=False))
        return parse_vlm(out), out


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
    ap.add_argument("--vlm-gpu", type=int, default=1)
    ap.add_argument("--n-per", type=int, default=50)
    args = ap.parse_args()
    outd = f"{REPO}/results/pilot_rev3/{args.detector}"
    os.makedirs(outd, exist_ok=True)
    ddev = torch.device(f"cuda:{args.det_gpu}")
    rng = random.Random(SEED)

    print("[load] detector + InternVL ...")
    model, transform = DET.load_detector(args.detector, ddev)
    vlm = InternVL(args.vlm_gpu)

    cand = P.gather_samples(rng)
    samples = []
    counts = {m: 0 for m in METHODS}
    for (method, vid, nnn) in cand:
        if counts[method] >= args.n_per:
            continue
        b = build(method, vid, nnn)
        if b is None:
            continue
        samples.append(b); counts[method] += 1
        if all(counts[m] >= args.n_per for m in METHODS):
            break
    print(f"[sampling] valid: {counts}")

    # 1) InternVL proposals
    recs = []
    for rec, fake, real, mask in samples:
        proposed, raw_txt = vlm.propose(fake)
        tr = true_regions(mask)
        recs.append(dict(rec=rec, proposed=sorted(proposed), true=sorted(tr),
                         fake=fake, real=real, mask=mask, raw=raw_txt[:300]))
    print("[vlm] proposals done")

    # 2) necessity per proposed region: gt Poisson repair of the region box, detector drop
    imgs, keys = [], []
    for i, r in enumerate(recs):
        imgs.append(r["fake"]); keys.append((i, "orig"))
        for reg in r["proposed"]:
            bm = box_mask(REGION_BOXES[reg], r["fake"].shape)
            imgs.append(poisson_paste(r["fake"], r["real"], bm))
            keys.append((i, reg))
    probs = DET.pfake_batch(model, transform, imgs, ddev)
    pmap = {}
    for (i, k), p in zip(keys, probs):
        pmap.setdefault(i, {})[k] = float(p)

    # metrics
    tp = fp = fn = 0                       # micro region-level
    frame_hit = 0
    pass_overlaps, fail_overlaps = [], []
    n_frames_with_true = 0
    for i, r in enumerate(recs):
        prop, tru = set(r["proposed"]), set(r["true"])
        tp += len(prop & tru); fp += len(prop - tru); fn += len(tru - prop)
        if tru:
            n_frames_with_true += 1
            if prop & tru:
                frame_hit += 1
        po = pmap[i]["orig"]
        for reg in r["proposed"]:
            drop = po - pmap[i][reg]
            ov = overlap_frac(REGION_BOXES[reg], r["mask"])
            r.setdefault("region_eval", []).append(dict(region=reg, drop=drop, mask_overlap=ov,
                                                         passed=bool(drop >= NECESSITY_DROP)))
            (pass_overlaps if drop >= NECESSITY_DROP else fail_overlaps).append(ov)

    def m(x): return float(np.mean(x)) if len(x) else float("nan")
    # continuous localization signal: correlation of per-region repair drop with mask overlap
    all_drops = [re_["drop"] for r in recs for re_ in r.get("region_eval", [])]
    all_ovs = [re_["mask_overlap"] for r in recs for re_ in r.get("region_eval", [])]
    if len(all_drops) >= 3 and np.std(all_drops) > 0 and np.std(all_ovs) > 0:
        pearson = float(np.corrcoef(all_drops, all_ovs)[0, 1])
        ro = np.argsort(np.argsort(all_drops)); rv = np.argsort(np.argsort(all_ovs))
        spearman = float(np.corrcoef(ro, rv)[0, 1])
    else:
        pearson = spearman = float("nan")
    S = dict(
        detector=args.detector, vlm="InternVL3-8B", n=len(recs), counts=counts,
        region_recall=tp / (tp + fn) if (tp + fn) else float("nan"),
        region_precision=tp / (tp + fp) if (tp + fp) else float("nan"),
        frame_detection_recall=frame_hit / n_frames_with_true if n_frames_with_true else float("nan"),
        mean_true_regions_per_frame=m([len(r["true"]) for r in recs]),
        mean_proposed_regions_per_frame=m([len(r["proposed"]) for r in recs]),
        n_pass=len(pass_overlaps), n_fail=len(fail_overlaps),
        mean_mask_overlap_PASS=m(pass_overlaps),
        mean_mask_overlap_FAIL=m(fail_overlaps),
        necessity_drop_thr=NECESSITY_DROP,
        pearson_drop_vs_overlap=pearson,
        spearman_drop_vs_overlap=spearman,
        n_proposed_regions=len(all_drops),
    )
    # strip images before dumping
    dump = []
    for r in recs:
        dump.append(dict(rec=r["rec"], proposed=r["proposed"], true=r["true"],
                         region_eval=r.get("region_eval", []), raw=r["raw"]))
    json.dump({"summary": S, "samples": dump}, open(f"{outd}/vlm.json", "w"), indent=2)

    lines = [
        f"Pilot 1 rev3 -- InternVL proposal recall + localization [{args.detector}]",
        "=" * 70,
        f"VLM: InternVL3-8B   detector: {args.detector}   samples: {S['n']} {S['counts']}",
        "",
        "Step 10, proposal quality vs GT mask (region boxes on the aligned crop):",
        f"  region recall            : {S['region_recall']:.3f}",
        f"  region precision         : {S['region_precision']:.3f}",
        f"  frame detection recall   : {S['frame_detection_recall']:.3f}  "
        f"(>=1 true region proposed)",
        f"  true regions / frame     : {S['mean_true_regions_per_frame']:.2f}",
        f"  proposed regions / frame : {S['mean_proposed_regions_per_frame']:.2f}",
        "",
        "Step 11, localization -- does the per-region repair drop track GT-mask overlap:",
        f"  Pearson  (drop vs mask overlap) : {S['pearson_drop_vs_overlap']:.3f}   "
        f"(n={S['n_proposed_regions']} proposed regions)",
        f"  Spearman (drop vs mask overlap) : {S['spearman_drop_vs_overlap']:.3f}",
        f"  PASS proposals (n={S['n_pass']:3d})  mean mask overlap : {S['mean_mask_overlap_PASS']:.3f}",
        f"  FAIL proposals (n={S['n_fail']:3d})  mean mask overlap : {S['mean_mask_overlap_FAIL']:.3f}",
        f"  (a proposal PASSES if gt-repair of its box drops p(fake) by >= {NECESSITY_DROP})",
    ]
    open(f"{outd}/vlm.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {outd}/vlm.json  {outd}/vlm.txt")


if __name__ == "__main__":
    main()
