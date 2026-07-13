"""
Pilot 2R VLM ceiling test. Adds InternVL semantic predicates to the enriched attribution
vector and isolates their contribution to seen accuracy and, critically, to cross-generator
transfer, where the deterministic-only enrichment lost ground.

AWQ note. LMDeploy 0.14.0 fails to build the InternVL3 vision model under transformers
4.56 ('dict' object has no attribute 'attn_implementation'), and downgrading transformers
would break the frozen detectors, CLIP, and SD that this project depends on. Per PILOTS_2
the sanctioned fallback is fp16, used here, with the throughput cost that a smaller DF40
sample is extracted (bounded per-generator frame counts, batched InternVL).

Semantic feature per frame. InternVL proposes {region, predicate} pairs from the fixed
codebook. Encoded as a 21-d vector, per-predicate fire count over regions (9) plus a
per-region any-predicate indicator (12).

Compares, on the SAME bounded sample, seen vs held-out under Option A:
  CLIP (ref)   D_det (phase1 + DISCERN)   D_full (phase1 + DISCERN + VLM-semantic)

Run:
  HF_HOME=~/.cache/huggingface \
  /data/umar/miniconda3/envs/GenD/bin/python -u scripts/pilot2r_vlm.py --clip-gpu 2 --vlm-gpu 3
"""
import os, sys, json, random, argparse
import numpy as np
import cv2

REPO = "/data/umar/Repos/DFB_NeSyNS"
CACHE = f"{REPO}/results/pilot2r/features_vlm.npz"
sys.path.insert(0, f"{REPO}/scripts")
sys.path.insert(0, f"{REPO}/preprocessing")
from pilot2_attribution import dct_profile, predicate_attributes, FAMILY, sample_frames
from pilot1_vlm import REGION_BOXES, PREDICATES, VLM_PROMPT
from pilot2r_attribution import discern_vec, family_A, EDITS, FAMS, HELD_OUT
import forensic_helpers as FH

SEED = 0
N_TRAIN = 25
N_EVAL = 25


def parse_pairs(txt):
    """Return list of (region, predicate) from the VLM JSON, validated to the codebook."""
    import re, json as _j
    m = re.search(r"\[.*\]", txt, re.S)
    out = []
    if not m:
        return out
    try:
        arr = _j.loads(m.group(0))
    except Exception:
        return out
    for o in arr:
        if isinstance(o, dict):
            r = str(o.get("region", "")).strip()
            p = str(o.get("artifact", "")).strip()
            if r in REGION_BOXES:
                out.append((r, p if p in PREDICATES else None))
    return out


def semantic_vec(pairs):
    """21-d: per-predicate fire count over regions (9) + per-region any-predicate flag (12)."""
    reg_list = list(REGION_BOXES)
    pc = np.zeros(len(PREDICATES), np.float32)
    rf = np.zeros(len(reg_list), np.float32)
    for r, p in pairs:
        rf[reg_list.index(r)] = 1.0
        if p is not None:
            pc[PREDICATES.index(p)] += 1.0
    return np.concatenate([pc, rf])


class InternVLBatch:
    def __init__(self, gpu):
        import torch
        from transformers import AutoModel, AutoTokenizer
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        self.torch = torch
        self.dev = f"cuda:{gpu}"
        mid = "OpenGVLab/InternVL3-8B"
        self.tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(mid, dtype=torch.bfloat16, trust_remote_code=True,
                                               low_cpu_mem_usage=True).eval().to(self.dev)
        self.tf = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                             T.ToTensor(),
                             T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.cfg = dict(max_new_tokens=128, do_sample=False)

    def propose_batch(self, bgr_list):
        import torch
        pvs = [self.tf(_pil(b)).to(torch.bfloat16).to(self.dev) for b in bgr_list]
        pixel_values = torch.stack(pvs)
        num_patches = [1] * len(bgr_list)
        qs = [VLM_PROMPT] * len(bgr_list)
        with torch.no_grad():
            resp = self.model.batch_chat(self.tok, pixel_values,
                                         num_patches_list=num_patches, questions=qs,
                                         generation_config=self.cfg)
        return resp


def _pil(bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def build(clip_gpu, vlm_gpu):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    dev = f"cuda:{clip_gpu}"
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(dev)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    vlm = InternVLBatch(vlm_gpu)
    rng = random.Random(SEED)

    rows = []
    for gen in FAMILY:
        fam = family_A(gen)
        if gen not in EDITS and gen not in HELD_OUT:
            for f in sample_frames(gen, "train", N_TRAIN, rng):
                rows.append((gen, fam, "train", f))
        for f in sample_frames(gen, "test", N_EVAL, rng):
            role = "eval_abstain" if gen in EDITS else ("eval_heldout" if gen in HELD_OUT else "eval_seen")
            rows.append((gen, fam, role, f))
    print(f"[data] {len(rows)} frames", flush=True)

    clip_f, dct_f, pred_f, disc_f, sem_f, valid = [], [], [], [], [], []
    B = 16
    batch_imgs, batch_paths = [], []

    @torch.no_grad()
    def clip_embed(bgr_list):
        pil = [Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)) for x in bgr_list]
        inp = proc(images=pil, return_tensors="pt").to(dev)
        return clip.get_image_features(**inp).cpu().numpy()

    def flush_batch():
        resp = vlm.propose_batch(batch_imgs)
        cf = clip_embed(batch_imgs)
        for k, img in enumerate(batch_imgs):
            dct_f.append(dct_profile(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
            pred_f.append(predicate_attributes(img)[0])
            disc_f.append(discern_vec(img))
            sem_f.append(semantic_vec(parse_pairs(resp[k])))
            clip_f.append(cf[k])

    for gen, fam, role, f in rows:
        img = cv2.imread(f)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        batch_imgs.append(img); batch_paths.append((gen, fam, role))
        if len(batch_imgs) == B:
            flush_batch(); valid += batch_paths; batch_imgs, batch_paths = [], []
            if len(valid) % 320 == 0:
                print(f"  featurized {len(valid)}/{len(rows)}", flush=True)
    if batch_imgs:
        flush_batch(); valid += batch_paths

    data = dict(
        clip=np.stack(clip_f).astype(np.float32), dct=np.stack(dct_f).astype(np.float32),
        pred=np.stack(pred_f).astype(np.float32), disc=np.stack(disc_f).astype(np.float32),
        sem=np.stack(sem_f).astype(np.float32),
        gens=np.array([v[0] for v in valid]), fams=np.array([v[1] for v in valid]),
        roles=np.array([v[2] for v in valid]))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **data)
    print(f"[cache] wrote {CACHE}  sem-dim {data['sem'].shape[1]}", flush=True)
    return data


def evaluate(data):
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    y = np.array([FAMS.index(f) if f in FAMS else -1 for f in data["fams"]])
    roles = data["roles"]
    tr = roles == "train"; ev_s = roles == "eval_seen"; ev_h = roles == "eval_heldout"
    det = np.concatenate([data["pred"], data["disc"]], axis=1)
    full = np.concatenate([data["pred"], data["disc"], data["sem"]], axis=1)
    feats = {"CLIP": data["clip"], "D_det": det, "D_full": full, "VLM_only": data["sem"]}
    out = {}
    for name, X in feats.items():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        s, h = [], []
        for seed in [0, 1, 2]:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xs, Xh = sc.transform(X[tr]), sc.transform(X[ev_s]), sc.transform(X[ev_h])
            clf = (MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed)
                   if name in ("D_det", "D_full", "VLM_only")
                   else LogisticRegression(max_iter=2000, random_state=seed))
            clf.fit(Xtr, y[tr])
            s.append(float((clf.predict(Xs) == y[ev_s]).mean()))
            h.append(float((clf.predict(Xh) == y[ev_h]).mean()))
        out[name] = dict(seen=float(np.mean(s)), seen_sd=float(np.std(s)),
                         held=float(np.mean(h)), held_sd=float(np.std(h)),
                         gap=float(np.mean(s) - np.mean(h)))
    _, c = np.unique(y[ev_s], return_counts=True)
    return out, float(c.max() / c.sum()), dict(n_tr=int(tr.sum()), n_s=int(ev_s.sum()), n_h=int(ev_h.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-gpu", type=int, default=2)
    ap.add_argument("--vlm-gpu", type=int, default=3)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if os.path.exists(CACHE) and not args.rebuild:
        d = np.load(CACHE, allow_pickle=True); data = {k: d[k] for k in d.files}
        print(f"[cache] loaded {CACHE}")
    else:
        data = build(args.clip_gpu, args.vlm_gpu)

    res, chance, cnt = evaluate(data)
    outd = f"{REPO}/results/pilot2r"; os.makedirs(outd, exist_ok=True)
    json.dump(dict(results=res, chance=chance, counts=cnt), open(f"{outd}/vlm_ceiling.json", "w"), indent=2)
    lines = ["Pilot 2R VLM ceiling test -- Option A, fp16 InternVL semantic predicates",
             "=" * 74,
             "AWQ unavailable (lmdeploy 0.14 x transformers 4.56 incompat); fp16 fallback, "
             "smaller sample (throughput cost).",
             f"frames: train {cnt['n_tr']}, eval-seen {cnt['n_s']}, eval-heldout {cnt['n_h']}   chance {chance:.3f}",
             "",
             f"{'vector':<12}{'seen':>16}{'held-out':>16}{'gap':>8}",
             "-" * 52]
    for k in ["CLIP", "VLM_only", "D_det", "D_full"]:
        r = res[k]
        lines.append(f"{k:<12}{r['seen']:.3f}+/-{r['seen_sd']:.3f}  {r['held']:.3f}+/-{r['held_sd']:.3f}  {r['gap']:+.3f}")
    dd, df = res["D_det"], res["D_full"]
    lines += ["", "VLM contribution (D_full vs D_det):",
              f"  seen  {dd['seen']:.3f} -> {df['seen']:.3f}  ({df['seen']-dd['seen']:+.3f})",
              f"  held  {dd['held']:.3f} -> {df['held']:.3f}  ({df['held']-dd['held']:+.3f})",
              f"  gap   {dd['gap']:+.3f} -> {df['gap']:+.3f}"]
    open(f"{outd}/vlm_ceiling.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {outd}/vlm_ceiling.txt")


if __name__ == "__main__":
    main()
