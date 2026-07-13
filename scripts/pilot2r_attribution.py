"""
Pilot 2R, attribution gate re-run, Option A plus enriched predicates (PILOTS_2.md).

Tests the predicate ceiling. Phase-1 D (13 deterministic attributes, Option B) reached
0.598 seen with the smallest seen-to-held-out gap. Two named depressors are addressed
here.
- Option A. Edits (e4e, stargan, starganv2, styleclip, all test-only) are removed from the
  trained families and become an abstention set. EFS is clean synthesis only. Trained
  families, face_swap, face_reenactment, entire_face_synthesis.
- Enriched predicate vector. The 13 phase-1 attributes concatenated with the DISCERN
  parsing-free forensic bank, radial-FFT spectral, SRM, PPNC, CCNC, multiscale noise, which
  Study 1B showed separates real vs fake at AUC 0.94 to 0.99 and survives compression.

This first cut is DETERMINISTIC-only, no InternVL semantic predicates yet, so it is
decision-free and needs no LMDeploy. The VLM-semantic enrichment is the follow-on ceiling
test once the AWQ validation is done.

Baselines, three seeds, seen vs held-out under Option A.
  A CLIP linear probe   B raw DCT   C predicate count   D enriched-predicate MLP
  E retrieval kNN over the enriched predicate vector
Abstention behavior on the edit set is reported as confidence and neighbor-distance gaps.

Run:
  HF_HOME=~/.cache/huggingface \
  /data/umar/miniconda3/envs/GenD/bin/python scripts/pilot2r_attribution.py --gpu 2
"""
import os, sys, json, random, argparse, glob
import numpy as np
import cv2

REPO = "/data/umar/Repos/DFB_NeSyNS"
DF = "/data/umar/Datasets/df40"
CACHE = f"{REPO}/results/pilot2r/features.npz"
sys.path.insert(0, f"{REPO}/scripts")
sys.path.insert(0, f"{REPO}/preprocessing")
from pilot2_attribution import dct_profile, predicate_attributes, FAMILY, list_frames, sample_frames
import forensic_helpers as FH

# Option A: edits become the abstention set, EFS is clean synthesis only
EDITS = {"e4e", "stargan", "starganv2", "styleclip"}
FAMS = ["face_swap", "face_reenactment", "entire_face_synthesis"]
HELD_OUT = ["deepfacelab", "heygen", "CollabDiff", "MidJourney", "whichfaceisreal"]  # test-only, non-edit
SEED = 0
N_TRAIN = 80
N_EVAL = 50


def family_A(gen):
    if gen in EDITS:
        return "abstention"
    return FAMILY[gen]          # face_edit never occurs; Option B EFS members are clean here


def discern_vec(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    parts = [np.asarray(FH.extract_spectral(gray), np.float32).ravel(),
             np.asarray(FH.extract_srm(gray), np.float32).ravel(),
             np.asarray(FH.extract_ppnc(gray), np.float32).ravel(),
             np.asarray(FH.extract_ccnc(bgr), np.float32).ravel(),
             np.asarray(FH.extract_multiscale_noise(gray), np.float32).ravel()]
    v = np.concatenate(parts)
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def build_features(gpu):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image
    dev = f"cuda:{gpu}"
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(dev)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    rng = random.Random(SEED)

    rows = []
    for gen in FAMILY:
        fam = family_A(gen)
        if gen not in EDITS and gen not in HELD_OUT:
            for f in sample_frames(gen, "train", N_TRAIN, rng):
                rows.append((gen, fam, "train", f))
        for f in sample_frames(gen, "test", N_EVAL, rng):
            if gen in EDITS:
                role = "eval_abstain"
            elif gen in HELD_OUT:
                role = "eval_heldout"
            else:
                role = "eval_seen"
            rows.append((gen, fam, role, f))
    print(f"[data] {len(rows)} frames")

    clip_f, dct_f, pred_f, disc_f, valid = [], [], [], [], []
    bufp, bufkey = [], []

    @torch.no_grad()
    def flush(paths):
        pil = [Image.open(p).convert("RGB") for p in paths]
        inp = proc(images=pil, return_tensors="pt").to(dev)
        return clip.get_image_features(**inp).cpu().numpy()

    for gen, fam, role, f in rows:
        img = cv2.imread(f)
        if img is None:
            continue
        dct_f.append(dct_profile(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        pred_f.append(predicate_attributes(img)[0])
        disc_f.append(discern_vec(img))
        valid.append((gen, fam, role)); bufp.append(f)
        if len(bufp) == 64:
            clip_f.append(flush(bufp)); bufp = []
            if len(valid) % 640 == 0:
                print(f"  featurized {len(valid)}")
    if bufp:
        clip_f.append(flush(bufp))

    data = dict(
        clip=np.concatenate(clip_f).astype(np.float32),
        dct=np.stack(dct_f).astype(np.float32),
        pred=np.stack(pred_f).astype(np.float32),
        disc=np.stack(disc_f).astype(np.float32),
        gens=np.array([v[0] for v in valid]), fams=np.array([v[1] for v in valid]),
        roles=np.array([v[2] for v in valid]))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **data)
    print(f"[cache] wrote {CACHE}  (disc dim {data['disc'].shape[1]})")
    return data


def evaluate(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    y = np.array([FAMS.index(f) if f in FAMS else -1 for f in data["fams"]])
    roles = data["roles"]
    tr = roles == "train"; ev_s = roles == "eval_seen"; ev_h = roles == "eval_heldout"
    ev_a = roles == "eval_abstain"

    med = np.median(data["pred"][tr], axis=0)
    cnt = (data["pred"] > med).astype(np.float32)
    enriched = np.concatenate([data["pred"], data["disc"]], axis=1)   # D and E vector

    feats = {"A_clip": data["clip"], "B_dct": data["dct"], "C_count": cnt,
             "D_enriched": enriched, "E_knn": enriched}
    results = {}
    for name, X in feats.items():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        accs_s, accs_h, conf_s, conf_a = [], [], [], []
        for seed in [0, 1, 2]:
            sc = StandardScaler().fit(X[tr])
            Xtr, Xs, Xh, Xa = sc.transform(X[tr]), sc.transform(X[ev_s]), sc.transform(X[ev_h]), sc.transform(X[ev_a])
            if name == "E_knn":
                clf = KNeighborsClassifier(n_neighbors=15)
            elif name == "D_enriched":
                clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=seed)
            else:
                clf = LogisticRegression(max_iter=2000, random_state=seed)
            clf.fit(Xtr, y[tr])
            accs_s.append(float((clf.predict(Xs) == y[ev_s]).mean()))
            accs_h.append(float((clf.predict(Xh) == y[ev_h]).mean()))
            if hasattr(clf, "predict_proba"):
                conf_s.append(float(clf.predict_proba(Xs).max(1).mean()))
                conf_a.append(float(clf.predict_proba(Xa).max(1).mean()))
        results[name] = dict(
            seen_mean=float(np.mean(accs_s)), seen_std=float(np.std(accs_s)),
            held_mean=float(np.mean(accs_h)), held_std=float(np.std(accs_h)),
            gap=float(np.mean(accs_s) - np.mean(accs_h)),
            conf_seen=float(np.mean(conf_s)) if conf_s else float("nan"),
            conf_abstain=float(np.mean(conf_a)) if conf_a else float("nan"))
    vals, cnts = np.unique(y[ev_s], return_counts=True)
    chance = float(cnts.max() / cnts.sum())
    counts = dict(n_train=int(tr.sum()), n_eval_seen=int(ev_s.sum()),
                  n_eval_heldout=int(ev_h.sum()), n_eval_abstain=int(ev_a.sum()))
    return results, chance, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if os.path.exists(CACHE) and not args.rebuild:
        d = np.load(CACHE, allow_pickle=True); data = {k: d[k] for k in d.files}
        print(f"[cache] loaded {CACHE}")
    else:
        data = build_features(args.gpu)

    results, chance, counts = evaluate(data)
    outd = f"{REPO}/results/pilot2r"; os.makedirs(outd, exist_ok=True)
    # phase-1 Option-B reference (from results/pilot2)
    try:
        b1 = json.load(open(f"{REPO}/results/pilot2/attribution.json"))["results"]
    except Exception:
        b1 = {}
    S = dict(results=results, chance=chance, counts=counts, families=FAMS,
             phase1_optionB=b1)
    json.dump(S, open(f"{outd}/attribution2r.json", "w"), indent=2)

    lab = {"A_clip": "A CLIP", "B_dct": "B DCT", "C_count": "C pred-count",
           "D_enriched": "D enriched-MLP", "E_knn": "E retr-kNN"}
    lines = ["Pilot 2R -- DF40 attribution, Option A + enriched DISCERN predicates (no VLM yet)",
             "=" * 76,
             f"families: {FAMS}   (edits = abstention set, excluded from training)",
             f"frames: train {counts['n_train']}, eval-seen {counts['n_eval_seen']}, "
             f"eval-heldout {counts['n_eval_heldout']}, eval-abstain {counts['n_eval_abstain']}",
             f"chance: {chance:.3f}",
             "",
             f"{'baseline':<16}{'seen':>16}{'held-out':>16}{'gap':>8}{'conf seen/abstain':>22}",
             "-" * 78]
    for k in ["A_clip", "B_dct", "C_count", "D_enriched", "E_knn"]:
        r = results[k]
        conf = f"{r['conf_seen']:.2f}/{r['conf_abstain']:.2f}" if r['conf_seen'] == r['conf_seen'] else "-"
        lines.append(f"{lab[k]:<16}{r['seen_mean']:.3f}+/-{r['seen_std']:.3f}  "
                     f"{r['held_mean']:.3f}+/-{r['held_std']:.3f}  {r['gap']:+.3f}  {conf:>18}")
    # comparison to phase-1 D
    d1 = b1.get("D_pred", {})
    De = results["D_enriched"]
    lines += ["", "vs phase-1 Option-B D (0.598 seen / 0.575 held):",
              f"  enriched D: seen {De['seen_mean']:.3f} (was {d1.get('seen_mean', float('nan')):.3f}), "
              f"held {De['held_mean']:.3f} (was {d1.get('held_mean', float('nan')):.3f}), gap {De['gap']:+.3f}"]
    best_raw_seen = max(results["A_clip"]["seen_mean"], results["B_dct"]["seen_mean"])
    best_raw_held = max(results["A_clip"]["held_mean"], results["B_dct"]["held_mean"])
    lines += ["", "Gate reading (enriched D and E):",
              f"  D seen closes gap to best raw feature ({best_raw_seen:.3f}): "
              f"{'mostly' if De['seen_mean'] >= best_raw_seen - 0.05 else 'NO, still '+format(best_raw_seen-De['seen_mean'],'.3f')+' below'}",
              f"  D held-out beats best raw held ({best_raw_held:.3f}): "
              f"{'YES' if De['held_mean'] > best_raw_held else 'NO'}",
              f"  abstention signal (conf seen>abstain for D): "
              f"{results['D_enriched']['conf_seen']:.2f} vs {results['D_enriched']['conf_abstain']:.2f}"]
    open(f"{outd}/attribution2r.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[written] {outd}/attribution2r.txt")


if __name__ == "__main__":
    main()
