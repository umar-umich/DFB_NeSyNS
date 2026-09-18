#!/usr/bin/env python3
"""Experiment 5, part 1: the FreqBlender generator (FPNet), trained locally against the upstream implementation.

    <timm python> analysis/tbiom/freqblender/fpnet.py facebank
    CUDA_VISIBLE_DEVICES=1 <timm python> analysis/tbiom/freqblender/fpnet.py train --epochs 20
    CUDA_VISIBLE_DEVICES=1 <timm python> analysis/tbiom/freqblender/fpnet.py generate --weights <tar>

WHAT THIS IS, PRECISELY (finish-line brief, Experiment 5: "distinguish any local reproduction from the official
pretrained configuration and record which is used").
  * The upstream checkout `/data/umar/Repos/FreqBlender` at the pinned revision 7c8bc7f is CALLED IN PLACE: its network
    (`net/aeNet/conv_autoencoder_pixelshuffle.conv_autoencoder`), its four losses (`utils/loss_function.loss_function`),
    its DCT convention and its config (`config/base.json`) are imported,
    not reimplemented. Nothing in that checkout is edited.
  * The FACIAL-FIDELITY teacher is the official MobileFaceNet + MTCNN shipped in the repository, with a face bank we
    build from our own training-split frames.
  * The AUTHENTICITY teacher is NOT the official one either: the released `resnet34.tar` is behind the same Baidu Pan
    wall, so `analysis/tbiom/train_authenticity_teacher.py` trained a split-safe replacement (ResNet-34, ImageNet init,
    FF++ TRAIN reals + four manipulations, validation AUROC 0.972, index 1 = fake), which is what the README sanctions
    ("or, you can choose to simply train a weight yourself"). It is loaded here with CLIP normalisation restored: it was
    trained on CLIP-normalised 224-px crops, while upstream's loss hands its teacher [0,1] images, so the teacher is
    wrapped in that normalisation rather than being fed a distribution it never saw.
  * The official released FPNet weights are NOT used: they are behind a Baidu Pan share that cannot be fetched
    non-interactively (audited in tbiom/docs/freqblender-source-audit.md). FPNet is therefore TRAINED HERE. Every
    result must be labelled a local reproduction of FreqBlender, never "the released FreqBlender".
DEVIATIONS FROM UPSTREAM, ALL DELIBERATE AND ALL RECORDED IN THE REPORT
  1. GEOMETRY (v2, 2026-09-18). Inputs are our stored 256-px DISCERN face crops -- the exact frames the detector
     trains on -- resized to upstream's 400 px, with NO re-cropping. v1 used upstream's own `face_cut` (the
     81-landmark box plus w/8 margins) and was withdrawn: that box covers only ~32% of the stored crop's area, so
     every pseudo-fake was a visibly tighter face than the ordinary samples it was mixed with, and since arm D
     substitutes them into FAKE slots only, crop scale became a label-correlated nuisance the detector could learn
     instead of the frequency artefact (`logs/tbiom/freqblender/v1_facecut_confounded/README.md`). Applying FPNet to
     the whole crop keeps a pseudo-fake pixel-aligned with the sample it replaces, with no seam and no scale cue, and
     FPNet is TRAINED on that same geometry so generator and application never disagree. The cost is that the face
     occupies a smaller share of the 400-px grid than in upstream's dlib crops, which shifts where facial structure
     sits in the DCT bands; the PI-loss boundaries (50, 400) are upstream's and are left unchanged.
  2. SPLIT SAFETY. Upstream trains on FF++ videos 000-999 indiscriminately. We use ONLY the 720 FF++ source ids of our
     training split (both members of each Deepfakes pair are in the same official split, asserted below), so no
     validation or test source video is seen by the generator, its face bank or its teachers.
  3. The face bank is keyed by the real video id. Upstream names its bank files by enumeration index, which coincides
     with the id only when all 1000 videos are present; with 720 it would silently mismatch identities.
  4. Epoch budget: upstream runs 100 epochs with StepLR(step 5, gamma 0.5), i.e. the learning rate is already 1/64 of
     its initial value by epoch 30. `--epochs` is fixed BEFORE launch and reported with the loss curve.
GENERATION. Upstream's pseudo-fake, read off its own authenticity loss (`synthetic_fake_M1M2`):
    x_pseudo = IDCT( DCT(real) * M1 + DCT(fake) * M2 ),  (M1, M2, M3) = FPNet(DCT(fake))
i.e. the real face keeps its facial-attribute band and borrows the forgery-trace band of a real manipulation. The
quality/noise band M3 is dropped. Generated images are written at 256 px, the size the detector's loader reads.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

FB = Path("/data/umar/Repos/FreqBlender")
DFB = Path("/data/umar/Repos/DFB_NeSyNS")
EXT = Path("/data/umar/Repos/DISCERN_Ext")
PRE = Path("/data/umar/Datasets/preprocessed/FaceForensics++")
REAL = PRE / "original_sequences/youtube/c23"
FAKE = PRE / "manipulated_sequences/Deepfakes/c23"
OUT = DFB / "logs/tbiom/freqblender"
GEN = Path("/data/umar/Datasets/preprocessed/FaceForensics++/freqblender_pseudofake")
N_FRAMES = 8                       # upstream num_frames
IMAGE_SIZE = 400                   # upstream image_size


def train_ids() -> list[str]:
    """The 720 FF++ source ids of OUR training split, read from the detector's own split lists."""
    f = EXT / "eval_adaptation/data/splits/FFpp_train/FFpp-real.txt"
    ids = sorted({line.strip().split("/")[-2] for line in open(f) if line.strip()})
    assert len(ids) == 720, len(ids)
    return ids


def pairs_for(ids: list[str]) -> list[tuple[str, str]]:
    """(real id, Deepfakes dir) for every training video, asserting the donor id is in the same split."""
    have = set(ids); out = []
    for v in ids:
        d = sorted(FAKE.glob(f"frames/{v}_*"))
        if not d:
            continue
        name = d[0].name
        assert name.split("_")[1] in have, f"{name}: donor id outside the training split"
        out.append((v, name))
    return out


def upstream():
    """Import the upstream modules in place (cwd must be the checkout: it reads config/base.json on import).

    `net/mobileface/infer.py` calls `parse_args()` at import time, so our own flags have to leave sys.argv first.
    """
    os.chdir(FB); sys.path.insert(0, str(FB)); del sys.argv[1:]
    import torch  # noqa: E402
    from utils.funcs import load_json  # noqa: E402
    from utils.energy_contrast import face_cut  # noqa: E402
    return torch, load_json("config/base.json"), face_cut


def authenticity_teacher(torch, dev):
    """Our split-safe FF++ teacher, wrapped so it sees the normalisation it was trained with.

    `utils/loss_function.py` calls `resnet(resize224(img)/255)`, i.e. [0,1] images. The checkpoint from
    `analysis/tbiom/train_authenticity_teacher.py` is a torchvision ResNet-34 with a 2-class head (index 1 = fake)
    trained on CLIP-normalised crops, so the wrapper applies CLIP's mean/std before the network. Frozen, eval mode.
    """
    import torch.nn as nn
    from torchvision.models import resnet34
    ck = torch.load(FB / "net/resnet/checkpoints/resnet34.tar", map_location="cpu")
    assert ck["class_order"] == "index 1 = fake", ck["class_order"]
    net = resnet34(); net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(ck["model"])

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("m", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
            self.register_buffer("s", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        def forward(self, x):                      # x in [0, 1]
            return self.net((x - self.m) / self.s)

    t = Teacher().to(dev).eval()
    for p in t.parameters():
        p.requires_grad = False
    print(f"authenticity teacher: local split-safe ResNet-34, val AUROC {ck['val_auroc']:.4f}", flush=True)
    return t


def pair_400(cv2, real_root, fake_root, v, fdir, frame, size=IMAGE_SIZE):
    """The real and donor-fake stored crops at upstream's 400 px, whole-crop (v2 geometry), with scaled landmarks.

    No re-crop and no jitter: the blend must land back on exactly the pixels of the stored crop it replaces. The
    81-point landmarks are in 256-px crop coordinates, so they are scaled by size/256 for the fidelity teacher.
    """
    lm = np.load(real_root / "landmarks81" / v / f"{frame}.npy").astype(np.float32) * (size / 256.0)
    r = cv2.cvtColor(cv2.imread(str(real_root / "frames" / v / f"{frame}.png")), cv2.COLOR_BGR2RGB)
    f = cv2.cvtColor(cv2.imread(str(fake_root / "frames" / fdir / f"{frame}.png")), cv2.COLOR_BGR2RGB)
    rs = lambda x: cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA if x.shape[0] > size else cv2.INTER_LINEAR)
    return rs(r), rs(f), lm


# ----------------------------------------------------------------------------- face bank
def build_facebank():
    import cv2
    torch, cfg, _ = upstream()
    bank = FB / "net/mobileface/face_bank"; bank.mkdir(exist_ok=True)
    ids = train_ids(); n = 0
    for v in ids:
        frames = sorted((REAL / "frames" / v).glob("*.png"))
        done = False
        for fr in frames[:4]:                       # first frame with a landmark file, as upstream uses frame 000
            if not (REAL / "landmarks81" / v / (fr.stem + ".npy")).exists():
                continue
            img = cv2.resize(cv2.imread(str(fr)), (IMAGE_SIZE, IMAGE_SIZE))   # v2: the whole stored crop, no re-crop
            cv2.imwrite(str(bank / f"{v}.png"), img); n += 1; done = True
            break
        if not done:
            print("no landmark for", v)
    print(f"face bank: {n} identities in {bank}")


# ----------------------------------------------------------------------------- dataset
class Pairs:
    """(DCT(real), DCT(fake), (real id, donor id, landmarks)) at 400 px, exactly upstream's item format."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2, torch
        from torch_dct import dct_2d
        v, fdir, frame = self.items[i]
        ri, fi, lm = pair_400(cv2, REAL, FAKE, v, fdir, frame)
        dct = lambda x: dct_2d(torch.tensor(np.ascontiguousarray(x) / 255).permute(2, 0, 1).float(), norm="ortho")
        return dct(ri), dct(fi), (int(v), int(fdir.split("_")[1]), torch.tensor(lm))


def items_for(pairs):
    items = []
    for v, fdir in pairs:
        frames = sorted(p.stem for p in (REAL / "frames" / v).glob("*.png"))
        kept = 0
        for fr in frames:
            if kept >= N_FRAMES:
                break
            if (REAL / "landmarks81" / v / f"{fr}.npy").exists() and (FAKE / "frames" / fdir / f"{fr}.png").exists():
                items.append((v, fdir, fr)); kept += 1
    return items


# ----------------------------------------------------------------------------- training (upstream loop)
def train(epochs, batch, workers, limit):
    torch, cfg, _ = upstream()
    from net.aeNet.conv_autoencoder_pixelshuffle import conv_autoencoder
    from utils.loss_function import loss_function
    torch.set_num_threads(4)
    seed = 5
    import random
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    items = items_for(pairs_for(train_ids()))
    if limit:
        items = items[:limit]
    ds = Pairs(items)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True,
                                         drop_last=True, worker_init_fn=lambda _: torch.set_num_threads(1))
    dev = torch.device("cuda")
    model = conv_autoencoder().to(dev)
    resnet = authenticity_teacher(torch, dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    run = OUT / "fpnet"; (run / "weights").mkdir(parents=True, exist_ok=True)
    hist, best = [], None
    print(f"FPNet: {len(items)} pairs from {len(set(i[0] for i in items))} training videos, "
          f"{epochs} epochs, batch {batch}", flush=True)
    for ep in range(epochs):
        t0, tot = time.time(), []
        for real, fake, extra in loader:
            real, fake = real.to(dev), fake.to(dev)
            M1, M2, M3 = model(fake)
            FF, AD, QA, PI = loss_function(M1, M2, M3, real, fake, resnet, extra, cfg)
            loss = FF + AD + QA + PI
            opt.zero_grad(); loss.backward(); opt.step()
            tot.append([float(x) for x in (FF, AD, QA, PI, loss)])
        sched.step()
        m = np.mean(tot, 0)
        hist.append({"epoch": ep, "FF": m[0], "AD": m[1], "QA": m[2], "PI": m[3], "loss": m[4],
                     "lr": opt.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60})
        print(f"epoch {ep}: loss {m[4]:.4f} (FF {m[0]:.4f} AD {m[1]:.4f} QA {m[2]:.4f} PI {m[3]:.4f}) "
              f"lr {opt.param_groups[0]['lr']:.2e} {(time.time() - t0) / 60:.1f} min", flush=True)
        if best is None or m[4] < best:
            best = m[4]
            torch.save({"model": model.state_dict(), "epoch": ep, "loss": float(m[4])}, run / "weights/best.tar")
        torch.save({"model": model.state_dict(), "epoch": ep, "loss": float(m[4])}, run / "weights/last.tar")
        json.dump({"epochs": epochs, "batch": batch, "n_pairs": len(items), "history": hist, "best_loss": float(best),
                   "config": cfg, "revision": "7c8bc7f"}, open(run / "train_record.json", "w"), indent=1)
    # The completion marker is what the Experiment 5 chain waits on, so a smoke run must never write it: a
    # 1-epoch generator once got as far as training a detector arm before the mistake was caught.
    if limit or epochs < 25:
        print(f"smoke run (epochs={epochs}, limit={limit}): DONE not written", flush=True)
    else:
        (run / "DONE").write_text(f"loss {best:.6f} epochs {epochs} pairs {len(items)}\n")


# ----------------------------------------------------------------------------- generation
def generate(weights, batch, workers):
    """Pseudo-fakes for every eligible real training frame; written at 256 px beside the SBI pool convention."""
    import cv2
    torch, cfg, _ = upstream()
    from net.aeNet.conv_autoencoder_pixelshuffle import conv_autoencoder
    from torch_dct import dct_2d, idct_2d
    torch.set_num_threads(4)
    dev = torch.device("cuda")
    model = conv_autoencoder().to(dev)
    ck = torch.load(weights); model.load_state_dict(ck["model"]); model.eval()
    print(f"FPNet weights {weights} (epoch {ck.get('epoch')}, loss {ck.get('loss')})", flush=True)

    pool = [l.strip() for l in open(DFB / "logs/tbiom/sbi/eligible_pool.txt") if l.strip()]
    donor = {v: d for v, d in pairs_for(train_ids())}
    todo = []
    for key in pool:
        v, frame = key.split("/")[-2], key.split("/")[-1]
        if v not in donor or not (FAKE / "frames" / donor[v] / frame).exists():
            continue
        if not (REAL / "landmarks81" / v / frame.replace(".png", ".npy")).exists():
            continue
        todo.append((key, v, donor[v], frame))
    print(f"{len(todo)} of {len(pool)} pool frames have a donor manipulation and landmarks", flush=True)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, i):
            _key, v, fdir, frame = todo[i]
            ri, fi, _lm = pair_400(cv2, REAL, FAKE, v, fdir, frame.replace(".png", ""))
            d = lambda x: dct_2d(torch.tensor(np.ascontiguousarray(x) / 255).permute(2, 0, 1).float(), norm="ortho")
            return d(ri), d(fi), i

    loader = torch.utils.data.DataLoader(DS(), batch_size=batch, num_workers=workers,
                                         worker_init_fn=lambda _: torch.set_num_threads(1))
    GEN.mkdir(parents=True, exist_ok=True); n = 0
    with torch.no_grad():
        for real, fake, idx in loader:
            real, fake = real.to(dev), fake.to(dev)
            M1, M2, M3 = model(fake)
            blend = torch.clip(idct_2d(real * M1 + fake * M2, norm="ortho") * 255, 0, 255)
            for k, i in enumerate(idx.tolist()):
                key, v, fdir, frame = todo[i]
                img = blend[k].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                p = GEN / v; p.mkdir(exist_ok=True)
                cv2.imwrite(str(p / frame), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)); n += 1
            if n % 2560 < batch:
                print(f"  {n}/{len(todo)}", flush=True)
    index = [f"{v}/{frame}" for _, v, _, frame in todo]
    (OUT / "pseudofake_index.txt").write_text("\n".join(index) + "\n")
    json.dump({"weights": str(weights), "epoch": ck.get("epoch"), "loss": ck.get("loss"), "n": n,
               "dir": str(GEN)}, open(OUT / "generate_record.json", "w"), indent=1)
    print(f"wrote {n} pseudo-fakes to {GEN}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["facebank", "train", "generate"])
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--weights", default=str(OUT / "fpnet/weights/best.tar"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "facebank":
        build_facebank()
    elif a.stage == "train":
        train(a.epochs, a.batch, a.workers, a.limit)
    else:
        generate(a.weights, a.batch, a.workers)


if __name__ == "__main__":
    main()
