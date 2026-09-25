"""
G0 check for the I-JEPA branch.

Setup on the GPU server:
    git clone https://github.com/facebookresearch/ijepa.git
    wget https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar

Run:
    python check_ijepa_predictor.py                      # parts 1 and 2 only
    python check_ijepa_predictor.py /path/to/real_faces  # adds part 3 (224x224 FF++ real face crops)
"""
import glob
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

REPO = Path("ijepa")
CKPT = Path("IN1K-vit.h.14-300e.pth.tar")
sys.path.insert(0, str(REPO))
from src.helper import init_model  # noqa: E402
from src.masks.utils import apply_masks  # noqa: E402


def strip(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}


# Part 1. Are the predictor weights in the file?
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("Top-level keys:", sorted(ck.keys()))
for k in ("encoder", "predictor", "target_encoder"):
    if k not in ck:
        sys.exit(f"FAIL: '{k}' is missing from the checkpoint")
    n = sum(v.numel() for v in ck[k].values())
    print(f"{k:15s} {len(ck[k]):4d} tensors {n / 1e6:8.1f} M parameters")

# Part 2. Do they load exactly into the published architecture?
dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = dict(patch_size=14, model_name="vit_huge", crop_size=224, pred_depth=12, pred_emb_dim=384)
enc, pred = init_model(device=dev, **cfg)
tgt, pred_rand = init_model(device=dev, **cfg)  # pred_rand keeps random weights as a control
for name, model, sd in (
    ("encoder", enc, ck["encoder"]),
    ("predictor", pred, ck["predictor"]),
    ("target_encoder", tgt, ck["target_encoder"]),
):
    model.load_state_dict(strip(sd), strict=True)
    print(f"{name}: loaded with strict=True")
for m in (enc, pred, tgt, pred_rand):
    m.eval()

if len(sys.argv) < 2:
    sys.exit("Parts 1 and 2 passed. Pass a folder of real face crops to run part 3.")

# Part 3. Does the predictor actually predict hidden face regions?
paths = sorted(glob.glob(f"{sys.argv[1]}/*.png") + glob.glob(f"{sys.argv[1]}/*.jpg"))[:32]
if len(paths) < 8:
    sys.exit("Need at least 8 face crops for part 3")
tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths]).to(dev)
B, G = len(imgs), 16  # 224 / 14 = 16x16 patch grid


def block(r0, r1, c0, c1):
    hidden = [r * G + c for r in range(r0, r1) for c in range(c0, c1)]
    visible = [i for i in range(G * G) if i not in set(hidden)]
    return (torch.tensor(visible, device=dev).repeat(B, 1),
            torch.tensor(hidden, device=dev).repeat(B, 1))


@torch.no_grad()
def errors(ctx, hid):
    h = tgt(imgs)
    h = F.layer_norm(h, (h.size(-1),))  # targets are layer-normalized in I-JEPA training
    h_hid = apply_masks(h, [hid])
    z = enc(imgs, [ctx])  # the hidden block never enters the context encoder
    mean_visible = apply_masks(h, [ctx]).mean(1, keepdim=True).expand_as(h_hid)
    return {
        "trained predictor": F.smooth_l1_loss(pred(z, [ctx], [hid]), h_hid).item(),
        "trained, another face's context": F.smooth_l1_loss(pred(z.roll(1, 0), [ctx], [hid]), h_hid).item(),
        "random predictor": F.smooth_l1_loss(pred_rand(z, [ctx], [hid]), h_hid).item(),
        "mean of visible patches": F.smooth_l1_loss(mean_visible, h_hid).item(),
    }


# Blocks of 56 of 256 patches (22%), close to the 15-20% blocks used in pretraining
passed = True
for name, box in {"upper face": (2, 9, 4, 12), "lower face": (9, 16, 4, 12)}.items():
    res = errors(*block(*box))
    print(f"\n{name}")
    for k, v in res.items():
        print(f"  {k:34s} {v:.4f}")
    trained = res["trained predictor"]
    ok = all(trained < v for k, v in res.items() if k != "trained predictor")
    passed &= ok
    print(f"  {'PASS' if ok else 'FAIL'}: trained predictor beats all three controls")

print("\nG0", "PASSED" if passed else "FAILED")