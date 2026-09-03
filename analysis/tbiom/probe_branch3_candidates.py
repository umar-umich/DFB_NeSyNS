"""The HONEST protocol: fit on FF++ c23 ONLY, test zero-shot on each OOD set.

The within-dataset CV numbers (FF++ 0.954, CDFv2 0.961, CDFv3 0.921, DFDCP 0.938) are NOT
comparable to the framework's OOD numbers, because each fold's training data came from the same
dataset as its test data. Our framework never sees a Celeb-DF or DFDC frame in training. This
reproduces that constraint: one model, fit on FF++ frames, applied unchanged everywhere else.

Video-level AUROC as well as frame-level, since the framework is judged at video level.
"""
import numpy as np, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import pandas as pd

ROOT = Path("/data/umar/Datasets/preprocessed")
REAL = ("original_sequences", "Celeb-real", "YouTube-real", "original_videos", "/real/")

def load(dirs, cap=200):
    X, y, v = [], [], []
    for d in dirs:
        p = ROOT / d / "forensic_features"
        if not p.is_dir(): continue
        lab = 0 if any(h.strip("/") in d for h in REAL) else 1
        for i, f in enumerate(sorted(p.glob("*.pt"))):
            if i >= cap: break
            try: b = torch.load(f, map_location="cpu", weights_only=False)
            except Exception: continue
            a = np.asarray(b["features"], dtype=np.float32)
            if a.ndim != 2 or a.shape[1] != 83: continue
            X.append(a); y.append(np.full(len(a), lab)); v.append([f"{d}/{f.stem}"]*len(a))
    if not X: return None, None, None
    return (np.nan_to_num(np.concatenate(X), nan=0., posinf=0., neginf=0.),
            np.concatenate(y), np.concatenate(v))

FF = ["FaceForensics++/original_sequences/youtube/c23",
      "FaceForensics++/manipulated_sequences/Deepfakes/c23",
      "FaceForensics++/manipulated_sequences/Face2Face/c23",
      "FaceForensics++/manipulated_sequences/FaceSwap/c23",
      "FaceForensics++/manipulated_sequences/NeuralTextures/c23"]
OOD = {
 "Celeb-DF-v2": ["Celeb-DF-v2/Celeb-real","Celeb-DF-v2/YouTube-real","Celeb-DF-v2/Celeb-synthesis"],
 "Celeb-DF-v3 FaceSwap": ["Celeb-DF-v3/Celeb-real","Celeb-DF-v3/YouTube-real",
      "Celeb-DF-v3/Celeb-synthesis/FaceSwap/BlendFace","Celeb-DF-v3/Celeb-synthesis/FaceSwap/InSwapper",
      "Celeb-DF-v3/Celeb-synthesis/FaceSwap/SimSwap"],
 "DFDCP": ["DFDCP/original_videos","DFDCP/method_A","DFDCP/method_B"],
 "DFD": ["FaceForensics++/original_sequences/actors/c23",
         "FaceForensics++/manipulated_sequences/DeepFakeDetection/c23"],
 "UADFV": ["UADFV/real","UADFV/fake"],
}

Xf, yf, _ = load(FF)
print(f"fit on FF++ c23 only: {len(yf)} frames ({int((yf==0).sum())}R/{int((yf==1).sum())}F)\n")
models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
          "GBM": GradientBoostingClassifier(random_state=0, n_estimators=100)}
for m in models.values(): m.fit(Xf, yf)

print(f"{'test set':22s} {'n frames':>9s} {'logistic frame':>15s} {'GBM frame':>10s} "
      f"{'logistic video':>15s} {'GBM video':>10s}")
for name, dirs in OOD.items():
    X, y, v = load(dirs)
    if X is None or len(np.unique(y)) < 2: print(f"{name:22s} insufficient"); continue
    row = [f"{name:22s}", f"{len(y):>9d}"]
    vids = {}
    for mn, m in models.items():
        p = m.predict_proba(X)[:,1]
        row.append(f"{roc_auc_score(y,p):>15.4f}" if mn=="logistic" else f"{roc_auc_score(y,p):>10.4f}")
        g = pd.DataFrame({"v":v,"p":p,"y":y}).groupby("v").agg(p=("p","mean"),y=("y","max"))
        vids[mn] = roc_auc_score(g.y.values, g.p.values)
    row.append(f"{vids['logistic']:>15.4f}"); row.append(f"{vids['GBM']:>10.4f}")
    print(" ".join(row))
print("\nSDXL 6-stat reference (FF++ val, within-domain probe): 0.7047")
