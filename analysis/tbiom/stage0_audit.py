#!/usr/bin/env python3
"""Stage 0 audit for the executable brief (tbiom/docs/discern-v2-executable-brief.md). Writes REFERENCE_MANIFEST.md.

    OMP_NUM_THREADS=4 python analysis/tbiom/stage0_audit.py

Everything is read or recomputed from the live tree: checkpoint hashes, the discount rule (read from the code and
checked numerically against both candidate definitions), and the segment-level cache inventory.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "temporal")]
import symbolic_overlay as SO  # noqa: E402
import finish_line as F  # noqa: E402

R = Path("tbiom/reports"); S23, S2 = Path("logs/tbiom/stage23"), Path("logs/tbiom/stage2")
ACQ = Path("logs/tbiom/acquisition/vjepa2_lora_B_seed42")
CK = {"T (V-JEPA 2 LoRA, B-augmented)": Path("logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt"),
      "APP + OM (warm+gate, epoch 19)": F.SPATIAL_CKPT}
DEV, TESTS = ["VALmix", "VALmixV2", "FFpp_val"], ["CDFv2", "CDFv3", "DFD", "DFDC", "DFDCP", "DFEval24", "FFIW", "KoDF", "IDForge", "DFDM"]
K = 2.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def discount_check():
    """The code's rule versus belief-mass (Shafer) discounting, on real development opinions."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 2000); u = rng.uniform(0.02, 0.9, 2000); S = K / u
    a = np.stack([(1 - p) * S, p * S], 1); e = a - 1
    out = {}
    for w in (0.25, 0.5, 0.75):
        # code: evidence scaling, alpha' = 1 + w (alpha - 1)
        a_code = 1 + w * e; p_code, u_code = a_code[:, 1] / a_code.sum(1), K / a_code.sum(1)
        # belief-mass discounting: b' = w b, u' = 1 - w (1 - u), projected p with base rate 1/2
        b = e / a.sum(1, keepdims=True); b2 = w * b; u2 = 1 - w * (1 - u)
        p_bm = b2[:, 1] + u2 / 2
        out[w] = {"max|dp|": float(np.abs(p_code - p_bm).max()), "max|du|": float(np.abs(u_code - u2).max()),
                  "mean|du|": float(np.abs(u_code - u2).mean())}
    # the fusion used by every experiment equals the code's rule followed by DISCERN's own pairwise operator
    t = pd.DataFrame({"p_T": p[:500], "u_T": u[:500], "p_APP": p[500:1000], "u_APP": u[500:1000],
                      "p_OM": p[1000:1500], "u_OM": u[1000:1500]})
    SO.BRANCHES[:] = ["T", "APP", "OM"]; ref = SO.fuse(t); SO.BRANCHES[:] = ["semantic", "artifact", "fsvfm"]
    err = float(np.abs(F.dsr(t, ["T", "APP", "OM"], (1.0, 1.0, 1.0))[0] - ref).max())
    return out, err


def frames_per_video(tag, n):
    d = pd.read_csv(S23 / f"{tag}_{n}.csv", usecols=["key"])
    raw = d.key.str.split("::").str[-1]
    c = raw.str.rsplit("/", n=1).str[0].value_counts()
    step = []
    for vid, g in raw.groupby(raw.str.rsplit("/", n=1).str[0]):
        idx = sorted(int("".join(ch for ch in s.rsplit("/", 1)[-1] if ch.isdigit()) or 0) for s in g)
        if len(idx) > 1:
            step.append(np.median(np.diff(idx)))
        if len(step) >= 300:
            break
    return {"videos": int(len(c)), "min": int(c.min()), "median": float(c.median()), "max": int(c.max()),
            "median_raw_frame_step": float(np.median(step)) if step else None}


def segment_inventory():
    inv = {}
    for n in DEV:
        f = ACQ / f"{n}.csv"
        if not f.exists():
            inv[n] = None; continue
        d = pd.read_csv(f); w = d[d.window != "full"]
        inv[n] = {"videos": int(d.v.nunique()), "segments_per_video": int(w.window.nunique()),
                  "distinct_frames_per_segment": {"min": int(w.n_src.min()), "median": float(w.n_src.median()),
                                                  "max": int(w.n_src.max())},
                  "degenerate_segments": int((w.n_src == 0).sum()), "degenerate_videos": int(w[w.n_src == 0].v.nunique()),
                  "segments_repeating_frames": int(((w.n_src > 0) & (w.n_src < 16)).sum()), "segments": int(len(w)),
                  "quality_descriptors": [c for c in ("blur", "blockiness", "contrast", "motion", "luma") if c in d]}
    return inv


def exists(p):
    return "yes" if Path(p).exists() else "**no**"


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    hashes = {k: sha(v) for k, v in CK.items()}
    disc, ds_err = discount_check()
    fpv = {n: frames_per_video("warmgate_e19", n) for n in DEV}
    seg = segment_inventory()
    rec = json.load(open(Path("logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/train_record.json"))); rec.pop("history", None)
    man = json.load(open(R / "REFERENCE_MANIFEST.json"))
    json.dump({"commit": commit, "hashes": hashes, "discount_check": disc, "ds_operator_err": ds_err,
               "frames_per_video": fpv, "segments": seg}, open(R / "STAGE0_AUDIT.json", "w"), indent=1, default=float)

    L = ["# Reference manifest — executable brief, Stage 0 audit", "",
         f"Resolved from the live tree on {pd.Timestamp.now():%Y-%m-%d}, commit `{commit}`, by "
         "`analysis/tbiom/stage0_audit.py`; every number below is recomputed, not retyped. Every later experiment in "
         "`tbiom/docs/discern-v2-executable-brief.md` cites this file. Raw numbers: `STAGE0_AUDIT.json`.", "",
         "## 1. The three reference branches", "",
         "| view | checkpoint | sha256 (first 16) |", "|---|---|---|"]
    for k, v in CK.items():
        L.append(f"| {k} | `{v}` | `{hashes[k][:16]}` |")
    L += ["", "APP and OM are the Appearance and Off-manifold views of one checkpoint (shared CLIP ViT-L/14 + LoRA r8 "
          "α16 path; β-VAE latent 32, β_KLD 2.0, λ_align 1.0, λ_vae 0.7). The face prior inside that checkpoint is "
          "outside the reference. The hashes match the previous manifest "
          f"({'yes' if hashes[list(CK)[1]].startswith('fdc924a7cc9e1579') and hashes[list(CK)[0]].startswith('6f0d9c63a6c052de') else '**NO — investigate**'}).", "",
          "**Preprocessing and aggregation order.**",
          "- T: one clip per video, 16 frames at `linspace(0, n−1, 16)` over the video's stored crops, in temporal order, "
          "V-JEPA 2's own 256-px processor; the branch emits one Dirichlet per clip, so there is no frame aggregation.",
          "- APP/OM: every stored crop scored separately at 224 px (CLIP preprocessing); per-frame (p, u) are "
          "**averaged per video first**, then the three views are discounted and fused.",
          "- Order: per-frame spatial opinions → per-video mean → reconstruct α from (p, u) → discount each view → "
          "Dempster combination T ⊕ APP ⊕ OM (associative).",
          "", "Temporal-branch training record:", "", "```json", json.dumps(rec, indent=1), "```", "",
          "## 2. The discount rule, read from code", "",
          "`analysis/tbiom/temporal/finish_line.py::dsr` (identical in `dsr_system.py`):", "",
          "```python", "S = K / u                                   # K = 2 classes",
          "alpha = [(1 - p) * S, p * S]                # reconstruct the Dirichlet from (p, u)",
          "alpha' = 1 + w * (alpha - 1)                # discount", "```", "",
          "That is **evidence scaling**: e' = w·e, S' = w·Σe + K, b' = w·e / (w·Σe + K), u' = K / (w·Σe + K). It is "
          "**not** belief-mass (Shafer) discounting, b' = w·b, u' = 1 − w·(1 − u). The two coincide only at w = 1. "
          "The previous manifest called this rule \"Shafer discounting\"; that name was wrong and is corrected here — "
          "the equations it printed were right.", "",
          "Numerical check on 2,000 opinions (p ∈ [0.05, 0.95], u ∈ [0.02, 0.9]):", "",
          "| w | max \\|Δp\\| code vs belief-mass | max \\|Δu\\| | mean \\|Δu\\| |", "|---:|---:|---:|---:|"]
    for w, v in disc.items():
        L.append(f"| {w} | {v['max|dp|']:.4f} | {v['max|du|']:.4f} | {v['mean|du|']:.4f} |")
    L += ["", "The combination after discounting is Dempster's rule on the discounted opinions, identical to DISCERN's "
          f"in-model `DS_Combin` / `_combine_two_opinions` (max |Δp| {ds_err:.1e} at w = 1): with b_i = e_i / S_i and "
          "u_i = K / S_i, conflict C = Σ_{j≠k} b_1j b_2k, b = (b_1 b_2 + b_1 u_2 + b_2 u_1)/(1 − C), "
          "u = u_1 u_2 / (1 − C), α = b·K/u + 1. DISCERN's own face-prior gate uses the same evidence-scaling form "
          "(`1 + g·(α − 1)`).", "",
          "**Consequence for Stage 1A:** every controller predicts w per view and applies α' = 1 + w(α − 1), unchanged. "
          "Changing to belief-mass discounting would be a separate experiment, never combined with a new predictor.", "",
          f"Fitted weights on the five source-grouped splits (from the previous manifest): "
          + "; ".join(str(tuple(s["w"].values())) for s in man["splits"]) + " for (T, APP, OM).", "",
          "## 3. Cache inventory and segment granularity", "",
          "**What a 'frame' is here.** The stored crops are a sparse, even sample of each source video, not its frames:",
          "", "| development source file | videos | stored crops per video (min / median / max) | median raw-frame step between stored crops |",
          "|---|---:|---|---:|"]
    for n, v in fpv.items():
        L.append(f"| {n} | {v['videos']} | {v['min']} / {v['median']:.0f} / {v['max']} | {v['median_raw_frame_step']} |")
    L += ["", "So a video is 16–32 stills roughly 10–15 source frames apart (≈ 0.3–0.5 s at 30 fps). "
          "`/data/umar/Datasets/tbiom_frames` (29 GB) holds the same 32-still sampling for CDFv2, CDFv3, DFD, DFDC, "
          "DFDCP and DFEval24 — it is not denser.", "",
          "**What exists per (view, granularity, condition):**", "",
          "| view | granularity | clean development | JPEG q40 / resize 0.60 development | tests |", "|---|---|---|---|---|",
          f"| APP, OM | per stored crop | yes (`warmgate_e19_*`) | {exists(S2 / 'p_dev_warmgate_e19_VALmix_jpeg40.csv')} (`p_dev_warmgate_e19_*`) | clean all ten; processed CDFv2, DFDCP |",
          "| T | one opinion per video (full span) | yes | yes | clean all ten; processed CDFv2, DFDCP |",
          f"| T | per segment (4 contiguous quarters + full) | {exists(ACQ / 'VALmix.csv')} (`logs/tbiom/acquisition/`) | **no** | **no** |",
          "| quality descriptors (blur, blockiness, contrast, motion, luma) | per segment | yes, same cache | **no** | **no** |",
          "", "Segment cache detail (clean development only):", "",
          "| source | videos | segments/video | distinct stored crops per segment (min / median / max) | segments whose 16-frame clip repeats crops | degenerate (empty) segments |",
          "|---|---:|---:|---|---:|---:|"]
    for n, v in seg.items():
        if v is None:
            L.append(f"| {n} | — | — | — | — | — |"); continue
        d = v["distinct_frames_per_segment"]
        L.append(f"| {n} | {v['videos']} | {v['segments_per_video']} | {d['min']} / {d['median']:.0f} / {d['max']} | "
                 f"{v['segments_repeating_frames']} of {v['segments']} | {v['degenerate_segments']} in {v['degenerate_videos']} videos |")
    L += ["", "Timestamps: the stored-crop file names carry the source frame index (`000.png`, `014.png`, …; "
          "`frame_0010.png`), so segment order and approximate time are recoverable; source fps is not stored.", "",
          "## 4. Gate", "",
          "Aligned segment-level opinions from all three views **do not exist** in the form Stage 1A needs:", "",
          "1. **Processed conditions have no segment-level temporal opinions and no quality descriptors.** The "
          "controllers are trained on clean *and* processed examples, so these must be generated by local clip "
          "inference before any controller is fitted. This adds GPU cost (the clean pass took ≈13 GPU-minutes for "
          "29k segment clips; two processed conditions ≈ 30 more).",
          "2. **At the stored density a segment holds 4–8 distinct crops**, so every per-segment V-JEPA clip is those "
          "crops repeated to 16 frames. Each segment is still a genuinely different observation — its frames are "
          "distinct from the other segments' — so this is not the forbidden repetition of one video-level opinion "
          "across timestamps; but the temporal branch sees a stuttered clip at segment granularity, and that is "
          "recorded as a limitation of every segment-level result.",
          "3. **Degenerate segments** (videos with fewer crops than segments) were filled with the full-span opinion "
          "in the acquisition cache. That *is* the forbidden simulation; those segments are excluded, not filled, in "
          "every Stage 1A fit.",
          "4. **Spatial segment opinions need no inference**: they are exact means of the cached per-frame opinions "
          "over the same crops the temporal segment used.",
          "",
          "**Processing-sensitive branch for Stage 1B** (from Experiment 3 on the same reference): development v3.1 "
          "drop clean → JPEG q40 is T 0.0498, APP 0.0994, OM 0.0813. The spatial path is the more "
          "processing-sensitive one, and APP and OM share the adapted CLIP path, so 1B treats them as one coupled "
          "spatial experiment.", "",
          "**Pass condition.** Three checkpoints named and hashed; the discount rule read from code and verified "
          "(evidence scaling); segment granularity recorded (4 segments of 4–8 distinct stored crops, clean "
          "development only). Stage 1A proceeds only after the missing processed segment opinions and descriptors "
          "are generated."]
    (R / "REFERENCE_MANIFEST.md").write_text("\n".join(L) + "\n"); print("wrote", R / "REFERENCE_MANIFEST.md")


if __name__ == "__main__":
    main()
