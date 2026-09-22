#!/usr/bin/env python3
"""Stages 4 and 7: the reference manifest and the algebraic checks, before any dataset scoring.

    OMP_NUM_THREADS=8 python analysis/tbiom/grouped/audit.py

Stage 4 records what is actually on disk -- commit, checkpoints and their hashes, branch order, the opinion algebra
as the caches really carry it, fold and grouping structure, and the exposure history. Stage 7 exercises every
operator on constructed opinions where the answer is known independently, so an implementation error shows up here
and not as a dataset result.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "temporal")]
import ops  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
from fusion_combos import REN  # noqa: E402

R = Path("tbiom/reports")
CK = {"T (V-JEPA 2 LoRA, B-augmented, seed 42)":
      "logs/tbiom/temporal/branches/vjepa2_lora_B_seed42/best.pt",
      "T (V-JEPA 2 LoRA, B-augmented, seed 1337)":
      "logs/tbiom/temporal/branches/vjepa2_lora_B_seed1337/best.pt",
      "APP + OM (warm+gate, epoch 19) — deployed reference":
      "/data/umar/Repos/DISCERN_Ext/runs/discern_ext_train/warm_dicome_gate_seed42/checkpoints/"
      "discern_ext-best-epoch=19-val_auroc_video=0.9956.ckpt"}


def sha(p, n=16):
    p = Path(p)
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


# --------------------------------------------------------------------------- Stage 7
def checks():
    rng = np.random.default_rng(7)
    out = []

    def rec(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    # random admissible opinions
    u = rng.uniform(0.02, 0.9, 4000)
    p = rng.uniform(0, 1, 4000) * (1 - u) + u / 2
    o, bad = ops.opinion_from_pu(p, u)
    rec("constructed opinions are admissible", not bad.any(), f"{int(bad.sum())} violations")
    rec("b1 + b0 + u = 1", np.allclose(o.sum(-1), 1), f"max dev {np.abs(o.sum(-1) - 1).max():.2e}")
    rec("p recovered from the opinion", np.allclose(ops.p_of(o), p), f"max dev {np.abs(ops.p_of(o) - p).max():.2e}")
    e = ops.evidence_of(o)
    rec("evidence round trip e -> opinion -> e", np.allclose(ops.opinion_of_evidence(e), o),
        f"max dev {np.abs(ops.opinion_of_evidence(e) - o).max():.2e}")
    rec("alpha_k = e_k + 1 and u = 2/sum(alpha)", np.allclose(o[..., 2], 2 / (e.sum(-1) + 2)),
        f"max dev {np.abs(o[..., 2] - 2 / (e.sum(-1) + 2)).max():.2e}")
    r = 0.37
    rec("evidence scaling is e' = r e, not belief-mass discounting",
        np.allclose(ops.evidence_of(ops.discount(o, r)), r * e),
        "belief-mass discounting would give b' = r b, u' = 1 - r(1-u)")

    o2, _ = ops.opinion_from_pu(rng.uniform(0, 1, 4000) * (1 - u) + u / 2, u)
    for nm, f in (("Dempster", ops.dempster2), ("Yager", ops.yager2), ("PCR6", ops.pcr6_2)):
        z = f(o, o2)
        rec(f"{nm}: masses non-negative", (z >= -1e-12).all(), f"min {z.min():.2e}")
        rec(f"{nm}: masses sum to 1", np.allclose(z.sum(-1), 1), f"max dev {np.abs(z.sum(-1) - 1).max():.2e}")
        rec(f"{nm}: two-source symmetry", np.allclose(z, f(o2, o)),
            f"max dev {np.abs(z - f(o2, o)).max():.2e}")
    c = ops.cautious_n(o, o2)
    rec("cautious: masses sum to 1", np.allclose(c.sum(-1), 1), f"max dev {np.abs(c.sum(-1) - 1).max():.2e}")
    rec("cautious: idempotent on admissible opinions", np.allclose(ops.cautious_n(o, o), o),
        f"max dev {np.abs(ops.cautious_n(o, o) - o).max():.2e}")
    rec("cautious: exchange invariant", np.allclose(c, ops.cautious_n(o2, o)), "")
    rec("cautious: duplicating a view inside the group changes nothing",
        np.allclose(ops.cautious_n(o, o2, o2), c), "the property Dempster does not have")
    rec("Dempster: duplicating a view DOES change the result (the double count being avoided)",
        not np.allclose(ops.dempster2(ops.dempster2(o, o2), o2), ops.dempster2(o, o2)), "")
    rec("evidence average: adds the Dirichlet prior once",
        np.allclose(ops.evidence_of(ops.evidence_average(o, o2)), (e + ops.evidence_of(o2)) / 2),
        "an average of probabilities would not satisfy this")

    vac = np.tile(np.array([0.0, 0.0, 1.0]), (4000, 1))
    rec("vacuous is the Dempster identity", np.allclose(ops.dempster2(o, vac), o),
        f"max dev {np.abs(ops.dempster2(o, vac) - o).max():.2e}")
    rec("grouped-R with a vacuous temporal view reduces to the cautious spatial pair",
        np.allclose(ops.fuse({"APP": o, "OM": o2, "T": vac}, "grouped-R"), c), "")
    rec("grouped-R with an absent temporal view reduces to the same",
        np.allclose(ops.fuse({"APP": o, "OM": o2}, "grouped-R"), c), "absence handled as pass-through")
    rec("grouped-R is not three-source Dempster",
        not np.allclose(ops.fuse({"T": o, "APP": o2, "OM": o}, "grouped-R"),
                        ops.fuse({"T": o, "APP": o2, "OM": o}, "Dempster-R")), "")
    rec("grouped-R is not the global cautious rule",
        not np.allclose(ops.fuse({"T": o, "APP": o2, "OM": o}, "grouped-R"),
                        ops.fuse({"T": o, "APP": o2, "OM": o}, "CAUT-R")), "")
    rec("grouped-R is APP/OM exchange invariant",
        np.allclose(ops.fuse({"T": o, "APP": o2, "OM": o}, "grouped-R"),
                    ops.fuse({"T": o, "APP": o, "OM": o2}, "grouped-R")), "")

    # named limit cases
    def one(p_, u_):
        return ops.opinion_from_pu(np.array([p_]), np.array([u_]))[0]
    cases = {"matching strong": (one(0.98, 0.02), one(0.97, 0.03)),
             "opposite strong": (one(0.98, 0.02), one(0.02, 0.02)),
             "one strong one weak": (one(0.98, 0.02), one(0.55, 0.80)),
             "very small vacuity": (one(0.9, 1e-6), one(0.85, 1e-6))}
    lim = []
    for nm, (a, b) in cases.items():
        st = {}
        lim.append((nm, float(ops.p_of(ops.dempster2(a, b, st))[0]), float(ops.p_of(ops.yager2(a, b))[0]),
                    float(ops.p_of(ops.pcr6_2(a, b))[0]), float(ops.p_of(ops.cautious_n(a, b))[0]),
                    float(ops.dempster2(a, b)[0, 2]), float(ops.pcr6_2(a, b)[0, 2])))
    rec("PCR6 keeps near-zero vacuity under opposite strong opinions",
        lim[1][6] < 0.01, f"u_PCR6 = {lim[1][6]:.2e} against u_Dempster = {lim[1][5]:.2e} — the brief's warning, confirmed")
    return out, lim


def main():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    dev, dproc, tests = D.load("vjepa2_lora_B_seed42")

    # opinion-algebra audit on the real caches, per view
    algebra = {}
    for v in ops.VIEWS:
        p, u = dev[f"p_{v}"].to_numpy(), dev[f"u_{v}"].to_numpy()
        o, bad = ops.opinion_from_pu(p, u)
        rough = np.abs(np.where(p < u / 2, u / 2 - p, np.where(p > 1 - u / 2, p - (1 - u / 2), 0.0)))
        algebra[v] = {"rows": int(len(p)), "outside_admissible": int(bad.sum()),
                      "max_violation": float(rough.max()), "roundoff_only": bool(rough.max() < 1e-6),
                      "u_min": float(u.min()), "u_max": float(u.max()),
                      "sum_dev_max": float(np.abs(o.sum(-1) - 1).max())}

    tests_list = sorted(tests)
    folds = [(k, len(Fd), len(Sd)) for k, Fd, Sd in D.splits(dev)]
    d40 = Path("logs/tbiom/df40/meta.json")
    meta = json.load(open(d40)) if d40.exists() else {}
    real_pools = {}
    for n, m in meta.items():
        real_pools.setdefault(m["source"], []).append(n)

    L = ["# Grouped-fusion reference manifest (Stage 4)", "", __doc__.split("\n\n", 2)[2].strip(), "",
         f"Repository commit `{commit}`; working tree {'DIRTY' if dirty else 'clean'} at audit time"
         + (f" ({len(dirty.splitlines())} paths)" if dirty else "") + ".", "",
         "## 1. Frozen checkpoints", "", "| role | path | sha256 (16) |", "|---|---|---|"]
    for k, p in CK.items():
        L.append(f"| {k} | `{p}` | `{sha(p)}` |")
    sc = Path("logs/tbiom/grouped/seed_checkpoints.json")
    if sc.exists():
        for tag, s in json.load(open(sc)).items():
            L.append(f"| second-seed spatial, {tag} (v3.1-selected epoch {s['epoch']}) | "
                     f"`{Path(s['checkpoint']).name}` | `{sha(s['checkpoint'])}` |")
    L += ["", "The deployed warm+gate recipe exists at seed 42 only; no second warm+gate checkpoint was ever trained "
          "and this task forbids training one. The seed check therefore runs inside SBI arm B, the one three-branch "
          "family trained at several seeds, where the seed is the only difference between the members.", ""]

    L += ["## 2. Branch order and the shared path", "",
          "| view | encoder | input | shares a path with |", "|---|---|---|---|",
          "| T | V-JEPA 2 ViT-L, LoRA | 16-frame clip | — |",
          "| APP | CLIP ViT-L, LoRA | stored crop | OM |",
          "| OM | beta-VAE head on the CLIP ViT-L features | stored crop | APP |", "",
          "Appearance and Off-manifold are two heads on one CLIP ViT-L feature path; the temporal view is a separate "
          "encoder on a separate input. That is the partition the primary operator uses, and §2 of the confirmation "
          "report tests it against the two wrong partitions.", "",
          "Aggregation order, unchanged from every other report: per-frame opinions, mean per video, then fuse.", "",
          "## 3. Opinion algebra, as the caches carry it", "",
          "| view | rows | outside u/2 <= p <= 1-u/2 | max violation | roundoff only | u range | max |b+u-1| |",
          "|---|---:|---:|---:|---|---|---:|"]
    for v, a in algebra.items():
        L.append(f"| {v} | {a['rows']} | {a['outside_admissible']} | {a['max_violation']:.2e} | "
                 f"{'yes' if a['roundoff_only'] else '**NO — substantive**'} | "
                 f"[{a['u_min']:.4f}, {a['u_max']:.4f}] | {a['sum_dev_max']:.2e} |")
    L += ["", "Opinions are reconstructed from the cached (p, u) pairs. Nothing is projected into the admissible "
          "region: the table reports violations instead, and separates roundoff from a substantive inconsistency by "
          "magnitude. alpha_k = e_k + 1, u = 2/sum(alpha), b_k = e_k/sum(alpha), p = b_fake + u/2, and p is the "
          "Dirichlet mean; all four identities are exercised in the algebraic checks below.", "",
          "Reliability enters as evidence scaling e'_k = r e_k. Belief-mass (Shafer) discounting is a different "
          "operation and is not used anywhere in this study.", "",
          "## 4. Folds, grouping, and sample counts", "",
          "| split | fitting rows | selection rows |", "|---|---:|---:|"]
    for k, nf, ns in folds:
        L.append(f"| {k} | {nf} | {ns} |")
    L += ["", f"Five source-grouped folds, `GroupShuffleSplit` on the source id, seeds 0-4 — the existing v3.1 "
          f"splits, unchanged. Development sets: {', '.join(sorted(set(dev.source)))}.", "",
          f"Processed development conditions: {', '.join(sorted(dproc))}, each linked to its clean source and "
          "resampled with it.", "",
          f"Test sets ({len(tests_list)}): {', '.join(tests_list)}.", "",
          f"DF40: {len(meta)} generator sets. Genuine pools are shared within a source — "
          + "; ".join(f"{s}: {len(v)} generator sets draw on one real pool" for s, v in sorted(real_pools.items()))
          + ". Bootstrap resampling therefore groups by source video, never by generator set, or the reused real "
            "videos would be counted as independent.", "",
          "## 5. Exposure history", "",
          "| data | status |", "|---|---|",
          "| CDFv2 | informed development (selection and diagnosis) — not untouched |",
          "| DeepfakeEval2024 | informed development — not untouched |",
          "| DF40 | **has informed operator design**: the grouped operator was checked against it before this task |",
          "| FaceShifter | read once after the recipe was frozen (`SEALED_BENCHMARKS.md`) |",
          "| the other seven test sets | scored after each decision was recorded, never fitted on |", "",
          "**There is no untouched confirmation set for this study.** DF40 informed the choice of the grouped "
          "operator, so a DF40 result cannot also confirm it; re-partitioning exposed examples would not change "
          "that. What the confirmation stage can establish is whether the effect survives a second detector seed and "
          "whether the wrong partitions lose — neither of which is a generalisation claim to unseen generators. "
          "Every finding here is labelled exploratory for that reason, and the exact deficit is: no generator family "
          "held out from operator design.", ""]

    res, lim = checks()
    npass = sum(1 for _n, ok, _d in res if ok)
    L2 = ["# Grouped-fusion algebraic and mechanism checks (Stage 7)", "",
          "Every operator exercised on constructed opinions where the answer is known independently of any dataset. "
          "Run before dataset scoring, so an implementation error surfaces here.", "",
          f"**{npass} of {len(res)} checks pass.**", "", "| check | result | detail |", "|---|---|---|"]
    for n, ok, d in res:
        L2.append(f"| {n} | {'pass' if ok else '**FAIL**'} | {d} |")
    L2 += ["", "## Limit cases, p_fused by rule", "",
           "| case | Dempster | Yager | PCR6 | cautious | u Dempster | u PCR6 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for nm, dp, yg, pc, ca, ud, up in lim:
        L2.append(f"| {nm} | {dp:.4f} | {yg:.4f} | {pc:.4f} | {ca:.4f} | {ud:.2e} | {up:.2e} |")
    L2 += ["", "The `opposite strong` row is the brief's warning made concrete: PCR6 redistributes the conflicting "
           "mass to the classes and leaves the vacuity at the product of two near-zero vacuities, so it returns a "
           "near-0.5 probability with almost no vacuity — confidently undecided. Dempster returns the same "
           "probability with the conflict visible. Any arm with a PCR6 outer step is therefore at risk on the "
           "uncertainty endpoint by construction, which is why the pre-registration disqualifies an arm on vacuity "
           "error-ranking regardless of its detection number.", "",
           "PCR6 is not associative, so source-family metadata is preserved throughout and no regrouping invariance "
           "is claimed for the PCR6-outer arms.", ""]
    (R / "GROUPED_REFERENCE_MANIFEST.md").write_text("\n".join(L) + "\n")
    (R / "EXP_GROUPED_AUDIT.md").write_text("\n".join(L2) + "\n")
    json.dump({"commit": commit, "dirty": bool(dirty), "algebra": algebra,
               "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in res],
               "checkpoints": {k: sha(v) for k, v in CK.items()}},
              open(R / "EXP_GROUPED_AUDIT.json", "w"), indent=1)
    print(f"{npass}/{len(res)} checks pass")
    for n, ok, d in res:
        if not ok:
            print("  FAIL:", n, d)
    print("wrote GROUPED_REFERENCE_MANIFEST.md, EXP_GROUPED_AUDIT.md")


if __name__ == "__main__":
    main()
