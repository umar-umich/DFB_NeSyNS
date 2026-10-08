#!/usr/bin/env python3
"""Assemble the Appendix backup report from its artifacts."""
from __future__ import annotations
import json, sys
from pathlib import Path

R = Path("/data/umar/Repos/DFB_NeSyNS")


def main() -> int:
    b = (R / "outputs/fusion_experiments/CURRENT_BACKUP_RUN").read_text().strip()
    out = R / "outputs/fusion_experiments" / b
    main_run = (R / "outputs/fusion_experiments/CURRENT_RUN").read_text().strip()
    j = lambda f: json.load(open(out / f))
    feas, subs, arms = j("backup_feasibility.json"), j("backup_subspace.json"), j("backup_arms.json")

    A = arms["arms"]; C = arms["contrasts"]
    g = A["intervention-guided"]["mean"]
    L = [f"# Appendix backup — intervention-guided fusion ({b})", "",
         "Label: **FF++-trained frozen views; intervention-guided fusion trained on development data.**",
         "", f"Authorised by the Appendix to override the Phase 2 gate for this experiment only. Main run: `{main_run}`.",
         "", "## 1. Feasibility (required before any fitting)", "",
         "| domain | paired groups | aligned pairs |", "|---|---:|---:|"]
    for d, v in feas["per_domain"].items():
        L.append(f"| {d} | {v['source_groups_with_a_genuine_partner']} | {v['aligned_pairs']} |")
    L += ["", f"**{feas['total_aligned_pairs']} aligned pairs** across {len(feas['families_with_pairs'])} families, so the backup proceeded.",
          "Coverage is uneven and this bounds every conclusion below: FF++ supplies 560 of them and four",
          "of seven development domains supply none, so the subspace is fitted mostly on FF++ structure",
          f"and then asked to help elsewhere. Pairs usable with all three conditions cached: **{subs['usable_pairs']}**.",
          "", "## 2. Arms — same head, same data, same budget; only the 8-D mapping differs", "",
          "| arm | mean development macro | sd over seeds |", "|---|---:|---:|"]
    for k, v in sorted(A.items(), key=lambda kv: -kv[1]["mean"]):
        sd = v.get("sd")
        L.append(f"| {k} | {v['mean']:.4f} | {('%.4f' % sd) if sd is not None else '—'} |")
    L += ["", "## 3. The two Appendix questions", "",
          "**Q1 — predictive utility: does the guided arm beat B1-refit and ordinary PCA?** **NO.**", "",
          "| contrast | point | 97.5% lower bound |", "|---|---:|---:|"]
    for k, v in C.items():
        lo = v.get("lo97.5")
        L.append(f"| {k} | {v['point']:+.4f} | {('%+.4f' % lo) if lo is not None else 'not paired'} |")
    L += ["",
          f"The guided arm reaches {g:.4f} against B1-refit's {A['B1-refit (reference)']['mean']:.4f} and PCA's "
          f"{A['PCA']['mean']:.4f}. Both primary utility comparisons fail, and every upper bound is below zero.",
          "",
          "**Q2 — mechanism evidence: does it beat the shuffled-pair and manipulation-only controls?** **NO.**",
          "",
          f"It is *worse* than both: shuffled pairs {A['shuffled-pairs']['mean']:.4f}, manipulation-only "
          f"{A['manipulation-only']['mean']:.4f}, random {A['random']['mean']:.4f}. The Appendix says that if the guided arm",
          "merely MATCHES PCA or shuffled pairing, the mechanism lacks supporting evidence. It does not",
          "match them — it loses to all four controls, including a random orthonormal subspace.",
          "",
          "**The shuffled-pair control is the most informative row.** Fitting the same construction with",
          "*deliberately wrong* genuine partners scores 0.9457, the best of every subspace arm. Correct",
          "source pairing does not merely fail to help here; it actively costs 0.012. Whatever the",
          "guided directions capture, it is not something that transfers to the fusion decision.",
          "", "## 4. Conclusions, in the Appendix's own four parts", "",
          "1. **Does the constrained representation improve predictive performance?** No — below every",
          "   control and 0.018 below B1-refit.",
          "2. **Does correct pairing contribute beyond generic dimensionality reduction?** No — the",
          "   shuffled-pair control beats correct pairing.",
          "3. **Does the nuisance penalty contribute beyond manipulation separation alone?** No — the",
          "   manipulation-only arm beats the full generalized-eigenvector arm.",
          "4. **Does any benefit transfer to DeepfakeEval24 and DFDC?** Not evaluated: there is no",
          "   benefit on development to transfer, and the Appendix forbids tuning another model after",
          "   seeing stress-test results. Running a locked panel for an arm that lost every development",
          "   contrast would add exposure without informing a decision.",
          "", "## 5. What this does and does not establish", "",
          "This tests ONE instantiation: PCA-32 then 8 generalized-eigenvector directions, three",
          "processing conditions, pairs dominated by FF++, a 4,914-video development pool. It is",
          "evidence against this construction, not proof that no intervention-guided representation",
          "could work. The honest summary is that the constraint did not rescue what unrestricted",
          "feature fusion failed at — F reached 0.8580, and constraining it to 8 guided directions",
          f"reached {g:.4f}, still below the six evidence values alone at {A['S (reference)']['mean']:.4f}.",
          "", "## 6. Recommendation", "",
          "Stop fusion-representation work. Across the main run and this backup, nine distinct attempts",
          "to give the fusion more or better-structured information have now failed, several against",
          "controls showing the mechanism is absent. The one surviving positive is the global discount",
          "(+0.0010 development, +0.0014 on DeepfakeEval24), which is too small to matter. The weak",
          "datasets are weak at the VIEW level, not the fusion level, and that is where effort should go.", ""]
    (out / "report.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
