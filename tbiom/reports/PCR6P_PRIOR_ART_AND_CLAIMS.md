# PCR6+ prior art and claim-to-evidence map

Section 8 of `tbiom/docs/pcr6plus-fusion-brief.md`.

## 1. Availability statement

Unlike the grouped-fusion study, **primary sources were obtained and read for this one.** Each row below says which.
Papers downloaded during this task, with the sha256 of the exact file read:

| work | file | sha256 (32) |
|---|---|---|
| Dezert, Dezert & Smarandache 2021 (PCR5+/PCR6+) | https://isif.org/files/isif/2024-01/last.pdf | `d87ae493e7c409d1b194ed701b6fe746` |
| Bezirganyan et al., Discounted Belief Fusion, AISTATS 2025 | https://arxiv.org/pdf/2412.18024 | `0d6ab4fd233747b9dceb05f6872b12a0` |
| Xu et al., Reliable Conflictive Multi-View Learning, AAAI 2024 | https://ojs.aaai.org/index.php/AAAI/article/download/29546/30911 | read, text extracted |

## 2. Comparison

| work | what it establishes | relation to this study | status |
|---|---|---|---|
| **PCR5/PCR6** (Smarandache & Dezert; Martin & Osswald) | conflicting mass redistributed proportionally to the sources that produced it; PCR6 = PCR5 for two sources; neither is associative | `PCR6-R` is the direct S-source rule, eq. (15). The historical rows in earlier reports were a pairwise fold and are relabelled `PCR6-pairwise-R`; the two differ by up to 1.8e-01 | **read** (eq. 9-15 of the 2021 paper) |
| **PCR6+** (T. Dezert, J. Dezert, Smarandache, JAIF 16(1) 2021, 48-73) | binary keeping indexes discard focal elements that contain all others in a partial conflict, restoring neutrality of the vacuous BBA | **This is the operator we evaluate.** It is established prior work and is credited as such; no novelty is claimed for it. Our contribution is limited to applying it to this three-view detector and reporting what it does | **read**, eq. (15), (22)-(24), (26) implemented directly |
| **Dempster's rule** | conjunctive, conflict normalised away; associative | mandatory baseline row, labelled as the DS standard and the DiCoME-family reference. Verified associative in our implementation (direct vs fold: 7.8e-16) | **read** (via the 2021 paper's restatement) |
| **Yager's rule** | conflict assigned to Theta rather than redistributed | comparator arm. Not associative — implemented as a direct S-source rule, not a fold (direct vs fold: 5.3e-01) | **read** (restatement) |
| **Reliable Conflictive Multi-View Learning / ECML** (Xu et al., AAAI 2024) | conflictive opinion aggregation implemented as an **average pooling layer** over view opinions, with a proof that fused uncertainty increases for conflictive instances | **Our `Averaging-R` arm is essentially this rule.** It is therefore not merely a "simple control" but a published method, and it is the strongest arm on several of our cells. Any claim for PCR6+-R must clear it, which the pre-registration already requires | **read** |
| **Discounted Belief Fusion** (Bezirganyan et al., AISTATS 2025) | pairwise degree of conflict DC = PD x CC (projected-probability distance times conjunctive certainty), collected in an agreement matrix, turned into a **per-modality discount** with strictness parameter lambda, then order-invariant fusion | **Closest prior art to the conflict-by-source idea.** They decompose conflict *per pair of sources* and use it to discount. The difference in this study is only where the quantity is used: DBF changes the fusion; we leave the decision untouched and use conflict shares purely as an error-ranking score. That is a narrower use, and a weaker claim | **read** |
| **DiCoME** | two-view evidential detector combined with Dempster's rule | baseline row; our own reproduction is in the repo | partially verified (our training; not the published description) |
| **DISCERN confidence-modulated fusion, IBDC** | this project's earlier fusion work | not used here; no auxiliary loss or extra branch is added | in-repo |

## 3. Claim-to-evidence map

| category | item | evidence |
|---|---|---|
| **Established operator, credited** | PCR6+ and its neutrality property | Dezert et al. 2021, eq. (26). Reproduced on our frame: `EXP_PCR6P_AUDIT.md` §3 |
| **Our implementation choice** | applying PCR6+ on a binary frame, where the keeping indexes reduce to kappa(Theta)=0 | derived from eq. (22)-(24), verified in 17/17 unit tests |
| **Our implementation choice** | direct S-source Yager rather than a pairwise fold | associativity check in this study |
| **Measured finding** | under evidence scaling, a view discounted to r=0 leaves PCR6+ exactly unchanged (0.00e+00) but shifts PCR6 by up to 2.8e-01 | `EXP_PCR6P_AUDIT.md` §3 |
| **Measured finding** | historical `PCR6-R` rows were a non-associative fold, not the direct rule | `EXP_PCR6P_AUDIT.md` §5 |
| **Measured finding** | detection and reliability outcomes | `EXP_PCR6P_DEVELOPMENT.md`, `EXP_PCR6P_EXPOSED.md`, summarised in `PCR6P_OUTCOME.md` |
| **Unverified hypothesis** | that conflict attributed by source pair is a better error ranker than vacuity, entropy or margin | tested and **not supported** — see the outcome report. DBF already uses pairwise conflict for discounting, so even a positive result would have been a narrow application claim |

## 4. Provenance

No prior report was deleted, renamed or relabelled. The historical `PCR6-R` rows are relabelled only in tables
produced by this study; `EXP_FUSION_OPERATORS.md`, `EXP_DF40_OPERATORS.md` and the grouped-fusion reports are
unchanged.
