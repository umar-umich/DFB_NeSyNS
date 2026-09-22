# Prior art and claim-to-evidence map — dependency-partitioned evidential fusion

Stage 9 of `tbiom/docs/grouped-fusion-brief.md`. The operator under examination is

```
omega_S = CautiousNormalized(omega'_APP, omega'_OM)      # idempotent t-norm inside the shared CLIP path
omega_F = Dempster_two_source(omega_S, omega'_T)         # conjunctive rule across the separate encoder
```

with evidence scaling `e'_k = r_v e_k` applied per view beforehand. Exact equations and the numerical policies are in
`analysis/tbiom/grouped/ops.py`; the algebraic checks that exercise them are in `EXP_GROUPED_AUDIT.md`.

## 1. Availability statement, first

**No primary source was retrieved or read during this task.** This machine has no network access in this session and
the papers below are not in the repository. Every entry is therefore written from the model's recollection of the
literature and is marked `UNVERIFIED`. Recollection is not a citation: each row states what must be checked against
the actual paper before any of it appears in a submission, and no distinction below may be relied on until it has
been. Where the recollection is too thin to state a distinction at all, the row says so rather than inventing one.

## 2. Comparison table

| work | what it establishes (recalled) | relation to this operator | status |
|---|---|---|---|
| Dempster's rule; Shafer, *A Mathematical Theory of Evidence* (1976) | conjunctive combination with conflict normalised away; assumes distinct (independent) bodies of evidence | the outer step **is** this rule, unchanged. No novelty is claimed for it | UNVERIFIED — standard, but the independence wording should be quoted exactly |
| Smets, canonical decomposition / conjunctive weight function | every separable belief function factors into simple support functions with canonical weights; Dempster multiplies those weights | the representation this operator is written in. Not novel here | UNVERIFIED |
| Denœux, *Conjunctive and disjunctive combination of belief functions induced by non-distinct bodies of evidence* (2008) | the **cautious rule**: minimum of canonical weights; idempotent; the principled choice for non-distinct sources | the inner step **is** this rule (normalised binary variant). Not novel here | UNVERIFIED — must check the normalisation convention used for the binary frame |
| Quost, Masson, Denœux, *Classifier fusion in the Dempster–Shafer framework using optimized t-norm based combination rules*, IJAR 2011, DOI 10.1016/j.ijar.2010.11.008 | a **family of t-norm combination rules between the cautious rule and Dempster's**, with the rule selected/optimised per classifier pair from data | **This is the closest prior art and probably subsumes the construction.** Choosing min for one pair and product for another is a point in exactly that family. If the paper allows a different t-norm per pair of sources — which is what "optimized t-norm based combination" suggests — then this operator is an instance of it, not a new rule, and the only thing left that could be new is the *criterion* for the choice (read off the architecture rather than fitted) | UNVERIFIED — **highest priority to obtain.** The claim stands or falls here |
| Chebbah, Martin, Ben Yaghlane, *Combining partially independent belief functions*, DSS 2015, DOI 10.1016/j.dss.2015.02.017 | estimates a degree of independence between sources and combines accordingly, interpolating between independence-assuming and dependence-tolerant behaviour | same problem, data-driven answer instead of an architectural one. If this covers partitioned application, the contribution narrows further | UNVERIFIED |
| Kallel, Le Hégarat-Mascle, cautious-adaptive rule (2009) | adapts between cautious and conjunctive behaviour according to measured conflict/dependence | the conflict-interpolating variant explored earlier (`EXP_FUSION_NOVEL.md`) sits here. That variant is **excluded from this queue** by the brief and is not claimed | UNVERIFIED |
| PCR5/PCR6, Smarandache & Dezert | conflicting mass redistributed proportionally to the sources that caused it; PCR6 = PCR5 for two sources | comparator arms only (`PCR6-R`, `GCP-AO-R`). The two-source equivalence is stated wherever PCR6 appears | UNVERIFIED — equivalence is standard and is also verified numerically in this repo for the binary case |
| DiCoME (the baseline this project reproduces) | two-view evidential detector combined with Dempster's rule | the mandatory baseline row. `Dempster-R` is labelled as the DS / DiCoME-family reference throughout and is never presented as the proposed method | partially verified — our own reproduction is in the repo; the published method description is not |
| DISCERN confidence-modulated fusion, IBDC, per-stream supervision | this project's earlier fusion work | not used here; no auxiliary loss or extra branch is added in this queue | in-repo |

## 3. What could have been claimed, and what the evidence supports

| candidate claim | evidence | verdict |
|---|---|---|
| A new combination **operator** | the construction is a t-norm chosen per partition, which Quost et al. (2011) appears to already define as a family | **not supportable** without the primary source, and probably not supportable with it |
| A new **criterion** for choosing the t-norm — read off the encoder graph rather than fitted | the falsification controls test it: the architecture's partition beats both wrong partitions on DF40 (`EXP_GROUPED_EXPOSED.md` §2) | possible in principle, but see the row below, which removes its empirical support |
| The partition **improves detection** | on development no arm clears the pre-registered bar; grouped-R is −0.0016 against Dempster-R with the CI spanning zero in 5/5 splits, and it fails the vacuity guard at −0.0055 (`EXP_GROUPED_DEVELOPMENT.md` §3, §4) | **not supported** |
| The DF40 reenactment gain is the partition at work | under its own fitted discount the appearance weight is **zero in 4 of 5 folds**, and grouped-R is then bit-for-bit the two-view `T+OM` fusion. Per-family differences against `T+OM` are −0.0001, +0.0001, −0.0003, +0.0002 (`EXP_GROUPED_EXPOSED.md` §0, §3) | **not supported — the gain is dropping the appearance view, not partitioning the dependence** |
| The mechanism is real | grouped-R rescues 0.7428 of the "both CLIP heads wrong together, temporal right" cases against Dempster-R's 0.5558, and loses the reverse case 0.5796 against 0.7173 (`EXP_GROUPED_DEVELOPMENT.md` §5) | supported as a **description of a trade**, not as an advantage |

## 4. Recommendation

Report the construction as an **architecture-informed application of existing rules with a negative outcome**, not as
a new operator. Specifically:

1. Do not claim a new fusion operator. The inner rule is Denœux's, the outer is Dempster's, and applying different
   t-norms to different pairs is, to the best of the available recollection, already the subject of Quost et al.
   (2011). Obtain that paper before writing anything stronger.
2. Do not claim a detection advantage. It fails its own pre-registered gate on development and its cross-manipulation
   gain is reproduced exactly by a two-view baseline.
3. The reportable finding is the **diagnostic** one, and it is genuinely useful: on the reenactment generators where
   both CLIP heads sit at chance they are also wrong *together*, because they share a feature path, and a rule that
   treats their agreement as two independent confirmations is misled by it. The cleanest demonstration of that is not
   a new operator but the fitted discount itself — given the freedom, the fit sets the appearance weight to zero.
4. Keep `Dempster-R` as the deployed rule and as a labelled baseline row. Nothing measured here justifies replacing
   it, and nothing measured here justifies presenting it as ours.

## 5. Provenance

No prior report was deleted, renamed or relabelled for this study. `EXP_FUSION_NOVEL.md` (commit `2b3f752`), which
reported the single-seed, interval-free version of this result, is retained unchanged and is superseded by the
documents listed in `grouped_fusion_preregister.yaml`.
