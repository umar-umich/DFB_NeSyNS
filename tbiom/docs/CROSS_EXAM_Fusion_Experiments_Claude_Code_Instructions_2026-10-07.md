# CROSS-EXAM: implementation instructions for fusion experiments

**For Claude Code — revised 7 October 2026, revision 3**

This full revision supersedes revisions 1 and 2. Fixed decisions: fusion heads train on multi-domain development data; overall development macro AUROC is the primary objective; PE-Core-L/336 LN-only is the baseline row; DeepfakeEval24 is the primary difficult-dataset stress test and DFDC is the secondary stress test, both previously exposed; Phase 2 has a hard statistical gate; Phase 4 adds a subspace after existing frozen suppression; no mixture-of-opinions experiment; 72-hour total execution cap.

Implement the following experiments in the existing T-BIOM / CROSS-EXAM research repository. Continue the existing experiment record; do not rebuild the project. Inspect the repository, implement the necessary code and configurations, run correctness checks and the feasible experiments, and produce a reproducible results report. Do not stop after proposing a plan. Record unavailable data, checkpoints, or compute precisely; never fabricate completed runs or results.

The dissertation is parked. Do not edit the dissertation, its figures, or its architecture claims. These are research variants to evaluate, not established improvements. If creating scientific figures for the experiment report, export **PNG only**, with restrained publication colors and concise labels. Do not generate SVG, PDF, or PowerPoint figures.

## 1. Objective and fixed context

Test whether fusion loses useful information when each view is reduced to two evidential outputs, and whether retaining small feature vectors plus controlled cross-view interactions improves generalization. Separately test whether intervention-guided subspaces help retain manipulation information while reducing sensitivity to processing nuisances.

### Lock one baseline before running anything

**B0 is the recorded PE-Core-L/336, layer-norm-only adapted APP configuration whose reported development macro AUROC is 0.9329 and ten-set mean is 0.9212**, with its exact associated OM and temporal checkpoints and PCR6+-R settings. These numbers identify a historical row, not a new selection contest. Do not substitute the LoRA+LN row, CLIP, a later J2b checkpoint, or a 448-resolution run because it scores better.

| View | B0 configuration to recover from that row's manifest |
|---|---|
| APP | PE-Core-L/336, layer-norm-only adaptation |
| OM | The row's beta-VAE orthogonal residual of the appearance feature |
| T | The row's V-JEPA 2 ViT-L/16 with LoRA and 16-frame clips |
| Fusion | The row's direct three-source PCR6+-R, pooling, calibration split, and operating-point rule |

Record exact run ID, checkpoint hashes, preprocessing, suppression state, and configuration in `baseline_lock.json`. The authoring workspace does not contain the training repository, so no checkpoint path or J2b verdict is asserted here. Locate and verify the recorded row before executing learned-fusion comparisons. If its identity cannot be resolved, complete the audit and stop comparative runs; do not select a replacement by performance.

Do not replace the OM residual with the VAE latent vector by assumption. Keep the existing nuisance suppression frozen; Phase 4 explicitly tests an additional projection **after** that suppression. If the locked row has no implemented suppression despite the proposal text, record that fact and do not invent an existing-suppression control.

The proposal separately specifies **PE-Core at 448 resolution with layer-norm-based fine-tuning**. The dissertation stays unchanged. This three-day experiment uses the recorded 336 baseline only; a 448 replication is outside its scope.

### Training stages and J2b status

- **Stage 1: separate view training with EDL.** Preserve the recorded training of appearance adaptation and its head, CLIP LoRA where applicable to historical CLIP runs, MR-VAE and the LFS head, and V-JEPA 2 LoRA and the TD head. No new Stage 1 training is authorized by this brief.
- **Proposed Stage 2: joint training with one-sided IBDC, updating the three evidential heads only.** Call this an accepted existing stage only if the repository's **J2b verdict explicitly passed**. Record the verdict artifact and checkpoint provenance. A failed, missing, or ambiguous verdict means it is not an accepted baseline component. Even a passed J2b does not silently replace the fixed B0 row.
- **New fusion experiments:** freeze all selected upstream extractors, adapters, backbone layer norms, MR-VAE, and existing nuisance suppression. Explicitly identify newly trainable projectors and fusion heads. Their development-data training is a changed protocol, as specified below.

Two outputs are sufficient to parameterize a binary Dirichlet opinion. The hypothesis is that they can discard information needed to decide *how views should interact*. Do not claim that binary classification intrinsically requires more than two evidence values, that vector coordinates are independent evidence counts, or that a larger head alone establishes novelty.

## 2. What the existing report already established

Treat these as historical reference values to verify against saved artifacts, not exact acceptance targets for a different split or checkpoint:

- Stronger standalone branches have not consistently improved fusion. Adapted appearance improves overall performance but hurts some reenactment cases; a stronger temporal backbone also degraded the fused result.
- Reliability is fitted on five source-grouped development folds. The reported grid is `{0, .25, .5, .75, 1}^3`, constrained by `max(w)=1`. Reliability currently acts through evidence scaling, `e' = r e`.
- Constant fitted temporal weight does not imply constant temporal influence in nonlinear PCR6 fusion.
- A jointly selected vacuity calibration improved development reenactment but reduced the ten-set average from approximately **0.9212 to 0.9035**. Generalization is the problem, not merely optimizing the development score.
- Class-specific evidence multipliers did not clearly outperform a comparable random-search control. Additional parameters are not sufficient evidence of a useful mechanism.
- In the matched-mean-vacuity comparison, the probability-preserving vacuity arm won only **4/20** cells; mean differences were approximately **−0.0015 overall / −0.0017 reenactment**. Do not retain “preserving probability is necessarily better” as a supported explanation. Matching mean vacuity is also not matching every video's vacuity.
- Discounting all three views together exposed a useful region excluded by `max(w)=1`; reported development macro AUROC increased approximately **0.9332 → 0.9351**. This motivates a small global-discount control before a larger architecture.
- Quality/input controllers, conflict conditioning, and per-class weights have already been explored. Do not present another generic reliability router as an untested new idea.
- The latest review reports that **logistic stacking already lost: 0.8736 clean and 0.7564 at JPEG q40**. Verify its dataset, split, inputs, and checkpoint before a numerical comparison; these values are not assumed to be the same macro metric as B0. S is another learned-stacking control, not an unprecedented mechanism. F/A/I increase capacity, so overfitting is a serious, empirically motivated expectation to test.
- Near-perfect FF++ training predictions do not prove that fusion has nothing to learn, but provide weak error/disagreement supervision. This brief therefore explicitly chooses development-data head training and reports that protocol change.

Reported repository anchors to inspect, not assumed current APIs:

```text
analysis/tbiom/grouped/ops.py
analysis/upgrade/screen_*.json
analysis/upgrade/final/*_s0.json
analysis/independent_views/INDEPENDENT-VIEWS-REPORT.md
upgrade/joint_calibration.py
tbiom/docs/reenactment-calibration-preregistration.md
```

Find their actual locations with `rg`. Read relevant `AGENTS.md` files and existing experiment instructions. Preserve uncommitted work and old outputs. Write new experiment outputs to a separate versioned directory. Record the Git commit, configuration, checkpoint hashes, software versions, and data-manifest hash for every run.

## 3. Execution order and experiment matrix

Prioritize **Phase 0 → Phase 1 → Phase 2 → hard gate → Phase 4**. Phase 3 and Phase 5 are optional lower-priority work within the same time cap, and only after the hard gate passes. Phase numbers are retained for continuity. Do not implement optional modules preemptively at the expense of the core comparison.

| Phase | ID | Experiment | Role |
|---|---|---|---|
| 0 | AUDIT | Algebra, baseline/J2b identity, exposure and split audit | Required prerequisite |
| 1 | B0 | Fixed recorded PCR6+-R configuration | Historical reproduction plus matched refitting control |
| 1 | B1 | Global evidence discount | Cheapest primary intervention |
| 2 | L | Reproduce existing logistic stacking under matched data access | Low-capacity negative/control baseline |
| 2 | S | MLP on six evidence values | Learned score-only stacking |
| 2 | F | Capacity-matched feature-augmented MLP | Hard falsification test for retained information |
| 4 | P | Additional intervention-guided subspace and mandatory controls | Conditional core; requires valid pairs and Phase 2 pass |
| 3 | A/I | Additive vectors versus low-rank interactions | Optional, after Phase 2 pass, if time remains |
| 5 | G | Group-aware loss on a fixed selected architecture | Optional, multi-domain data only, after Phase 2 pass |

There is **no mixture-of-opinions phase** in this scope. Do not implement one.

Primary vector dimension is `d=8`; optional pairwise rank is `r=2`. Drop the dimension sweep for this bounded pilot. Reuse existing feature caches. No larger backbone or backbone retraining.

Use three predeclared seeds for the Phase 2 gate, reusing established project seeds when available; otherwise use `42, 1024, 3407`. One seed may be used for debugging/screening but cannot pass the gate. A deterministic B0/B1 need not be rerun as if stochastic. Record every attempted configuration and failure.

## 4. Phase 0: audit the opinion implementation

For classes `G` (genuine) and `F` (fake), with a uniform base rate:

\[
\alpha_k=e_k+1,\quad S=e_G+e_F+2,\quad
p_F=\alpha_F/S,\quad u=2/S,\quad b_k=e_k/S.
\]

A valid opinion satisfies `b_G + b_F + u = 1` with nonnegative masses. If a calibration keeps `p=p_F` fixed while changing `u`, then

\[
b_F=p-u/2,\qquad b_G=1-p-u/2,
\qquad 0<u\leq 2\min(p,1-p).
\]

Thus `p=.99` permits `u<=.02`. Multiplying `u` by an arbitrary temperature can produce negative beliefs. Inspect every calibration path for clipping, reconstruction, renormalization, and whether probability is truly preserved. Count invalid pre-repair opinions and quantify the effect of repair. Do not silently rewrite old experiment results. Reproduce legacy behavior under a clearly named legacy setting, then implement a validated setting separately.

For evidence discounting `e'=r e`, verify:

\[
D=r(1-u)+u,\qquad u'=u/D,\qquad b'_k=r b_k/D,
\qquad p'_F=(r e_F+1)/(r(e_F+e_G)+2).
\]

This changes both strength and, generally, the predictive mean. Call it **evidence discounting**, not logit temperature scaling. At `r=0`, it gives the vacuous opinion `(b_G,b_F,u)=(0,0,1)` and `p_F=.5`.

Audit the exact PCR6+ variant in the repository. Preserve direct three-source fusion if that is the baseline; sequential pairwise fusion is a different operator. Build a small independent reference implementation from the actual documented rule. Do not impose properties the rule does not possess, such as idempotence under duplicated sources.

Document the order of frame/crop pooling, evidence conversion, discounting, and fusion. Mean features followed by a nonlinear head is not equivalent to mean head outputs. Preserve the native baseline and make any aggregation changes explicit.

**Gate:** resolve invalid masses, split overlap, inconsistent checkpoints, or unexplained baseline discrepancies before claiming comparative results. A documented rounding difference is not a blocker; a changed input pipeline is.

## 5. Shared data and evaluation protocol

### Exact training-data decision and comparison lanes

**Train S, F, and any permitted A/I/P/G fusion parameters on the existing multi-domain development pool, using nested source-grouped splitting. Do not train these heads on FF++ train in this experiment.** Upstream checkpoints remain those trained under the recorded FF++ protocol. Any supervised subspace statistics use only eligible development-training pairs. Do not silently supplement scarce pairs with external data or FF++ pairs.

Use this exact report label: **“FF++-trained frozen views; fusion trained on development data.”** This is not an FF++-only training claim. Equal data access with B1 makes it a fair fusion comparison; it does not make external FF++-only methods trained under a different protocol matched competitors.

| Result lane | Parameter/data access | Permitted interpretation |
|---|---|---|
| Historical B0 and published FF++-only methods | Their documented original access; B0 includes its reported low-dimensional dev calibration | Historical context, with access explicitly listed |
| Matched B0/B1/L/S/F and conditional variants | Same fixed frozen views, same development pool, same outer/inner partitions; fit fusion parameters only inside allowed partitions | Main within-system comparison under development-data adaptation |
| External baselines adapted to the same pool, if already available | Explicitly matched adaptation data and declared budget | Separate matched-protocol comparison; do not launch broad baseline retraining within this cap |

Retain `B0-historical` predictions for reproduction. For the main experiment, `B0-matched` uses exactly the locked architecture/checkpoints but refits its reliability weights inside each outer training partition. Old weights selected with access to an outer assessment fold must not enter that fold's matched comparison. B1 uses this same nested refitting protocol. Distinguish historical reproduction from nested out-of-fold performance in every table.

### Splits and leakage

1. First reserve the DeepfakeEval24 and DFDC stress-test source groups as specified below, then freeze the exact eligible development-pool membership in a manifest before training. Group by original source video and all fake derivatives, processed copies, clips, and crops; merge linked sources/identities where required. Inspect overlap with upstream training and evaluation sources.
2. Use five source-grouped outer folds. Reuse the established folds only if they satisfy these constraints. An outer held-out fold is used for assessment, never fitting, early stopping, calibration, or selection within that fold. If rebuilding is necessary, create one deterministic manifest and document the departure from historical calibration folds.
3. Inside each outer training partition, use three source-grouped inner folds, stratified by domain/class where feasible. Neural heads train on inner-training groups. Select hyperparameters and epochs using mean inner-validation **overall macro AUROC**. Refit on the whole outer training partition for the fixed selected epoch count, then predict the outer assessment fold once. B0/B1 grid selection and logistic stacking use these same inner splits. For fixed global coefficients, select the grid candidate by its mean inner-validation macro AUROC; the larger permissible training pool is available equally, although methods use it differently.
4. Fit normalization, PCA, nuisance/subspace statistics, thresholds, and any calibration on permitted training/inner-validation partitions only. After selection, refit preprocessing on outer-training data. Outer assessment samples are excluded. Existing suppression remains the frozen baseline artifact; no refitting it on all development folds.
5. Define the macro metric and eligible domains once from the manifest. Report an equal-domain average of within-domain AUROCs, averaged over outer folds and then seeds. Do not concatenate scores across independently fitted outer-fold models to compute a single AUC: different calibration scales can create spurious cross-fold rankings. If a domain lacks both classes in a fold, fix the split before training or predeclare an eligible-domain rule; never drop an inconvenient domain after results are seen.
6. A leave-one-generator-out extension must exclude that generator from training and inner selection, including subspace fitting; it is optional within the cap, not a substitute for the main fixed five-fold protocol.
7. All ten old benchmarks **and DF40 are already exposed**. Label their new results as exposed-set evaluation. Exclude source groups used for head training/calibration from any claimed independent evaluation. A development-sample result cannot reappear as a test result just because it belongs to one of the ten datasets.
8. Dataset/family metadata define training groups and reporting strata only. Do not require source or generator identities at inference.

Write `data_access_manifest.json` with each dataset, source group, role, allowed fitted modules, and exposure status. Refuse cache/split mismatches. Five outer fits are not five stochastic seeds; maintain both indices explicitly.

### Targeted difficult-dataset evaluation: DeepfakeEval24 and DFDC

**Use DeepfakeEval24 as the primary stress-test dataset and DFDC as the secondary stress-test dataset.** The research priority is to improve the known weak cases highlighted by the user, where performance has often remained below 0.90 AUROC. Recover the exact video subsets and historical metrics from the repository rather than assuming every checkpoint is below that level. Use the project's canonical dataset IDs/aliases and existing video-level evaluation protocol.

Both datasets are **previously exposed**. Report this as **targeted evaluation on known difficult datasets**, not untouched confirmation or proof of fresh-dataset generalization. No new dataset acquisition or untouched-dataset search is required for this run.

Before fitting any new fusion parameter, reserve the complete specified DeepfakeEval24 and DFDC video evaluation subsets and exclude their source groups, derivatives, and duplicate content from this run's development/adaptation pool. If the existing development pool includes either dataset, remove those evaluation groups and freeze a revised manifest shared by B0-matched, B1, L, S, F, and all conditional variants. Preserve original historical results separately; scores on a changed development pool are not directly comparable with the historical development macro. Record remaining domain/source counts and rebuild grouped folds only if required. If insufficient eligible development data remain, report that limitation instead of training on the stress-test labels. Log prior checkpoint/data exposure; this exclusion cannot undo historical adaptive use of these benchmarks.

**Model selection remains overall macro AUROC on the remaining development pool.** DeepfakeEval24 is the primary stress-test endpoint, not an early-stopping or hyperparameter-selection dataset. DFDC and the other fixed exposed benchmarks test whether any benefit is broader than one dataset. Neither stress-test outcome may rescue a failed Phase 2 gate or trigger another tuning round within this run.

For final evaluation, use three source-grouped inner folds on the permitted full development pool to select the already allowed hyperparameters/epoch counts, then refit each locked candidate on that pool, once per prescribed neural seed. Refit B0/B1 reliability coefficients using their declared grids on the same permitted data. Fit thresholds from development out-of-fold predictions under the established rule. Do not fit normalization, subspaces, calibration, or thresholds on the stress-test data. Complete refits before hour 66.

Freeze configurations, checkpoints, preprocessing, seed handling, dataset/subset manifests, thresholds, exclusions, and comparison list in `stress_test_lock.json`. Run one final evaluation round on both target datasets and the fixed broader benchmark panel. Include B0-matched, B1-refit, L, S, F, and at most the single conditional variant chosen on development data. Use all three prescribed neural seeds and report the mean of their metrics, not the best seed. Reuse saved predictions for intervals and breakdowns; do not retrain in response to the stress-test outcomes.

Report DeepfakeEval24 and DFDC separately: video AUROC, paired source-group intervals for changes from B0-matched and B1-refit, genuine FPR at the frozen development-derived threshold, NLL, and failure counts. Add manipulation-family and processing strata where annotations support them; do not invent labels. Also report the broader fixed-panel macro and per-dataset changes. Lead with the DeepfakeEval24 result, show DFDC next, and disclose any regression elsewhere. A gain on one difficult dataset is a targeted gain, not automatically an overall improvement. Insufficient evidence is inconclusive. Do not set 0.90 as a required pass threshold merely because it describes past performance.

### Cached features

Cache the actual pre-classifier APP representation, OM orthogonal residual, and temporal representation, together with native two-class evidence. Use the same videos, clips, frames, crops, and masks across methods. Do not hard-code the encoder feature dimensions.

Each record must resolve to a video/source identifier, label, permitted training metadata, view identifiers, feature arrays, evidence arrays, valid-frame/clip masks, preprocessing settings, and checkpoint/configuration hashes. Training-only paired interventions can be cached separately.

Use the existing native video-level opinion aggregation for score inputs. For vector inputs, specify an identical deterministic masked pooling rule for all feature-based methods. The primary choice is masked mean of the selected pre-head features, with temporal pooling following its existing interface. Do not mix stochastic training crops with a different cached baseline unnoticed.

F receives the same six native evidence values as S, plus feature information. This tests the incremental value of retained representations. If vector pooling differs from native score pooling, also evaluate an aggregation-matched score control by applying the frozen original heads to the corresponding pooled features, where the interface permits it. Do not attribute all gain specifically to fusion dimensionality without this control; additional temporal/spatial summary information can also explain it.

### Metrics and selection

Reuse the existing benchmark list and macro-averaging definition. Report per dataset and manipulation family, as well as aggregate:

- AUROC and genuine-video FPR under the established validation-selected threshold rule;
- predictive NLL, Brier score, and ECE with a fixed documented binning scheme;
- reenactment performance and worst eligible group performance;
- selective risk/AURC and coverage curves for models exposing an uncertainty score;
- trainable parameters, training budget, inference cost, and seed variability.

Fit each model's operating threshold on the same allowed calibration partition using the same rule, then freeze it. Raw thresholds need not match across models with different score scales. Do not retune thresholds on test labels.

**The primary selection objective is overall development macro AUROC**, under the fixed domain/fold weighting above. Use it for all grid choices, neural early stopping, architecture choices, and candidate ranking. Break ties within `1e-4` by lower predictive NLL, then lower parameter count, then the fixed config ID. Reenactment, genuine FPR, processing robustness, NLL, and calibration are secondary outcomes; none can rescue a failed primary gate. Do not optimize a reenactment-specific objective in this run. The historical −0.005 macro-AUROC guardrail may be reported as a secondary noninferiority check, not used to select a lower-primary-score winner.

For the development gate and final evaluation, compute paired source-group bootstrap intervals with 2,000 replicates, retaining all related derivatives together. For development, preserve domain/outer-fold strata and average within-fold/domain AUROCs over the three seeds; do not treat seed predictions as independent videos or average predictions into an unregistered ensemble. Recompute nonlinear metrics within each bootstrap replicate. Keep fitted models fixed: these intervals quantify evaluation-sample uncertainty, not the whole hyperparameter-selection process. Report across-seed variability separately. For cross-dataset aggregate metrics, preserve the chosen dataset weights and bootstrap source groups within datasets; handle shared sources consistently if present. The development intervals remain conditional on fitted models and prior exposure to the pool; they are an operational continuation gate, not evidence of generalization to previously unexamined datasets.

A positive screening point estimate is a reason to confirm, not a significance claim. A final improvement claim requires a predeclared primary comparison with uncertainty evidence; otherwise label it inconclusive. A noninferiority claim requires the corresponding confidence bound to exceed −0.005, not just the point estimate. Enumerate planned primary contrasts and disclose multiplicity; do not advertise the best of many cells as a single prespecified test.

## 6. Phase 1: global-discount baseline

Implement:

\[
r_v=\gamma w_v,\qquad \max_v w_v=1,\qquad
\gamma\in\{1,\tfrac12,\tfrac14,\tfrac18\}.
\]

Run two comparisons:

- **B1-fixed:** keep the corresponding nested B0-matched relative weights fixed and select only `gamma` using the allowed inner validation folds. Historical full-pool weights are prohibited in outer assessment.
- **B1-refit:** jointly select relative weights and `gamma` with the same inner splits and primary macro-AUROC objective. This is the **predeclared B1 comparator for the Phase 2 gate**, regardless of which B1 arm later looks best on outer folds. Record the larger search budget; do not attribute all difference to the parameterization.

`gamma=1` with the same relative weights must reproduce its corresponding B0 exactly; refitted weights need not equal historical weights. Preserve B1-refit as the fixed comparator for every later head. If global discounting already solves the observed weakness, that is a useful result; do not manufacture architectural novelty.

## 7. Phase 2: does retained information help?

**L — logistic stacking:** recover the previous implementation and repeat it under the new matched nested data-access protocol. Retain its historical result separately and verify the reported clean/q40 metrics. Do not claim a new neural head beats stacking by comparing different splits or inputs.

**S — score-only MLP:** input the six native nonnegative evidence values, transformed by `log1p` and train-fold standardization. Use a small regularized MLP with two output preactivations, converted to evidence with softplus. Keep the class order explicit. Do not add source IDs or generator metadata.

**F — feature-augmented MLP:** input exactly S's six values plus small per-view projected features. Use the same final EDL objective, stopping policy, data, and tuning budget as S. Count the projectors in the parameter budget. Adjust S's hidden width to match F's total trainable parameter count within approximately 10%, reporting the exact difference. Use the same available number of nonlinear layers where practical.

Run S and F both with no auxiliary per-view loss for the primary information comparison. If auxiliary losses are subsequently added to F, report the no-auxiliary control separately. Include an uninformative, fixed random-feature augmentation control if the observed gain is small or parameter matching is imperfect; its seed must be fixed before results are inspected.

Use bounded projected features, initially:

\[
s_v=\tanh(W_v\,\mathrm{LN}(h_v)+a_v)\in\mathbb R^8.
\]

Here LN is a newly specified, parameter-free normalization in the fusion module unless explicitly configured otherwise. It does not unfreeze backbone layer norms. All dataset-level standardization is fitted on training data only.

**Hard gate — no exceptions for an interesting plot:** use the locked five-outer-fold predictions for three prespecified seeds. Compute the differences in overall development macro AUROC for **F−S** and **F−B1-refit**. Both point estimates must be positive and both one-sided **97.5% paired source-group bootstrap lower bounds must exceed zero**. These two Bonferroni-adjusted bounds target 95% simultaneous coverage for the two planned comparisons, subject to bootstrap assumptions; they do not remove prior development-set exposure or model-selection uncertainty.

If either bound is at or below zero, a required seed/fold is missing, or capacity/data-access matching failed, **skip Phases 3, 4, and 5 entirely**. Report the failure/inconclusive result and retain B1 as the cheap intervention. Do not run a limited interaction diagnostic, try another feature dimension, or tune F after inspecting the outer-fold gate. Test-set results cannot override the gate. If the gate passes, prioritize Phase 4; interactions and group loss remain optional within the original clock.

## 8. Phase 3: vector evidence with low-rank interactions

**Optional and disabled unless the Phase 2 hard gate passed.** Run only after core Phase 4 work or a documented lack of valid pairs, and only within the same 72-hour budget.

These vectors are learned signed features. Do not sum their coordinates as independent pseudo-counts.

For each class `k`, implement an additive head A:

\[
g_k=c_k+\sum_v a_{vk}^{\top}s_v.
\]

Then implement I by adding three pairwise interactions:

\[
g_k=c_k+\sum_v a_{vk}^{\top}s_v
 +\sum_{i<j}s_i^{\top}U_{ij,k}V_{ij,k}^{\top}s_j,
\quad U_{ij,k},V_{ij,k}\in\mathbb R^{d\times r},\quad r=2.
\]

The final opinion is:

\[
e_k=\operatorname{softplus}(g_k),\qquad \alpha_k=e_k+1.
\]

Interactions may increase or decrease a class preactivation; evidence becomes nonnegative only after softplus. APP and OM are structurally dependent. The model must not assume that three views, or their vector coordinates, are independent observations.

Optionally add per-view auxiliary evidential heads:

\[
e_v=\operatorname{softplus}(B_v s_v+c_v).
\]

Use the same auxiliary setting in A and I. Start with auxiliary coefficient `0.1`; include coefficient `0` as an explicit ablation for a promising result. Do not sweep a large coefficient grid.

A concrete common loss, unless the repository already implements a better documented equivalent, is:

\[
\ell_{\rm pred}(\alpha,y)=-\log(\alpha_y/\textstyle\sum_k\alpha_k),
\quad \tilde\alpha=y+(1-y)\odot\alpha,
\]

\[
\ell_{\rm EDL}=\ell_{\rm pred}
+\lambda_{\rm KL}(t)\,\mathrm{KL}
  [\operatorname{Dir}(\tilde\alpha)\|\operatorname{Dir}(\mathbf1)],
\]

\[
\mathcal L=\ell_{\rm EDL}^{\rm fused}
+\lambda_{\rm aux}\frac13\sum_v\ell_{\rm EDL}^{v}
+\lambda_{\rm wd}\|\theta\|_2^2.
\]

This same final-head EDL loss also applies to S and F in Phase 2, with no auxiliary term. Here `y` in the tilde-alpha expression is a one-hot class vector. Reuse the established EDL loss and annealing schedule when available, documenting whether it uses this predictive-mean NLL, expected log loss, or squared error. These are different losses: never silently compare methods trained with different versions. If there is no established schedule, use a linear KL ramp over the first 10 epochs, ending at 1, and predeclare it.

Record the J2b verdict before calling IBDC an accepted existing stage. Do not add a new IBDC ablation in this bounded pilot. Preserve any IBDC provenance actually belonging to the locked upstream checkpoints, and use the same new-head objective in A and I. A proposed IBDC design in the dissertation is not evidence that its implementation passed.

Required controls: I versus A; a parameter-matched additive/nonlinear head; and zeroed interactions with identical remaining weights. Initialize one interaction factor near zero and the other nonzero, or use small random factors: initializing both factors to zero prevents learning through that product.

## 9. Phase 4: intervention-guided subspaces

**Disabled unless the Phase 2 hard gate passed.** This is an **add-on after the existing frozen nuisance suppression**, not a replacement, and not a second fit of the original suppressor. Reuse the frozen upstream features from B0. Add PCA/subspace selection at the fusion input only, preserving native evidence inputs and the downstream F head across all Phase 4 rows. Do not mix this with the optional A/I interaction change in the primary Phase 4 comparison.

Run only when authentic/manipulated pairs can be linked to the same original content and aligned frames/crops. A pair linked only by dataset membership is insufficient. At inference, the model must require only the queried video.

Use label-preserving processing interventions already supported by the data pipeline. Primary conditions: identity, JPEG quality 50, and resize to 0.5 followed by the standard input resize. Apply the same sampled parameters and aligned geometry to the genuine/manipulated member of each pair. Record the exact ordering relative to face cropping and normalization. Avoid severe transformations that erase essentially all manipulation information.

Fit on the corresponding inner/outer **development-training sources only**, using the same nested protocol as F. Sample source groups uniformly, balance manipulation families where feasible, and cap the initial pilot at 1,024 source groups using a fixed seed. Log coverage and pair counts. Multiple manipulated derivatives of one source must not dominate covariance estimation merely because there are more of them.

For each view, fit a training-only PCA to at most 32 dimensions after the declared feature preprocessing. Let the resulting vector be `z_v(x)`. Define:

\[
\delta_{s,m,a}=z_v(a(x_s^m))-z_v(a(x_s^0)),\qquad
\bar\delta_{s,m}=\mathbb E_a\delta_{s,m,a},
\]

\[
C_{\rm man}=\mathbb E_{s,m}[\bar\delta_{s,m}\bar\delta_{s,m}^{\top}],
\]

\[
C_{\rm proc}=\mathbb E_{s,m,a}[(\delta_{s,m,a}-\bar\delta_{s,m})
(\delta_{s,m,a}-\bar\delta_{s,m})^{\top}],
\]

\[
C_{\rm real}=\mathbb E_{s,a}[(z_v(a(x_s^0))-z_v(x_s^0))
(z_v(a(x_s^0))-z_v(x_s^0))^{\top}].
\]

`C_man` is deliberately an **uncentered second moment**, preserving a consistent manipulation shift. `C_proc` captures instability of the paired difference; `C_real` penalizes processing drift in genuine content. Neither proves causal identification.

For the primary implementation, normalize each nonzero matrix by its mean diagonal, `trace(C)/p`, where `p` is the PCA dimension. Keep an exactly zero matrix zero and log it. Then solve:

\[
\widehat C_{\rm man}q=\lambda
(\widehat C_{\rm proc}+\widehat C_{\rm real}+10^{-3}I)q.
\]

Use a symmetric generalized eigensolver in float64; retain up to eight leading directions. A rank-deficient or zero-signal case must be flagged and evaluated with an explicitly documented fallback, never silently promoted as a learned meaningful subspace. Standardize projected coordinates using training statistics before tanh.

Freeze the fitted subspace in the primary experiment. Train the same downstream F fusion head for every Phase 4 row, including its unchanged six native evidence inputs. Keep the existing suppression artifact frozen in all rows. Do not allow an unrestricted trainable projection to immediately undo the constrained subspace and then claim the constraint caused the gain.

Mandatory controls:

| Row | Existing suppression | Additional fusion feature mapping | Purpose |
|---|---|---|---|
| P0 / F | Frozen, unchanged | F's learned 8-D projector only | **Existing suppression control**, no intervention-guided addition |
| P-PCA | Frozen, unchanged | Fixed ordinary PCA to 8-D | Fixed-subspace control |
| P-random | Frozen, unchanged | Fixed random 8-D orthonormal subspace within PCA-32 | Dimensionality/random-direction control |
| P-shuffled | Frozen, unchanged | Proposed subspace fitted with shuffled genuine partners | Pair-specificity control |
| P-guided | Frozen, unchanged | Proposed aligned-pair subspace | Tested addition |

Use the same downstream architecture, six-score bypass, data, and training budget. The four fixed-subspace rows have identical downstream trainable parameter counts. P0's learned projector adds parameters; report the difference and do not use P-guided versus P0 alone to establish the subspace mechanism. For P-random use the three prespecified seeds and report their mean, not the best. Shuffle genuine partners across different training source groups while matching available family/processing strata. No replacement-suppression arm is included in this scope.

Evaluate clean performance and processing conditions absent from fitting, for example JPEG quality 40 and resize 0.6, if these are genuinely unused in selection. Such nearby processing tests are robustness probes, not proof of unseen-generator generalization. Include leave-one-generator-out evaluation if supported.

If no valid pairs exist, skip P with an explicit reason. Do not synthesize identity-mismatched pairs or call an unpaired covariance heuristic the same experiment. Name the method **intervention-guided representation learning**, not an identified causal representation, unless a separate identifiability argument is actually established.

## 10. Phase 5: group-aware training

Run only if the Phase 2 hard gate passed, time remains after the core work, and the adaptation pool contains multiple eligible domains. This experiment explicitly uses development-data-trained heads; it is not justified as a multi-domain objective on FF++-only training. Apply it to one architecture selected by development macro AUROC, comparing `eta=0` with `eta=.25`:

\[
\mathcal L=(1-\eta)R_{\rm avg}+\eta\max_g R_g
+\text{the unchanged auxiliary/KL/weight penalties}.
\]

Use predictive NLL for `R_g`. Define groups from training dataset/domain × manipulation family; original source-video IDs remain grouping units for leakage control, not thousands of tiny optimization groups. Coarsen or exclude groups with fewer than 50 independent training sources according to a rule fixed from counts, not outcomes.

Construct each family risk from balanced fake examples of that family and an appropriate genuine reference. If genuine examples are reused across family risks, normalize their total contribution explicitly. Use balanced batches containing the eligible groups, or a documented stable group-risk estimator; do not take a noisy maximum over whichever tiny groups happen to appear in a minibatch.

Keep architecture, training duration, and loss coefficients fixed. Report the cost of robust training, including average-performance tradeoffs. Do not add this loss to I alone while comparing against A trained differently.

## 11. Training budget and reproducibility

### Hard 72-hour wall-clock cap

The clock starts when Claude Code begins this brief, including audit, coding, extraction, queue time, training, and reporting. Record UTC start/deadline in the run manifest. There is no automatic extension, clock reset after a failed job, or new backbone run.

- **Hours 0–8:** baseline/J2b/exposure audit, DeepfakeEval24/DFDC subset reservation, algebra tests, split lock, and B0/B1.
- **Hours 8–48:** matched logistic/S/F comparison, all required folds/seeds, and the hard gate.
- **Hours 48–66:** only after a gate pass, Phase 4 and mandatory controls; optional A/I/G only if core work is complete and time remains.
- **Hours 66–72:** freeze all choices, run the final difficult-dataset and broader-panel evaluation round, produce report/artifacts. No new training search after hour 66.

These are budget targets, not permission to skip correctness checks. At hour 48, an incomplete Phase 2 gate does not authorize later phases. At the deadline, checkpoint/stop this run's jobs safely, record the outcome, and report completed and skipped work with reasons. Do not terminate unrelated jobs. If the cap prevents all controls or seeds, label the comparison incomplete; no improvement claim. Reaching a failure gate early should shorten the run, not fill the remaining time with new experiments.

### Common optimization settings

Allocate the three prescribed seeds to final outer-fold fits. Use the first seed for the shared inner hyperparameter search to limit cost, applying the selected hyperparameters/epoch count to all three outer-fit seeds. Record this distinction; do not report three-seed hyperparameter searches that were never run.

Prefer the repository's established optimizer, epoch limit, and batch size. Use inner-validation **overall macro AUROC** for early stopping and selection in this run, with the declared tie-break rules. If absent, start cached-head pilots with AdamW, learning rate `1e-3`, weight decay `1e-4`, maximum 50 epochs, patience 8 on inner-validation overall macro AUROC, and batch size 256 subject to memory. These are starting implementation defaults, not established optimal settings.

Use identical stopping and tuning budgets for comparable neural models. A small shared learning-rate grid `{3e-4, 1e-3}` is sufficient for the initial comparison if a grid is needed. Count auxiliary-loss checks, reliability fitting, and failed configurations in the search log. No feature-dimension sweep is included. Do not combine every option into a Cartesian sweep.

All upstream networks stay in evaluation mode during extraction and frozen training; confirm dropout/batch-statistic behavior. Cache hashes must include preprocessing, checkpoint, layer selection, pooling, and intervention settings. Use float32 for loss/evidence arithmetic even if extraction uses mixed precision. Use float64 for algebra reference tests and eigensolvers.

Run on available authorized compute. Do not launch paid cloud capacity, new backbone fine-tuning, or unrelated infrastructure work. If resources are insufficient, complete runnable code/configuration, local checks, and an exact resume command; clearly distinguish implemented, tested, and experimentally evaluated status.

## 12. Required correctness checks

Write focused tests for mathematical and protocol risks, reusing existing tests where possible:

- Opinion/evidence round trips; valid masses; `r=1` identity; `r=0` vacuity; analytic discounting equals direct evidence scaling; invalid fixed-probability/vacuity pairs are rejected or explicitly repaired and logged.
- B1 at `gamma=1` equals B0 on the same cached inputs. PCR6+ optimized code agrees with an independent reference on ordinary, near-vacuous, and strongly conflicting opinions. Verify permutation symmetry only if the documented direct rule requires it.
- Class ordering remains consistent through losses, checkpoints, exported predictions, and metrics.
- Split checks catch a real source video and its fake/processed derivative crossing partitions. Projection fitting and normalization reject outer-evaluation samples.
- Masked pooling ignores padded elements and documents feature-pooling versus opinion-pooling differences.
- Low-rank bilinear implementation matches the explicit dense matrix `U @ V.T`; zero interactions recover A exactly; initialization allows nonzero gradients.
- New fusion parameters receive gradients; every frozen upstream parameter and existing suppression artifact remains unchanged. Verify any actually used IBDC provenance rather than assuming acceptance.
- Covariance matrices are symmetric; the generalized-eigen denominator is positive definite after ridge; eigenvectors satisfy the expected metric normalization. A small synthetic signal/nuisance example recovers the known useful direction when it exists.
- Tiny-data smoke runs save, reload, and reproduce predictions from a checkpoint and its preprocessing artifact.
- A failed/incomplete Phase 2 gate prevents scheduling Phases 3–5. The deadline prevents launching new jobs beyond the budget. Stress-test evaluation refuses to run without the final configuration lock; fitting rejects any reserved stress-test source group.

Do not add tests that merely repeat implementation statements or impose invalid fusion identities. Correctness tests do not establish empirical improvement.

## 13. Deliverables and completion criteria

Adapt names to existing repository conventions. The following is a proposed organization, not a claim about existing files or commands:

```text
configs/fusion_experiments/       # immutable configs and experiment matrix
<existing_package>/fusion/       # discount, vector, interaction, subspace modules
tests/fusion/                    # focused algebra/protocol tests
analysis/fusion_experiments/     # orchestration, evaluation, paired comparisons
outputs/fusion_experiments/<run_id>/
    config.yaml
    provenance.json
    baseline_lock.json
    data_access_manifest.json
    split_manifest.json
    gate_decision.json
    stress_test_lock.json
    metrics.json
    predictions.parquet
    training_log.jsonl
    checkpoint_and_preprocessing/
    figures/*.png
    report.md
```

Provide documented entry points for: audit; cache extraction; subspace fitting; training one configuration; evaluating a frozen checkpoint; aggregating a predeclared comparison; and resuming incomplete runs. Reuse existing CLIs rather than creating duplicates. Include actual tested commands in the final handoff, with required paths and device options. Never present a proposed command as already available.

The report must contain:

1. Baseline reproduction, exact configurations, and any departures from historical numbers.
2. A complete experiment table with dataset/family metrics, seed counts, parameter counts, search budgets, uncertainty intervals, and completion status.
3. The primary contrasts: B1−B0-matched, F−S, and F−B1-refit; conditional contrasts P-guided−P0 and fixed-subspace controls, I−A, and G−the same architecture without robust loss. Report both gate bounds and the machine-readable gate decision.
4. Genuine FPR and calibration effects alongside AUROC, including losses on reenactment or other groups.
5. The six-output bottleneck hypothesis assessed as supported, unsupported, or inconclusive, with aggregation and capacity caveats.
6. Explicit separation of development-data adaptation, historical FF++-only results, and targeted exposed-set stress testing: DeepfakeEval24 first, DFDC second, then the broader panel (including DF40 if part of the fixed panel). Include source exclusions, the J2b verdict, and frozen baseline identity. Do not claim untouched-data confirmation.
7. Negative results, validity violations, missing-data limitations, elapsed time, and each skipped phase with its gate/budget reason.
8. A recommended next experiment justified by the results, not by architectural complexity.

Mark the bounded core implementation complete when baseline preservation, required mathematical checks, Phase 0–2 runnable interfaces, and reproducibility artifacts are in place. Conditional modules skipped by the gate are intentionally outside the executed scope, not missing mandatory work. Mark an experiment evaluated only after its actual results exist. Stop broad experimentation when the next stage lacks support or resources; retain the useful implementation and a precise resume path. Do not convert an inconclusive result into a dissertation claim.

## 14. Scientific positioning

The proposed contribution to investigate is the combination of retained view representations, controlled interactions, and manipulation-sensitive/processing-stable subspace selection under source-grouped transfer evaluation. Learned stacking, bilinear fusion, EDL, and generalized eigenvectors are established building blocks. Do not claim that assembling them is automatically novel or that causal effects have been identified.

Relevant primary research to inspect before writing a novelty claim or reproducing a competing method:

- [Beyond Magnitude: Scale-Invariant Evidential Fusion for Multi-View Classification](https://proceedings.mlr.press/v306/liu26r.html).
- [Are Independently Estimated View Uncertainties Comparable? Unified Routing for Trusted Multi-View Classification](https://arxiv.org/abs/2604.09288).
- [Flexible Evidential Deep Learning — NeurIPS 2025 proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/abf0ea3ae33d1a931483e327ff8d94f8-Abstract-Conference.html).
- [Causal Representation Learning Made Identifiable by Grouping Observational Variables](https://proceedings.mlr.press/v235/morioka24a.html).
- [NeurIPS 2024 analysis of evidential deep learning](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3177be226ee12e34d6ba3b5e6fe6a5b-Abstract-Conference.html).

Read the actual method and assumptions before implementing it; these references do not specify replacement APIs or confer guarantees on this system.


## Appendix: Backup experiment for intervention-guided fusion
1. Purpose and authorization
Test whether an intervention-guided representation improves fusion even when unrestricted feature fusion F fails the Phase 2 gate.
Failure of F does not establish that a constrained representation will fail: the purpose of the constraint is to reduce reliance on nuisance variation and limit overfitting.
This appendix overrides the Phase 2 gate only for the backup experiment described below. It does not authorize reopening Phases 3 or 5, increasing backbone capacity, or repeating the earlier hyperparameter search.
Let the current experiment finish and preserve its results before starting this backup. If Phase 4 already completed with the controls below, reuse those results rather than rerunning it.
2. Scope and budget
Use a separate experiment identifier and output directory, with a maximum additional budget of 24 wall-clock hours, including preparation, fitting, evaluation, and reporting.
Keep frozen:
- The locked B0 view checkpoints and preprocessing.
- Existing nuisance suppression.
- Backbone adapters, backbone layer norms, and MR-VAE.
- The original source-grouped split assignments wherever compatible.
Use the same eligible development pool as the main experiment. DeepfakeEval24 and DFDC remain excluded from parameter fitting, preprocessing-statistic estimation, calibration, and model selection.
This experiment must be labelled:
FF++-trained frozen views; intervention-guided fusion trained on development data.

Do not introduce a learned causal graph, mixture model, reliability router, or additional backbone training.
3. Feasibility check
Before fitting anything, verify that eligible development-training data contain genuine/manipulated pairs derived from the same original content, with corresponding frames and compatible crops.
Record:
- Number of independent source groups.
- Available manipulation families.
- Pair-alignment procedure.
- Any unmatched or rejected samples.
- Whether coverage differs substantially across families or domains.
If valid pairs are unavailable, stop this backup and report the missing requirement. Do not substitute unrelated genuine videos or bring in an additional training dataset.
Pairs are required only to fit the subspace. The fusion head may train on all eligible development-training examples, including unpaired videos. Inference must require only the queried video.
4. Hypothesis and controlled interventions
The hypothesis is:
Useful fusion directions retain a consistent genuine-to-manipulated difference while responding less to controlled processing changes.

Use the previously specified conditions:
1. Identity processing.
2. JPEG quality 50.
3. Resize by 0.5, followed by the standard input resize.
Apply identical processing parameters and aligned geometry to each genuine/manipulated pair. Apply transformations consistently across frames within a clip.
For view \(v\), compute:
\[
\delta_{v,s,m,a}
=
z_v(a(x_s^m))-z_v(a(x_s^0)),
\]
where \(x_s^0\) is the genuine source, \(x_s^m\) its manipulated derivative, and \(a\) the processing condition.
Use the Phase 4 definitions of \(C_{\mathrm{man}}\), \(C_{\mathrm{proc}}\), and \(C_{\mathrm{real}}\), including their normalization and ridge term. Fit the generalized-eigenvector projection exactly as specified there:
\[
\widehat C_{\mathrm{man}}q
=
\lambda
\left(
\widehat C_{\mathrm{proc}}
+\widehat C_{\mathrm{real}}
+10^{-3}I
\right)q.
\]
Fit preprocessing and covariance statistics on the permitted training partition only. Use PCA dimension at most 32 and output dimension 8. Freeze the resulting subspace during fusion-head training.
This projection is an addition after existing nuisance suppression. It does not replace or refit that suppression.
5. Required comparison arms
Use the same downstream F architecture, including its six native evidence inputs, loss, optimizer, and training budget.
Arm	Feature mapping	Question
Existing F	Original learned projector	Does the constrained approach outperform unrestricted feature fusion?
PCA	Fixed ordinary PCA, 8 dimensions	Is dimensionality reduction sufficient?
Random	Fixed random orthonormal 8-D subspace within PCA-32	Is the result specific to the selected directions?
Shuffled pairs	Intervention-guided projection fitted using incorrect genuine partners	Does correct source pairing matter?
Manipulation-only	Leading directions of \(C_{\mathrm{man}}\), without the nuisance denominator	Does explicitly penalizing processing sensitivity help?
Intervention-guided	Full generalized-eigenvector projection	Does the proposed mechanism help?


Also retain matched B1-refit and S as reference comparators.
For shuffled pairs, select genuine partners from different source groups while matching available domain and processing strata. Keep all partners within the permitted training partition.
Use the same downstream trainable architecture across fixed-subspace arms. Report the extra projector parameters in existing F separately. Use the prescribed seeds; never select the best random subspace retrospectively.
Reuse compatible saved predictions and checkpoints from the main experiment. Refit only when the data, preprocessing, or comparison protocol differs.
6. Selection and stopping rules
Use the existing nested source-grouped protocol and three prescribed final-fit seeds. Retain overall development macro AUROC as the primary selection objective.
Do not add a new hyperparameter sweep. Reuse the main experiment’s permitted settings and selection procedure.
The backup has two separate questions:
Predictive utility: Does the intervention-guided arm improve development macro AUROC over B1-refit and the ordinary-PCA control?
Mechanism evidence: Does it outperform the shuffled-pair and manipulation-only controls while reducing sensitivity to processing changes?
Report paired source-group bootstrap intervals for all planned differences. For the two primary utility comparisons, use the same one-sided 97.5% lower-bound rule as the main brief. Report other contrasts as secondary and disclose their multiplicity.
A reduced nuisance-sensitivity score alone is insufficient: a collapsed or uninformative representation can also be insensitive.
At the budget limit, stop and report. Missing folds, seeds, or required controls make the corresponding conclusion incomplete. Do not extend the experiment to find a positive result.
7. Diagnostics that distinguish the proposed mechanism from generic regularization
On outer-assessment data, report both representation-level and prediction-level behavior.
Representation diagnostics
- Genuine-to-manipulated separation.
- Variation of the paired difference across processing conditions.
- Genuine-feature drift under processing.
- Retained feature variance and evidence of collapse.
Normalize comparisons consistently across arms; arbitrary coordinate scales must not determine the result.
Prediction diagnostics
For each genuine or fake video, measure the change in fused fake probability under processing:
\[
d_p(x,a)=
\left|p_F(a(x))-p_F(x)\right|.
\]
For aligned pairs, also measure the change in the prediction gap:
\[
d_{\mathrm{gap}}(s,m,a)=
\left|
[p_F(a(x_s^m))-p_F(a(x_s^0))]
-
[p_F(x_s^m)-p_F(x_s^0)]
\right|.
\]
Report these alongside clean and processed AUROC, NLL, genuine FPR, and vacuity changes.
Do not force unchanged confidence when processing destroys useful information. Increased uncertainty may be appropriate even when the underlying genuine/fake label is unchanged.
JPEG quality 40 and resize 0.6 may be reused as processing robustness probes. Identify them as previously exposed conditions where applicable.
8. Final difficult-dataset evaluation
Before inspecting backup-model results on DeepfakeEval24 or DFDC, lock the configurations, checkpoints, thresholds, preprocessing, and comparison list.
Evaluate the complete required comparison panel in one final round:
- DeepfakeEval24: primary stress-test dataset.
- DFDC: secondary stress-test dataset.
- The fixed broader panel: check for regressions.
All comparisons must use identical eligible videos. Report exclusions and failures consistently.
These remain previously exposed datasets. The backup is already an adaptive follow-up to the main experiment; do not describe its results as untouched confirmation. Do not tune another model after seeing its stress-test results.
9. Interpretation and deliverables
Use “intervention-guided fusion” or “processing-stable manipulation subspaces.”
Controlled processing changes support analysis of the detector’s response to those changes. They do not, by themselves, identify a latent causal graph or isolate the causal effect of manipulation from every generation-pipeline difference.
Conclude separately whether:
1. The constrained representation improves predictive performance.
2. Correct pairing contributes beyond generic dimensionality reduction.
3. The nuisance penalty contributes beyond manipulation separation alone.
4. Any benefit transfers to DeepfakeEval24 and DFDC.
If the guided arm matches PCA or shuffled pairing, report that the proposed mechanism lacks supporting evidence, even if it beats unrestricted F.
Deliver the configurations, provenance, predictions, focused correctness checks, comparison table, and a concise report. Export any figures as PNG only. Keep the dissertation and main experiment record unchanged.