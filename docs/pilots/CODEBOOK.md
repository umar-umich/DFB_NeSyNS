# Predicate codebook and Verified Evidence Graph schema, revision 3

Revision 3 change. Verified confidence normalization constants are refit on a calibration split disjoint from evaluation data before headline numbers.

This is the fixed alphabet. It is three things at once. The vocabulary the VLM may output. The node types of the Verified Evidence Graph. The symbols the logic program reasons over. Freeze it before both pilots so nothing is redesigned mid experiment.

Revision 2 changes. A whole_face global region is added because generator fingerprints are often image wide and a spatially local gate would systematically reject them. Frequency and noise predicates may bind to whole_face and are verified by spectral interventions, not spatial repair. The node schema gains intervention_type, co_verified_with, and control offsets. Verified confidence is redefined as a normalized effect size against the real repair offset distribution. A landmarking failure policy is fixed. face_boundary interventions are masked to the face side of the seam.

Design commitments already settled.
- Predicates are bound to deterministic facial landmark regions. No free text, no grounding model on the critical path.
- Nodes are structured records with a generous named attribute list. Prune by ablation later, since dropping is free and adding means re running verification.
- Evidence enters the reasoner as probability, not Boolean, so the proof carries a confidence. The confidence is calibrated by measurement, never by construction.

---

## 1. Node schema

Every node is one instance of one predicate type over one region. Fields below.

- predicate_type. One entry from the predicate table in section 3.
- region_id. One entry from the landmark region table in section 2. Global predicates use whole_face.
- region_box. The deterministic bounding box from landmarking, pixel coordinates. Null for whole_face.
- intervention_type. Which intervention verified this node. One of gt_region_swap, diffusion_repair, spectral_notch, checkerboard_suppress, residual_renormalize.
- verified. Boolean, whether the node passed the counterfactual gate. Nodes that fail are dropped from the graph but logged for the mask overlap of rejected versus confirmed.
- verified_confidence. Probability in zero to one. A normalized effect size, the calibrated p(fake) drop on the fake normalized against the real repair offset distribution for the same intervention type and the matched corruption reference point. Normalization constants are refit on a calibration split disjoint from any evaluation data before any headline attribution numbers are produced. Pilot measured constants never score evaluation data. This is the fact probability the reasoner consumes.
- co_verified_with. List of node ids verified by the same spatial intervention on the same region. Spatial repair verifies all predicates on a region jointly, so this field makes the confound explicit and reportable. Empty for spectrally intervened predicates, which are verified individually.
- control_offsets. The measured wrong region drop and real repair delta recorded alongside the node, for audit.
- attributes. The typed attribute record for that predicate type, from section 3. Each attribute is tagged Boolean or probabilistic.
- provenance. Which VLM or deterministic probe proposed it, which detector verified it, timestamp. Needed for the three entity independence argument if an adapter is ever added.

Landmarking failure policy. If landmark confidence falls below the frozen threshold on a frame, the frame emits a single abstention node and no predicates are proposed. This is decided now so it is not an ad hoc call during a run.

---

## 2. Landmark regions

Bound to a standard 68 or 468 point landmarking scheme. Keep the region set small and anatomically named so a human can read a proof.

- left_eye
- right_eye
- inter_ocular, the bridge and glabella between the eyes
- nose
- nasolabial, the folds from nose to mouth corners
- mouth
- jawline
- left_cheek
- right_cheek
- forehead
- hairline
- chin
- face_boundary, the outer blend seam between face and background or neck. Interventions here are masked strictly to the face side of the seam so background pixels are never altered and detector changes cannot be background driven.
- whole_face, global region for image wide fingerprints. Not a spatial repair target. Verified only by spectral interventions.

---

## 3. Predicate types and their attributes

Grouped by the forensic family they probe. Each predicate lists its named attributes and the Boolean or probabilistic tag. Attribute lists are intentionally generous.

### 3.1 Blending and boundary predicates

Spatially localized. Verified by region repair, ground truth swap where a paired real frame exists, diffusion repair otherwise.

boundary_artifact
- artifact_strength, probabilistic
- spatial_extent, probabilistic, fraction of region perimeter affected
- gradient_sharpness, probabilistic, edge transition steepness
- symmetry_break, probabilistic, left right asymmetry of the seam
- co_located_with_landmark, Boolean

blend_seam_visible
- seam_contrast, probabilistic
- seam_continuity, probabilistic
- orientation_consistency, probabilistic

### 3.2 Frequency and noise predicates

May bind to any region or to whole_face. When bound to whole_face, verified by predicate targeted spectral interventions, notch the cited band, suppress the checkerboard peak, renormalize the residual, each followed by re running the detector. These predicates are verified individually, so predicate level necessity is claimed only here.

frequency_anomaly
- hf_energy_ratio, probabilistic, high frequency over total
- dominant_band, probabilistic, normalized peak frequency
- band_concentration, probabilistic, how peaked the spectrum is
- checkerboard_score, probabilistic, periodic upsampling signature
- radial_spectrum_slope, probabilistic

noise_inconsistency
- residual_std, probabilistic, SRM style residual dispersion
- cross_region_noise_gap, probabilistic, this region versus a reference region
- noise_periodicity, probabilistic

### 3.3 Geometry and identity predicates

Spatially localized, deterministic probes.

geometry_inconsistency
- symmetry_deviation, probabilistic
- proportion_deviation, probabilistic, against golden ratio priors
- landmark_displacement, probabilistic
- pose_geometry_mismatch, probabilistic

identity_drift
- id_embedding_distance, probabilistic, region identity versus whole face identity
- inner_outer_id_gap, probabilistic, inner face versus outer face identity
- temporal_id_stability, probabilistic, reserved for the video extension, tag inactive for the image version

### 3.4 Texture and physiology predicates

Spatially localized, VLM proposed. VLM proposal recall against ground truth masks is measured in the verification pilot. If recall is near zero these predicates are droppable without harming the framework.

texture_anomaly
- skin_texture_regularity, probabilistic
- pore_detail_loss, probabilistic
- specular_consistency, probabilistic, highlight plausibility

physiology_violation
- eye_reflection_mismatch, probabilistic, catchlight consistency across eyes
- teeth_regularity, probabilistic
- iris_shape_plausibility, probabilistic

### 3.5 Global context predicate

lighting_inconsistency
- shadow_direction_gap, probabilistic
- color_temperature_gap, probabilistic
- global_local_light_mismatch, probabilistic

---

## 4. Relation types

Edges of the graph. These carry the interactions the rules reason over.

- co_located(a, b). Two predicates over the same or adjacent regions.
- co_verified(a, b). Two predicates admitted by the same spatial intervention. Structural, derived from co_verified_with. Rules may not treat co verified predicates as independent corroboration.
- supports(node, family). This evidence raises the probability of a family.
- contradicts(node, family). This evidence lowers the probability of a family.
- causes(a, b). One artifact type mechanistically implies another, for example boundary_artifact causes identity_drift in a swap.
- corroborates(a, b). Two independent forensic families agree in the same region, a strong signal since it is cross method agreement. Independence requires that a and b are not co_verified.

Relation confidences are probabilistic where the reasoner supports it.

---

## 5. Generator families for attribution

The reasoning targets. Bind the DF40 grouping to these before the pilot and pre register it in writing before any results exist. Default coarse grouping.

- face_swap
- face_reenactment
- diffusion_edit
- entire_face_synthesis

Assignment principle for boundary cases. Family is defined by manipulation scope, not generative backbone. A diffusion based face swap is face_swap. Log the rationale per generator.

Hold out at least one generator per family for the transfer test. The claim scope is unseen generators within known families. A genuinely novel family is unattributable by construction and abstention is the designed behavior for it.

---

## 6. Example rule, for illustration only

Rules are written over verified nodes and their attributes.

Verified(boundary_artifact, region in face_boundary or jawline) with high artifact_strength
AND Verified(identity_drift) with high inner_outer_id_gap
AND absence of Verified(frequency_anomaly, whole_face) with high checkerboard_score
supports family face_swap and contradicts family diffusion_edit.

A new generator that breaks this can be handled by editing this one rule. Rule edits are evaluated in a measured study, pre registered predicted behavior change, accuracy recovered, side effect rate on other families, and human time, against few shot fine tuning of the same input MLP.

---

## 7. Boolean versus probabilistic summary

Almost everything is probabilistic, since the whole point of using Scallop or DeepProbLog is to carry confidence through to the attribution. The only Boolean fields are structural, verified and co_located_with_landmark. If the pilot shows a probabilistic field adds no separation, demote it in ablation.
