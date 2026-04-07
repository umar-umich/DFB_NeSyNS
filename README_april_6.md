
Attribute Audit: Redundancy, Relevance, and Faster Alternatives

  Category 1: REMOVE — Irrelevant to Deepfake Detection (54 attributes)

  These add noise without forensic signal. A VLM saying "blonde_hair" vs "brown_hair" tells you nothing about whether the face
   is manipulated.

  ┌───────────────┬───────────────────────────────────────────────────────────────────┬───────┬───────────────────────────┐
  │     Group     │                            Attributes                             │ Count │        Why Remove         │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Hair color    │ black_hair, blonde_hair, brown_hair, red_hair, white_hair,        │ 6     │ Hair color doesn't        │
  │               │ dyed_hair                                                         │       │ indicate manipulation     │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Hair style    │ hair_parted, ponytail, bun_hair, braided_hair                     │ 4     │ Hairstyle is irrelevant   │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Eye color     │ brown_eyes, blue_eyes, green_eyes, hazel_eyes, black_eyes         │ 5     │ Color doesn't change in   │
  │               │                                                                   │       │ deepfakes                 │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Eyelashes     │ long_eyelashes, thick_eyelashes, sparse_eyelashes                 │ 3     │ Too fine-grained, LLaVA   │
  │               │                                                                   │       │ unreliable here           │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Nose detail   │ upturned_nose, long_nose, crooked_nose, flat_nose                 │ 4     │ Fine nose geometry —      │
  │               │                                                                   │       │ LLaVA is noisy            │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Ears          │ large_ears, small_ears, protruding_ears                           │ 3     │ Ears rarely visible in    │
  │               │                                                                   │       │ face crops                │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Neck          │ long_neck, short_neck                                             │ 2     │ Outside face region       │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Subjective    │ attractive                                                        │ 1     │ Subjective, no forensic   │
  │               │                                                                   │       │ value                     │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Skin texture  │ oily_skin, dry_skin, freckled_skin                                │ 3     │ Too fine-grained for      │
  │               │                                                                   │       │ LLaVA                     │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Accessories   │ round_glasses, rectangular_glasses, rimless_glasses,              │       │ Fine-grained accessory    │
  │ detail        │ thick_frame_glasses, stud_earrings, hoop_earrings,                │ 11    │ types don't help          │
  │               │ dangling_earrings, choker, pendant_necklace, hair_clip, hair_band │       │                           │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Makeup detail │ red_lipstick, pink_lipstick, nude_lipstick, mascara, blush,       │ 7     │ Too granular for forensic │
  │               │ foundation, contour_makeup                                        │       │  use                      │
  ├───────────────┼───────────────────────────────────────────────────────────────────┼───────┼───────────────────────────┤
  │ Background    │ complex_background, natural_lighting, artificial_lighting,        │       │ Background details don't  │
  │ detail        │ side_lighting, bokeh_background                                   │ 5     │ reliably indicate         │
  │               │                                                                   │       │ manipulation              │
  └───────────────┴───────────────────────────────────────────────────────────────────┴───────┴───────────────────────────┘

  Category 2: DUPLICATES to Merge (22 attributes → 11)                                                                        
   
  These are near-opposite pairs where keeping both wastes capacity. Keep one or compute their difference as a consistency     
  rule.                                                                                                                     
                                                                                                                              
  ┌─────────────────────────────┬──────────────────────────────────┬───────────────┬──────────────────────────────────────┐   
  │            Pair             │               Keep               │     Drop      │                Reason                │
  ├─────────────────────────────┼──────────────────────────────────┼───────────────┼──────────────────────────────────────┤   
  │ large_forehead /            │ neither — use landmark distance  │ both          │ InsightFace landmarks give exact     │
  │ small_forehead              │ instead                          │               │ measurement                          │
  ├─────────────────────────────┼──────────────────────────────────┼───────────────┼──────────────────────────────────────┤   
  │ thick_eyebrows /            │ thick_eyebrows                   │ thin_eyebrows │ Anti-correlated; one suffices        │
  │ thin_eyebrows               │                                  │               │                                      │   
  ├─────────────────────────────┼──────────────────────────────────┼───────────────┼──────────────────────────────────────┤
  │ bushy_eyebrows /            │ neither — overlaps with          │ both          │ Redundant with thick/thin            │   
  │ sparse_eyebrows             │ thick/thin                       │               │                                      │   
  ├─────────────────────────────┼──────────────────────────────────┼───────────────┼──────────────────────────────────────┤
  │ large_nose / small_nose     │ large_nose                       │ small_nose    │ Anti-correlated                      │   
  ├─────────────────────────────┼──────────────────────────────────┼───────────────┼──────────────────────────────────────┤   
  │ pointed_nose / broad_nose   │ broad_nose                       │ pointed_nose  │ Anti-correlated                      │
  ├─────────────────────────────┼──────────────────────────────────┼───────────────┼──────────────────────────────────────┤   
  │ full_lips / thin_lips       │ full_lips                        │ thin_lips     │ Anti-correlated                      │
  ├─────────────────────────────┼──────────────────────────────────┼───────────────┼──────────────────────────────────────┤   
  │ wide_mouth / small_mouth    │ wide_mouth                       │ small_mouth   │ Anti-correlated                      │
  ├─────────────────────────────┼──────────────────────────────────┼───────────────┼──────────────────────────────────────┤   
  │ large_eyes / small_eyes     │ large_eyes                       │ small_eyes    │ Anti-correlated                      │
  ├─────────────────────────────┼──────────────────────────────────┼───────────────┼──────────────────────────────────────┤   
  │ round_cheeks /              │ round_cheeks                     │ hollow_cheeks │ Anti-correlated                      │
  │ hollow_cheeks               │                                  │               │                                      │   
  ├─────────────────────────────┼──────────────────────────────────┼───────────────┼──────────────────────────────────────┤
  │ fair_skin / pale_skin       │ fair_skin                        │ pale_skin     │ Semantically identical               │   
  ├─────────────────────────────┼──────────────────────────────────┼───────────────┼──────────────────────────────────────┤
  │ light_makeup / no_makeup    │ no_makeup                        │ light_makeup  │ no_makeup + heavy_makeup covers      │
  │                             │                                  │               │ range                                │   
  └─────────────────────────────┴──────────────────────────────────┴───────────────┴──────────────────────────────────────┘
                                                                                                                              
  Category 3: REPLACE with Faster Extractors (48 attributes)

  These features can be extracted 100-1000x faster by specialized models. You already have InsightFace, MediaPipe, and        
  DeepFace installed.
                                                                                                                              
  ┌──────────────┬─────────────────────────────────────┬───────────────┬───────────────────────────────────┬─────────────┐    
  │   Feature    │           FaceBench Attrs           │     Count     │        Faster Alternative         │    Speed    │ 
  │    Group     │                                     │               │                                   │             │    
  ├──────────────┼─────────────────────────────────────┼───────────────┼───────────────────────────────────┼─────────────┤ 
  │ Gender       │ male, female                        │ 2             │ InsightFace app.get() → gender    │ ~5ms/frame  │ 
  ├──────────────┼─────────────────────────────────────┼───────────────┼───────────────────────────────────┼─────────────┤    
  │              │ baby_face, young_looking,           │               │ InsightFace → continuous age      │             │    
  │ Age          │ middle_aged, elderly_looking,       │ 5             │ score                             │ ~5ms/frame  │    
  │              │ age_ambiguous                       │               │                                   │             │    
  ├──────────────┼─────────────────────────────────────┼───────────────┼───────────────────────────────────┼─────────────┤ 
  │ Ethnicity    │ east_asian, african, caucasian,     │ 5             │ DeepFace                          │ ~20ms/frame │ 
  │              │ hispanic, south_asian               │               │ analyze(actions=['race'])         │             │    
  ├──────────────┼─────────────────────────────────────┼───────────────┼───────────────────────────────────┼─────────────┤ 
  │              │ neutral_expression, happy, sad,     │               │ DeepFace                          │             │    
  │ Expression   │ angry, surprised, fearful,          │ 8             │ analyze(actions=['emotion']) or   │ ~20ms/frame │ 
  │              │ disgusted, contemptuous             │               │ HSEmotion                         │             │    
  ├──────────────┼─────────────────────────────────────┼───────────────┼───────────────────────────────────┼─────────────┤ 
  │ Action Units │ All 25 AUs                          │ 25            │ py-feat or LibreFace (CVPR 2024)  │ ~15ms/frame │    
  ├──────────────┼─────────────────────────────────────┼───────────────┼───────────────────────────────────┼─────────────┤ 
  │ Face shape / │ oval_face, round_face, square_face, │ 6 → derive    │ MediaPipe 468 landmarks → compute │             │    
  │  geometry    │  heart_shaped_face, long_face,      │ from          │  ratios                           │ ~3ms/frame  │    
  │              │ diamond_face                        │ landmarks     │                                   │             │
  ├──────────────┼─────────────────────────────────────┼───────────────┼───────────────────────────────────┼─────────────┤    
  │              │ strong_jaw, narrow_jaw,             │ 5 → derive    │ MediaPipe landmarks → jaw angles, │             │
  │ Structural   │ pointed_chin, round_chin,           │ from          │  chin ratios                      │ ~3ms/frame  │    
  │              │ double_chin                         │ landmarks     │                                   │             │
  └──────────────┴─────────────────────────────────────┴───────────────┴───────────────────────────────────┴─────────────┘    
                                                            
  FaceBench: ~500ms/frame (LLM mode) or ~50ms/frame (vision-only mode)                                                        
  InsightFace + DeepFace + MediaPipe: ~30ms/frame total for ALL the above
                                                                                                                              
  Category 4: KEEP in FaceBench — Requires VLM Understanding (40 attributes)                                                  
                                                                                                                              
  These genuinely need VLM-level holistic understanding that specialized models can't provide:                                
                                                            
  ┌──────────────┬─────────────────────────────────────────────────────────┬───────┬──────────────────────────────────────┐   
  │    Group     │                       Attributes                        │ Count │             Why VLM-only             │
  ├──────────────┼─────────────────────────────────────────────────────────┼───────┼──────────────────────────────────────┤   
  │ Facial hair  │ beard, mustache, goatee, sideburns, stubble,            │ 6     │ Fine-grained facial hair detection   │
  │              │ clean_shaven                                            │       │ requires holistic face understanding │
  ├──────────────┼─────────────────────────────────────────────────────────┼───────┼──────────────────────────────────────┤   
  │ Makeup       │ heavy_makeup, lipstick, eyeshadow, eyeliner             │ 4     │ Makeup presence/type needs global    │
  │              │                                                         │       │ context                              │   
  ├──────────────┼─────────────────────────────────────────────────────────┼───────┼──────────────────────────────────────┤
  │ Skin         │ smooth_skin, wrinkled_skin, acne, moles, scars,         │ 7     │ Skin condition requires detailed     │   
  │ condition    │ skin_blemishes, age_spots                               │       │ visual analysis                      │   
  ├──────────────┼─────────────────────────────────────────────────────────┼───────┼──────────────────────────────────────┤
  │              │ narrow_eyes, wide_set_eyes, double_eyelid,              │       │                                      │   
  │ Eye details  │ single_eyelid, hooded_eyes, bags_under_eyes,            │ 8     │ Eye morphology details               │
  │              │ dark_circles, puffy_eyes                                │       │                                      │   
  ├──────────────┼─────────────────────────────────────────────────────────┼───────┼──────────────────────────────────────┤
  │ Symmetry     │ symmetrical_face, asymmetrical_face                     │ 2     │ Holistic symmetry judgment           │
  ├──────────────┼─────────────────────────────────────────────────────────┼───────┼──────────────────────────────────────┤
  │ Hair state   │ gray_hair, receding_hairline, bald, bangs               │ 4     │ Age-linked hair features             │
  ├──────────────┼─────────────────────────────────────────────────────────┼───────┼──────────────────────────────────────┤   
  │ Accessories  │ eyeglasses, sunglasses, hat, face_mask, headphones      │ 5     │ Occlusion detection                  │
  ├──────────────┼─────────────────────────────────────────────────────────┼───────┼──────────────────────────────────────┤   
  │ Image        │ blurry_image, sharp_image                               │ 2     │ LLaVA's blur assessment complements  │
  │ quality      │                                                         │       │ DCT                                  │   
  ├──────────────┼─────────────────────────────────────────────────────────┼───────┼──────────────────────────────────────┤
  │ Lighting     │ bright_lighting, dim_lighting                           │ 2     │ Lighting condition affects forgery   │   
  │              │                                                         │       │ visibility                           │   
  └──────────────┴─────────────────────────────────────────────────────────┴───────┴──────────────────────────────────────┘
                                                                                                                              
  Category 5: ADD — Missing Features Critical for Deepfake Detection

  These are NOT in FaceBench but are highly relevant forensically and extractable with fast libraries:                        
   
  ┌────────────────────────────┬────────────┬────────────────────────────┬────────────────────────────────────────────────┐   
  │          Feature           │ Dimensions │         Extractor          │                 Why It Matters                 │
  ├────────────────────────────┼────────────┼────────────────────────────┼────────────────────────────────────────────────┤   
  │ Head pose (yaw, pitch,     │ 3          │ InsightFace / 6DRepNet     │ Faceswap often introduces subtle pose          │
  │ roll)                      │            │                            │ inconsistencies between face and context       │   
  ├────────────────────────────┼────────────┼────────────────────────────┼────────────────────────────────────────────────┤   
  │ Gaze direction (x, y per   │ 4          │ MediaPipe / L2CS-Net       │ Deepfakes frequently have inconsistent gaze    │
  │ eye)                       │            │                            │                                                │   
  ├────────────────────────────┼────────────┼────────────────────────────┼────────────────────────────────────────────────┤
  │ Face detection confidence  │ 1          │ InsightFace det_score      │ Fakes often have lower detection confidence    │   
  ├────────────────────────────┼────────────┼────────────────────────────┼────────────────────────────────────────────────┤   
  │ Landmark stability         │ 5          │ MediaPipe (temporal: std   │ Deepfakes have jittery landmarks across frames │
  │ (per-region jitter)        │            │ across N frames)           │                                                │   
  ├────────────────────────────┼────────────┼────────────────────────────┼────────────────────────────────────────────────┤   
  │ Eye aspect ratio (L/R)     │ 2          │ MediaPipe landmarks        │ L/R eye asymmetry is a deepfake indicator      │
  ├────────────────────────────┼────────────┼────────────────────────────┼────────────────────────────────────────────────┤   
  │ Mouth aspect ratio         │ 1          │ MediaPipe landmarks        │ Abnormal mouth shapes in fakes                 │   
  ├────────────────────────────┼────────────┼────────────────────────────┼────────────────────────────────────────────────┤
  │ Face-to-image area ratio   │ 1          │ InsightFace bbox           │ Scale consistency                              │   
  ├────────────────────────────┼────────────┼────────────────────────────┼────────────────────────────────────────────────┤
  │ 3DMM fitting residual      │ 1          │ DECA / InsightFace         │ How well does the face fit a 3D model? Fakes   │   
  │                            │            │                            │ fit poorly                                     │   
  ├────────────────────────────┼────────────┼────────────────────────────┼────────────────────────────────────────────────┤   
  │ Inter-pupillary distance   │ 1          │ MediaPipe landmarks        │ Structural consistency                         │   
  │ ratio                      │            │                            │                                                │
  ├────────────────────────────┼────────────┼────────────────────────────┼────────────────────────────────────────────────┤
  │ Jaw symmetry score         │ 1          │ MediaPipe landmarks        │ Fakeswap creates asymmetric jaws               │
  └────────────────────────────┴────────────┴────────────────────────────┴────────────────────────────────────────────────┘   
   
  ---                                                                                                                         
  Proposed Refined Feature Vector                           
                                                                                                                              
  Tier A: Fast-extracted features (precompute with InsightFace + MediaPipe + DeepFace)
                                                                                                                              
  # Demographics (InsightFace + DeepFace) — 4 features
  gender_score              # continuous [-1, 1] from InsightFace (more useful than binary)                                   
  age_score                 # continuous [0, 100] from InsightFace                                                            
  ethnicity_scores          # 5 softmax scores → keep top-1 confidence + entropy = 2 features                                 
                                                                                                                              
  # Expression (DeepFace / HSEmotion) — 8 features          
  emotion_scores[8]         # neutral, happy, sad, angry, surprise, fear, disgust, contempt                                   
                                                                                                                              
  # Action Units (LibreFace or py-feat) — 20 features
  au_intensities[20]        # AU1,2,4,5,6,7,9,10,12,14,15,17,20,23,24,25,26,28,43,45                                          
                                                                                                                              
  # Geometry from landmarks (MediaPipe 468 points) — 15 features                                                              
  head_pose[3]              # yaw, pitch, roll                                                                                
  gaze_direction[4]         # x,y per eye                                                                                     
  eye_aspect_ratio[2]       # left, right                   
  mouth_aspect_ratio[1]                                                                                                       
  interpupillary_ratio[1]                                                                                                     
  jaw_symmetry[1]
  face_area_ratio[1]                                                                                                          
  face_width_height_ratio[1]                                                                                                  
  chin_angle[1]
                                                                                                                              
  # Quality / confidence — 3 features                                                                                         
  det_confidence[1]         # face detection confidence
  landmark_confidence[1]    # MediaPipe face mesh confidence                                                                  
  blur_score[1]             # Laplacian variance (already in forensic features)
                                                                                                                              
  Total Tier A: ~50 features, precomputed at ~30ms/frame                                                                      
                                                                                                                              
  Tier B: VLM-only features (keep in FaceBench, precompute) — 40 features                                                     
                                                            
  The features from Category 4 above. These are the ones that genuinely need VLM holistic understanding and can't be reduced  
  to landmark geometry or a specialized classifier.         
                                                                                                                              
  Tier C: Consistency Rules (computed at training time, no precomputation) — 20 features                                      
   
  Keep your existing consistency rules — they're well designed. They operate on the combined Tier A + B features.             
                                                            
  Tier D: Forensic pixel features (already precomputed) — 30 features                                                         
                                                            
  Keep as-is. These are good.                                                                                                 
                                                            
  ---
  Summary: 211 → ~140 Features, 3x Faster Precomputation
                                                                                                                              
  ┌───────────────────────────┬───────────────────────────┬──────────────────────────────────────────┐
  │                           │          Before           │                  After                   │                        
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────┤
  │ FaceBench features        │ 211 (all from slow LLaVA) │ 40 (VLM-only)                            │                        
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────┤
  │ Fast-extracted            │ 0                         │ ~50 (InsightFace + MediaPipe + DeepFace) │                        
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────┤                        
  │ Consistency rules         │ 20                        │ 20 (unchanged)                           │                        
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────┤                        
  │ Forensic pixel            │ 30                        │ 30 (unchanged)                           │
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────┤                        
  │ Total                     │ 261                       │ ~140                                     │
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────┤                        
  │ FaceBench precompute time │ ~500ms/frame              │ ~50ms/frame (40 attrs only)              │
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────┤                        
  │ Fast model precompute     │ N/A                       │ ~30ms/frame                              │
  ├───────────────────────────┼───────────────────────────┼──────────────────────────────────────────┤                        
  │ Total precompute          │ ~500ms/frame              │ ~80ms/frame (6x faster)                  │
  └───────────────────────────┴───────────────────────────┴──────────────────────────────────────────┘                        
                                                            
  Key Wins                                                                                                                    
                                                            
  1. No duplicates — every feature carries unique signal                                                                      
  2. Head pose + gaze — the biggest missing features for deepfake detection, now included
  3. Continuous scores — age=34 and gender_score=0.92 are more informative than binary young/old male/female                  
  4. Much faster precomputation — 6x speedup, critical for augmented data                                                     
  5. Better AU coverage — dedicated AU models (LibreFace/py-feat) are much more accurate than LLaVA for AUs                   
  6. Leaner for causal graphs — fewer, more meaningful nodes = better graph discovery                                         
                                                                                                                              
  What to Do Next                                                                                                             
                                                                                                                              
  1. Write a unified precomputation script that runs InsightFace + MediaPipe + DeepFace on all frames and saves a single      
  fast_semantic/{video_id}.pt with shape (n_frames, 50)
  2. Trim FaceBench to the 40 VLM-only attributes and re-extract (or just index into existing 211-d vectors at load time)     
  3. Update CAUSAL_ATTRIBUTE_NAMES to use the new combined feature indices                                                    
  4. Update consistency rules to work with the new feature layout


Here's a summary of the changes:

  preprocessing/precompute_fast_semantic.py:
  - Added VLMSemanticExtractor class that wraps SemanticPrecomputer from precompute_semantic_features.py, overriding
  _get_attr_names() to return only the 64 VLM attributes from refined_attributes.py instead of all 211
  - Uses the same teacher-forcing approach: each attribute gets a clean yes/no prompt, batched through padded LLM forward
  passes — just 64 attrs instead of 211 (1 pass at attr_batch_size=64 vs 2+ passes)
  - main() now runs both extractors per video and concatenates: [fast(58) || vlm(64)] = 122-d
  - New CLI args: --no_vlm (skip VLM, fast-only), --vlm_attr_batch_size, --vlm_image_batch_size, --load_4bit, --load_8bit
  - Output .pt files now contain fast_dim and vlm_dim metadata

  augment_and_precompute.sh:
  - Step 2 passes --load_4bit and VLM batch size args to each GPU job

  Output format per video:
  {'features':      Tensor(n_frames, 122),   # [fast(58) || vlm(64)]
   'frame_paths':   List[str],
   'feature_names': List[str],               # 122 names
   'fast_dim':      58,
   'vlm_dim':       64}

  claude --resume 25b31781-3bbc-4e7e-af18-22ab43c328fc       
                                                          