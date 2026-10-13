# Orchestrator decision log

```
[2026-10-09T23:26:11-04:00] === orchestrator start ===
[2026-10-09T23:26:11-04:00] GPUs: 0, 46948 MiB 1, 69596 MiB 2, 40246 MiB 3, 36415 MiB 
[2026-10-09T23:28:42-04:00] === orchestrator start ===
[2026-10-09T23:28:43-04:00] GPUs: 0, 46948 MiB 1, 69596 MiB 2, 40246 MiB 3, 34985 MiB 
[2026-10-09T23:54:28-04:00] S IN-LOOP SMOKE: PASS, recorded on reaching the EXPORT stage rather than DONE.
[2026-10-09T23:54:28-04:00]   The rehearsal was a child of the ssh session and dies with it, mid-export. It had
[2026-10-09T23:54:28-04:00]   already cleared every stage this patch touches: the timestamp table reached the
[2026-10-09T23:54:28-04:00]   temporal stage (no slot-number fallback warning was printed), SeqTemporal and the
[2026-10-09T23:54:28-04:00]   VAE loss ran, a full epoch scored on WHOLE dev sets (v3.1 0.5595), and a checkpoint
[2026-10-09T23:54:28-04:00]   was selected. The export that follows is Dr Kutub's unmodified code, which the
[2026-10-09T23:54:28-04:00]   P+OM smoke ran to completion ("done in 23 min", exit 0). S in-loop is cleared.
[2026-10-09T23:54:33-04:00] === orchestrator start ===
[2026-10-09T23:54:33-04:00] GPUs: 0, 46948 MiB 1, 130350 MiB 2, 36726 MiB 3, 34985 MiB 
[2026-10-10T00:24:45-04:00] still waiting on spatial depth runs (b12/b8/b16 seed 1) (30 min)
[2026-10-10T00:38:21-04:00] === orchestrator start ===
[2026-10-10T00:38:21-04:00] GPUs: 0, 46948 MiB 1, 93962 MiB 2, 75214 MiB 3, 34985 MiB 
[2026-10-10T01:08:33-04:00] still waiting on spatial depth runs (b12/b8/b16 seed 1) (30 min)
[2026-10-10T01:13:06-04:00] CONCAT SMOKE: PASS -- ran to completion ("done in 34 min", exit 0), epoch 0 dev 0.5128.
[2026-10-10T01:13:06-04:00]   Its watcher had DEADLOCKED on itself: the watcher polled pgrep -f "om-mode concat
[2026-10-10T01:13:06-04:00]   --smoke", and its own parent shell carried that string in its command line, so the
[2026-10-10T01:13:06-04:00]   watcher waited for a process that was waiting for the watcher. Marker written by
[2026-10-10T01:13:06-04:00]   hand from the log; wait_for in this orchestrator now counts only python processes.
[2026-10-10T01:13:25-04:00] === orchestrator start ===
[2026-10-10T01:13:25-04:00] GPUs: 0, 46948 MiB 1, 93962 MiB 2, 14 MiB 3, 36415 MiB 
[2026-10-10T01:43:44-04:00] still waiting on spatial depth runs (b12/b8/b16 seed 1) (30 min)
[2026-10-10T02:14:01-04:00] still waiting on spatial depth runs (b12/b8/b16 seed 1) (60 min)
[2026-10-10T02:44:19-04:00] still waiting on spatial depth runs (b12/b8/b16 seed 1) (90 min)
[2026-10-10T03:14:37-04:00] still waiting on spatial depth runs (b12/b8/b16 seed 1) (120 min)
[2026-10-10T03:21:40-04:00] spatial depth runs (b12/b8/b16 seed 1) is finished
[2026-10-10T03:21:41-04:00] P+OM and its control is finished
[2026-10-10T03:21:41-04:00] committed: fusion: depth probes and the P off-manifold runs
[2026-10-10T03:21:41-04:00] depth verdict:
    PE spatial trunk, off-manifold with the encoder in the loop (full dev, residual arm):
      b8    0.7831
      b12   0.8053
      b16   0.8673
      last  0.8664
      b20   0.8613
      spread across depths: 0.0842
    
    block 16: seed 0 0.8673, seed 1 0.8547, |difference| 0.0126
    SEED NOISE (0.0126) EXCEEDS THE WHOLE DEPTH SPREAD (0.0060).
    The depth ordering is not measurable at one seed per cell. No further depth run is
    launched; reporting depth as indistinguishable is the honest result, and any future
    depth claim needs several seeds per cell rather than another layer.
[2026-10-10T03:21:41-04:00] launching P+OM (concat: [f_t ; f_a]) on GPU 3 -- the DISCERN-faithful construction
[2026-10-10T05:41:35-04:00] committed: fusion: the off-manifold residual as a second view rather than a replacement
[2026-10-10T05:41:35-04:00] launching S in-loop (real-timestamp temporal stage) on GPU 3
[2026-10-10T06:49:19-04:00] S in-loop FAILED (see logs/branchfusion/lanes/om_S_inloop.log)
[2026-10-10T07:18:35-04:00] S control FAILED
[2026-10-10T07:18:35-04:00] committed: fusion: branch S with the off-manifold front end and its control
[2026-10-10T07:18:35-04:00] completing the block-16 test-set extraction
[2026-10-10T07:18:47-04:00] committed: fusion: opinions for the encoder-in-loop off-manifold branch
[2026-10-10T07:18:48-04:00] gate verdict:
    Temporal off-manifold runs, PE-Core trunk, block 16 (development v3.1):
    
      ombvae_b16_cat                     off-manifold   0.8296  (epoch 9)
      ombvae_b16                         off-manifold   0.4942  (epoch 2)
    
    P16om: ombvae_b16_cat
      GATE 1 cannot be evaluated -- its no-OM control has not finished. HELD.
    P16om: ombvae_b16
      GATE 1 cannot be evaluated -- its no-OM control has not finished. HELD.
[2026-10-10T07:18:48-04:00] no branch passed both gates; the screen is NOT re-run and the deployed selection stands
[2026-10-10T07:19:41-04:00] committed: fusion: final correlation matrix, tables and the off-manifold campaign report
[2026-10-10T07:19:41-04:00] === orchestrator done ===
[2026-10-10T07:19:41-04:00] read reports/branch_fusion/OM_CAMPAIGN.md first; the decision log is logs/branchfusion/orchestrator/orchestrator.log
[2026-10-10T13:11:00-04:00] rerun: om_P_b16_control_v2
[2026-10-10T13:14:45-04:00] === finisher waiting on the rerun lane ===
[2026-10-10T14:15:43-04:00] finisher still waiting (60 min)
[2026-10-10T15:17:33-04:00] finisher still waiting (120 min)
[2026-10-10T15:35:22-04:00] om_P_b16_control_v2 FAILED again -- see logs/branchfusion/lanes/om_P_b16_control_v2.log
[2026-10-10T15:35:22-04:00] rerun: om_S_inloop_v2
[2026-10-10T16:17:58-04:00] finisher still waiting (180 min)
[2026-10-10T17:18:17-04:00] finisher still waiting (240 min)
[2026-10-10T18:18:36-04:00] finisher still waiting (300 min)
[2026-10-10T19:18:55-04:00] finisher still waiting (360 min)
[2026-10-10T19:27:21-04:00] om_S_inloop_v2 FAILED again -- see logs/branchfusion/lanes/om_S_inloop_v2.log
[2026-10-10T19:27:21-04:00] rerun: om_S_control_v2
[2026-10-10T20:19:13-04:00] finisher still waiting (420 min)
[2026-10-10T21:06:20-04:00] v3 start: om_P_b16_cat_v3 on GPU 1
[2026-10-10T21:06:20-04:00] v3 start: om_S_b16_cat_v3 on GPU 0
[2026-10-10T21:06:20-04:00] v3 start: om_S_b16_control_v3 on GPU 2
[2026-10-10T21:06:20-04:00] v3 start: om_P_b16_control_v3 on GPU 3
[2026-10-10T21:06:23-04:00] === v3 finisher waiting ===
[2026-10-10T22:06:40-04:00] v3 finisher still waiting (60 min)
[2026-10-10T23:06:59-04:00] v3 finisher still waiting (120 min)
[2026-10-11T00:07:17-04:00] v3 finisher still waiting (180 min)
[2026-10-11T00:40:52-04:00] om_S_b16_cat_v3 DONE
[2026-10-11T01:07:41-04:00] v3 finisher still waiting (240 min)
[2026-10-11T01:37:28-04:00] om_P_b16_control_v3 DONE
[2026-10-11T02:05:08-04:00] om_P_b16_cat_v3 DONE
[2026-10-11T02:08:34-04:00] v3 finisher still waiting (300 min)
[2026-10-11T02:10:26-04:00] om_S_b16_control_v3 DONE
[2026-10-11T02:10:26-04:00] === v3 LANE DONE ===
[2026-10-11T02:10:35-04:00] v3 lane finished
[2026-10-11T02:10:45-04:00] gate verdict:
    Temporal off-manifold runs, PE-Core trunk, block 16 (development v3.1):
    
      ombvae_b16_cat                     off-manifold   0.8409  (epoch 9)
      ombvae_b16_cat_seq                 off-manifold   0.8497  (epoch 9)
      omnone_b16                         NO-OM CONTROL  0.8678  (epoch 9)
      omnone_b16_seq                     NO-OM CONTROL  0.8665  (epoch 6)
    
    P16om: ombvae_b16_cat
      gate 1  0.8409 vs control 0.8678  -> FAIL
      gate 2  phi vs A +0.415 < 0.6  -> PASS
      P16om is reported, not registered: a branch that fails either gate would either duplicate A or be worse than the same network without the residual.
    S16om: ombvae_b16_cat_seq
      gate 1  0.8497 vs control 0.8665  -> FAIL
      gate 2  phi vs A +0.500 < 0.6  -> PASS
      S16om is reported, not registered: a branch that fails either gate would either duplicate A or be worse than the same network without the residual.
[2026-10-11T02:10:45-04:00] no branch passed both gates; the deployed selection stands
[2026-10-11T02:10:54-04:00] committed: fusion: the PE-Core off-manifold campaign re-run with the residual in the optimiser
[2026-10-11T02:10:54-04:00] === v3 finisher done ===
[2026-10-11T10:49:36-04:00] seed replicate start: om_P16ctl_seed1 on GPU 1
[2026-10-11T13:18:51-04:00] === robustness lane: Y16 vs A under jpeg:40 ===
[2026-10-11T13:18:53-04:00] robustness: A extract failed on VALmix
[2026-10-11T13:19:46-04:00] === robustness lane: Y16 vs A under jpeg:40 ===
[2026-10-11T13:54:27-04:00] === robustness lane done -- see reports/branch_fusion/ROBUSTNESS_Y16.md ===
[2026-10-11T14:13:20-04:00] === all-14 per-dataset grid: 16383 subsets across 32 shards ===
[2026-10-11T14:15:43-04:00] === robustness lane (full): both conditions, development and test ===
[2026-10-11T14:18:06-04:00] all-14 grid: shards done, merging
[2026-10-11T14:18:08-04:00] all-14 grid MERGE FAILED -- see logs/branchfusion/lanes/grid_all14_merge.log
[2026-10-11T14:18:37-04:00] === all-16 per-dataset grid: 65535 subsets across 64 shards ===
[2026-10-11T14:28:46-04:00] === grid switch: waiting for P16ctlB to register ===
[2026-10-11T14:44:21-04:00] grid switch: still waiting for P16ctlB (15 min)
[2026-10-11T14:55:20-04:00] === robustness (parallel): 52 passes across GPUs 1, 3 and 0 ===
[2026-10-11T14:59:56-04:00] grid switch: still waiting for P16ctlB (30 min)
[2026-10-11T15:15:35-04:00] grid switch: still waiting for P16ctlB (45 min)
[2026-10-11T15:27:22-04:00] all-16 grid retired EARLY: its rows are discarded by the 17-branch switch anyway, and at 4 subsets/min under extraction contention it was burning 64 cores for nothing
[2026-10-11T15:31:11-04:00] grid switch: still waiting for P16ctlB (60 min)
[2026-10-11T15:34:33-04:00] grid switch STOPPED: the all-subsets grid is not computable -- fit_weights searches 5^k - 4^k weight vectors, so all 2^17-1 subsets is 3024 core-years
[2026-10-11T15:40:06-04:00] === final grid: waiting for P16ctlB ===
[2026-10-11T15:55:53-04:00] final grid: still waiting for P16ctlB (15 min)
[2026-10-11T16:11:28-04:00] final grid: still waiting for P16ctlB (30 min)
[2026-10-11T16:27:01-04:00] final grid: still waiting for P16ctlB (45 min)
[2026-10-11T16:40:43-04:00] om_P16ctl_seed1 DONE
[2026-10-11T16:41:33-04:00] final grid: P16ctlB registered; 17 branches
[2026-10-11T16:41:33-04:00] grid all17eq: sizes 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17, weight grid equal, 128 shards
[2026-10-11T16:43:10-04:00] === om_P16ctl_seed1 lane done ===
[2026-10-11T16:53:53-04:00] robustness (parallel): extraction done; exporting Y16 opinions
[2026-10-11T16:54:34-04:00] === robustness (parallel) done ===
[2026-10-12T16:27:43-04:00] grid all17eq MERGE FAILED -- see logs/branchfusion/lanes/grid_all17eq_merge.log
[2026-10-12T16:27:43-04:00] grid all17: sizes 1,2,3,4,5,6, weight grid locked, 128 shards
[2026-10-12T17:16:57-04:00] packing DeepSpeak-2.0
[2026-10-12T17:18:20-04:00] DeepSpeak-2.0 packed
[2026-10-12T17:18:20-04:00] packing DeepSpeak-1.1
[2026-10-12T17:20:33-04:00] DeepSpeak-1.1 packed
[2026-10-12T17:20:33-04:00] === GenD packing done ===
[2026-10-12T17:26:22-04:00] === O CE export: artifact view from the simple-CE run, 16 datasets ===
[2026-10-12T17:26:28-04:00] O-CE CDFv2 FAILED
[2026-10-12T17:26:34-04:00] O-CE CDFv3 FAILED
[2026-10-12T17:26:40-04:00] O-CE DFD FAILED
[2026-10-12T17:26:46-04:00] O-CE DFDC FAILED
[2026-10-12T17:26:52-04:00] O-CE DFDCP FAILED
[2026-10-12T17:26:58-04:00] O-CE DFEval24 FAILED
[2026-10-12T17:27:04-04:00] O-CE FFIW FAILED
[2026-10-12T17:27:09-04:00] O-CE KoDF FAILED
[2026-10-12T17:27:16-04:00] O-CE IDForge FAILED
[2026-10-12T17:27:21-04:00] O-CE DFDM FAILED
[2026-10-12T17:27:27-04:00] O-CE VALmix FAILED
[2026-10-12T17:27:33-04:00] O-CE VALmixV2 FAILED
[2026-10-12T17:27:34-04:00] === O CE export: artifact view from the simple-CE run, 16 datasets ===
[2026-10-12T17:27:39-04:00] O-CE FFpp_val FAILED
[2026-10-12T17:27:45-04:00] O-CE UADFV FAILED
[2026-10-12T17:27:50-04:00] O-CE DeepSpeak-1.1 FAILED
[2026-10-12T17:27:53-04:00] O-CE CDFv2 FAILED
[2026-10-12T17:27:57-04:00] O-CE DeepSpeak-2.0 FAILED
[2026-10-12T17:27:57-04:00] O-CE DeepSpeak-2.0 FAILED
[2026-10-12T17:28:10-04:00] O-CE CDFv3 FAILED
[2026-10-12T17:28:27-04:00] O-CE DFD FAILED
[2026-10-12T17:28:45-04:00] O-CE DFDC FAILED
[2026-10-12T17:29:01-04:00] O-CE DFDCP FAILED
[2026-10-12T17:30:03-04:00] === O CE export: artifact view from the simple-CE run, 16 datasets ===
[2026-10-12T17:34:31-04:00] O-CE CDFv2 done
[2026-10-12T17:43:04-04:00] CE arm start: ce_P16ctl on GPU 0
[2026-10-12T17:43:04-04:00] CE arm start: ce_S16ctl on GPU 1
[2026-10-12T18:16:50-04:00] O-CE CDFv3 done
[2026-10-12T18:35:34-04:00] O-CE DFD done
[2026-10-12T18:57:15-04:00] O-CE DFDC done
[2026-10-12T19:00:20-04:00] O-CE DFDCP done
[2026-10-12T19:04:25-04:00] O-CE DFEval24 done
[2026-10-12T19:22:38-04:00] O-CE FFIW done
[2026-10-12T19:35:03-04:00] O-CE KoDF done
[2026-10-12T20:01:33-04:00] ce_P16ctl DONE
[2026-10-12T20:06:11-04:00] O-CE IDForge done
[2026-10-12T20:16:13-04:00] O-CE DFDM done
[2026-10-12T20:22:56-04:00] O-CE VALmix done
[2026-10-12T20:35:58-04:00] O-CE VALmixV2 done
[2026-10-12T20:39:55-04:00] O-CE FFpp_val done
[2026-10-12T20:40:45-04:00] O-CE UADFV done
[2026-10-12T20:44:22-04:00] CE arm start: ce_V on GPU 3
[2026-10-12T20:57:28-04:00] O-CE DeepSpeak-1.1 done
[2026-10-12T21:16:07-04:00] ce_S16ctl DONE
[2026-10-12T21:19:50-04:00] O-CE DeepSpeak-2.0 done
[2026-10-12T21:19:50-04:00] === O CE export done ===
[2026-10-12T22:03:05-04:00] ce_V DONE
[2026-10-12T22:03:05-04:00] === CE arm lane done ===
[2026-10-13T00:26:54-04:00] === scoring the five branches on UADFV DeepSpeak-1.1 DeepSpeak-2.0 ===
[2026-10-13T00:27:24-04:00] A trunk cached: UADFV
[2026-10-13T00:33:33-04:00] === chain: App CE head ===
[2026-10-13T00:34:23-04:00] App CE head DONE
[2026-10-13T00:34:23-04:00] chain: waiting for the new-set scoring
[2026-10-13T00:35:05-04:00] A trunk cached: DeepSpeak-1.1
[2026-10-13T00:43:50-04:00] A trunk cached: DeepSpeak-2.0
[2026-10-13T01:14:42-04:00] chain: scoring drained after 40 min
[2026-10-13T01:14:42-04:00] chain: regenerating tables
[2026-10-13T01:15:12-04:00] === scoring the new sets (UADFV,DeepSpeak-1.1,DeepSpeak-2.0) ===
[2026-10-13T01:17:19-04:00] === chain done -> reports/branch_fusion/PROPOSAL_TABLES.md ===
[2026-10-13T01:34:08-04:00] STT exported on the new sets
[2026-10-13T01:46:02-04:00] === overnight chain start ===
[2026-10-13T01:52:26-04:00] STT-order exported on the new sets
[2026-10-13T01:52:31-04:00] === new-set scoring v2 done ===
[2026-10-13T01:53:05-04:00] overnight: STT lane drained after 7 min
[2026-10-13T01:53:10-04:00] TD export FAILED
[2026-10-13T01:55:02-04:00] LFS exported: UADFV
[2026-10-13T02:26:41-04:00] LFS exported: DeepSpeak-1.1
[2026-10-13T03:05:22-04:00] LFS exported: DeepSpeak-2.0
[2026-10-13T03:05:42-04:00] overnight: regenerating tables
[2026-10-13T03:09:45-04:00] === overnight chain done -> reports/branch_fusion/PROPOSAL_TABLES.md ===
[2026-10-13T09:35:54-04:00] === finishing the new sets ===
[2026-10-13T09:35:56-04:00] LFS reduced into the ledger
[2026-10-13T09:43:05-04:00] TD exported
[2026-10-13T09:43:11-04:00] TD copy FAILED
[2026-10-13T09:43:13-04:00] proposal_tables FAILED
[2026-10-13T09:45:46-04:00] === new sets finished ===
[2026-10-13T16:35:21-04:00] === regenerating all tables over 12 test sets ===
[2026-10-13T16:35:23-04:00] regen proposal_tables FAILED
[2026-10-13T16:36:24-04:00] regen sota_table1 ok
[2026-10-13T16:37:24-04:00] regen table2_calibration ok
[2026-10-13T16:38:04-04:00] regen uncertainty_figs ok
[2026-10-13T16:39:17-04:00] regen forensic_metrics ok
[2026-10-13T16:40:34-04:00] regen sota_operating_points ok
```
