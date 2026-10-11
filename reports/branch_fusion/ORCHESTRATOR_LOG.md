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
```
