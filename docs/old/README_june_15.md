Hi, We have planned to go for NeurIPS based on this implementation (/data/umar/Repos/DFB_NeSyNS/training/detectors/nesy_defake_detector.py)
  but it was not of that level and finally I submitted it to IEEE IJCB. Now I'm planning to develop a new Idea for CVPR, WACV or a Special issue
  "Special Issue on Towards Responsible and Explainable Multi-Modal Fusion" in ACM Transactions on Multimedia Computing, Communications, and
  Applications. Please let me know how we can develop a solid idea and what would be the core contibutions and how we would justify it based on
  results outputs. We can first discuss what were the challenges and what we claimed that was not justifiable based on architecture or results.

   My honest recommendation

  Build Direction 1 first as a kill-test, alone, before writing a method section — it's cheap, training-free, and either gives you the anchor
  figure or saves you months. If it survives, Direction 2 is your method-section novelty and the strongest neuro-symbolic story. Direction 3 is
  high-risk/high-reward — keep it as a third contribution only if its ablation pays off; otherwise a clean 2-contribution paper (topological axis
  + learned grammar) is more defensible than three weak ones, which matches your own "fewer defensible contributions" constraint.

  Speculative flags (I won't oversell): wake–sleep library learning on forensic predicates is unproven in this domain and may not converge to
  human-meaningful programs — treat contribution 2.1 as speculative until the library is inspected. Active inference query-selection beating a
  random/greedy baseline is also unproven — that's exactly what its ablation tests.

  Two things to decide before I turn any of this into a runnable spec:
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
←  ☐ Lead idea  ☐ Synth data  ✔ Submit  →

Which direction should I develop into a concrete, runnable spec first?

Do you have (or can you obtain) a fully-synthetic face dataset? The swap-vs-synthetic axis is untestable without one.