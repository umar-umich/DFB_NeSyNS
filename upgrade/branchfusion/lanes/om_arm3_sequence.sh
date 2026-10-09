#!/bin/bash
# ARM 3 -- SCHEDULED, NOT LAUNCHED. OM on a SEQUENCE representation (our S, and Kutub's P).
#
# Different proposition from arms 1 and 2: the feature is a CLIP-level embedding, not a frame
# embedding, so "off-manifold" becomes "off the manifold of plausible clip dynamics" rather than of
# plausible faces. That is interesting, and it is also where the redundancy risk is worst -- the
# representation inherits the PE trunk (so it shares with A, like arm 1) AND is a sequence (so it
# shares with V and P). Run it AFTER arms 1 and 2, because arm 1 tells you whether the trunk-sharing
# prediction holds at all; if arm 1 fails as predicted, this arm's prior gets much worse and the
# money is better spent elsewhere.
#
# BLOCKER: this needs per-clip embeddings, and neither S nor P caches them today -- both export
# opinions only. `seq_branch.py` would need a `--dump-embeddings` pass writing the pooled clip vector
# per video, and P's trainer the equivalent. Budget that work before launching.
set -u
echo "ARM 3 is a stub: per-clip embeddings do not exist yet for S or P."
echo "Required first: seq_branch.py --dump-embeddings  (pooled clip vector per video, 13 datasets)"
exit 2
