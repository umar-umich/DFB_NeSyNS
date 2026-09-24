#!/usr/bin/env python3
"""Bounded RL pilot on the cached acquisition environment: PPO at budget 1 and 2, REINFORCE at budget 1.

    OMP_NUM_THREADS=8 python analysis/tbiom/acquisition/rl_pilot.py

Cached outputs only; no detector is retrained. The environment is exactly the one in `step1_controls.py`: the same
five source-grouped folds, the same candidate windows, the same `conf` within-branch aggregation, the same DS-R
fusion with weights fitted per fold, and the same frozen threshold (EER of the no-acquisition score on the fold's
FF++ val videos).

State (all of it available at inference): the current fused probability, the per-view vacuities, the spread and
standard deviation of the per-view probabilities, each candidate window's quality descriptors and their deviation
from the video's mean, which windows have been inspected, and the remaining budget. A window's cached detector
output is never in the state until that window has been chosen.

Actions: inspect one un-inspected window, or stop. Reward: the drop in log loss caused by the observation, minus a
fixed cost per extra clip. The cost is chosen on the FITTING folds only and reported.

Note on provenance: the frozen go/no-go in `tbiom/docs/decision-2026-09-18.md` concluded DO NOT proceed to PPO,
because the learnability condition failed. This pilot was requested afterwards and is run as an explicit override of
that gate; it is reported as a pilot, not as a satisfied criterion.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "temporal"), str(HERE.parent / "grouped")]
import step1_controls as S1  # noqa: E402
import exp1_fusion_audit as A  # noqa: E402
import dsr_system as D  # noqa: E402
import finish_line as F  # noqa: E402
import confirm as C  # noqa: E402
import score as SC  # noqa: E402
from health import eer  # noqa: E402

R, RULE, NW = Path("tbiom/reports"), "conf", S1.N_WIN
COSTS = (0.0, 0.002, 0.005, 0.01, 0.02)
torch.set_num_threads(4)


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


class Env:
    """Fully enumerated episode outcomes for one pool: every reachable observation set is precomputed."""

    def __init__(self, pool, y, w):
        self.pool, self.y, self.n = pool, y, len(pool.v)
        self.none = pool.score(np.zeros((self.n, 0), int), w)
        self.ll0 = logloss(self.none, y)
        self.s1 = np.stack([pool.score(np.full((self.n, 1), j), w, RULE) for j in range(NW)], 1)
        self.ll1 = np.stack([logloss(self.s1[:, j], y) for j in range(NW)], 1)
        self.s2, self.ll2 = {}, {}
        for i in range(NW):
            for j in range(NW):
                if i == j:
                    continue
                key = tuple(sorted((i, j)))
                if key in self.s2:
                    continue
                sc = pool.score(np.tile(np.array(key), (self.n, 1)), w, RULE)
                self.s2[key] = sc
                self.ll2[key] = logloss(sc, y)
        f = pool.features([set()] * self.n)                       # per-window inference-available features
        self.base = f[0][:, :8]                                   # p/u per view, spread, std
        self.qw = np.stack([x[:, 8:-NW - 1] for x in f], 1)       # (n, NW, q-features)

    def state(self, visited, budget):
        """visited: (n, NW) 0/1. Returns the flat state matrix."""
        b = np.full((self.n, 1), budget, float)
        return np.concatenate([self.base, self.qw.reshape(self.n, -1), visited, b], 1).astype(np.float32)

    def dim(self):
        return self.state(np.zeros((self.n, NW)), 1).shape[1]

    def score_of(self, chosen):
        """chosen: list of frozensets per video -> fused score."""
        out = np.empty(self.n)
        for i, c in enumerate(chosen):
            c = tuple(sorted(c))
            out[i] = self.none[i] if not c else (self.s1[i, c[0]] if len(c) == 1 else self.s2[c][i])
        return out

    def ll_of(self, chosen):
        out = np.empty(self.n)
        for i, c in enumerate(chosen):
            c = tuple(sorted(c))
            out[i] = self.ll0[i] if not c else (self.ll1[i, c[0]] if len(c) == 1 else self.ll2[c][i])
        return out


class Net(nn.Module):
    def __init__(self, d, na, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi, self.v = nn.Linear(hidden, na), nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def rollout(env, net, budget, cost, greedy=False, gen=None):
    """One vectorised episode over every video in the pool. Action NW == stop."""
    n = env.n
    visited = np.zeros((n, NW))
    chosen = [set() for _ in range(n)]
    done = np.zeros(n, bool)
    traj = []
    for step in range(budget):
        st = torch.from_numpy(env.state(visited, budget - step))
        logits, val = net(st)
        mask = torch.from_numpy(np.concatenate([visited, np.zeros((n, 1))], 1)).bool()
        logits = logits.masked_fill(mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(1) if greedy else dist.sample()
        ll_before = env.ll_of(chosen)
        act = a.cpu().numpy()
        for i in range(n):
            if not done[i] and act[i] < NW:
                chosen[i].add(int(act[i])); visited[i, act[i]] = 1
        stop = act == NW
        done = done | stop
        ll_after = env.ll_of(chosen)
        rew = (ll_before - ll_after) - cost * (~stop & ~np.isnan(ll_after)).astype(float)
        rew[stop] = 0.0
        traj.append({"state": st, "action": a, "logp": dist.log_prob(a), "value": val,
                     "reward": torch.from_numpy(rew.astype(np.float32)), "done": torch.from_numpy(done.copy())})
        if done.all():
            break
    return chosen, traj


def train(env_fit, budget, cost, algo, seed, iters=60, clip=0.2, lr=3e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(env_fit.dim(), NW + 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    curve = []
    for it in range(iters):
        _c, traj = rollout(env_fit, net, budget, cost)
        ret = torch.zeros_like(traj[-1]["reward"])
        rets = []
        for t in reversed(traj):
            ret = t["reward"] + ret
            rets.append(ret.clone())
        rets = list(reversed(rets))
        loss = 0.0
        for t, G in zip(traj, rets):
            adv = (G - t["value"]).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            if algo == "ppo":
                logits, val = net(t["state"])
                mask = torch.zeros_like(logits, dtype=torch.bool)
                dist = torch.distributions.Categorical(logits=logits.masked_fill(mask, -1e9))
                ratio = torch.exp(dist.log_prob(t["action"]) - t["logp"].detach())
                pl = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
                vl = ((val - G) ** 2).mean()
                ent = dist.entropy().mean()
                loss = loss + pl + 0.5 * vl - 0.01 * ent
            else:                                                  # REINFORCE with a learned baseline
                pl = -(t["logp"] * adv).mean()
                vl = ((t["value"] - G) ** 2).mean()
                loss = loss + pl + 0.5 * vl
        opt.zero_grad(); loss.backward(); opt.step()
        curve.append(float(sum(t["reward"].mean() for t in traj)))
    return net, curve


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    t0 = time.time()
    win, full_t = S1.load()
    dev = D.load("vjepa2_lora_B_seed42")[0]
    dev = dev[dev.v.isin(full_t.index)].reset_index(drop=True)
    base = dev.set_index("v")
    out = {"rule": RULE, "iters": a.iters, "seeds": a.seeds, "cost_grid": list(COSTS), "folds": []}
    for k, Fd, Sd in D.splits(dev):
        w = F.fit_w([Fd], S1.VIEWS)
        ef = Env(S1.Pool(Fd.v, base, win), Fd.y.to_numpy(), w)
        es = Env(S1.Pool(Sd.v, base, win), Sd.y.to_numpy(), w)
        ys = Sd.y.to_numpy()
        ff = np.where(Sd.source == "FFpp_val")[0]
        tau = float(eer(ys[ff], es.none[ff])[1])
        g = A.groups_for(Sd)
        draws = np.stack(SC.group_boot(g, 2000, 20260924 + k)).astype(float)
        ref = C.auc_weighted(es.none, ys, draws)
        fold = {"split": k, "n_fit": ef.n, "n_sel": es.n, "tau": tau, "w": list(w), "arms": {}}
        for algo, budget in (("ppo", 1), ("ppo", 2), ("reinforce", 1)):
            name = f"{algo.upper()} (k={budget})"
            # cost chosen on the FITTING fold only, by fitting-fold AUROC of the greedy policy
            best = None
            for c in COSTS:
                net, _cv = train(ef, budget, c, algo, seed=0, iters=max(10, a.iters // 3))
                ch, _ = rollout(ef, net, budget, c, greedy=True)
                s = ef.score_of(ch)
                auc = A.fast_auc(s[ef.y == 0], s[ef.y == 1])
                if best is None or auc > best[1]:
                    best = (c, auc)
            cost = best[0]
            per_seed, curves = [], []
            for sd in range(a.seeds):
                ts = time.time()
                net, cv = train(ef, budget, cost, algo, seed=sd, iters=a.iters)
                ch, _ = rollout(es, net, budget, cost, greedy=True)
                s = es.score_of(ch)
                d = C.auc_weighted(s, ys, draws) - ref
                pred = (s >= tau).astype(int)
                nclip = float(np.mean([len(c_) for c_ in ch]))
                per_seed.append({"seed": sd, "auroc": A.fast_auc(s[ys == 0], s[ys == 1]),
                                 "delta": float(np.nanmean(d)),
                                 "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                                 "fpr": float((pred[ys == 0] == 1).mean()),
                                 "clips_per_video": nclip, "train_s": time.time() - ts})
                curves.append(cv)
            fold["arms"][name] = {"cost": cost, "cost_selected_on": "fitting fold AUROC",
                                 "per_seed": per_seed, "curves": curves}
            print(f"  split {k} {name}: cost {cost} | AUROC "
                  f"{np.mean([x['auroc'] for x in per_seed]):.4f} "
                  f"(spread {np.ptp([x['auroc'] for x in per_seed]):.4f}) | "
                  f"clips {np.mean([x['clips_per_video'] for x in per_seed]):.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        # null-oracle permutation spread, to settle which figure is right
        rng = np.random.default_rng(1234)
        nulls = []
        for _ in range(20):
            perm = np.stack([es.s1[rng.permutation(es.n), j] for j in range(NW)], 1)
            L = np.stack([logloss(perm[:, j], ys) for j in range(NW)], 1)
            sc = perm[np.arange(es.n), L.argmin(1)]
            nulls.append(A.fast_auc(sc[ys == 0], sc[ys == 1]))
        fold["null_oracle"] = {"auroc_mean": float(np.mean(nulls)), "auroc_sd": float(np.std(nulls)),
                               "auroc_min": float(np.min(nulls)), "auroc_max": float(np.max(nulls)),
                               "base": float(A.fast_auc(es.none[ys == 0], es.none[ys == 1])), "n_perms": 20}
        out["folds"].append(fold)
    out["wall_clock_s"] = time.time() - t0
    json.dump(out, open(R / "EXP_ACQUISITION_RL.json", "w"), indent=1, default=float)
    print(f"wrote {R / 'EXP_ACQUISITION_RL.json'} in {out['wall_clock_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
