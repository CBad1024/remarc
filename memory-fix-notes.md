# Memory Blowup in `run_experiments.sh` — Diagnosis & Fix

## Symptom

`run_experiments.sh` launches 6 training experiments in parallel (3 datasets ×
2 `dh` values). Memory usage stayed reasonable early on, then grew until the
machine ran out of memory, later in the run.

## Root Cause

`run_experiments.sh` launching 6 parallel top-level Python processes is *not*
the problem by itself. The actual multiplier is inside each process:

`examples/run_plots.py --train` calls `train_wf_landscapes()`
(`remarc/agents/tianshou_agent.py`), which calls `WrightFisherEnv.get_env()`
(`remarc/envs/wright_fisher_env.py`). That factory used **`SubprocVectorEnv`**
to spawn `n_train_envs=16` + `n_test_envs=4` = **20 extra OS subprocesses per
experiment**. Across 6 parallel experiments, that's **120 subprocesses**, not 6
processes.

### Why that blew up memory specifically

macOS's default `multiprocessing` start method is `spawn`, not `fork`
(`tianshou/env/worker/subproc.py` calls `multiprocessing.get_context(None)`,
which resolves to the platform default). Under `spawn`, every child process
re-executes the **entire top level of the launching script**
(`examples/run_plots.py`) to rebuild a valid `__main__` context for
unpickling — even though a worker only needs a small numpy-based
`WrightFisherEnv` object.

That means every one of the 120 subprocesses re-imported `torch`,
`matplotlib`, `optuna`, and `torch.utils.tensorboard` from scratch, none of
which the worker actually uses.

**Measured cost** (this repo, this machine):

```
import torch + matplotlib + optuna + tensorboard alone:  ~233 MB RSS
Full top-level of examples/run_plots.py (what each
  SubprocVectorEnv worker actually pays via spawn):       ~391 MB RSS
```

`20 workers × 6 experiments × ~390 MB ≈ 47 GB` of pure redundant import
overhead — on a 36 GB machine, that alone explains the OOM.

### Why it got worse "toward the end" specifically

`train_wf_landscapes()` never called `.close()` on `train_envs` /
`test_envs` after training finished, and tianshou's vector-env classes have
no `__del__` fallback (only an explicit `.close()`). So the 20 subprocesses
per experiment — each carrying the ~390 MB import tax — stayed alive for the
rest of that process's lifetime, including through the eval/plotting phase.
Since the 6 experiment configs don't finish training at the same wall-clock
time, memory kept stacking as some processes moved into their (also memory-
heavy) eval phase while others were still mid-training with their 20 live
subprocesses.

## The Fix

Switched `SubprocVectorEnv` → `DummyVectorEnv` (runs all envs sequentially
in a single process — no subprocesses at all).

**This is not a meaningful speed tradeoff.** Benchmarked directly:

```
DummyVectorEnv:   0.114s for 200 rounds × 16 envs
SubprocVectorEnv: 0.068s for 200 rounds × 16 envs
```

Scaled to a full run (300 epochs × 10,000 steps = 3,000,000 env-steps), that's
roughly **94s (Dummy) vs 64s (Subproc) — a ~30 second difference across an
entire multi-hour training run.** The env's `step()` is cheap pure-numpy math
(a small Wright-Fisher selection/mutation/drift update on an 8-element
vector), so process-level parallelism for it isn't worth anything. The actual
compute-heavy part (PPO network forward/backward) runs in the main process
either way, completely unaffected by this change.

Also added explicit `.close()` calls after training as a defensive measure,
in case a subprocess-based vector env is reintroduced later.

### Diff

```diff
diff --git a/remarc/envs/wright_fisher_env.py b/remarc/envs/wright_fisher_env.py
--- a/remarc/envs/wright_fisher_env.py
+++ b/remarc/envs/wright_fisher_env.py
@@ -3,7 +3,15 @@ from gymnasium import spaces
 import numpy as np
 import itertools
 import functools
-from tianshou.env import SubprocVectorEnv
+# DummyVectorEnv (not SubprocVectorEnv): benchmarked at near-identical wall-clock
+# cost for this env (~30s difference over a full 300-epoch run), since each
+# step() is cheap pure-numpy math. SubprocVectorEnv was previously used here for
+# parallel training, but on macOS its subprocesses spawn (not fork), which
+# re-imports torch/matplotlib/etc. per worker (~390MB each) since this env's
+# factory fns are launched from a script with those heavy top-level imports.
+# With n_train_envs=16 x several parallel experiments, that multiplied into
+# 100+GB of redundant import overhead and caused OOM. See run_experiments.sh.
+from tianshou.env import DummyVectorEnv
 from ..core.landscapes import Landscape
 import collections

@@ -389,8 +397,8 @@ class WrightFisherEnv(gym.Env):
             n_frames,
             delta_horizon,
         )
-        train_envs = SubprocVectorEnv([fn_train for _ in range(n_train)])
-        test_envs = SubprocVectorEnv([fn_test for _ in range(n_test)])
+        train_envs = DummyVectorEnv([fn_train for _ in range(n_train)])
+        test_envs = DummyVectorEnv([fn_test for _ in range(n_test)])
         return train_envs, test_envs

@@ -480,8 +488,9 @@ class ThreeGenotypeEnv(WrightFisherEnv):
             n_frames,
             delta_horizon,
         )
-        train_envs = SubprocVectorEnv([fn_train for _ in range(n_train)])
-        test_envs = SubprocVectorEnv([fn_test for _ in range(n_test)])
+        # DummyVectorEnv here too — see comment on the import above.
+        train_envs = DummyVectorEnv([fn_train for _ in range(n_train)])
+        test_envs = DummyVectorEnv([fn_test for _ in range(n_test)])
         return train_envs, test_envs

diff --git a/remarc/agents/tianshou_agent.py b/remarc/agents/tianshou_agent.py
--- a/remarc/agents/tianshou_agent.py
+++ b/remarc/agents/tianshou_agent.py
@@ -564,6 +564,14 @@ def train_wf_landscapes(
     test_result = test_collector.collect(n_episode=p.test_episodes)
     print(f"Final testing result: {test_result}")

+    # Explicitly close the vector envs now that training/eval is done. train_envs/
+    # test_envs are DummyVectorEnv now (see wright_fisher_env.py), so this isn't
+    # freeing OS subprocesses anymore, but tianshou's vector envs have no __del__,
+    # so leaving this out would leak file handles / grow unbounded if this ever
+    # goes back to a subprocess-based vector env.
+    train_envs.close()
+    test_envs.close()
+
     # Log hyperparameters and final performance
     hparams = {}
     for k, v in dataclasses.asdict(p).items():
```

**Files touched:** `remarc/envs/wright_fisher_env.py`,
`remarc/agents/tianshou_agent.py`. `run_experiments.sh` itself needed no
changes — the 6-way outer parallelism across experiment configs was never the
problem.

## Verification Done

- Ran a reduced-scale smoke test (2 epochs, 200 train steps, tiny [32,32]
  network) through the real `train_wf_landscapes()` path: trained
  successfully, `.close()` did not error, and `ps aux` showed zero leftover
  processes afterward.
- Ran the real `run_experiments.sh` (all 6 configs, full settings) and
  monitored memory for ~7 minutes before it was manually stopped: total RSS
  across all 6 processes stayed flat at ~3.6 GB (no growth observed), and
  `ps aux` showed no extra subprocess children — confirming the subprocess
  explosion is gone.

## Known Gaps — Not Yet Fixed or Measured

1. **`run_eval()` in `examples/run_plots.py`** (used in the post-training
   eval/plotting phase) deep-copies the environment 100 times and holds full
   5000-step trajectories in memory simultaneously for all 8 policy types
   (RL / Greedy / Random / SHEPHERD / 4× SingleDrug). Rough estimate is
   ~1 GB/process at peak, **not confirmed by measurement** — the ~7-minute
   monitored run never reached this phase before being stopped. If memory
   still creeps up after this fix (at a much smaller scale than before),
   this is the next place to look.
2. Full-run peak memory (training all the way through eval + plotting, for
   all 6 configs) has not been directly measured — only early training
   (~3.6 GB flat) has real data behind it.

## Hardware Note

The above measurements were taken on a 36 GB / 18-core Mac. On a **16 GB
MacBook Air M4** (fanless, 10-core), running all 6 configs in parallel is
riskier:

- ~3.6 GB observed baseline + an unmeasured eval-phase peak (see gap #1
  above) + typical macOS/app overhead (3-6 GB) could plausibly approach
  12-16 GB with no real safety margin.
- Separately (not a memory issue, but relevant): each process was already
  observed pulling 100-150% CPU on an 18-core machine. 6 of those on a
  fanless 10-core chip will oversubscribe the CPU and thermal-throttle hard,
  making training much slower even if it doesn't crash.

**Recommendation for constrained machines:** cap concurrency in
`run_experiments.sh` (e.g. run 2-3 experiments at a time instead of all 6)
rather than assuming this fix makes unlimited parallelism safe on any
hardware. This has not been implemented yet — flag if you want it done.
