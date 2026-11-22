# Isaac Lab: SRAM-resident training (prototype)

`isaaclab_sram` runs the complete RSL-RL PPO training loop of `Isaac-Cartpole-Direct` inside a single persistent,
cooperative CUDA kernel. One launch covers:
- physics, rewards, terminations and resets;
- the policy and value MLPs and Gaussian action sampling;
- GAE and advantage normalization;
- the PPO epochs and mini-batches, with an analytic backward pass;
- the gradient all-reduce, KL-adaptive learning rate, gradient clipping and Adam.

The kernel runs one CTA per SM. Each CTA keeps its environments, their rollout buffer and a replica of the networks
in shared memory and registers for the whole run. The host only launches the kernel and reads back statistics and
weights.

The numerics follow RSL-RL 5.5.1 and the Newton / MuJoCo-Warp (`physics=newton_mjwarp`) cartpole, and the task and
PPO settings come from Isaac Lab's own configuration objects. See [the report](../../docs/sram/SRAM_TRAINING.md)
for results, verification, scaling and the analysis of larger and perceptive tasks.

| File | Contents |
| --- | --- |
| `isaaclab_sram/cartpole.py` | `CartpoleTask`, `CartpolePhysics` and `PPOConfig`, built from Isaac Lab's env and RSL-RL agent configurations |
| `isaaclab_sram/megakernel.py` | `SramPPOCartpole` trainer: launch planning, RSL-RL network initialization, training, statistics |
| `isaaclab_sram/cuda_rt.py` | Minimal NVRTC and CUDA driver-API runtime: compiles the kernel at run time and caches the cubin |
| `isaaclab_sram/kernels/ppo_cartpole.cu` | The megakernel |
| `isaaclab_sram/kernels/sram_device.cuh` | Device helpers: Philox RNG, ELU, butterfly warp reductions, keyed Feistel permutation, `wrap_to_pi` |
| `isaaclab_sram/kernels/cartpole_physics.cuh` | Closed-form Newton / MuJoCo-Warp `implicitfast` cart-pole step |
| `isaaclab_sram/export.py` | Writes the trained weights and Adam state as an RSL-RL run directory |
| `isaaclab_sram/rl_backend.py` | `--rl_library sram` backend for Isaac Lab's unified training entry point |
| `scripts/sram/train_sram.py` | Training command line (in the repository, not the package) |
| `scripts/sram/verify_megakernel.py` | Numerical verification against independent references (in the repository, not the package) |

## Installation

The package is not a member of the uv-locked workspace: adding it to the root `pyproject.toml` would require
re-locking `uv.lock`, so `uv sync` and `uv run` do not install it. Use one of these instead:

* `./isaaclab.sh -i` installs it as an editable core submodule.
* In an environment you manage, run `pip install -e source/isaaclab_sram`. A later `uv sync` removes it again,
  because it is not in `uv.lock`.
* Without installing it, put it on the path with `export PYTHONPATH=$PWD/source/isaaclab_sram:$PYTHONPATH`. The
  scripts in `scripts/sram/` and the tests do this themselves; `isaaclab_sram.__version__` then reports `0.0.0`.

It needs a CUDA GPU and Linux. The kernel is compiled with NVRTC through `cuda.bindings`, which torch's Linux
wheels install; no `nvcc` is required. Compiled cubins are cached in `~/.cache/isaaclab_sram/cubin`; set
`ISAACLAB_SRAM_CUBIN_CACHE` to use another directory.

## Training

```bash
# standalone: trains without a simulator and writes an RSL-RL run directory under logs/sram/
./isaaclab.sh -p scripts/sram/train_sram.py --num_envs 4096 --max_iterations 150

# Isaac Lab's unified entry point: registers --rl_library sram, then runs the stock train.py;
# the run goes to logs/rsl_rl/cartpole_direct/<timestamp>_sram
./isaaclab.sh -p scripts/sram/train_sram.py --via-isaaclab-cli --rl_library sram --task Isaac-Cartpole-Direct \
    --num_envs 4096 --max_iterations 150 physics=newton_mjwarp
```

Both write `model_<iteration>.pt`, `run.json`, `params/` and TensorBoard events. The stock
`scripts/reinforcement_learning/play.py` and `scripts/benchmarks/play.py` load the run with
`--rl_library rsl_rl --checkpoint <path>`; `scripts/sram/eval_policy.py` evaluates any RSL-RL checkpoint in the stock
Newton environment with deterministic actions.

From Python:

```python
from isaaclab_sram.cartpole import CartpoleTask, PPOConfig, isaaclab_cartpole_configs
from isaaclab_sram.megakernel import SramPPOCartpole

# Isaac Lab's own Isaac-Cartpole-Direct env and RSL-RL agent configurations (no simulator is started)
env_cfg, agent_cfg = isaaclab_cartpole_configs()
trainer = SramPPOCartpole(CartpoleTask.from_env_cfg(env_cfg), PPOConfig.from_rsl_rl_cfg(agent_cfg), num_envs=4096, seed=1)
result = trainer.train(150)               # one launch; train(150, iters_per_launch=1) gives bitwise-identical weights
print(result["gpu_time_s"], result["stats"]["mean_episode_return"][-1])
actor, critic = trainer.trained_models()  # rsl_rl MLPModel modules with the trained weights
```

## Verification

```bash
./isaaclab.sh -p scripts/sram/verify_megakernel.py --num_envs 4096 [--storage auto|smem|global] [--warmup_iters N]
```

The script trains one iteration with the kernel's debug build and checks:
- the physics against a float64 closed-form reference;
- the rewards, resets, networks, log-probabilities and GAE against RSL-RL;
- the first mini-batch gradient against torch autograd;
- every one of the iteration's 20 optimizer steps against RSL-RL's loss code and `torch.optim.Adam`. Each step
  starts from the kernel's exact state before it (parameters, Adam moments, learning rate).

It exits non-zero if any check fails. `--warmup_iters N` checks a later iteration, with non-zero Adam state and an
adapted learning rate.

The tests run this script for several launch configurations and also check bitwise determinism, the device
helpers, the launch planner and edge cases. They need pytest and a CUDA GPU, and skip otherwise:

```bash
./isaaclab.sh -p -m pytest source/isaaclab_sram/test
```

## Limits

* The kernel is specialized for `Isaac-Cartpole-Direct` with `physics=newton_mjwarp`: `[32, 32]` ELU actor and
  critic, a scalar action std, the clipped value loss and no observation normalization.
  `PPOConfig.from_rsl_rl_cfg` rejects other configurations.
* It has only been verified on an RTX PRO 6000 Blackwell (sm_120, 188 SMs). The planner queries the device's
  per-block shared-memory limit (101,376 bytes there). On that GPU the rollout buffer stays in shared memory up to
  11,092 environments (59 per SM), then moves to global memory (L2, then DRAM), up to 317,720 environments.
* Results are not bitwise comparable with stock Isaac Lab, only statistically:
  * The random streams differ from torch's.
  * Each SM fills every mini-batch from its own transitions (stratified per SM) instead of RSL-RL's single global
    permutation.
  * The kernel computes in fp32; stock training enables TF32 matmuls.
* For a fixed seed, the parameters, optimizer state and environment state are bitwise reproducible, however the
  iterations are split across launches and wherever the rollout buffer lives. The exceptions are two logging
  statistics, `episode_return_sum` and `step_reward_sum`: the kernel sums them with float atomics, so they can
  differ in the last bits.
* `scripts/reinforcement_learning/train.py` only accepts `--rl_library sram` when it is started through
  `scripts/sram/train_sram.py --via-isaaclab-cli`. That wrapper adds the backend to the private
  `isaaclab_rl.entrypoints.dispatch._BACKEND_MODULES` table at run time. The backend rejects options that need a
  simulator, such as `--video`, `--distributed` and resuming from `--checkpoint`.
* The free-running 20-update replay in `verify_megakernel.py` is informational, because ~1e-7 rounding differences
  are amplified through 20 sequential non-linear updates (and through samples whose probability ratio sits on a PPO
  clip boundary). The gating check is the teacher-forced replay of every update.
