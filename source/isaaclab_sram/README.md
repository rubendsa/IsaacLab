# Isaac Lab: SRAM-oriented training (prototype)

This package contains a persistent Cartpole training megakernel, fused operations in the native Kuka-Allegro
camera PPO learner, and an experimental persistent native critic. The Kuka paths retain the task's trainable
visual encoder, state observations, privileged critic and full PPO rollout. The persistent critic keeps its
parameter and optimizer working state in shared memory during its update; images, rollouts and the actor still
use global allocations. The complete visual training run is not in SRAM. See
[the technical report](../../docs/sram/SRAM_TRAINING.md) for the memory budget, research,
measurement protocol and current evidence.

`isaaclab_sram` runs the complete RSL-RL PPO training loop of `Isaac-Cartpole-Direct` inside a single
persistent, cooperative CUDA kernel. That covers physics, rewards, terminations and resets, the policy and
value MLPs, Gaussian action sampling, GAE, advantage normalization, the PPO epochs and mini-batches with an
analytic backward pass, gradient all-reduce, KL-adaptive learning rate, gradient clipping and Adam. The kernel
runs one CTA per SM, and each CTA keeps its environments, their rollout buffer (when it fits) and a replica of the
networks in shared memory and registers for the whole run. The host only launches the kernel and reads back
statistics and weights.

The Cartpole numerics target RSL-RL 5.5.1 and the Newton / MuJoCo-Warp (`physics=newton_mjwarp`) cartpole. The task
and PPO settings come from Isaac Lab's own configuration objects. Cross-block reduction buffers are global memory;
L2 caching is not a guaranteed residency mechanism. Parameter and environment checkpoints are loaded and saved
at kernel boundaries, and compiler spill information must be inspected separately.

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
| `isaaclab_sram/spatial_softmax.py` | Fused spatial-softmax forward/backward, including temperature gradients, for the camera actor |
| `isaaclab_sram/ppo_kernels.py` | Fused PPO objective and GAE operations |
| `isaaclab_sram/perception.py` | Native camera PPO integration with selectable learner operations |
| `isaaclab_sram/persistent_perception.py` | Camera PPO with a separate persistent native critic update |
| `isaaclab_sram/persistent_critic.py` | Native critic checkpoint packing, residency checks and cooperative launch |
| `isaaclab_sram/kernels/ppo_native_critic.cu` | All critic forward/backward, clipping and Adam steps in one launch |
| `isaaclab_sram/compact_history.py`, `compact_storage.py` | Lossless state-history storage and native rollout adapter |
| `scripts/sram/train_sram.py` | Training command line (in the repository, not the package) |
| `scripts/sram/train_kuka_perception.py` | Real Kuka-Allegro camera rollout and training command |
| `scripts/sram/benchmark_persistent_critic.py` | Repeated native critic updates from an identical resumed checkpoint and synthetic rollout |
| `scripts/sram/verify_megakernel.py` | Numerical verification against independent references (in the repository, not the package) |
| `scripts/sram/projection/measure_update_cost_cnn.py` | Native CNN learner on synthetic inputs: eager/CUDA-graph benchmark and memory accounting |

## Installation

The root `pyproject.toml` and `uv.lock` include this package as an editable dependency. Run commands from the
repository root with `uv run`; uv installs the locked environment. `uv sync` can prepare it explicitly. In an
environment you manage separately, install the package with `uv pip install -e source/isaaclab_sram`.

It needs a CUDA GPU and Linux. The Cartpole kernel is compiled with NVRTC through `cuda.bindings`, which torch's
Linux wheels install. No `nvcc` is required. Compiled cubins are cached in `~/.cache/isaaclab_sram/cubin`; set
`ISAACLAB_SRAM_CUBIN_CACHE` to use another directory.

## Kuka-Allegro perception training

The runner resolves `Isaac-Lift-KukaAllegro-Camera` with `physics=newton_mjwarp`, `renderer=newton_renderer` and
`presets=single_camera,rgb64` (or `depth64`). It keeps the native CNN, learned spatial-softmax temperatures, privileged
critic, running observation normalizers, 23 actions and fixed-rate PPO recipe. `sram` selects fused spatial-softmax,
GAE and PPO-objective kernels and fused Adam; `torch` selects stock RSL-RL. `sram_graph` additionally captures a
complete minibatch update, including CNN backward and Adam, in a CUDA graph. The CNN is trained end to end in all
three paths. Graph replay still uses global memory and is distinct from a persistent megakernel.

```bash
OMP_NUM_THREADS=1 uv run python scripts/sram/train_kuka_perception.py \
    --backend torch --camera rgb64 --num_envs 256 --iterations 100 --warmup_iterations 5 \
    --seed 42 --output logs/sram/kuka_rgb64_torch
OMP_NUM_THREADS=1 uv run python scripts/sram/train_kuka_perception.py \
    --backend sram --camera rgb64 --num_envs 256 --iterations 100 --warmup_iterations 5 \
    --seed 42 --output logs/sram/kuka_rgb64_sram
OMP_NUM_THREADS=1 uv run python scripts/sram/train_kuka_perception.py \
    --backend sram_graph --camera rgb64 --num_envs 256 --iterations 100 --warmup_iterations 5 \
    --seed 42 --output logs/sram/kuka_rgb64_sram_graph
```

Use a quiet GPU for paired timings. The examples preserve native reset initialization and compare the same task
and model. For a quick execution check, use `--num_envs 64 --iterations 3 --warmup_iterations 1
--smoke_reset_bank 16`; the reset-bank override changes initialization coverage and is explicitly recorded. Such a
short smoke run does not establish a successful lift policy or native time to convergence.

Outputs include `model_<iteration>.pt`, `params/`, `run.json`, TensorBoard events and `perception_summary.json`.
The summary checks actual camera observations, CNN weight changes, finite parameters, checkpoint action replay
and input-sensitive actions, and separates collection/learning timings. `--checkpoint PATH` resumes a native
RSL-RL checkpoint. See the technical report for the remaining long-run learning and memory-profiling gates.
The graph backend omits unused action samples during optimization, so its rollout random stream differs from
stock. Checkpoints must be loaded before its first update. Capture warmup restores parameters and Adam state;
normalizer buffers retain stable addresses across subsequent updates.

The experimental `sram_critic` backend supports at most 64 environments and the
native 32-step, five-epoch, eight-minibatch recipe. It runs all 40 critic optimizer
steps in one cooperative launch, retaining the critic's parameters, gradients and
Adam moments in block-owned shared memory. The actor and critic have disjoint
parameters and separate gradient clipping, so the actor can use the same
minibatches in a subsequent phase. The native CNN remains trainable. Critic
inputs, activation/reduction buffers and launch-boundary checkpoints still use
global memory; this backend is a step toward a complete resident visual learner.
The wrapper rejects devices whose queried shared-memory/occupancy limits cannot
support its cooperative launch. It copies current native optimizer state at
every boundary, including resumed moments and individual parameter step counts.

`--compact_history` optionally stores each processed state frame once, plus the
initial history prefix and reset boundaries. It discovers the live observation
manager's term ordering, reconstructs exact minibatches, and leaves raw camera
pixels unchanged. A history transition that cannot be represented losslessly
raises an error before the learner consumes it. Normalizer updates reconstruct
complete state groups to preserve their reduction order. The stored frames and
reconstructed minibatches still use global allocations.

```bash
OMP_NUM_THREADS=1 uv run python scripts/sram/train_kuka_perception.py \
    --backend sram_critic --camera rgb64 --num_envs 64 --iterations 30 --warmup_iterations 5 \
    --compact_history --ieee_fp32 --seed 42 --output logs/sram/kuka_rgb64_persistent_critic
```

Use `--ieee_fp32` on every compared backend to disable TF32 library operations.
The custom critic uses IEEE float32 arithmetic and a float64 sum of squared
gradients for clipping; floating-point reduction order can differ from PyTorch.
Neither a smaller stored rollout nor a persistent critic proves that the full
camera learner is resident, or that global buffers avoid DRAM traffic.

The fused perception backend currently supports CUDA float32 model operations, the fixed-learning-rate Gaussian
camera recipe, clipped values and global advantage normalization. It rejects mixed precision, distributed or
recurrent training, RND, symmetry, adaptive learning-rate schedules and other unsupported algorithm settings.
The code uses Triton supplied by the Linux PyTorch environment. Convolutions and GEMMs still use PyTorch's CUDA
libraries in the selective-fusion paths; the persistent critic replaces its own MLP and optimizer operations.
All paths retain global rollout buffers and native checkpoint tensors.

## Cartpole training

```bash
# Standalone: trains without a simulator and writes an RSL-RL run directory under logs/sram/
uv run python scripts/sram/train_sram.py --num_envs 4096 --max_iterations 150

# Isaac Lab's unified entry point: registers --rl_library sram, then runs the stock train.py.
# The run goes to logs/rsl_rl/cartpole_direct/<timestamp>_sram.
uv run python scripts/sram/train_sram.py --via-isaaclab-cli --rl_library sram --task Isaac-Cartpole-Direct \
    --num_envs 4096 --max_iterations 150 physics=newton_mjwarp
```

Both commands write `model_<iteration>.pt`, `run.json`, `params/` and TensorBoard events. The stock
`scripts/reinforcement_learning/play.py` and `scripts/benchmarks/play.py` load the run with
`--rl_library rsl_rl --checkpoint <path>`. The benchmark and evaluation scripts are also in `scripts/sram/`. For
timings, run them on a quiet GPU.

To train from Python:

```python
from isaaclab_sram import CartpoleTask, PPOConfig, isaaclab_cartpole_configs
from isaaclab_sram.megakernel import SramPPOCartpole

# Isaac Lab's own Isaac-Cartpole-Direct env and RSL-RL agent configurations (no simulator is started)
env_cfg, agent_cfg = isaaclab_cartpole_configs()
trainer = SramPPOCartpole(
    CartpoleTask.from_env_cfg(env_cfg), PPOConfig.from_rsl_rl_cfg(agent_cfg), num_envs=4096, seed=42
)
result = trainer.train(150)  # all 150 PPO iterations in a single kernel launch
print(f"mean episode return {result['stats']['mean_episode_return'][-1]:.2f} (max ~5)")
actor, critic = trainer.trained_models()  # RSL-RL MLPModel modules holding the trained weights
```

* `train(n, iters_per_launch=None)` runs `n` iterations. By default they all run in one launch; pass
  `iters_per_launch` to split them across launches. It returns the GPU time, environment steps per second and
  per-iteration statistics: episode return and length, losses, KL, learning rate, action std and gradient norms.
* `describe()` reports the launch plan: CTAs, environments per CTA, shared memory per CTA and where the rollout
  buffer lives.
* `trained_models()` returns the RSL-RL actor and critic for evaluation or export.

Compare policies on episode return, whose maximum is about 5. A policy that does nothing still reaches the full
episode length of 300 steps, with a return of about -20.

## Verification

```bash
uv run python scripts/sram/verify_megakernel.py --num_envs 4096 [--storage auto|smem|global] [--out report.json]
```

The script trains one iteration with the kernel's debug build. It then checks the physics against a float64
closed-form reference, and the rewards, resets, networks, log-probabilities and GAE against RSL-RL. It compares
the first mini-batch gradient with torch autograd, and replays every one of the iteration's 20 optimizer steps
with RSL-RL's loss code and `torch.optim.Adam`, each starting from the kernel's exact state before that step
(parameters, Adam moments, learning rate). It exits non-zero if any check fails. `--warmup_iters N` checks a later
iteration (non-zero Adam state, adapted learning rate).

The tests run this script for several launch configurations and check bitwise determinism. They need pytest
(`--extra test`) and a CUDA GPU, and skip otherwise:

```bash
uv run --extra test python -m pytest source/isaaclab_sram/test
```

## Cartpole limits

* The kernel is specialized for `Isaac-Cartpole-Direct` with `physics=newton_mjwarp`, `[32, 32]` ELU actor and
  critic, a scalar action std, the clipped value loss and no observation normalization.
  `PPOConfig.from_rsl_rl_cfg` rejects other configurations.
* The planner queries the device's per-block shared-memory limit (101,376 bytes on sm_120). The maximum environment
  count depends on the GPU, launch configuration, rollout length and scratch layout. `describe()` reports the
  actual decision. A global rollout is backed by device memory even when its accesses benefit from L2 caching.
* Results are not bitwise comparable with stock Isaac Lab, only statistically:
  * The random streams differ from torch's.
  * Each SM fills every mini-batch from its own transitions (stratified per SM) instead of RSL-RL's single
    global permutation.
* For a fixed seed, the parameters, optimizer state and environment state are bitwise reproducible, however the
  iterations are split across launches and wherever the rollout buffer lives. The exceptions are two logging
  statistics, `episode_return_sum` and `step_reward_sum`: the kernel sums them with float atomics, so they can
  differ in the last bits.
* `scripts/reinforcement_learning/train.py` only accepts `--rl_library sram` when it is started through
  `scripts/sram/train_sram.py --via-isaaclab-cli`. That wrapper adds the backend to the private
  `isaaclab_rl.entrypoints.dispatch._BACKEND_MODULES` table at run time. The backend rejects options that need a
  simulator, such as `--video`, `--distributed` and resuming from `--checkpoint`.
* `verify_megakernel.py` checks every update of the iteration teacher-forced from the kernel's own state
  (parameters, Adam moments, learning rate); its free-running 20-update replay is informational, because ~1e-7
  rounding differences are amplified through 20 sequential non-linear updates (and through samples whose
  probability ratio sits on a PPO clip boundary).
