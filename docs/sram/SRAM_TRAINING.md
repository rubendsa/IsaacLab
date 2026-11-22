# SRAM-oriented training in Isaac Lab 3.0

Branch: `sram-resident-training`. Base: Isaac Lab `v3.0.0`, commit
`e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`. Research checked on 2026-11-22.

The opportunity is to keep repeatedly used learner intermediates on chip and remove launch and synchronization
overhead from PPO. This branch contains a persistent Cartpole training megakernel, fused operations for the native
Kuka-Allegro camera learner, a real-environment runner and a separate learner benchmark. The Cartpole implementation
can keep its rollout, environment state, parameters and optimizer
working state in shared memory and registers during a launch. Real Kuka RGB64 measurements below test selective
fusion with the original trainable CNN. The native camera rollout cannot fit entirely in the GPU's SRAM.

For Ruben: the useful experiment is to measure how much a fused learner saves on the real Kuka-Allegro lift recipe,
then measure the effect on complete training iterations and time to task success. The native RGB64 rollout alone
contains 1.5 GiB of images at 4,096 environments. The credible destination is a learner that streams compact inputs
and keeps each computation tile on chip, with a separately measured small-model experiment for fully persistent
training. Neither CUDA graphs nor a persistent kernel imply zero device-memory traffic.

## What is implemented, and what each result would prove

| Path | Work performed | What it can establish |
| --- | --- | --- |
| `source/isaaclab_sram/isaaclab_sram/megakernel.py` and `kernels/ppo_cartpole.cu` | Persistent cooperative CUDA kernel: analytic Cartpole physics, rollout, GAE, PPO backward, gradient reduction and Adam | Whether a sufficiently small complete learning loop benefits from persistent on-chip state |
| `scripts/sram/verify_megakernel.py` | Numerical comparison with float64 closed-form physics, PyTorch autograd and RSL-RL optimizer references | Numerical agreement for the tested Cartpole configurations; not Kuka physics or visual learning |
| `scripts/sram/projection/measure_update_cost_cnn.py` | Native Kuka actor/critic learner on synthetic observations; eager versus captured CUDA graphs, precision modes, memory and operation accounting | Learner throughput and launch-overhead opportunity for the native architecture; not rendered observations or policy quality |
| `source/isaaclab_sram/isaaclab_sram/perception.py`, `spatial_softmax.py`, `ppo_kernels.py` | Native trainable CNN and critic with fused spatial softmax, PPO objective, GAE and Adam | Whether selective fusion improves the original learner while preserving its gradient computation |
| `scripts/sram/train_kuka_perception.py` | Real Kuka camera rollouts and native PPO with `torch`, `sram` and `sram_graph` backends, checkpoint and integration checks | Actual camera/task execution and iteration throughput; longer evaluation is still needed for learning quality and time to success |

The branch includes [real camera run summaries](data/perception/README.md),
[isolated kernel measurements](data/kuka_kernel_microbenchmark.json) and the [device query](data/gpu.json). Historical
`data/locomotion_footprint.json` and `data/rl/` describe different workloads; they cannot establish Kuka RGB speed or
fit.

## What SRAM means here

Each SM has its own registers and shared memory. Shared memory is scoped to the thread block and does not preserve
values across ordinary kernel launches. Keeping a block alive can preserve its own state, subject to register and
shared-memory capacity. Compiler spills may use device memory. GPU global allocations are backed by device DRAM
even if many accesses hit L2. These are distinct residency claims. See the
[CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
and [shared-memory lifetime](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-c-programming-guide/index.html#shared).

| Claim | Evidence needed |
| --- | --- |
| Intermediate activations stay on chip within a fused region | Kernel source/layout, actual shared-memory and register usage, spill counts and memory profiling |
| Parameters and optimizer state persist on chip for a training launch | Persistent allocation/lifetime, no parameter or moment loads/stores inside the iteration loop, and no spills |
| Rollouts stay on chip | Capacity calculation and verified launch plan, including all fields, padding and scratch space |
| Global buffers are usually served from L2 | L2 hit rate and DRAM-byte counters for the measured run |
| The entire native simulator, renderer and learner use only SRAM | All of the above for every participating component; this branch does not satisfy this claim |

CUDA's persisting-cache policies preferentially retain data; they do not turn global allocations into guaranteed
SRAM storage. A buffer fitting in L2 is a capacity observation, not proof that its accesses avoid DRAM. See
[NVIDIA L2 cache control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html).

Blackwell products have different shared-memory and occupancy limits. The planner must query the actual GPU
rather than applying one
Blackwell product's limits to all Blackwell GPUs. Shared-memory capacity, register pressure, block residency and
barrier overhead all constrain fusion. See the
[Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html).

The [actual device query](data/gpu.json) on this RTX PRO 6000 Blackwell reports 188 SMs,
100 KiB shared memory per SM, 99 KiB per block, 18.36 MiB aggregate shared memory,
47 MiB aggregate register files and 128 MiB L2. The native 2.28 GiB observation rollout
exceeds even the sum of these physically separate on-chip stores. Their capacities
cannot be pooled into a single directly addressable SRAM allocation.

## The Cartpole megakernel

One cooperative launch executes the requested number of training iterations. Each block owns a fixed slice of
environments and a replica of the small actor/critic. The loop computes actions, advances an analytic two-degree-
of-freedom Cartpole, collects rewards, resets environments, computes GAE, then runs PPO minibatches and Adam.
Cross-block gradient reduction and normalization use global scratch buffers and grid synchronization.

The implementation is specialized for four observations, one action and two `[32, 32]` ELU networks. Its analytic
Cartpole follows the task's Newton/MuJoCo-Warp settings; it does not execute the general Newton solver in the kernel.
It is a useful complete-loop demonstration, but substituting this physics for Kuka's articulated contacts would
change the task.

| State | Cartpole location during a non-debug launch |
| --- | --- |
| Environment and episode state | Shared memory; loaded from and saved to global checkpoint buffers at launch boundaries |
| Rollout fields | Shared memory when the queried launch plan fits; otherwise global memory |
| Weights | Replicated shared-memory/register working copies, with global input/output buffers |
| Adam moments | Per-owner-thread working values; global input/output buffers at launch boundaries |
| Activations and local gradient scratch | Shared memory/registers; inspect compiled register/spill reports |
| Minibatch permutation | Shared-memory cache when the planner can fit it; otherwise recomputed keyed permutation |
| Gradient partials and reduced gradients | Global memory, potentially served by L2; no guaranteed L2-only placement |
| Statistics, debug output, profiling output | Global memory; debug/profiling builds change traffic |

Initialization, checkpoint export and launch boundaries also transfer data. The kernel uses fixed-order gradient
reduction, but two logging sums use floating-point atomics. Bitwise parameter determinism tests and numerical
comparisons should be reported separately from claims about stock RSL-RL equivalence: random streams and the
per-block stratified minibatch permutation differ from the stock global permutation.

A fresh 150-iteration run at 4,096 environments and seed 42 measured the following
on the same GPU. Both exported policies were evaluated in stock Newton Cartpole
on 8,192 held-out episodes with evaluation seed 10002:

| Cartpole path | Process wall time | Training loop | Held-out return (maximum ≈5) |
| --- | ---: | ---: | ---: |
| Stock Isaac Lab/RSL-RL | 30.64 s | 21.13 s | 4.9455 ± 0.0697 |
| Persistent analytic megakernel | 3.28 s | 0.0738 s | 4.9502 ± 0.0685 |

Process wall time, including imports, setup and checkpoint export, improved 9.34×.
The loop comparison uses rounded host iteration times for stock and CUDA events
for the megakernel; it excludes compilation/setup and substitutes specialized
analytic physics for the general engine. NVRTC's cache was warm. This is a small
complete-loop feasibility result, **not a predicted Kuka speedup**. See the
[raw stock run](data/cartpole/stock_torch_s42.json),
[megakernel run](data/cartpole/sram_s42.json) and
[commands/source hashes](data/cartpole/manifest.json).

The compiled kernel used 223 registers and reported 32 bytes local memory per
thread, with 72,696 bytes dynamic shared memory per block. The local-memory report
and global cross-block reductions prevent a zero-DRAM claim even for this path.

## Native Kuka-Allegro perception workload

The target is `Isaac-Lift-KukaAllegro-Camera` with `presets=single_camera,rgb64`,
`physics=newton_mjwarp` and `renderer=newton_renderer`. The local
[agent configuration](../../source/isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/agents/rsl_rl_ppo_cfg.py)
and [camera configuration](../../source/isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/camera_cfg.py)
define:

- A 23-action policy and 32 control steps per rollout; five PPO epochs and eight minibatches, with fixed learning
  rate `7e-5`.
- A trainable RGB actor: uint8 `3 × 64 × 64` images, three ELU convolutions with channels `[16, 32, 32]`, kernels
  `[8, 4, 3]`, strides `[4, 2, 1]`, learned-temperature spatial softmax, and an MLP `[512, 256, 128]`.
- An actor with 699 MLP inputs: 170 policy/history values, 465 proprioception/history values and 64 encoded visual
  coordinates. A privileged critic consumes 1,595 state/history/point-cloud inputs and uses the same hidden widths.
- Running normalization for one-dimensional observation groups. Images remain uint8 in rollout storage and are
  converted/scaled inside the model.

This is a perception workload with additional state inputs. In particular, `policy` contains object orientation,
and the critic's point cloud comes from known object geometry and pose. It is not a pixels-only lift benchmark.
Preserving this routing matters when comparing implementations.

The following are **calculated lower bounds**, not allocator measurements. They count unique RGB images and the
three float32 observation groups once per transition, without actions, values, advantages, old log probabilities,
optimizer state, activations, simulation buffers or rendering workspaces:

| Environments | RGB64 image rollout, 32 steps | Float32 state/history rollout | Combined lower bound |
| ---: | ---: | ---: | ---: |
| 256 | 96 MiB | 49.84 MiB | 145.84 MiB |
| 1,024 | 384 MiB | 199.38 MiB | 583.38 MiB |
| 4,096 | 1,536 MiB | 797.50 MiB | 2,333.50 MiB |

Formulas: `N × 32 × 3 × 64 × 64` bytes for images and `N × 32 × 1595 × 4` bytes for state/history. Even the
256-environment input rollout exceeds a 128 MiB L2 cache. Registers and per-SM shared memory are also partitioned
resources, so adding their advertised capacities does not create one freely addressable memory pool.

For these layer shapes, an analytic parameter count is about 1.53 million including the action standard deviations
and spatial-softmax temperatures. Float32 parameters, gradients and two Adam moments require about 23.31 MiB before
any replicas, activations or workspace. The resolved model and its allocator measurements remain authoritative.

Storing only encoded features would be an algorithm change if the encoder remains trainable: PPO re-evaluates old
images under current encoder parameters in each minibatch. A frozen encoder, compressed images, smaller networks,
fewer environments or shorter rollouts are valid separate experiments, but each changes the native comparison and
must be labeled. Recomputing feature tiles during backward can trade arithmetic for activation storage while
retaining the original image inputs and trainable encoder.

## Megakernel design options

The persistent Cartpole implementation explores the strongest form of fusion: simulator plus learner, including
the entire optimizer loop. For native visual manipulation, the practical next boundary is the learner update.
The simulator and renderer produce global observation tensors; a learner can stream those inputs into tiles,
keep short-lived activations on chip, and return actions or updated parameters. This reduces traffic without
claiming that the full rollout or renderer is resident.

Three primary-source precedents help choose that boundary:

1. [NVlabs tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) demonstrates fully fused small MLPs and runtime
   integration of model device functions into larger kernels. Its documentation also warns that large widths or
   models can make fusion slower and that the fully fused MLP requires substantial shared memory. It provides a
   useful small-network precedent, not a turnkey implementation of this spatial-softmax CNN and PPO update.
2. [Hazy Research's Llama megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) uses persistent
   execution, shared-memory paging and explicit dependency counters to reduce launch gaps and pipeline work.
   Weights still arrive from global memory. Its
   [released implementation](https://github.com/HazyResearch/Megakernels) is an inference example; its published
   speedups cannot be transferred to robot training or CNN backward passes.
3. Recent research on
   [megakernel residency and resource costs](https://arxiv.org/abs/2611.00495) and
   [tile-level dependencies between separate kernels](https://arxiv.org/abs/2611.16248) shows why persistent execution
   should compete with optimized separate kernels. These are inference studies, but motivate a CUDA-graph and
   library-kernel baseline before committing to a large custom scheduler.

A CNN implementation must handle convolution forward and both gradients, ELU, spatial softmax and its trainable
temperatures, policy/value losses, gradient clipping, normalizer updates and Adam. Fusion that preserves all of
those operations is a stronger test than timing a small MLP over precomputed frozen image features. Large GEMMs
and convolutions should retain efficient tensor-core tiling; reducing launches at the expense of arithmetic
throughput can lose overall.

The implemented perception path preserves the native convolution/GEMM libraries and trainable CNN. It fuses
spatial-softmax probabilities and coordinate expectations within each feature-map tile; backward recomputes
probabilities, including gradients for the learned temperatures. Separate fused kernels cover GAE and the PPO
objective/backward, and PyTorch's fused Adam applies optimizer updates. This is selective fusion across several
kernels, not one persistent visual training megakernel. Convolution activations, inputs/outputs, gradient partials
and persistent learner state still use global memory. The native camera recipe uses fixed learning rate and
float32 model state with TF32 library operations; unsupported mixed precision, recurrent/distributed training,
RND, symmetry, adaptive scheduling and alternate objectives fail explicitly.

For a subsequent complete persistent visual learner, 64 environments are a
plausible capacity experiment. Losslessly reconstructing repeated state histories
reduces estimated rollout plus float32 parameters, gradients and Adam moments to
50.70 MiB, before scratch, padding and normalizers. At 128 environments this grows
to 78.08 MiB, already above aggregate registers/shared memory. These are design
calculations, not implemented storage reductions or proof of residency: each SM
has its own address space and cross-SM state exchange still needs a mechanism.

That experiment would retain the native CNN, actor/critic, 23 actions and all 40
PPO optimizer steps, reconstruct histories and recompute activation tiles inside
a persistent cooperative learner. The simulator would supply observations at an
explicit boundary. Start with cache-backed state and measure DRAM traffic before
asserting SRAM residency. Warp-NN has convolution backward support but materializes
global layer arrays; tiny-cuda-nn does not directly implement this architecture.
Neither provides a ready-made complete visual PPO megakernel. The branch's
working perception implementation is the selective fusion described above.

## How fast could it be?

On one RTX PRO 6000 Blackwell Server Edition (188 SMs, driver 580.178.04, CUDA 13.0,
PyTorch 2.11.0, RSL-RL 5.5.1), the first paired real RGB64 runs measured:

| Native Kuka RGB64, 4,096 environments | Iteration median (p10–p90) | Learner median | Transitions/s |
| --- | ---: | ---: | ---: |
| Stock RSL-RL | 1.526 s (1.374–1.595) | 563 ms | 85,882 |
| Fused operations | 1.307 s (1.300–1.353) | 433 ms | 100,322 |
| Fused operations + CUDA graph | 1.317 s (1.304–1.357) | 432 ms | 99,491 |

Each run uses seed 42, native reset initialization, 30 iterations and five warmup
iterations. Total iteration timing synchronizes CUDA and includes physics,
rendering, collection and learning. The phase split uses host timer attribution;
startup/compilation, logging and checkpoint writes are outside the steady-state
numbers. Raw rows and source hashes are in [the run summaries](data/perception/README.md).
The short pair gives **1.17× iteration throughput** and **1.30× learner throughput**.
Stock timings varied substantially, so this is preliminary. Graph replay adds no
measurable benefit at this batch size; at 64 environments it reduced the fused
learner from about 146 ms to 28 ms in a three-iteration execution smoke test.

All three native-scale runs produced real RGB observations, changed CNN weights,
kept parameters finite and loaded checkpoints into an unmodified native actor.
Maximum action disagreement in IEEE float32 was below `4.2e-7`; TF32 comparisons
are separately recorded because library matmul and fused reductions round
differently. RGB and depth graph smoke runs also passed. None established lift
success: the native task success metric remained zero through these short runs.

![Real RGB64 camera observations from 16 native Kuka environments](data/perception/kuka_rgb64_camera_preview.jpg)

Isolated hot paths show where fusion helps. These are 100-repeat CUDA-event
medians with **both implementations captured in CUDA graphs**, excluding capture
and compilation. They use native minibatch/rollout shapes and do not include CNN
convolutions, gathering, optimizer updates, simulation or rendering:

| Operation | Native | Fused | Speedup |
| --- | ---: | ---: | ---: |
| PPO objective forward/backward, 16,384 × 23 actions | 0.144 ms | 0.0226 ms | 6.37× |
| Spatial softmax forward/backward, 16,384 × 32 × 4 × 4 | 0.373 ms | 0.0501 ms | 7.44× |
| GAE and global normalization, 32 × 4,096 | 0.462 ms | 0.0140 ms | 33.15× |

The [raw microbenchmark](data/kuka_kernel_microbenchmark.json) also includes eager
controls, wall times and compiled resources. Its eight Triton kernels report zero
spill slots, 12–76 registers per thread and 0–128 bytes shared memory per block.
This supports the residency of their local intermediates; global inputs, outputs
and partial reductions remain. DRAM-byte profiling has not been performed.

Separate three quantities when interpreting these results:

- **Learner speedup:** same collected transitions, same network, PPO settings and precision, eager versus optimized.
- **Iteration speedup:** collection plus learning, including physics, cameras, reset and observation handling.
- **Time to success:** evaluation against the same lift criterion across several seeds, including startup and
  compilation where those costs matter to the use case.

For a measured learner fraction `f` and learner-only speedup `s`, the iteration speedup is
`1 / ((1 - f) + f / s)` when collection time is unchanged. For example, if learning consumes half the iteration,
doubling learner throughput gives `1.33×` overall; even a zero-time learner would only yield `2×`. These are
illustrative arithmetic, not results from this branch. If fused inference also improves collection, measure that
term separately rather than crediting the update benchmark for it.

The CNN benchmark already supports synchronized host timing, CUDA-event spans, profiler kernel counts,
activation/storage accounting and optional comparison with recorded labperf collection times. Its synthetic
images permit controlled learner experiments, but do not exercise camera rendering, manipulation contacts,
observation history evolution or convergence. Mixed-precision results must be separated from the FP32/TF32
baseline; they combine a precision change with scheduling changes.

## Reproduce and validate

Validation on this GPU passed 102 Cartpole tests and nine perception tests, run
one file per process. The perception tests compare outputs, gradients and complete
native PPO updates, including all model parameters, normalizer buffers and Adam
state. Graph tests resume nonzero optimizer state and change observations across
updates to detect stale captured buffers. Multi-GPU execution was not tested.

The Cartpole verifier uses float64 autograd for its first-minibatch gradient
oracle: at 65,536 environments the CPU float32 reduction erred by `2.38e-6`, while
the CUDA result was within `1.39e-8` of float64. The existing acceptance tolerance
was retained. A deliberate one-step-late timeout mutation passed the former
state-derived check and is rejected by the independent episode-counter check.

Run from the repository root in the uv-managed environment. The experimental package is an editable dependency
in the root project and lockfile. Use the package's
[README](../../source/isaaclab_sram/README.md) for its API and supported Cartpole settings.

```bash
# Numerical verification of the persistent Cartpole loop.
uv run python scripts/sram/verify_megakernel.py --num_envs 4096 --out /tmp/cartpole-parity.json

# Persistent training plus export and evaluation in the stock Newton Cartpole environment.
uv run python scripts/sram/train_sram.py --num_envs 4096 --max_iterations 150 --eval

# Native Kuka learner architecture on synthetic RGB and state observations.
OMP_NUM_THREADS=1 uv run python scripts/sram/projection/measure_update_cost_cnn.py \
    --tasks dex_rgb64 dex_state --modes tf32 bf16 --repeats 20 --out /tmp/kuka-learner.json

# Paired real-environment runs, preserving the native camera task and reset settings.
OMP_NUM_THREADS=1 uv run python scripts/sram/train_kuka_perception.py \
    --backend torch --camera rgb64 --num_envs 4096 --iterations 200 --warmup_iterations 5 \
    --seed 42 --output logs/sram/kuka_rgb64_torch
OMP_NUM_THREADS=1 uv run python scripts/sram/train_kuka_perception.py \
    --backend sram --camera rgb64 --num_envs 4096 --iterations 200 --warmup_iterations 5 \
    --seed 42 --output logs/sram/kuka_rgb64_sram
OMP_NUM_THREADS=1 uv run python scripts/sram/train_kuka_perception.py \
    --backend sram_graph --camera rgb64 --num_envs 4096 --iterations 200 --warmup_iterations 5 \
    --seed 42 --output logs/sram/kuka_rgb64_sram_graph
```

Use `--camera depth64` for the corresponding depth preset. A small execution smoke test can use `--num_envs 64
--iterations 3 --warmup_iterations 1 --smoke_reset_bank 16`; the explicit reset-bank override changes initialization
coverage and is recorded in the summary. Do not quote it as native-task learning evidence. Full comparisons omit
that override and hold the task, seed, reset distribution, network, precision and PPO settings fixed.

Each real-environment run writes an RSL-RL checkpoint, `params/`, `run.json`, TensorBoard events and
`perception_summary.json`. The summary records observation shapes/dtypes, rendered-image statistics, whether CNN
weights changed, finite parameters, checkpoint action replay, image sensitivity and per-iteration timings. The
`torch` backend is the reference learner; `sram` selects the fused learner operations;
`sram_graph` captures each minibatch update. All retain global state. The graph path
omits unused optimization-time action samples, changing subsequent rollout RNG
streams without changing the objective. Use `--iterations 30` to reproduce the
initial short comparison above.

Before publishing a numerical claim, archive the GPU model and count, driver/CUDA/PyTorch/RSL-RL versions, base
commit, branch commit, command, seed, observation shapes/dtypes, model dimensions, renderer/physics selectors,
precision, rollout size, optimizer settings, warmup, repeats and timing boundaries. Run on an otherwise idle GPU
and report median and spread. Compare equivalent successful updates, not compilation or a skipped backward pass.

For a perception implementation, require a real rendered-observation rollout, finite forward/backward values,
nonzero visual-encoder parameter updates, input-sensitive actions and a reloadable checkpoint. Then compare
learning curves and held-out lift success to the native recipe. A short smoke run proves execution, not task
mastery. SRAM claims additionally require compiled shared-memory/register/spill inspection and profiler evidence
for global-memory traffic. These remain separate acceptance gates so a passing synthetic benchmark cannot stand
in for successful visual manipulation training.
