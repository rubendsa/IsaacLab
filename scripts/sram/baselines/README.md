# Cartpole environment and learner ablations

These scripts compare the stock RSL-RL learner with an analytic PyTorch Cartpole environment, its
`torch.compile` variants, and the native Isaac Lab environment. They help separate simulator, environment,
learner and launch costs for Cartpole. They do not establish performance or learning quality for the
Kuka-Allegro visual task. See [the technical report](../../../docs/sram/SRAM_TRAINING.md) for current measurements,
timing boundaries, and the distinction between local SRAM working sets and global storage.

Earlier exploratory timings used a shared GPU and changing kernel revisions. Their raw logs are not included
here, so this README makes no numerical speedup or learning-equivalence claim from those runs. Kernel counts
alone cannot attribute a speedup to SRAM residency, and a global reduction buffer is not guaranteed to remain
in L2. Use the report's archived evidence and rerun these ablations before drawing additional conclusions.

## Files

| File | Content |
|---|---|
| `torch_cartpole_vecenv.py` | `CartpoleTorchVecEnv`, an `rsl_rl.env.VecEnv` with closed-form MuJoCo-Warp `implicitfast` cart-pole physics. It has two modes, `eager` and `compiled` (`torch.compile`). The CLI adds `--check` (numerical cross-checks) and `--bench` (`env.step` micro-benchmark). |
| `train_rsl_rl_analytic.py` | Mirrors `isaaclab_rl/entrypoints/backends/train_rsl_rl.py` but runs the stock `OnPolicyRunner` on the analytic env. Instrumentation options: per-iteration metrics, `--breakdown` for synchronized phase timing, `--profile_iteration` for a kernel census. `--env isaaclab` runs the *real* env (torch or warp frontend) under the same instrumentation. |
| `README.md` | This file. |

Neither script edits any Isaac Lab, RSL-RL or `isaaclab_sram` file. The optional instrumentation only wraps
runner methods in-process. `isaaclab_sram` is imported from `source/isaaclab_sram` through `sys.path`, and its
constants come from `isaaclab_sram.cartpole.CartpoleTask`.

## The analytic VecEnv

Every control step mirrors `DirectRLEnv.step` + `CartpoleEnv` + `RslRlVecEnvWrapper.step`:

1. Effort `f = 100·a`. Actions are not clipped (`clip_actions=None`) and no noise is added.
2. Two substeps (decimation 2) at `dt = float32(1/120)` of the closed-form 2-DoF cart-pole. The integrator
   matches MuJoCo-Warp `implicitfast` as configured by Newton 1.6.1, and the equations, constants and float32
   operation order are those of the megakernel's `cartpole_substep` (`isaaclab_sram/kernels/cartpole_physics.cuh`):
   `(M + dt·diag(kv,0))·qacc = [-mp·l·sinθ·θ̇² + clamp(-kv·ẋ, ±400) + f, mp·g·l·sinθ]`, then `q̇ += dt·qacc`,
   then `q += dt·q̇`. The feed-forward effort is **not** clamped. The ±400 N limit clamps only the damping
   (actuator) force, and its implicit `dt·kv` term stays in the mass matrix.
3. Increment `episode_length`. `terminated = |x| > 3` (there is no pole-angle termination), and
   `time_out = episode_length >= 300`.
4. Compute the reward from the post-physics, pre-reset state, with the same op order as `compute_rewards`
   (`wrap_to_pi` on θ only).
5. Reset finished envs in the same step. The draws are `torch.rand·(hi-lo)+lo` in Isaac Lab's order
   (x, ẋ, θ, θ̇), clamped to the soft joint limits.
6. Return the observation `[x, θ, ẋ, θ̇]` of the post-reset state, with θ unwrapped.
7. Wrapper conventions: `dones = (terminated|time_out).long()` and `extras["time_outs"]`. Like Isaac Lab, the
   same `extras` dict object is returned every step, and `extras["log"]["Metrics/success_rate"]` keeps its value
   on steps without resets.

Two quirks are reproduced on purpose:

* **First-reset zero velocities.** The first `reset()` happens in the constructor, as with the RSL-RL wrapper.
  At that point Newton's `soft_joint_vel_limits` are still 0, so the sampled velocities are clamped to 0.
  Disable this with `--no_first_reset_quirk`.
* **`init_at_random_ep_len`.** The stock runner handles it through the `episode_length_buf` setter.

Seeding follows `DirectRLEnv` (`configure_seed(seed)` before the first reset). Verify first-reset behavior and
initial network weights against the installed Isaac Lab/Newton revision using the checks below.

The two modes run the same Python function (`make_env_step`):

* **`--mode eager`**: plain PyTorch operations.
* **`--mode compiled`**: the whole step (physics, reward, termination, reset sampling, obs, success metric) is
  fused with `torch.compile(fullgraph=True)`.
  * `max-autotune-no-cudagraphs` is the default.
  * `reduce-overhead` also CUDA-graph captures the step, with
    static-address buffers, and clones the outputs, because RSL-RL keeps the previous observation alive across
    `env.step`.
  * The compile runs in the env constructor, under inference mode and with inference-tensor actions, because
    dynamo guards on both. CPU and CUDA RNG states are restored afterwards, so iteration 0 is not charged for
    compilation.

## Commands

Start in the repository root. The commands select its uv-managed environment and put logs in a scratch directory:

```bash
ablation_repo=$(git rev-parse --show-toplevel)
ablation_scripts="$ablation_repo/scripts/sram/baselines"
mkdir -p /tmp/sram_ablation
cd /tmp/sram_ablation
export OMP_NUM_THREADS=1

# Record the real engine, then compare recorded transitions and analytic implementations.
uv run --project "$ablation_repo" python "$ablation_scripts/record_newton_rollouts.py" \
    --actions random --resets on
uv run --project "$ablation_repo" python "$ablation_scripts/record_newton_rollouts.py" \
    --actions random --resets off --action_amp 10 --steps 120
uv run --project "$ablation_repo" python "$ablation_scripts/torch_cartpole_vecenv.py" \
    --check --megakernel \
    --dataset ref_random_resetson_n4096.npz ref_random_resetsoff_n4096_amp10.npz --out check.json
uv run --project "$ablation_repo" python "$ablation_scripts/torch_cartpole_vecenv.py" --bench --out bench.json

# Stock learner with eager, compiled and native environment execution.
uv run --project "$ablation_repo" python "$ablation_scripts/train_rsl_rl_analytic.py" --mode eager
uv run --project "$ablation_repo" python "$ablation_scripts/train_rsl_rl_analytic.py" --mode compiled
uv run --project "$ablation_repo" python "$ablation_scripts/train_rsl_rl_analytic.py" \
    --mode compiled --compile_mode reduce-overhead
uv run --project "$ablation_repo" python "$ablation_scripts/train_rsl_rl_analytic.py" \
    --mode compiled --policy_compile_mode max-autotune-no-cudagraphs
uv run --project "$ablation_repo" python "$ablation_scripts/train_rsl_rl_analytic.py" --env isaaclab
```

Add `--frontend warp` for the alternate native frontend. Use `--breakdown` for synchronized phase timing, or
`--max_iterations 12 --profile_iteration 10 --eval_steps 0` for a kernel census. Instrumentation perturbs wall
time; compare these results separately from uninstrumented training.

Evaluate a saved policy in the actual Newton environment, replacing the checkpoint path with the produced run:

```bash
uv run --project "$ablation_repo" python "$ablation_repo/scripts/benchmarks/play.py" \
    --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 4096 --num_steps 600 \
    --checkpoint logs/rsl_rl/cartpole_direct/REPLACE_WITH_RUN/model_149.pt \
    --output_path play_out --visualizer none physics=newton_mjwarp
```

Each training run writes `ablation_metrics.json` to its log directory. The file holds every iteration's runner
values (collection/learning time, steps/s, Train/mean_reward, LR, std, losses), the steady-state medians, the
optional breakdown or profile, and a deterministic evaluation of the final policy on the analytic env
(4096 envs × 600 steps, using the play benchmark's episode bookkeeping). The TensorBoard events and
`model_*.pt` checkpoints are the stock ones.

## Validation and known differences

The analytic environment implements the task's closed-form physics, reward, termination, timeout, reset and
observation bookkeeping; it does not execute the native Newton solver. A float64 implementation of the same
equations tests numerical implementation. Recorded Newton transitions additionally test agreement with the
native engine. Archive the check report and the recorded dataset's source/configuration alongside any claim.

- Analytic resets draw random values for all environments before masking; native resets draw for finished
  environments. Inductor also uses a different random stream. Matching seeds does not guarantee matching
  trajectories after reset sampling diverges.
- Native Newton integrates in world coordinates, so floating-point rounding can differ from the analytic
  joint-coordinate update. Compare step errors, rewards and done flags with the recorded-engine check.
- The cart joint-limit constraint is omitted. Under the normal task configuration, termination occurs before
  that limit is reached. High-force tests must explicitly exclude or account for joint-limit contacts.
- The persistent megakernel uses different random streams and stratified per-block minibatches. Compare
  learning statistically; a successful analytic run does not establish stock numerical identity.
- Episode return, rather than episode length alone, distinguishes a successful Cartpole policy from a
  stationary policy that survives to the timeout. Evaluate exported policies in the native engine too.

Before publishing results, archive commands, source revision, installed versions, GPU/process information,
warmup, raw rows, precision and timing boundaries. Repeat across seeds for learning claims. A faster analytic
Cartpole experiment cannot demonstrate an all-SRAM visual learner.
