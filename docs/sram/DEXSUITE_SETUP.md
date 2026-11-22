# Perceptive workloads on-chip: a DexSuite setup

Design study for SRAM-resident training of a camera-based dexterous-manipulation task. The summary is in
[SRAM_TRAINING.md](SRAM_TRAINING.md#perceptive-workloads-a-dexsuite-setup); this document has the data flow, capacity,
timing model and build plan in full. Nothing here is implemented yet: it is a design with measured inputs.

**Scope.** Isaac-Lift-KukaAllegro-Camera with `presets=single_camera,rgb64`, `physics=newton_mjwarp` and `renderer=newton_renderer`, at 4,096 envs. One iteration is 32 control steps, or 131,072 transitions.

**Evidence tags.**
- **[M]** measured in this study. Every [M] timing was taken on a GPU shared with other agents, inside pmon-gated quiet windows, so treat it as indicative.
- **[L]** an earlier, unpublished benchmark study (labperf) on the same GPU model: runs were sequential, but external GPU contexts were observed at times, so GPU exclusivity was not established. Its data are not part of this branch.
- **[D]** derived. **[A]** assumed.

Stock timings are given as native ADR difficulty 0 (early training) → full-ADR endpoint (late training).

**Paths.** Task files are under `source/isaaclab_tasks/isaaclab_tasks/core/lift/config/kuka_allegro/`. `isaaclab/...` and `isaaclab_newton/...` are relative to `source/isaaclab/` and `source/isaaclab_newton/`, and `kernels/...` to `source/isaaclab_sram/isaaclab_sram/`. `rsl_rl/...` means the installed RSL-RL 5.5.1 package.

**Data.** The [M] values are in [`data/dexsuite/`](data/dexsuite/): the per-phase timing of the stock runs (`phase_timing.json`), the CNN update cost stock vs CUDA graphs (`update_cost_cnn_*.json`, from `scripts/sram/projection/measure_update_cost_cnn.py`), the convolution micro-benchmark, the footprint and FLOP analyses, and the Stream design's time model. The profiling scripts that produced the other files are not part of this branch.

## Three facts that shape the design

1. **The stock camera recipe fails on learning, not throughput.**
   - kuka_rgb_64 converged on 0/3 seeds within 30,000 iterations: 3.93 B transitions per seed, censored at 50,201-54,237 s. Best held-out success was 0.000 / 0.047 / 0.594.
   - kuka_state converged on 3/3 seeds: median 1,262 s and 170.4 M transitions, held-out success 0.859 / 0.859 / 0.828 [L].
   - In the RGB runs, ADR stalled at 0.20-0.68 while action std grew to 1.8-2.05 [L, parsed logs].
   - A faster copy of this recipe only fails sooner. No per-iteration multiplier becomes a time-to-policy multiplier.
2. **Time is the constraint, not bytes.**
   - Stock rgb64 takes 1,445 → 1,565 ms per iteration [M].
   - The ADR0 split: MJWarp physics 399 ms (GPU 64% busy), Newton render 221 ms (216 ms of it ray tracing, the rest camera pose), Isaac Lab managers 287 ms (CPU- and launch-bound, GPU 7-14% busy), act 45 ms, PPO update 484 ms.
   - Of the update, 203 ms is legacy FP32 cuDNN convolution running at 17.5 TFLOP/s [M].
   - Moving all stock image traffic on chip would save at most about 14 ms per iteration [D].
   - Fusing only the learner does little. Capturing the whole stock learner in one CUDA graph takes the update from 484 to 431 ms [M, `scripts/sram/projection/measure_update_cost_cnn.py`]. The fused camera-PPO backend of the concurrent `sram-resident-training` branch shows the same end to end: learning 486 → 433 ms, iteration 1,466 → 1,421-1,438 ms (1.02-1.03x) [that branch's `docs/sram/data/perception/`].
3. **Pixels are the only large object, and they need not stay resident.**
   - A uint8 rgb64 frame is 12,288 B per env-step, or 1.61 GB per rollout.
   - Everything else that persists is 2,404 B per env plus a 13-31 MB learner [D]. That fits in L2.
   - The irreducible piece is the CUDA-core Newton tracer: 2.2-2.5 G primary rays/s, 6.76 → 7.68 ms per 4,096-env frame, so 216 → 246 ms per iteration of ray tracing [M].
   - This caps the per-transition gain on this path at 6.4-6.7x, against ~3-26x (66x optimistic) projected for locomotion if a per-world solver port succeeds.

## Recommended setup

- **Teacher (T): the stock state PPO recipe, unchanged** (`agents/rsl_rl_ppo_cfg.py:64-97`).
  - A converged teacher exists from the earlier, unpublished study: seed 2 of the stock state recipe. It is not included in this branch; training it took a median 1,262 s (21 min) there.
  - It is from iteration 1,300, and its `actor_state_dict` carries a 1,595-input normalizer (checked on CPU).
- **Student (S): the stock camera actor**, `SpatialSoftmaxCNNModel` with 546,206 parameters (`agents/rsl_rl_ppo_cfg.py:50-61`).
  - It is trained with RSL-RL `Distillation` semantics (`rsl_rl/algorithms/distillation.py:116-203`).
  - The student acts stochastically. The frozen teacher labels the same observation with its mean action.
  - The loss is MSE, with gradients accumulated over `gradient_length` steps. There is no reward, critic, GAE or PPO epoch.
  - The teacher's inputs are unchanged: `SingleCameraObservationsCfg` inherits `StateObservationCfg` (`camera_cfg.py:234,248`; `kuka_allegro_env_cfg.py:111`; `kuka_allegro_camera_env_cfg.py:52-53`).
- **Exact mode (E): stock end-to-end PPO**, with the uint8 rollout in a DRAM ring streamed through SMEM (the "Stream" design scored at the end). It is used for RSL-RL parity and for an optional PPO fine-tune that starts from the distilled student.
- **Fallback:** if the student plateaus, add an auxiliary head that regresses object position and surface points (from the "Latent" design) to the student loss.

## Tier map

| Tier | Contents |
|---|---|
| Registers (256 KB/SM, 47 MiB) | MMA accumulators for the conv implicit-GEMM and MLP tiles. The per-CTA conv weight gradient: 20,592 fp32 values, 80 regs/thread at 256 threads, flushed at each optimizer step. Philox counters. Solver temporaries (fused tier). |
| SMEM (99 KB/block, 18.4 MiB), shared between phases | **Encoder phase**, 83 KB total: 2-slot uint8 image ring 24.6 KB, one image's conv maps 23.1 KB, dconv 6.7 KB, conv1 weights 12.3 KB, double-buffered conv2/3 K-chunks 16 KB. **GEMM phase:** double-buffered TF32 tiles. **Physics phase** (fused tier): 1-2 worlds of lean solver scratch at 29-47 KB each. |
| L2 persisting window (sized to the set, not to the 80 MiB maximum) | Student learner (fp32 weights, grads and Adam state) 8.74 MB. Frozen teacher plus normalizer 3.96 MB. Fixed tables 5.9 MB. In the fused tier, also the persistent env state at 2,404 B/env (9.85 MB at 4,096 envs). **Totals:** S 28.5 MB; T 47.2 MB (includes the 31.45 MB PPO learner); E 40.2 MB. |
| L2, normal | The render scene while rendering (38.4 MB: 237,734 unique triangles plus mesh BVHs). Act and update transients, at most 28 MB per 2,048-sample micro-batch. Per-CTA dW slots for the deterministic reduction (15.5 MB). The current image micro-batch. |
| DRAM (95 GiB, 1.6 TB/s) | The uint8 image ring, 1.61 GB at 4,096 envs (used by S with exact Distillation semantics, and by E). The per-frame RGBA render output, 67.5 MB. The teacher rollout in stock fp32 layout, 875 MB, about 3.3 ms per iteration to stream, exact. In the graph tier only, the stock process's per-env buffers (MJWarp/Newton state, renderer and manager buffers): about 368 KB/env, plus a 480 MB (458 MiB) contact buffer. Checkpoints. |

## Per-phase data flow (student stage)

One iteration is one CUDA graph with no host syncs. For each control step t = 0..31:

1. **Render node.**
   - The unchanged Newton `SensorTiledCamera` graph (`isaaclab_newton/renderers/newton_warp_renderer.py:627-651`), 1 spp, with the same shading.
   - It reads device `body_q` and writes RGBA.
2. **Act** (persistent kernel, 188 CTAs of about 22 envs each).
   - **Repack:** RGBA → uint8 3×64×64 into ring slot (t, env). This is the same copy the stock observation term makes.
   - **Encoder:**
     - conv1-3 run as TF32 `mma.sync` implicit GEMMs, with x/255-0.5 folded into the epilogue;
     - the fold is valid because conv1 has no padding (`agents/models.py:93-97`);
     - ELU, then spatial softmax with learned per-channel temperatures, produces 32 keypoints (64 coordinates).
   - **Student MLP:** 699→512→256→128→23 as M=4,096 GEMM phases gives μ.
   - **Action:** a = μ + 0.1ε. The std stays fixed because MSE does not train it.
   - **Teacher MLP:** on the 1,595-D privileged row, it gives the label a*.
   - **Normalizer:** accumulate moments, then a grid barrier.
3. **Physics and MDP.**
   - **Graph tier:** the stock MJWarp graph runs 4 × [target = q + 0.1a; 2 substeps]. Device-side manager kernels handle rewards, terminations, the reset bank, goals, ADR, the history with first-push backfill, and noise.
   - **Fused tier:** CTAs pull worlds from a queue and step them in SMEM waves: 2 worlds/SM, 11 waves per control step. They write the raw frame and `body_q` back to L2.
   - A grid barrier follows either way.

**Update** (exact `Distillation.update`, `distillation.py:143-203`, with E ∈ {1, 2} epochs):

4. For each t:
   - Stream slot t (50 MB) through SMEM in 2,048-image micro-batches.
   - Run the encoder forward, the MLP forward, MSE against the stored a*, the MLP backward, and the encoder backward with conv recompute.
   - Every 16 steps: a fixed-order dW reduction, then sharded Adam over the 546,206 parameters.
   - Per-step backward with accumulation equals stock's single backward over the 16-step graph, up to fp32 reduction order.
5. **End of iteration.**
   - Update the student normalizer (`distillation.py:192`), write logs on device, and checkpoint asynchronously.
   - Every 25 iterations (finer than the earlier study's 50-100-iteration cadence, to resolve S_student), run held-out evaluation with the earlier study's protocol: 128 envs, difficulty 1.0.
   - The output is a stock-format camera actor.

**Optional streaming variant** (only after parity is shown):
- Each frame is used once, inside step 2, with no ring.
- This saves 8-17 ms per iteration but moves optimizer steps into the middle of the rollout, so it has to be re-validated against the P0 learning curve.

**Teacher stage:**
- The same rollout loop with actor and critic GEMMs.
- Then GAE (0.64 ms when graph-captured [M]), 5 epochs × 4 minibatches of 32,768, and the normalizer (`rsl_rl/algorithms/ppo.py:190-191, 241, 309-310, 327-330`).
- Most of this kernel work is shared with the locomotion L2-resident design.

**Mode E:**
- Images go to keyed-permutation ring slots, so each 16,384-sample minibatch is a contiguous range.
- 2,048-image micro-batches stream through the encoder tile with recompute and deterministic dW.

## Capacity (rgb64)

| Working set | Tier | Per env | 4,096 envs | 16,384 envs | Tag |
|---|---|---|---|---|---|
| Student learner + frozen teacher | L2 persisting | - | 12.7 MB | 12.7 MB | D |
| Persistent env state (fused tier) | L2 persisting | 2,404 B | 9.85 MB | 39.4 MB | D |
| Fixed tables (lean; stock bank + monitor is 6.55 MB) | L2 persisting | - | 5.9 MB | 5.9 MB | D/M |
| Render scene | DRAM; L2 while rendering | - | 38.4 MB | 38.4 MB | D |
| Act/update transients + dW slots | L2 | - | 43.5 MB | 43.5 MB | D |
| RGBA render output | L2 → DRAM | 16,480 B | 67.5 MB | 270 MB | M |
| uint8 image ring (32 steps) | DRAM | 393 KB | 1.61 GB | 6.44 GB | D |
| Student 1-D rollout (635-D fp32 + labels, or lean raw frames) | DRAM or L2 | 87 KB or 15 KB | 357 MB or 62 MB | 1.43 GB or 248 MB | D |
| Encoder tile | SMEM | - | 83 of 99 KB/block | same | D |
| Physics scratch (fused tier) | SMEM | 29-47 KB per in-flight world | 376 in flight, 11 waves/step | 44 waves/step | D |
| Teacher PPO learner / stock-layout rollout | L2 / DRAM | 213.7 KB | 31.45 MB / 875 MB | 31.45 MB / 3.5 GB | M |
| Mode E learner / pixel traffic per iteration | L2 / DRAM | - | 24.4 MB / 9.7-19.3 GB (6-12 ms) | 24.4 MB / x4 | M/D |
| Graph tier: stock per-env buffers + contact buffer | DRAM | about 368 KB | 1.5 GB + 0.48 GB fixed | 6.0 GB + 0.48 GB | M |
| Reference: stock RSL-RL rgb64 | DRAM | 607 KB rollout | 2.49 GB + 5.12 GB update peak | about 45K-env process limit | M |

What this means:
- **SMEM alone never holds a configuration** once a learner is counted: the student set is 18.6 MB before any env state (28.5 MB at 4,096 envs), and the teacher's PPO learner alone is 31.45 MB, against 18.4 MiB (19.3 MB) of SMEM split across 188 SMs.
- **The student persisting set** is 28.5 MB at 4,096 envs and 58 MB at 16,384. It fits the persisting window up to about 27K envs [D], and DRAM never binds (about 240K envs).
- **Throughput per transition is assumed flat above 4,096 envs** once fused [A]: the model scales the tracer and kernels linearly, and nothing above 4,096 envs was measured. Keep the 4,096 envs the recipes were tuned at.

## Per-iteration time (ms per 131,072 transitions, rgb64, 4,096 envs, ADR0 → endpoint)

| Phase | Stock E2E PPO [M] | P0 stock-code DAgger [D] | G graph tier [D/A] | F fused, expected [D/A] | F fused, optimistic [D/A] |
|---|---|---|---|---|---|
| Physics (32×4×2 substeps) | 399 → 483 | 399 → 483 | 399 → 483 | 102.5 | 32 |
| Newton render (ray trace, without camera pose) | 216 → 246 | 216 → 246 | 216 → 246 | 216 → 246 | 216 → 246 |
| Managers / MDP | 287 | 287 | 36-39 | 8 | 3 |
| Act + learner | 45 + 484 | 45 + 65-95 | 28-53 | 28-53 | 28 |
| Storage, logging, syncs | 16 | 16 | 1-2 | 1-2 | 1 |
| **Total** | **1,445 → 1,565** | **1,026 → 1,176** | **680 → 820** | **356 → 411** | **281 → 310** |
| Transitions/s | 90.7k → 83.7k | 111-128k | 160-193k | 319-368k | 423-467k |
| Speedup per transition (same ADR) | 1.0x | 1.33-1.41x | 1.9-2.1x | 3.8-4.1x | 5.0-5.2x |

**Assumptions.**
- **Stock phases** come from different instrumentation modes, so they sum to about 0.5% above the un-instrumented total.
- **G managers.** This is the measured GPU kernel time of the manager phases (35.6-39.1 ms) [M]. It is assumed reachable once host syncs are gone: `isaaclab/envs/manager_based_rl_env.py:242`, plus the ones in commands, ADR and the logger. G also includes the K1 student kernel.
- **Learner, exact Distillation semantics: 2.1 TFLOP per iteration** [D].
  - Act: student forward 0.435 + teacher forward 0.258 TFLOP.
  - Update: student forward+backward 1.12 + conv recompute 0.30 TFLOP.
  - Rates: conv at 35-80 TFLOP/s [A], which is 2-4.6x the measured cuDNN 17.5; MLP at 93 TFLOP/s [A, about 41% of the 229 TFLOP/s TF32 `mma.sync` peak]; tail factor x1.1.
  - Ring traffic: 3.2 GB, or 2 ms.
  - The streaming variant comes to 20-36 ms.
- **P0 update.** 1.12 TFLOP at the measured cuDNN rate, plus per-step launches and syncs.
- **F physics.** 0.289 TFLOP per iteration [D, per-substep FLOP model from the DexSuite footprint analysis, full-gravity contact statistics] at 5% / 12% of FP32 peak, with 2x / 1.5x padding [A, the locomotion scenarios].
  - The expected case needs 0.098 µs per world-substep, against MJWarp's 0.245 µs of kernel time.
  - If the port only matches MJWarp's kernel time (257 → 300 ms), F rises to 510-607 ms, a 2.6-2.8x speedup.
- **Ceiling with the stock tracer and everything else free:** 216-246 ms, which is 533-605k transitions/s or 6.4-6.7x.

**Teacher stage** (state PPO, per 131,072 transitions):

| Configuration | ms per iteration | Speedup |
|---|---|---|
| Stock [M] | 874 → 932 | 1x |
| G (graph tier) [D] | 573 | 1.5x |
| F expected [D] | 202 | 4.3x |
| F optimistic [D] | 99 | 8.8x |
| F with physics at MJWarp-kernel parity [D] | 357 | 2.4x |

**Mode E** (exact stock E2E PPO), from the Stream design's time model [D]:

| Configuration | ms per iteration | Speedup |
|---|---|---|
| Learner fused only | 1,060 | 1.36x |
| Plus fused managers | 763-798 | 1.8-1.9x |
| Plus fused physics: expected (optimistic; 870 if the physics stays at stock MJWarp speed) | 466 (348; 870) | 3.1x |

These are per-iteration gains only. Mode E alone is not expected to converge (0/3 in 30,000 iterations). It turns a 30,000-iteration run from about 12.0 h (1,445 ms per iteration at ADR difficulty 0; the earlier study's runs averaged 1,638 ms, about 13.7 h) into 3.9 h.

## Time to a converged camera policy (rgb64)

S_student is the number of transitions the student needs to reach success ≥ 0.8 on two consecutive evaluations. It has not been measured. The table brackets it at 0.25x / 1x / 2x the teacher's 170.4 M [A], and P0 is what measures it.

| Path | Teacher | Student | Total |
|---|---|---|---|
| Stock E2E PPO [L] | - | - | no policy (0/3, about 14-15 h each) |
| P0 stock code [D] | 1,262 s [L]; checkpoint already exists | 372 / 1,489 / 2,978 s | 27 / 46 / 71 min |
| G graph tier [D] | 745 s | 266 / 1,065 / 2,130 s | 17 / 30 / 48 min |
| F fused, expected [D] | 263 s | 134 / 534 / 1,068 s | 7 / 13 / 22 min |
| F fused, optimistic [D] | 129 s | 101 / 403 / 806 s | 4 / 9 / 16 min |

Stock end-to-end PPO did not reach the gate within 30,000 iterations (at least 50,201 s, ~14 h, in the earlier,
unpublished study); distillation is projected to reach it in 27-71 min on stock code. These are different algorithms,
so the two are not a speedup of one another.

How to read this table:
- **The algorithm change produces almost all of the speedup over stock**, and that speedup rests on S_student.
- **The kernel tiers add 1.5-1.6x (G) and 3.2-4.1x (F expected)** on top of the stock-code pipeline at the same S_student.
- **The teacher already exists**, so the first camera policy costs only the student phase: about 6-50 min on stock code.
- **If P0 fails, none of these rows apply.** The per-transition gains still carry over to whatever algorithm replaces distillation, because the render and physics floor does not depend on the algorithm.

## What has to be built

Value of each step on its own, per transition against distillation on stock code (from the table above): K1 alone
≈ 1.1×, the device-side MDP alone ≈ 1.35×, both (the graph tier) ≈ 1.5×, and on-chip physics (F) 2.9-3.8×. K1 goes
first anyway because it has a fast kill criterion and its encoder is shared with mode E.

| Phase | Work | Effort [A] | Gate / measurable output |
|---|---|---|---|
| **P0: learning gate (first measurable prototype)** | Stock `DistillationRunner`. Teacher: the earlier study's seed-2 state checkpoint, or a state teacher trained with the stock recipe (`Distillation.load` accepts `actor_state_dict`, `distillation.py:225-248`). Student: `CNN_POLICY_CFG` with init_std 0.1. Obs: student [policy, proprio, base_image]; teacher [policy, proprio, perception]. 32 steps with gradient_length 16 (15 leaves 2 of 32 steps unused; rsl_rl warns, `distillation.py:93-102`). E in {1, 2}, lr in {1e-3, 3e-4}. A standalone launcher script; no changes to Isaac Lab or RSL-RL are needed. | 1-3 days; at most 400 M transitions (about 1 h) per config | Measures S_student, success ≥ 0.8 twice (the earlier study's protocol), the gap to the teacher, and the phase split. Go if the student reaches held-out success ≥ 0.8 on two consecutive evaluations within 340 M transitions (~50 min of student training on stock code). |
| P0b | Time the stock rgb64 render at 4,096 envs with a 40 MB persisting L2 window, against 6.76 ms/frame without one. | 10 GPU-min | The render changes by less than 5%. |
| P0 fallback (only if the student plateaus below 0.8) | Train the teacher past 1,300 iterations. Add the auxiliary object-position head. PPO fine-tune from the student with the stock camera recipe. Latent-PPO A/B (the "Latent" design). | days | Same success gate as P0. |
| **K1: first kernel prototype** | Fused student step: uint8 tile → TF32 implicit-GEMM conv1-3 → spatial softmax with temperature gradients (the `sram-resident-training` branch has a fused implementation) → MLP → MSE → backward with recompute and deterministic dW → Adam. The same encoder core serves mode E. | 3-6 weeks | Time per 4,096-image step, against the cuDNN encoder forward+backward at 1.97 ms [M]. Target ≤ 0.6 ms for the whole step; kill above about 0.93 ms (15% of TF32). FP32 within 1e-6 of `scripts/sram/projection/measure_update_cost_cnn.py`; TF32 within the ~2e-5 floor. No register spills, SMEM ≤ 99 KB. |
| G: graph tier | Device-side MDP: 7 rewards, 3 terminations, reset bank with 25% in-hand starts and success-targeted sampling, goals, ADR, DR, history backfill, counter-RNG noise. No host syncs. One CUDA graph per iteration, with the MJWarp and tracer graphs as nodes. Optional teacher learner megakernel reusing the `kernels/ppo_cartpole.cu` reduce/Adam code. | 1-2 months | Obs, reward and termination parity with the Isaac Lab managers under recorded actions. Student iteration 680-820 ms; teacher iteration 573 ms. |
| F: on-chip physics | MJWarp-equivalent per-world solver plus Newton collision, run in SMEM waves. First slice: FK, CRB, sparse LDL (nv 35, nM 222) and one Newton iteration on recorded constraint sets. | 3-6 months, research risk | The slice reaches ≤ 0.098 µs per world-substep, against MJWarp's 0.245. Then statistical equivalence: contacts (23.3 mean, 76 max), constraint rows, solver iterations, termination and success rates. |
| Mode E (optional, after K1) | DRAM pixel ring plus the exact PPO update kernel (the Stream design). | about 4-6 weeks beyond K1 | FP32 equivalence with RSL-RL at 1.2e-7. |
| Renderer prototype | An RT-core/OptiX path writing uint8 frames into L2 (50 MB per frame); alternatives: a static-layer cache plus screen-space culling, E=2 image reuse from the ring. | ~2 weeks to a rays/s measurement [A] | Go if ≥ 2× the current tracer's 2.2-2.5 G primary rays/s for 4,096 × 64×64 frames at matching image quality. This is the only way past the 6.4-6.7x stock-tracer ceiling per transition. |

## Open risks, ranked

1. **Learning gates everything.**
   - S_student has not been measured.
   - The teacher's held-out margin is thin: 0.828-0.859 against the 0.8 gate.
   - The student sees object position only through pixels, at about 1.4 cm per pixel at 1 m [D], against a 5 cm success radius.
   - The student has no image history, and the hand occludes the object.
2. **Render floor.** The unchanged tracer is 60-79% of a fused iteration. Getting past the 6.4-6.7x stock-tracer ceiling needs a different renderer (the optimistic fused tier already reaches 5.0-5.2x), and no alternative has been measured.
3. **Physics port.**
   - Efficiency: the expected case needs 2.5x MJWarp's own kernel speed. At mere parity, the student speedup falls to 2.6-2.8x.
   - Fidelity: contact-rich grasping is chaotic, so only statistical equivalence is possible.
4. **Conv efficiency on sm_120.**
   - sm_120 has `mma.sync` only, N is 16/32, SMEM is 99 KB, and dW needs 80 regs/thread.
   - The designs assume 35-80 TFLOP/s; cuDNN measured 17.5.
   - At stake: 10-25 ms per iteration in the student stage, and 60-130 ms in mode E.
5. **L2 is a cache, not storage.** The persisting window competes with the 38.4 MB scene and the 67.5 MB RGBA frame. P0b tests this.
6. **MDP parity.** History backfill, reset ordering, ADR timing and contact filtering are easy to get subtly wrong.
7. **Cooperative persistent kernels next to Warp-captured graphs.**
   - The cooperative kernels and the captured graphs must stay co-resident, or the grid barrier can hang.
   - Pointer swaps can force graph re-capture (`newton_warp_renderer.py:636-643`).
8. **Measurement.** All timings come from a shared GPU, or from labperf, where GPU exclusivity was not established.
9. **Scope.**
   - depth64 inherits a quirk: a 0.0 miss maps to -0.5, the same as zero distance (`agents/models.py:97`). labperf never trained depth64.
   - duo_camera doubles the render floor.
   - Reorient needs its own teacher.
   - B200-class parts have no RT cores and lower FP32 throughput [A], so this setup targets RTX-class GPUs.

## How the three candidate designs scored (1-10; for effort, higher means less work)

| Design | Speed to policy | Fidelity | Feasibility | Effort | Evidence | Role here |
|---|---|---|---|---|---|---|
| Pixels in DRAM, exact PPO (stream) | 2 | 9 | 7 | 3 | 8 | Mode E; shared encoder kernel |
| Latent rollout with privileged auxiliary encoder (latent) | 5 | 5 | 5 | 5 | 5 | Fallback auxiliary head; RL-only A/B |
| Teacher → streaming-DAgger student (distill) | 8 | 4 | 8 | 7 | 6 | Primary path, with exact Distillation semantics first |

- **Stream** is the most faithful and the best evidenced, but it speeds up a recipe that has not converged.
- **Latent** still learns the policy by PPO from scratch at the stock 7e-5 learning rate, and its larger gains rest on an unmeasured 3-10x renderer.
