# Landscape: on-chip memory and related work (sources)

Collected 2026-11-22 for the SRAM-resident training report. Tags: [P] primary/official source, [S] reputable press,
[C] vendor or self-reported claim, [R] rumor/unconfirmed, [M] measured on this machine, [D] derived arithmetic.
Numbers marked [C] or [R] are not independently verified.

| Fact | Value | Source |
|---|---|---|
| Groq LPU v1 SRAM per chip | 230 MB | Tom's Hardware 2026-03-19 (prior-gen); Groq product brief v1.5/v1.7 now 404 |
| Groq LPU v1 on-chip bandwidth | upwards of 80 TB/s (vs ~8 TB/s GPU HBM) | https://groq.com/lpu/ (dated 2025-03-07) [vendor claim] |
| NVIDIA-Groq deal date/structure | 2025-12-24, non-exclusive license + Ross/Madra join NVIDIA; Groq independent; no terms disclosed | groq.com newsroom 2025-12-24 |
| NVIDIA-Groq deal value | $20B (reported by CNBC; repeated in Senate letter; not confirmed by NVIDIA) | Warren/Blumenthal letter 2026-03-19 |
| Groq 3 LPU SRAM per chip | 500 MB (Tom's HW: 512 MB per die) | nvidia.com/en-us/data-center/lpx (accessed 2026-11-22); Tom's HW 2026-03-19 |
| Groq 3 LPU SRAM bandwidth | 150 TB/s | nvidia.com/en-us/data-center/lpx |
| Groq 3 LPU FP8 compute | ~1.2 PFLOPS (1.23 per Tom's HW) | The Register 2026-03-19; Tom's HW 2026-03-19 |
| LPX rack | 256 LPUs, 128 GB SRAM, 40 PB/s SRAM BW, 640 TB/s scale-up, 12 TB DDR5, 315 PFLOPS FP8 | nvidia.com LPX page; Tom's HW Hot Chips 2026-08-26 |
| LPX production status | 'now in full production' 2026-08-24; first cloud Nebius | nvidianews.nvidia.com press release 2026-08-24 |
| LPX third-party benchmark | 3,431 output tok/s Gemma 4 31B @100K ctx (~4x 870 tok/s next-fastest), concurrency 1 | Tom's HW 2026-08-26 (Artificial Analysis) |
| Rubin GPU | 224 SMs, up to 288 GB HBM4, up to 22 TB/s; L2/smem not disclosed | NVIDIA blog 2026-07-21 |
| Cerebras WSE-3 | 44 GB SRAM, 21 PB/s, 214 Pb/s fabric, 900k cores, 4T transistors, 5 nm, 125 PF | IEEE Spectrum 2024-03-13 |
| Cerebras SRAM per PE | 48 kB (44 GB/900k = 48.9 KB) | Cerebras SDK blog/slides (via search); derived |
| Cerebras WSE-3T | 44 GB SRAM, ~43.2 PB/s, 250 PF per wafer; CS-4 = 3 wafers; clock ~1.4->2.8 GHz (estimate) | cerebras.ai/chip; STH 2026-08-25; Next Platform 2026-08-19 |
| OpenAI-Cerebras | 750 MW through 2028 (company statements); >$10B (reported) | The Register 2026-01-15; Bloomberg/TechCrunch 2026-01-14 |
| d-Matrix Corsair | 2 GB SRAM @150 TB/s + 256 GB LPDDR5 @400 GB/s; 2.4 PF MXINT8 / 9.6 PF MXINT4 | The Register 2025-09-08 |
| d-Matrix Raptor | 32 GB 3D DRAM per card, ~100 TB/s, TSMC N4 logic on DRAM | ServeTheHome 2026-08-23 |
| Tenstorrent Blackhole p150 | 120 Tensix, 180 MB SRAM (1.5 MB/core), 32 GB GDDR6 @512 GB/s, 664 TFLOPS BlockFP8 (earlier 140 cores/210 MB) | docs.tenstorrent.com/aibs/blackhole (updated 2026-11-17); The Register 2025-11-27 |
| SambaNova SN40L | 520 MiB SRAM, 64 GiB HBM (~1.8-2 TB/s), up to 1.5 TiB DDR | arXiv 2405.07518 |
| Graphcore Bow IPU | 897 MiB SRAM @65 TB/s, 1472 tiles | Graphcore talk, ESO RTC4AO Nov 2023 |
| H100 SXM5 | 132 SMs; 256 KB RF/SM; 256 KB L1+smem/SM; 228/227 KB smem SM/block; 50 MB L2; 3.35 TB/s | Hopper tuning guide v13.5; Hopper in-depth blog 2022-03-22; H100 product page |
| B200 | 148 SMs (3rd-party); 228/227 KB smem; 256 KB L1+smem; 256 KiB TMEM/SM; 126 MB L2 (GB200); 192 GB HBM3E @8 TB/s | Blackwell tuning guide v13.5; PTX ISA 9.5; Blackwell Ultra blog 2025-08-22; Hazy 2025-05-27 |
| RTX PRO 6000 BSE SMs / CC | 188 SMs, CC 12.0 | local cudaDeviceGetAttribute (devquery_out.json) |
| RTX PRO 6000 BSE L2 | 128 MiB (134,217,728 B); max persisting L2 80 MiB | local device query |
| RTX PRO 6000 BSE smem | 100 KiB/SM (102,400 B); 99 KiB/block opt-in (101,376 B); 48 KiB default | local device query; CUDA Programming Guide 5.1 (2026-10-31) |
| RTX PRO 6000 BSE regs/threads | 65,536 regs/SM and /block; 1536 threads/SM; 24 blocks/SM; 48 warps/SM | local device query; CUDA Programming Guide 5.1 |
| RTX PRO 6000 BSE DRAM | 96 GB GDDR7, 1597 GB/s (2 x 12,481 MHz x 512 bit = 1597.6 GB/s) | NVIDIA product page; local device query |
| RTX PRO 6000 BSE max SM clock | 2430 MHz | local device query / nvidia-smi |
| RTX PRO 6000 smem bandwidth (measured, Workstation Ed.) | 115.79 B/clk/SM; >60 TB/s aggregate | Chips and Cheese 2025-06-28 |
| RTX PRO 6000 L2 bandwidth/latency (measured, Workstation Ed.) | ~8.7 TB/s; ~130 ns | Chips and Cheese 2025-06-28 |
| RTX PRO 6000 BSE smem aggregate (derived) | ~52.9 TB/s = 188 x 115.79 B x 2.43 GHz (~33x DRAM) | derived arithmetic [D] |
| RTX PRO 6000 BSE measured on this machine | shared memory 57.4 TB/s (4-wide reads; 98% of 58.5 TB/s peak at 128 B/clk/SM); L2 5.8-6.2 TB/s (streaming re-reads of 8-128 MB); DRAM 1.5-1.7 TB/s (0.5-1 GB); FP32 FFMA 112.8 TFLOP/s; `mma.sync` peaks 229 TFLOP/s TF32 and 458 TFLOP/s BF16/FP16 | micro-benchmarks on this machine, 2026-11-22 [M]; raw results in `microbench/` (the scripts are not part of this branch) |
| Total on-chip SRAM (derived) | H100 ~116 MB; B200 ~237 MB; RTX PRO 6000 BSE ~198.5 MiB (RF 47 + L1/smem 23.5 + L2 128) | derived arithmetic [D] |
| tcgen05/TMEM availability | sm_100a/sm_101a(=110a) and sm_100f/sm_110f families only; not sm_120 | PTX ISA 9.5 tcgen05.alloc Target ISA Notes |
| Cartpole PPO learnable state | 2,499 params = 9.8 KiB fp32; 39.0 KiB incl. grads + Adam m,v | derived from rsl_rl_ppo_cfg.py:28,34 and cartpole_direct_env_cfg.py:46-47 [D] |
| Cartpole rollout buffer | 3.0 MiB (4096 x 16 x 12 fp32 fields, as stored by the kernel) = 16.5 KiB per SM for 22 envs per SM | derived [D] |
| Hazy Llama-1B megakernel | H100 <1 ms/forward, 78% HBM BW; B200 <680 us, >3.5x vLLM | hazyresearch.stanford.edu 2025-05-27 |
| Kernel launch overhead (H100 dummy kernel) | 2.1 us (stream), 1.3 us (CUDA graph) | Hazy Research 2025-05-27 |
| Hazy TP Llama-70B megakernel | 23,468 vs 19,170 tok/s (>22% over SGLang), 8xH100 | hazyresearch.stanford.edu 2025-09-28 |
| MPK | up to 1.7x lower e2e latency vs kernel-per-operator | arXiv 2512.22219; OSDI '26 |
| Fully fused MLP (NRC) | 5-10x vs XLA TensorFlow; training 4-5x slower than inference (spills activations, CUTLASS weight-grad GEMMs) | Muller et al. ACM TOG 40(4) Art.36, Aug 2021 (arXiv 2106.12372) |
| Persistent RNNs | 2.8 TFLOP/s at minibatch 4 on TitanX; 16x less activation memory; 12x more params | Diamos et al. ICML 2016 (PMLR v48) |
| Isaac Gym | 2-3 orders of magnitude vs CPU-sim RL; Ant up to 700K env steps/s, Humanoid 200K (A100) | arXiv 2108.10470 (2021-08-24) |
| Isaac Lab paper multi-GPU | Franka cabinet >1.6M FPS, DextrAH >900K FPS on 8 GPUs; CPU single-core bottlenecks acknowledged | arXiv 2511.04831 (2025-11-06) |
| MuJoCo Playground CartpoleBalance | 718,626 +/- 6,894 PPO training steps/s on 1x A100 | arXiv 2502.08844 Table IV |
| Madrona Cartpole | 3.4e9 env steps/s (sim only) @1024K envs, RTX 4090; HideSeek 1.9M | Shacklett et al. SIGGRAPH 2023 (ACM TOG 42(4)) |
| Brax | 100-1000x RL speed/cost; millions of sim steps/s per accelerator | arXiv 2106.13281 |
| WarpDrive | 2.9M env steps/s (2000 envs x 1000 agents), >=100x CPU | arXiv 2108.13976 |
| Large Batch Simulation | >19,000 frames/s end-to-end on 1 GPU; 72,000 on 8 GPUs | arXiv 2103.07013 (ICLR 2021) |
| PureJaxRL | >10x single agent vs CleanRL; >4000x vmapped; 2048 CartPole agents in ~half of one CleanRL agent's time | chrislu.page/blog/meta-disco (2023-04) |
| PufferLib 2.0 | 300k-1.2M steps/s PPO on RTX 4090; envs >1M steps/s/CPU core | RLJ/RLC 2025 PufferLib 2.0 paper |
| PufferLib 4.0 | 15M sps standard model, >20M sps smaller models, single RTX 5090; ~5,000 lines CUDA-C | puffer.ai/blog/4.0 (release ~2026-04) [self-reported] |
| PufferLib 5.0 | 'Up to 60,000,000 step/second training' (~10k lines CUDA C; hardware/env unstated) | puffer.ai/docs.html (accessed 2026-11-22) [self-reported] |
| LeanRL speedups | PPO Atari 6.8x, PPO continuous 2.7x, SAC 5.7x, TD3 3.4x (H100) | github.com/pytorch-labs/LeanRL |
| rl-triton | 1.6-5.7x vs vectorized torch.compile for GAE/V-trace etc. | arXiv 2608.17641 (2026-08-18) |
| Mixture-of-Kittens | up to 2.37x layer, 1.41x e2e training on 512 GB300 | arXiv 2609.36070 (2026-09-28) |
| Cohere decode megakernel | 1.25-1.41x vs vLLM on H100 | cohere.com/blog/megakernels (2026-09-08) |
| Flapping Airplanes funding | $1B Series A at $6B (2026-10-21); $180M seed at $1.5B (Jan 2026) | AlphaSignal 2026-10-21 relaying Bloomberg |
| open-flapping-airplanes B200 | 162M GPT 4,000 steps: megakernel 48.1 s vs Megatron + Transformer Engine eager 62.4 s vs CUDA graphs 43.1 s | github.com/Bobliuuu/open-flapping-airplanes README (2026-11) |
| warp-nn | v0.4.1 2026-10-18 (first 0.1.0 2026-04-08); Apache-2.0; no perf claims | pypi.org/project/warp-nn |
| Isaac Lab v3.0.0 tag | commit e3f7d9ff, 2026-10-19 | local git (read-only) in IsaacLab |
