# Real Kuka-Allegro camera runs

The per-run JSON files are unmodified `perception_summary.json` outputs from
`scripts/sram/train_kuka_perception.py`, renamed to identify each run. The comparison
JSON and timing figure are derived from those reports. The main
[report](../../SRAM_TRAINING.md) gives interpretation and reproduction commands.

The headline `4096_200` runs use seed 42, the native reset bank, 200 iterations and
five warmup iterations. They ran sequentially in the order graph, stock, fused on
clean commit `c117928647ddad83c480c7a267c8f4225ab9ff6e`, with identical source
hashes. Each includes 26,214,400 transitions and 8,000 optimizer minibatches. The
derived [comparison](comparison_4096_200.json) contains full, early (5–29) and late
(150–199) timing windows, checks, task metrics and periodic GPU samples. Samples
are diagnostic snapshots, not energy measurements or hardware traffic counters.

The [timing figure](kuka_rgb64_200_timings.jpg) plots all 195 measured rows and
centered nine-iteration medians. These are sequential single-seed runs; the
percentiles describe within-run spread, not confidence intervals. Simulation
work evolves with policy trajectories. Reproduce the figure from the repository
root with:

```bash
uv run python scripts/sram/plot_perception_timings.py \
    --stock docs/sram/data/perception/torch_rgb64_4096_200.json \
    --fused docs/sram/data/perception/sram_rgb64_4096_200.json \
    --graph docs/sram/data/perception/sram_graph_rgb64_4096_200.json \
    --out /tmp/kuka_rgb64_200_timings.jpg
```

The `4096_30` runs use the native reset bank, RGB64 images, seed 42 and 30 training
iterations; the first five iterations are excluded from medians. They were run
sequentially in the order `torch`, `sram`, `sram_graph`. The short stock run has
variable learner timings, so its speedup should be checked against longer runs.

The smoke runs use 64 environments, three iterations and an explicitly reduced
16-state reset bank. They check execution and checkpoint compatibility, not lift
success or native-scale throughput. `sram_rgb64_smoke2` is the successful rerun
after the checkpoint comparison separated IEEE float32 from TF32 rounding.

The camera-preview JPEG shows 16 actual camera observations from the native 4,096-environment
graph run. It was saved outside the timed iterations.

Provenance includes source SHA-256 values, base commit, dirty status and installed
versions. The short runs predate the implementation commit. Absolute paths and
process identifiers describe the measurement host; checkpoints and TensorBoard
files are not embedded in this repository. In `torch_rgb64_4096_30` and
`sram_rgb64_smoke2`, `rollout_storage_bytes` counts **observations only**. Later
outputs name that field `rollout_observation_bytes` and separately count the full
rollout storage. Retaining the original files avoids rewriting measurement data.

Iteration times synchronize CUDA at the end of each update and include simulation,
rendering, collection and optimization. The collection/learning split follows
host timers and can attribute asynchronous work across that boundary. Startup,
compilation warmup, logging and checkpoint writes are excluded from steady-state
iteration timings. `train_wall_s` includes the runner's complete training call.
Only desktop service processes shared the GPU during these measurements; no
other training or benchmark job ran concurrently. No DRAM-byte counters were
collected, and no run here establishes a solved lift policy.
