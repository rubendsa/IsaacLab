# Real Kuka-Allegro camera runs

These are unmodified `perception_summary.json` outputs from
`scripts/sram/train_kuka_perception.py`, renamed to identify each run. The main
[report](../../SRAM_TRAINING.md) gives interpretation and reproduction commands.

The `4096_30` runs use the native reset bank, RGB64 images, seed 42 and 30 training
iterations; the first five iterations are excluded from medians. They were run
sequentially in the order `torch`, `sram`, `sram_graph`. The short stock run has
variable learner timings, so its speedup should be checked against longer runs.

The smoke runs use 64 environments, three iterations and an explicitly reduced
16-state reset bank. They check execution and checkpoint compatibility, not lift
success or native-scale throughput. `sram_rgb64_smoke2` is the successful rerun
after the checkpoint comparison separated IEEE float32 from TF32 rounding.

The JPEG shows 16 actual camera observations from the native 4,096-environment
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
