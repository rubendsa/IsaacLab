# Pure Warp investigation for native RGB64 perception

The optional [experiment](../../scripts/sram/experiment_warp_perception.py) implements the native
Kuka–Allegro RGB64 convolutional encoder and learned spatial softmax entirely in Warp kernels.
PyTorch constructs the actual configured model, supplies frozen parameters and inputs, and serves
as the independent reference. The candidate uses no injected CUDA, Triton, or neural-network
extension. It passes forward and vector–Jacobian-product checks on CPU and CUDA.

This is a component experiment. It does not implement the MLP, PPO objective, gradient clipping,
Adam, rollout collection, GAE, normalizers, or random sampling. It has no performance result and
makes no model-state residency claim. The scalar convolution baseline is unoptimized; it does not
measure the performance limit of Warp or its tile APIs.

## Reproduce

From the repository's configured environment:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/sram/experiment_warp_perception.py \
  --device cpu --batch 2 --output /tmp/warp_cpu.json
uv run python scripts/sram/experiment_warp_perception.py \
  --device cuda:0 --batch 64 --output /tmp/warp_cuda.json
```

The script uses the existing `warp-lang` dependency and adds no dependency. The tested installation
reports Warp 1.17.0. Each result records the exact script, native-model, configuration, and selected
Warp source hashes, package versions, and an unchanged-source check. A failed numerical assertion
exits unsuccessfully. No simulator is needed. CPU runs with CUDA hidden may print Warp's
no-CUDA-device initialization warning before completing successfully.

## Preserved native component and evidence

The experiment resolves `KukaAllegroSingleCameraPPORunnerCfg` and the actual native actor class.
It preserves uint8 RGB normalization, all three trainable convolution layers and ELUs, and the
32 learned log-temperatures:

| Layer | Native shape |
|---|---|
| Input | `[B, 3, 64, 64]`, scaled by `1/255` and shifted by `-0.5` |
| Convolution + ELU | `3 → 16`, kernel 8, stride 4, output 15 × 15 |
| Convolution + ELU | `16 → 32`, kernel 4, stride 2, output 6 × 6 |
| Convolution + ELU | `32 → 32`, kernel 3, stride 1, output 4 × 4 |
| Spatial softmax | Per-channel `exp(log_temperature).clamp(min=0.001)`; 32 x coordinates then 32 y coordinates |

The independent native reference supplies a random upstream gradient. Assertions compare the
normalized image, feature map, keypoints, gradients of all six convolution parameter tensors,
normalized-image gradient, feature gradient, and learned-temperature gradient. Every compared
reference tensor has nonzero norm. A second fixture exercises temperatures below and above the
clamp threshold, including adjacent FP32 log-temperature values; inactive temperature gradients
must be exactly zero and active gradients must be nonzero. Exact temperature equality is not
represented by the tested neighboring FP32 exponentials and is not claimed as a runtime test.

| Evidence | Native component maximum absolute difference | Clamp feature-gradient relative L2 |
|---|---:|---:|
| [CPU, B = 2](data/pure_warp/cpu_native_cnn.json) | 2.06e-7 | 1.42e-7 |
| [CUDA, B = 64](data/pure_warp/cuda_native_cnn.json) | 5.94e-6 | 1.38e-7 |

The ordinary component gates use `atol=2e-5, rtol=2e-4`, with a tighter image-normalization gate.
The separate near-clamp feature-gradient fixture uses `atol=2e-4, rtol=3e-4`: its gradient norm
exceeds 2,000 because the temperature is near 0.001. This fixture's CUDA maximum absolute
difference is 9.16e-5. These are numerical parity checks, not bitwise determinism claims;
Warp-generated convolution adjoints accumulate gradients with atomics. Batch 64 is the minibatch
size of the native 16-environment, 32-step, eight-minibatch configuration. This experiment does
not run its 40 optimizer steps. [Manifest](data/pure_warp/manifest.json)

## Autodiff finding and correction

An initial implementation put ELU directly after the runtime convolution-reduction loop in one
Warp kernel. Its forward outputs passed, but the normalized-image gradient failed at 20,346 of
24,576 elements, with maximum absolute difference 0.00267542. Inspection of the generated CPU
adjoint showed that its forward replay initialized the accumulator from the bias and did not
replay the reduction before evaluating ELU. This is evidence for the inspected installed build
and this kernel pattern, not a statement that all Warp nonlinear reductions are unsupported.
[Retained diagnostic](data/pure_warp/failed_fused_loop_diagnostic.json)

The corrected implementation saves each linear convolution's preactivation in a global array
and applies ELU in a separate Warp kernel. Tape then differentiates ELU at the actual
preactivation. All convolution gradients pass without relaxing their tolerances. The saved
preactivations add 5,264 FP32 values per image, plus their gradient arrays. Intermediate
activations, model parameters, and gradients remain global allocations. The softmax adjoint is
explicit and retains the native inclusive clamp derivative. Warp documents the need to preserve
primal values required by reverse differentiation. [Warp differentiability guide](https://nvidia.github.io/warp/v1.17/user_guide/differentiability.html)

## What pure Warp can support next

Warp's differentiable tile operations provide block-local register/shared-memory computation and
matrix multiplication. A bounded next optimization is a blocked convolution or native-width
MLP layer using those APIs, checked against the same FP32 reference before timing. The official
trainable tile-MLP example is a small, FP16 network; it does not establish feasibility for this
FP32 699 → 512 → 256 → 128 → 23 actor. FP32 storage and accumulation also do not by themselves
guarantee IEEE FP32 multiplication: MathDx/tile-matmul precision choices must be explicit and
validated against the native reference with TF32 disabled.
[Warp tiles](https://nvidia.github.io/warp/v1.17/user_guide/programming_model/tiles.html),
[tile matmul](https://nvidia.github.io/warp/v1.17/language_reference/_generated/warp.tile_matmul.html)

Tile fusion alone does not give the existing CUDA learner's persistent cooperative grid.
The [inspected Warp launch path](data/pure_warp/installed_source_audit.json) uses ordinary `cuLaunchKernel`;
no supported cooperative-grid launch or grid-wide synchronization API was found. `cluster_dim`
does not supply that guarantee.
An ad hoc software grid barrier on an ordinary launch would risk deadlock and is not a supported
substitute. A custom runtime/native backend could add the capability, but that is a separate
extension, outside this pure-Warp experiment. One block cannot hold the native 1,527,711
parameters: weights alone are about 5.83 MiB and weights/gradients/two Adam moments about
23.31 MiB. Aggregate GPU SRAM is not a single block-addressable pool.

A multi-kernel Warp learner with CUDA graph replay is a concrete possible route for the native
32-step rollout and five epochs of eight minibatches. It would still need these explicit
compatibility steps before claiming equivalence:

- Native PPO's clipped maxima split gradients at equal operands. The inspected Warp generic
  min/max adjoints route ties to the second operand, and generic clamp differs at its upper
  boundary. Implement and test the native derivatives, including standard-deviation clamping.
- Native Adam checkpoints contain per-parameter step counters. The stock Warp optimizer has one
  shared counter and a different floating-point expression order. Preserve the individual
  counters, moments, fused/default Adam math, and separate actor/critic norm clipping.
  [Warp Adam](https://nvidia.github.io/warp/v1.17/api_reference/_generated/warp.optim.Adam.html)
- Warp's random implementation uses PCG, whereas native Torch CUDA sampling uses Philox.
  Identical integer seeds do not imply identical samples or RNG-state advancement. Native
  permutation reuse, discarded training samples, and collection sampling require an explicit
  compatibility implementation or a clearly declared different random stream.
  [Warp runtime](https://nvidia.github.io/warp/v1.17/user_guide/runtime.html)
- Keep observation normalization frozen through all 40 updates, normalize advantages with the
  native unbiased standard deviation, and preserve native checkpoints. Tape buffers must be
  reset between minibatches; optimization itself must remain outside the differentiated graph.
  Isaac Lab disables simulation backward generation by default, so enable learner adjoints
  locally, as this script does.

NVIDIA's separate [Warp-NN project](https://github.com/NVIDIA/warp-nn) offers neural-network layers
and is another implementation option. It is not installed or used here, and its existence is
not evidence of persistent SRAM storage or native PPO/checkpoint/RNG equivalence.
