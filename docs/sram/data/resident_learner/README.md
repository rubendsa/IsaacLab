# Resident model and input correctness evidence

[Summary](summary.json) · [Artifact hashes](manifest.json) ·
[Technical report](../../SRAM_TRAINING.md#native-models-and-prepared-inputs-retained-together)

This archive covers the `sram_resident` native RGB64 learner at 16 environments. It retains both
networks' weights and Adam moments, 6,291,456 bytes of raw RGB and 4,622,336 bytes of prepared numeric
inputs through one cooperative launch. Global input tiles, activations, adjoints and reduction
buffers remain. GAE, normalizers, RNG compatibility calls, preparation and copyback remain outside.
It is neither an all-SRAM learner nor a zero-DRAM claim.

`matched_default16.json.gz` and `matched_fused16.json.gz` compare two resumed updates with actual
native PPO, changed observations and normalizers, unequal per-tensor steps, all model/normalizer/Adam
state and losses, and exact CPU/CUDA RNG. Every gate passes; maximum state error is `1.19e-7`.
`production_update_tests.txt.gz` records six passing production full-update tests, including reuse
and checkpoint reconstruction.

The driver reports 255 registers per thread, 91,200 dynamic plus 336 static shared bytes per CTA,
zero local memory and one resident CTA per SM. Compiler and SASS inspection finds zero stack,
spill loads/stores and local-memory instructions. The production CUDA adds comments to the validated
scratch source; the cubins are byte-identical. The compiler audit's historical
`runtime_validation_pending` field describes its creation time; the separate runtime records above
establish subsequent validation. The PTX and index audits support original-input loads at entry
and ownership of the retained data; they are not DRAM counter measurements.

`real16_camera_TIMING_EXCLUDED.json.gz` records three actual camera updates with the native reset
bank, compact history and IEEE float32. All integration checks pass: three resident launches,
changed CNN weights, finite parameters, image-sensitive actions and a native checkpoint replay
error of `5.96e-8`. **Every timing field in this raw smoke record is excluded**, because a process
monitor did not cover the run. It does not establish task success or a camera speedup.

The separately screened synthetic timing comparison is in
[unified_performance](../unified_performance/README.md). Raw artifacts here are gzip archives with
timestamp zero; the manifest records both compressed and original SHA-256 hashes. For example:

```bash
gzip -dc docs/sram/data/resident_learner/matched_fused16.json.gz | python -m json.tool
```
