* Added an experimental native RGB64 PPO backend with separate persistent actor and critic phases.
  The actor kernel includes trainable CNN, learned spatial-softmax temperatures, Gaussian policy,
  PPO backward, gradient clipping and all forty Adam steps while retaining its working state in shared memory.
  Rollout inputs, activations and cross-block reductions remain global allocations.
* Added a standalone actor benchmark with matched native Adam controls, restored resumed checkpoints,
  complete update parity, source fingerprints and a CUDA profiler entry point.
