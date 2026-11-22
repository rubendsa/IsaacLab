* Added the :mod:`isaaclab_sram` extension, a prototype that runs the complete RSL-RL PPO training loop of
  ``Isaac-Cartpole-Direct`` (physics, rewards and resets, policy and value networks, GAE, mini-batch updates and
  Adam) inside one persistent cooperative CUDA kernel. Small working sets fit in registers and shared memory;
  cross-block reductions, checkpoints and larger rollout buffers use global memory.
* Added :class:`~isaaclab_sram.megakernel.SramPPOCartpole`, which takes its task and PPO settings from Isaac Lab's
  configuration objects, initializes the networks like RSL-RL and returns the trained weights as RSL-RL
  ``MLPModel`` modules.
* Added :mod:`isaaclab_sram.export`, which writes the trained weights and Adam state as an RSL-RL run directory
  that the stock ``play.py`` entry points load.
* Added an ``--rl_library sram`` training backend (:mod:`isaaclab_sram.rl_backend`) and
  ``scripts/sram/train_sram.py``, which trains standalone or registers the backend and runs the stock
  ``train.py``.
* Added ``scripts/sram/verify_megakernel.py``, which checks a training iteration of the kernel against a float64
  physics reference, RSL-RL's networks and loss, torch autograd and ``torch.optim.Adam``.
* Added fused spatial-softmax forward/backward, PPO-objective and GAE operations for the native Kuka-Allegro camera
  PPO learner, optional CUDA graph replay of each minibatch update, and a real-environment training runner.
  The camera encoder remains trainable; image rollouts,
  parameters and optimizer state remain global allocations.
* Added a native CNN learner benchmark using synthetic observations, and a technical report distinguishing
  on-chip intermediates, persistent working state, measured runtime and end-to-end learning validation.
