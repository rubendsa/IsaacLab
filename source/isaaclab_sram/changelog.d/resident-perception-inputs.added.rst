* Added an experimental RGB64 native perception PPO backend for 1–16 environments
  that retains both networks' weights and Adam moments, raw RGB, and prepared
  numeric rollout inputs through one cooperative optimizer launch. Selected-input
  tiles, activations and reduction mailboxes remain global; normalization, GAE,
  RNG alignment, packing and checkpoint copyback remain outside the kernel.
