Changelog
---------

.. towncrier release notes start

0.1.0 (2026-11-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_sram.megakernel.SramPPOCartpole`, a persistent cooperative CUDA kernel that runs the
  complete RSL-RL PPO training loop of ``Isaac-Cartpole-Direct`` (Newton MuJoCo-Warp ``implicitfast`` physics,
  policy and value MLPs, GAE, mini-batch updates and Adam) with its working set in on-chip SRAM.
* Added an RSL-RL checkpoint exporter and a ``--rl_library sram`` backend for the unified training entry point.
