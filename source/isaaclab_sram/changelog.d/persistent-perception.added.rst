* Added a small-rollout native Kuka camera PPO backend that retained critic parameters, gradients and Adam
  moments in shared memory across all 40 optimizer steps of a cooperative kernel launch. The actor remained
  trainable through the native CNN; rollout inputs and cross-block communication still used global memory.
* Added lossless state-history rollout storage that preserved processed observations, reset backfill and
  native minibatch ordering while retaining each repeated frame only once.
