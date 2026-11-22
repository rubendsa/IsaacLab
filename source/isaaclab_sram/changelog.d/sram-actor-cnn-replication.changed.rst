* Distributed native RGB64 actor convolution work across resident filter replicas while
  preserving complete minibatch gradient reduction, actor-wide clipping, Adam state and
  canonical checkpoints. Added explicit canonical and replicated state resource accounting.
