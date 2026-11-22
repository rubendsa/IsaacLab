* Added a native actor/critic update that retained both networks' weights and Adam
  moments in one cooperative launch, with a phase-reused gradient bank and
  explicit global rollout/activation storage.
* Shared checkpoint preparation and one-use copyback between standalone and
  unified updates, preserving per-tensor Adam steps and rejecting stale plans
  and cross-stream scratch reuse.
