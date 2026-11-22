# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO integration for the real Kuka-Allegro camera tasks.

The camera encoder remains trainable through the complete PPO objective. Custom
CUDA kernels fuse GAE, Gaussian likelihoods, clipped PPO losses and their backward
pass; spatial softmax is fused separately. Convolution and matrix multiplication
use PyTorch's CUDA libraries. Rollouts, weights and optimizer state remain in GPU
global memory: this is SRAM tiling within kernels, not an all-SRAM learner.
"""

from __future__ import annotations

from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage


class SramPerceptionPPO(PPO):
    """Feed-forward camera PPO with fused CUDA training kernels and fused Adam.

    Uses the native RSL-RL rollout storage, CNN/MLP models, gradient clipping,
    normalizers and checkpoint format. The supported camera recipe has a fixed
    learning rate, Gaussian actions, clipped values and global advantage
    normalization. Unsupported configurations fail before training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        unsupported = []
        if self.rnd is not None or self.symmetry is not None:
            unsupported.append("RND or symmetry")
        if self.is_multi_gpu or self.actor.is_recurrent or self.critic.is_recurrent:
            unsupported.append("distributed or recurrent training")
        if self.schedule != "fixed":
            unsupported.append("non-fixed learning rate")
        if self.normalize_advantage_per_mini_batch or not self.use_clipped_value_loss:
            unsupported.append("per-minibatch normalization or unclipped value loss")
        if self.use_mixed_precision:
            unsupported.append("mixed precision")
        if not isinstance(self.actor.distribution, GaussianDistribution):
            unsupported.append("non-Gaussian actions")
        if type(self.optimizer) is not torch.optim.Adam:
            unsupported.append("non-Adam optimizer")
        if not str(self.device).startswith("cuda"):
            unsupported.append("non-CUDA device")
        if unsupported:
            raise ValueError(f"SRAM perception PPO does not implement: {', '.join(unsupported)}")

        from .spatial_softmax import install_fused_spatial_softmax

        self.fused_spatial_softmax_modules = install_fused_spatial_softmax(self.actor)
        self.optimizer = torch.optim.Adam(
            chain(self.actor.parameters(), self.critic.parameters()), lr=self.learning_rate, fused=True
        )

    def compute_returns(self, obs) -> None:
        """Compute globally normalized GAE using the fused CUDA kernel."""
        from .ppo_kernels import fused_gae

        st = self.storage
        last_values = self.critic(obs).detach()
        returns, advantages = fused_gae(st.rewards, st.dones, st.values, last_values, self.gamma, self.lam)
        st.returns.copy_(returns)
        st.advantages.copy_(advantages)

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load native checkpoints while retaining the fused optimizer implementation."""
        iteration_loaded = super().load(loaded_dict, load_cfg, strict)
        for group in self.optimizer.param_groups:
            group["fused"], group["foreach"] = True, False
            for parameter in group["params"]:
                state = self.optimizer.state[parameter]
                if "step" in state:
                    state["step"] = state["step"].to(parameter.device)
        return iteration_loaded

    def update(self) -> dict[str, float]:
        """Run PPO epochs, backpropagating fused objective gradients into the CNN."""
        loss_sums = torch.zeros(3, device=self.device)
        batches = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in batches:
            loss, value_loss, surrogate_loss, entropy = self._loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            loss_sums += torch.stack((value_loss.detach(), surrogate_loss.detach(), entropy.detach()))

        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        losses = (loss_sums / (self.num_learning_epochs * self.num_mini_batches)).tolist()
        self.storage.clear()
        return dict(zip(("value", "surrogate", "entropy"), losses, strict=True))

    def _loss(self, batch, sample_actions=True):
        from .ppo_kernels import fused_ppo_loss

        if sample_actions:
            # Preserve stock PPO's discarded sample and therefore its RNG behavior.
            self.actor(batch.observations, stochastic_output=True)
        else:
            # Normal.sample checks std on the host and cannot be CUDA-graph captured.
            self.actor.distribution.update(self.actor.mlp(self.actor.get_latent(batch.observations)))
        mean, std = self.actor.output_distribution_params
        values = self.critic(batch.observations)
        return fused_ppo_loss(
            mean,
            std,
            values,
            batch.actions,
            batch.old_actions_log_prob,
            batch.values,
            batch.returns,
            batch.advantages,
            self.clip_param,
            self.value_loss_coef,
            self.entropy_coef,
        )


class SramGraphPerceptionPPO(SramPerceptionPPO):
    """Replay one fused minibatch training step from a CUDA graph.

    The native minibatch permutation, camera CNN, optimizer state and running
    normalization remain intact. Static input copies and a graph replay replace
    Python launch overhead for each minibatch. Capture warmup restores all model
    parameters and Adam moments/counters before the first real training step.
    Rollout RNG streams differ from stock because update's discarded action
    sample is omitted. GPU global memory holds inputs, outputs and graph pools.
    """

    _BATCH_FIELDS = ("actions", "values", "returns", "advantages", "old_actions_log_prob")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for model in (self.actor, self.critic):
            if model.obs_normalization and not isinstance(model.obs_normalizer, EmpiricalNormalization):
                raise ValueError("CUDA graph PPO supports EmpiricalNormalization only")
        for group in self.optimizer.param_groups:
            group["capturable"] = True
        self._update_graph = None
        self._graph_losses = torch.zeros(3, device=self.device)

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore a checkpoint before capture, preserving capturable Adam."""
        if self._update_graph is not None:
            raise RuntimeError("Load checkpoints before the first graphed update")
        loaded = super().load(loaded_dict, load_cfg, strict)
        for group in self.optimizer.param_groups:
            group["capturable"] = True
        return loaded

    def update(self) -> dict[str, float]:
        """Gather native minibatches, replay training and update normalizers."""
        self._graph_losses.zero_()
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            if self._update_graph is None:
                self._capture_update(batch)
            else:
                self._copy_batch(batch)
            with torch.cuda.device(self.device):
                self._update_graph.replay()

        # Native normalization rebinds _std. Retain its address because captured
        # kernels reference that buffer across iterations.
        obs = self.storage.observations.flatten(0, 1)
        for model in (self.actor, self.critic):
            if model.obs_normalization:
                std = model.obs_normalizer._std
                model.update_normalization(obs)
                std.copy_(model.obs_normalizer._std)
                model.obs_normalizer._std = std
        losses = (self._graph_losses / (self.num_learning_epochs * self.num_mini_batches)).tolist()
        self.storage.clear()
        return dict(zip(("value", "surrogate", "entropy"), losses, strict=True))

    def _copy_batch(self, batch):
        for name, value in self._static_batch.observations.items():
            value.copy_(batch.observations[name])
        for name in self._BATCH_FIELDS:
            getattr(self._static_batch, name).copy_(getattr(batch, name))

    def _graph_step(self):
        loss, value, surrogate, entropy = self._loss(self._static_batch, sample_actions=False)
        self.optimizer.zero_grad(set_to_none=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._graph_losses += torch.stack((value.detach(), surrogate.detach(), entropy.detach()))

    def _capture_update(self, batch):
        self._static_batch = RolloutStorage.Batch(
            observations=batch.observations.clone(),
            **{name: getattr(batch, name).clone() for name in self._BATCH_FIELDS},
        )
        params = list(chain(self.actor.parameters(), self.critic.parameters()))
        # Initialize Adam lazily without performing an optimizer step. This makes
        # its zero-state addresses stable just as resumed nonzero moments are.
        for parameter in params:
            state = self.optimizer.state[parameter]
            if not state:
                state.update(
                    step=torch.zeros((), device=parameter.device),
                    exp_avg=torch.zeros_like(parameter),
                    exp_avg_sq=torch.zeros_like(parameter),
                )
        snapshot_params = [p.detach().clone() for p in params]
        snapshot_states = [{name: value.clone() for name, value in self.optimizer.state[p].items()} for p in params]
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(2):
                self._graph_step()
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        with torch.cuda.device(self.device):
            self._update_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._update_graph, stream=side):
                self._graph_step()
        torch.cuda.synchronize(self.device)
        with torch.no_grad():
            for parameter, saved_param, saved_state in zip(params, snapshot_params, snapshot_states, strict=True):
                parameter.copy_(saved_param)
                for name, value in saved_state.items():
                    self.optimizer.state[parameter][name].copy_(value)
            self._graph_losses.zero_()
