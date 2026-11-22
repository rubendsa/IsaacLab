# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export SRAM-trained Cartpole policies as standard RSL-RL ``OnPolicyRunner`` run directories.

The megakernel keeps the weights and the Adam moments as flat fp32 vectors with the layout of
:func:`isaaclab_sram.megakernel.pack_params`: per network ``[W1, b1, W2, b2, W3, b3]`` (``nn.Linear``
``[out, in]`` row-major), actor first, critic second, and the actor's scalar action std at
``OFF_STD = 2498``.  This module turns them into exactly what ``OnPolicyRunner.save`` writes
(rsl-rl-lib 5.5.1)::

    model_<iter>.pt = {
        "actor_state_dict":     MLPModel.state_dict()   # distribution.std_param, mlp.{0,2,4}.{weight,bias}
        "critic_state_dict":    MLPModel.state_dict()   # mlp.{0,2,4}.{weight,bias}
        "optimizer_state_dict": Adam(chain(actor.parameters(), critic.parameters())).state_dict()
        "iter":                 index of the last completed learning iteration (RSL-RL convention)
        "infos":                provenance dict (RSL-RL itself writes None)
    }

It also writes the ``run.json`` manifest (used by ``--checkpoint latest|best``) and
``params/{env,agent}.yaml``, as Isaac Lab's training entry point does.

The stock ``play.py`` entry points load checkpoints with ``OnPolicyRunner.load(path)``. That call uses
``load_cfg=None`` and ``strict=True``, so the actor, critic *and* optimizer state must match the runner's
modules exactly. :func:`verify_checkpoint` loads the written file the same way and checks the
round trip against the trainer bit for bit.

Torch places the Adam moments itself. The flat moment vectors are unpacked into shadow copies of the
actor and critic, using the same :func:`~isaaclab_sram.megakernel.unpack_params` as the weights. Each
tensor is then assigned to ``optimizer.state[p]`` and serialised with ``optimizer.state_dict()``. As a
result, every state index is the position of its parameter in ``chain(actor.parameters(),
critic.parameters())``. Index 0 is the actor's ``distribution.std_param``, which ``MLPModel``
registers before its MLP.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.metadata
import io
import itertools
import os
import warnings
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .megakernel import SramPPOCartpole

DEFAULT_TASK = "Isaac-Cartpole-Direct"
DEFAULT_AGENT_ENTRY_POINT = "rsl_rl_cfg_entry_point"
CHECKPOINT_KEYS = ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos")
CARTPOLE_OBS_DIMS = {"policy": 4}
CARTPOLE_NUM_ACTIONS = 1


# ------------------------------------------------------------------------------------------------
# RSL-RL model / optimizer construction (mirrors rsl_rl.algorithms.PPO.construct_algorithm)
# ------------------------------------------------------------------------------------------------
def _needs_version_shims(agent_cfg: Any) -> bool:
    """Whether :func:`~isaaclab_rl.rsl_rl.handle_deprecated_rsl_rl_cfg` has not been applied to *agent_cfg* yet.

    The shim is not idempotent: it deletes the legacy attributes it migrates, and a second call raises
    ``AttributeError``. Its footprint is therefore detected instead. For rsl-rl >= 5 the model configs lose
    ``stochastic`` (and related fields); for rsl-rl < 5 they lose ``distribution_cfg``. A legacy
    ``policy`` config is cleared in both cases.
    """
    from packaging import version

    from isaaclab_rl.rsl_rl.utils import is_missing

    def present(name: str) -> bool:
        return hasattr(agent_cfg, name) and not is_missing(getattr(agent_cfg, name))

    if present("policy"):
        return True
    v5 = version.parse(importlib.metadata.version("rsl-rl-lib")) >= version.parse("5.0.0")
    for name in ("actor", "critic", "student", "teacher"):
        if present(name) and hasattr(getattr(agent_cfg, name), "stochastic" if v5 else "distribution_cfg"):
            return True
    return False


def with_version_shims(agent_cfg: Any) -> Any:
    """Return a copy of *agent_cfg* with Isaac Lab's rsl-rl version shims applied exactly once.

    The stock entry points apply the shims once before the runner sees the config.
    """
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    cfg = copy.deepcopy(agent_cfg)
    if _needs_version_shims(cfg):
        cfg = handle_deprecated_rsl_rl_cfg(cfg, importlib.metadata.version("rsl-rl-lib"))
    return cfg


def agent_cfg_to_dict(agent_cfg: Any) -> dict:
    """Return the runner configuration dict that ``OnPolicyRunner`` receives for *agent_cfg*.

    That dict is ``to_dict(agent_cfg)`` after Isaac Lab's rsl-rl version shims
    (:func:`with_version_shims`).

    Args:
        agent_cfg: An ``RslRlOnPolicyRunnerCfg`` instance, shimmed or not, or an already converted dict.
    """
    if isinstance(agent_cfg, dict):
        return copy.deepcopy(agent_cfg)
    from isaaclab.utils import to_dict

    return to_dict(with_version_shims(agent_cfg))


def build_rsl_rl_models(
    agent_cfg: Any,
    *,
    obs_dims: dict[str, int] | None = None,
    num_actions: int = CARTPOLE_NUM_ACTIONS,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Build the actor and critic exactly as ``rsl_rl.algorithms.PPO.construct_algorithm`` does.

    The modules are created from a one-sample dummy observation batch, so no environment is needed.
    The global CPU RNG is left untouched; ``nn.Linear`` initialisation runs inside a forked RNG.

    Args:
        agent_cfg: Runner config (configclass or dict).
        obs_dims: Observation group name -> flat dimension. Defaults to the Cartpole ``{"policy": 4}``.
        num_actions: Action dimension.
        device: Device the modules are moved to (the runner uses ``agent_cfg.device``).
    """
    from rsl_rl.utils import resolve_class, resolve_obs_groups
    from tensordict import TensorDict

    cfg = agent_cfg_to_dict(agent_cfg)
    obs = TensorDict({k: torch.zeros(1, d) for k, d in (obs_dims or CARTPOLE_OBS_DIMS).items()}, batch_size=[1])
    # resolve_obs_groups warns about the (intentionally) empty Isaac Lab obs_groups and prints the result
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), torch.random.fork_rng(devices=[]):
        warnings.simplefilter("ignore", UserWarning)
        obs_groups = resolve_obs_groups(obs, copy.deepcopy(cfg.get("obs_groups") or {}), ["actor", "critic"])
        actor_cls, actor_kw = resolve_class(cfg["actor"])
        critic_cls, critic_kw = resolve_class(cfg["critic"])
        actor = actor_cls(obs, obs_groups, "actor", num_actions, **actor_kw).to(device)
        critic = critic_cls(obs, obs_groups, "critic", 1, **critic_kw).to(device)
    return actor, critic


def build_rsl_rl_optimizer(actor: torch.nn.Module, critic: torch.nn.Module, agent_cfg: Any) -> torch.optim.Optimizer:
    """Build the optimizer exactly as ``rsl_rl.algorithms.PPO.__init__`` does.

    The call is ``resolve_optimizer(name)(chain(actor.parameters(), critic.parameters()), lr=learning_rate)``.
    """
    from rsl_rl.utils import resolve_optimizer

    alg = agent_cfg_to_dict(agent_cfg)["algorithm"]
    return resolve_optimizer(alg.get("optimizer", "adam"))(
        itertools.chain(actor.parameters(), critic.parameters()), lr=float(alg["learning_rate"])
    )


def _require_adam(agent_cfg: Any) -> None:
    name = str(agent_cfg_to_dict(agent_cfg)["algorithm"].get("optimizer", "adam")).lower()
    if name != "adam":
        raise ValueError(f"the SRAM megakernel implements Adam only; the agent config asks for {name!r}")


def _chain_params(actor: torch.nn.Module, critic: torch.nn.Module) -> list[torch.nn.Parameter]:
    return list(itertools.chain(actor.parameters(), critic.parameters()))


def _unpack_to_parameter_tensors(
    flat: torch.Tensor, actor: torch.nn.Module, critic: torch.nn.Module
) -> list[torch.Tensor]:
    """Split a flat kernel-layout vector into one tensor per parameter, in ``chain(actor, critic)`` order."""
    from . import megakernel as mk

    shadow_actor, shadow_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    mk.unpack_params(flat, shadow_actor, shadow_critic)
    return [p.detach().clone() for p in _chain_params(shadow_actor, shadow_critic)]


def _pack_parameter_tensors(
    tensors: list[torch.Tensor], actor: torch.nn.Module, critic: torch.nn.Module
) -> torch.Tensor:
    """Inverse of :func:`_unpack_to_parameter_tensors`: per-parameter tensors -> flat kernel layout (CPU)."""
    from . import megakernel as mk

    shadow_actor, shadow_critic = copy.deepcopy(actor), copy.deepcopy(critic)
    shadow = _chain_params(shadow_actor, shadow_critic)
    if len(shadow) != len(tensors):
        raise ValueError(f"expected {len(shadow)} parameter tensors, got {len(tensors)}")
    with torch.no_grad():
        for p, t in zip(shadow, tensors):
            p.copy_(t.reshape(p.shape))
    return mk.pack_params(shadow_actor, shadow_critic)


# ------------------------------------------------------------------------------------------------
# checkpoint assembly
# ------------------------------------------------------------------------------------------------
def checkpoint_from_flat(
    params: torch.Tensor,
    *,
    agent_cfg: Any,
    learning_rate: float,
    iteration: int,
    exp_avg: torch.Tensor | None = None,
    exp_avg_sq: torch.Tensor | None = None,
    adam_step: int = 0,
    infos: dict | None = None,
    device: str | torch.device | None = None,
) -> dict:
    """Assemble an ``OnPolicyRunner.save``-compatible checkpoint dict from flat kernel-layout buffers.

    Args:
        params: Flat parameters (``megakernel.pack_params`` layout, at least ``N_PARAMS`` floats).
        agent_cfg: RSL-RL runner config the checkpoint will be loaded with.
        learning_rate: Current (KL-adaptive) learning rate. ``PPO.update`` writes it into every
            parameter group, and ``PPO.load`` reads it back from ``param_groups[0]["lr"]``.
        iteration: Value stored under ``"iter"``. This is the index of the last completed iteration,
            so ``max_iterations - 1`` after a full run, as in ``OnPolicyRunner``.
        exp_avg: Flat Adam first moments (same layout as *params*); required when ``adam_step > 0``.
        exp_avg_sq: Flat Adam second moments; required when ``adam_step > 0``.
        adam_step: Number of Adam steps taken. If 0, the optimizer state is empty, like a fresh torch Adam.
        infos: Value stored under ``"infos"``.
        device: Device of the saved tensors. Defaults to ``agent_cfg.device``, like the runner.
    """
    from . import megakernel as mk

    _require_adam(agent_cfg)
    cfg = agent_cfg_to_dict(agent_cfg)
    device = torch.device(device if device is not None else cfg.get("device", "cpu"))
    actor, critic = build_rsl_rl_models(cfg, device=device)
    mk.unpack_params(params, actor, critic)
    optimizer = build_rsl_rl_optimizer(actor, critic, cfg)
    for group in optimizer.param_groups:  # what PPO.update does after every adaptive-LR change
        group["lr"] = float(learning_rate)
    if adam_step > 0:
        if exp_avg is None or exp_avg_sq is None:
            raise ValueError("adam_step > 0 needs both exp_avg and exp_avg_sq")
        m = _unpack_to_parameter_tensors(exp_avg, actor, critic)
        v = _unpack_to_parameter_tensors(exp_avg_sq, actor, critic)
        for p, m_p, v_p in zip(_chain_params(actor, critic), m, v):
            # torch.optim.Adam (non-fused, non-capturable) keeps "step" as a CPU float32 scalar tensor and
            # the moments on the parameter's device; the key order matches Adam._init_group
            optimizer.state[p] = {
                "step": torch.tensor(float(adam_step), dtype=torch.float32),
                "exp_avg": m_p.to(p.device),
                "exp_avg_sq": v_p.to(p.device),
            }
    return {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": int(iteration),
        "infos": infos,
    }


def trainer_infos(trainer: SramPPOCartpole) -> dict:
    """Provenance stored in the checkpoint's ``"infos"`` (plain Python types only)."""
    from dataclasses import asdict

    return {
        "trainer": "isaaclab_sram.megakernel.SramPPOCartpole",
        "seed": int(trainer.seed),
        "num_envs": int(trainer.plan.num_envs),
        "completed_iterations": int(trainer.iteration),
        "adam_step": int(round(float(trainer.scal[1].item()))),
        "learning_rate": float(trainer.scal[0].item()),
        "num_ctas": int(trainer.plan.num_ctas),
        "rollout_storage": "smem" if trainer.plan.store_in_smem else "global",
        "ppo": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(trainer.ppo).items()},
    }


def checkpoint_from_trainer(
    trainer: SramPPOCartpole,
    agent_cfg: Any = None,
    *,
    iteration: int | None = None,
    infos: dict | None = None,
    device: str | torch.device | None = None,
) -> dict:
    """Checkpoint dict holding the trainer's current weights, Adam moments, Adam step and learning rate.

    Args:
        trainer: A :class:`~isaaclab_sram.megakernel.SramPPOCartpole` instance.
        agent_cfg: Runner config the checkpoint is built for. Defaults to the registered
            ``Isaac-Cartpole-Direct`` RSL-RL config.
        iteration: Stored ``"iter"``. Defaults to ``trainer.iteration - 1``, the RSL-RL convention:
            ``model_149.pt`` / ``iter=149`` after 150 iterations.
        infos: Extra provenance merged into :func:`trainer_infos`.
        device: Device of the saved tensors. Defaults to ``agent_cfg.device``.
    """
    if agent_cfg is None:
        from .cartpole import isaaclab_cartpole_configs

        _, agent_cfg = isaaclab_cartpole_configs(DEFAULT_TASK)
    lr = float(trainer.scal[0].item())
    step = int(round(float(trainer.scal[1].item())))
    it = max(trainer.iteration - 1, 0) if iteration is None else int(iteration)
    return checkpoint_from_flat(
        trainer.params.detach().cpu(),
        agent_cfg=agent_cfg,
        learning_rate=lr,
        iteration=it,
        exp_avg=trainer.adam_m.detach().cpu(),
        exp_avg_sq=trainer.adam_v.detach().cpu(),
        adam_step=step,
        infos={**trainer_infos(trainer), **(infos or {})},
        device=device,
    )


# ------------------------------------------------------------------------------------------------
# run directory
# ------------------------------------------------------------------------------------------------
def write_run_manifest(
    run_dir: str,
    *,
    task: str,
    agent_entry_point: str = DEFAULT_AGENT_ENTRY_POINT,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write Isaac Lab's ``run.json`` (library ``rsl_rl``) with the stock helper.

    The ``agent`` metadata must equal the ``--agent`` of the play command (default
    ``rsl_rl_cfg_entry_point``) for ``--checkpoint latest|best`` to select the run. Extra metadata keys
    do not affect that match.
    """
    from isaaclab_rl.entrypoints.common import write_run_manifest as _write

    _write(run_dir, library="rsl_rl", task=task, metadata={"agent": agent_entry_point, **(metadata or {})})


def write_run_dir(
    run_dir: str,
    checkpoint: dict,
    *,
    task: str = DEFAULT_TASK,
    env_cfg: Any = None,
    agent_cfg: Any = None,
    agent_entry_point: str = DEFAULT_AGENT_ENTRY_POINT,
    manifest_metadata: dict[str, str] | None = None,
    write_manifest: bool = True,
) -> str:
    """Write ``model_<iter>.pt`` (plus ``run.json`` and ``params/*.yaml``) like an RSL-RL training run.

    Returns:
        Absolute path of the written checkpoint.
    """
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    if write_manifest:
        write_run_manifest(run_dir, task=task, agent_entry_point=agent_entry_point, metadata=manifest_metadata)
    if env_cfg is not None or agent_cfg is not None:
        from isaaclab.utils.io import dump_yaml

        if env_cfg is not None:
            dump_yaml(os.path.join(run_dir, "params", "env.yaml"), env_cfg)
        if agent_cfg is not None:
            dump_yaml(os.path.join(run_dir, "params", "agent.yaml"), agent_cfg)
    path = os.path.join(run_dir, f"model_{int(checkpoint['iter'])}.pt")
    torch.save(checkpoint, path)
    return path


def export_trainer(
    trainer: SramPPOCartpole,
    run_dir: str,
    *,
    task: str = DEFAULT_TASK,
    env_cfg: Any = None,
    agent_cfg: Any = None,
    agent_entry_point: str = DEFAULT_AGENT_ENTRY_POINT,
    infos: dict | None = None,
    write_manifest: bool = True,
    verify: bool = True,
    device: str | torch.device | None = None,
) -> dict:
    """Export a trained :class:`~isaaclab_sram.megakernel.SramPPOCartpole` as an RSL-RL run directory.

    ``params/env.yaml`` and ``params/agent.yaml`` hold copies of the given configs, updated with the
    trainer's ``num_envs`` and seed so they describe the run that produced the weights.

    Returns:
        ``{"run_dir", "checkpoint", "iter", "learning_rate", "adam_step", "verification"}``.
    """
    if env_cfg is None or agent_cfg is None:
        from .cartpole import isaaclab_cartpole_configs

        default_env_cfg, default_agent_cfg = isaaclab_cartpole_configs(task)
        env_cfg = default_env_cfg if env_cfg is None else env_cfg
        agent_cfg = default_agent_cfg if agent_cfg is None else agent_cfg
    # describe the run that produced the weights, like the stock trainer's params/*.yaml (post version shims)
    env_cfg = copy.deepcopy(env_cfg)
    agent_cfg = with_version_shims(agent_cfg)
    env_cfg.scene.num_envs = int(trainer.plan.num_envs)
    env_cfg.seed = int(trainer.seed)
    env_cfg.log_dir = os.path.abspath(run_dir)
    agent_cfg.seed = int(trainer.seed)
    ckpt = checkpoint_from_trainer(trainer, agent_cfg, infos=infos, device=device)
    path = write_run_dir(
        run_dir,
        ckpt,
        task=task,
        env_cfg=env_cfg,
        agent_cfg=agent_cfg,
        agent_entry_point=agent_entry_point,
        manifest_metadata={"trainer": "isaaclab_sram"},
        write_manifest=write_manifest,
    )
    result = {
        "run_dir": os.path.dirname(path),
        "checkpoint": path,
        "iter": int(ckpt["iter"]),
        "learning_rate": float(ckpt["optimizer_state_dict"]["param_groups"][0]["lr"]),
        "adam_step": int(round(float(trainer.scal[1].item()))),
    }
    if verify:
        result["verification"] = verify_checkpoint(path, agent_cfg, trainer=trainer)
        if not result["verification"]["ok"]:
            raise RuntimeError(f"exported checkpoint failed verification: {result['verification']}")
    return result


def export_rsl_rl_run(trainer: SramPPOCartpole, run_dir: str, **kwargs: Any) -> str:
    """Shorthand for :func:`export_trainer` that returns only the checkpoint path.

    Keyword arguments are forwarded to :func:`export_trainer`.
    """
    return export_trainer(trainer, run_dir, **kwargs)["checkpoint"]


# ------------------------------------------------------------------------------------------------
# verification
# ------------------------------------------------------------------------------------------------
def _structure(ck: dict) -> dict:
    """Key / shape / dtype / device signature of a checkpoint (values excluded)."""

    def tensors(sd):
        return {k: (tuple(v.shape), str(v.dtype), v.device.type) for k, v in sd.items()}

    opt = ck["optimizer_state_dict"]
    return {
        "keys": sorted(ck.keys()),
        "actor": tensors(ck["actor_state_dict"]),
        "critic": tensors(ck["critic_state_dict"]),
        "param_groups": [{k: v for k, v in g.items() if k != "lr"} for g in opt["param_groups"]],
        "state": {int(i): tensors(s) for i, s in opt["state"].items()},
    }


def verify_checkpoint(
    path: str,
    agent_cfg: Any = None,
    *,
    trainer: SramPPOCartpole | None = None,
    reference: str | None = None,
) -> dict:
    """Load *path* the way ``OnPolicyRunner.load`` does (strict) and check it.

    Checks:
      * the top-level keys are exactly those ``OnPolicyRunner.save`` writes, and the file also loads with
        ``torch.load(weights_only=True)``;
      * the actor and critic ``load_state_dict(strict=True)`` into freshly built runner modules, and
        ``Adam.load_state_dict`` accepts the optimizer state;
      * with *trainer*: the loaded weights and Adam moments, read back through the loaded optimizer's
        per-parameter state, equal the kernel's flat buffers bit for bit, and the Adam step and LR match;
      * with *reference* (e.g. a stock ``model_149.pt``): the same key / shape / dtype / device structure.
    """
    from . import megakernel as mk

    if agent_cfg is None:
        from .cartpole import isaaclab_cartpole_configs

        _, agent_cfg = isaaclab_cartpole_configs(DEFAULT_TASK)
    cfg = agent_cfg_to_dict(agent_cfg)
    report: dict[str, Any] = {"checkpoint": os.path.abspath(path)}
    fails: list[str] = []

    ck = torch.load(path, weights_only=False, map_location=None)  # OnPolicyRunner.load
    if tuple(sorted(ck.keys())) != tuple(sorted(CHECKPOINT_KEYS)):
        fails.append(f"top-level keys {sorted(ck.keys())}")
    try:
        torch.load(path, weights_only=True, map_location="cpu")
        report["loads_with_weights_only"] = True
    except Exception as exc:  # noqa: BLE001  (informational)
        report["loads_with_weights_only"] = False
        report["weights_only_error"] = str(exc)[:200]

    device = torch.device(cfg.get("device", "cpu"))
    actor, critic = build_rsl_rl_models(cfg, device=device)
    optimizer = build_rsl_rl_optimizer(actor, critic, cfg)
    # == rsl_rl.algorithms.PPO.load(loaded_dict, load_cfg=None, strict=True)
    actor.load_state_dict(ck["actor_state_dict"], strict=True)
    critic.load_state_dict(ck["critic_state_dict"], strict=True)
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    report["strict_load"] = True
    report["iter"] = int(ck["iter"])
    report["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    params = _chain_params(actor, critic)
    report["optimizer_param_count"] = len(params)
    steps = {float(optimizer.state[p]["step"]) for p in params if p in optimizer.state}
    report["adam_step"] = int(steps.pop()) if len(steps) == 1 else (sorted(steps) if steps else 0)
    for p in params:
        if p in optimizer.state:
            for key in ("exp_avg", "exp_avg_sq"):
                if tuple(optimizer.state[p][key].shape) != tuple(p.shape):
                    fails.append(f"{key} shape {tuple(optimizer.state[p][key].shape)} != parameter {tuple(p.shape)}")

    if trainer is not None:
        n = mk.N_PARAMS
        flat_params = mk.pack_params(actor, critic)
        report["params_max_abs_diff"] = float((flat_params[:n] - trainer.params.detach().cpu()[:n]).abs().max())
        if report["params_max_abs_diff"] != 0.0:
            fails.append("params differ from the trainer")
        step = int(round(float(trainer.scal[1].item())))
        if report["adam_step"] != step:
            fails.append(f"adam step {report['adam_step']} != trainer {step}")
        if report["learning_rate"] != float(trainer.scal[0].item()):
            fails.append("learning rate differs from the trainer")
        if step > 0:
            for key, buf in (("exp_avg", trainer.adam_m), ("exp_avg_sq", trainer.adam_v)):
                flat = _pack_parameter_tensors([optimizer.state[p][key] for p in params], actor, critic)
                err = float((flat[:n] - buf.detach().cpu()[:n]).abs().max())
                report[f"{key}_max_abs_diff"] = err
                if err != 0.0:
                    fails.append(f"{key} differs from the trainer")
            # the std parameter is the first optimizer entry and carries the moments of flat index OFF_STD
            report["std_param_is_state_0"] = bool(
                params[0] is actor.distribution.std_param
                and float(ck["optimizer_state_dict"]["state"][0]["exp_avg"].reshape(-1)[0])
                == float(trainer.adam_m[mk.OFF_STD].item())
            )
            if not report["std_param_is_state_0"]:
                fails.append("std_param moments are not optimizer state 0")

    if reference is not None:
        ref = torch.load(reference, weights_only=False, map_location=None)
        same = _structure(ref) == _structure(ck)
        report["reference"] = os.path.abspath(reference)
        report["same_structure_as_reference"] = same
        if not same:
            report["structure"] = {"export": _structure(ck), "reference": _structure(ref)}
            fails.append("structure differs from the reference checkpoint")

    report["failures"] = fails
    report["ok"] = not fails
    return report
