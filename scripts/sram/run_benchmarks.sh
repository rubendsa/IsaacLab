#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Reproduce every measured number of docs/sram/SRAM_TRAINING.md, strictly sequentially: the numerical
# verification, the Cartpole runs (stock, ablations, SRAM) with evaluation, throughput and time to policy vs the
# environment count, the stock iteration times of the larger tasks and the PPO update cost.
#
# Run on a QUIET host: no other GPU jobs and no other CPU-heavy processes (stock Isaac Lab training is
# host-latency bound, so background load inflates its iteration times). Runs that overlap another GPU
# process are repeated (--quiet_retries); contention.log records the host every 5 s.
# About 1 hour on an RTX PRO 6000 Blackwell.
#
#   scripts/sram/run_benchmarks.sh [OUT_DIR]

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-${REPO}/.venv/bin/python}"
OUT="${1:-${REPO}/../sram_bench_$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-1 2 3}"
mkdir -p "${OUT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${REPO}"

others=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E "dcvagent|gnome|Xorg|remote-desktop" || true)
if [[ -n "${others}" && -z "${FORCE:-}" ]]; then
    echo "other GPU compute processes are running (set FORCE=1 to ignore):"; echo "${others}"; exit 1
fi
{
    date -Is; hostname; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv
    git -C "${REPO}" rev-parse HEAD; git -C "${REPO}" status --short | head -20
    "${PY}" -c "import torch, warp, newton, importlib.metadata as m; print('torch', torch.__version__, 'warp', warp.__version__, 'newton', newton.__version__, 'rsl-rl', m.version('rsl-rl-lib'))"
} > "${OUT}/provenance.txt" 2>&1

# contention monitor: other GPU compute processes and load average every 5 s (any foreign GPU process during a
# measurement invalidates it; see contention.log)
(
    while true; do
        echo "$(date -Is) load=$(cut -d' ' -f1 /proc/loadavg) gpu_apps=$(nvidia-smi --query-compute-apps=pid,process_name \
            --format=csv,noheader | grep -v -E "dcvagent|gnome|Xorg|remote-desktop" | tr '\n' ';')"
        sleep 5
    done
) > "${OUT}/contention.log" 2>&1 &
MONITOR=$!
trap 'kill ${MONITOR} 2>/dev/null' EXIT

echo "== 1/8 numerical verification"
"${PY}" scripts/sram/verify_megakernel.py --num_envs 4096 --seed 3 --out "${OUT}/verify_4096.json" > "${OUT}/verify_4096.log" 2>&1
"${PY}" scripts/sram/verify_megakernel.py --num_envs 4096 --seed 3 --warmup_iters 3 --out "${OUT}/verify_4096_warm.json" > "${OUT}/verify_4096_warm.log" 2>&1
# closed-form physics against recorded Newton/MuJoCo-Warp transitions (and the megakernel against the closed form)
mkdir -p "${OUT}/newton_check"
(
    cd "${OUT}/newton_check"
    "${PY}" "${REPO}/scripts/sram/baselines/record_newton_rollouts.py" --actions random --resets on > record_resets_on.log 2>&1
    "${PY}" "${REPO}/scripts/sram/baselines/record_newton_rollouts.py" --actions random --resets off --action_amp 10 \
        --steps 120 > record_resets_off.log 2>&1
    "${PY}" "${REPO}/scripts/sram/baselines/torch_cartpole_vecenv.py" --check --megakernel \
        --dataset ref_random_resetson_n4096.npz ref_random_resetsoff_n4096_amp10.npz --out check.json > check.log 2>&1
)

echo "== 2/8 training runs (stock torch, stock warp, analytic ablations, SRAM) x seeds ${SEEDS}"
# shellcheck disable=SC2086
"${PY}" scripts/sram/benchmark_cartpole.py run --out "${OUT}" --seeds ${SEEDS} --quiet_retries 3

echo "== 3/8 evaluation of every trained policy in the stock Newton env"
"${PY}" scripts/sram/benchmark_cartpole.py eval --out "${OUT}"

echo "== 4/8 throughput vs the environment count: SRAM megakernel and stock Isaac Lab"
"${PY}" scripts/sram/benchmark_cartpole.py scaling --out "${OUT}"
"${PY}" scripts/sram/profile_megakernel.py --num_envs 4096 --out "${OUT}/profile_4096.json" > "${OUT}/profile_4096.log" 2>&1
# does the rollout buffer's placement matter? shared memory vs global memory at the same env counts, twice
for rep in 1 2; do
    "${PY}" scripts/sram/benchmark_cartpole.py scaling --out "${OUT}/storage_ablation/rep${rep}" --env_counts 4096 8192 \
        --storages smem global --iterations 50
done
"${PY}" scripts/sram/benchmark_cartpole.py stock_scaling --out "${OUT}" --quiet_retries 3
# a second stock sweep: the report takes the faster of the two per env count (stock is host-latency bound)
"${PY}" scripts/sram/benchmark_cartpole.py stock_scaling --out "${OUT}/stock_rep2" --quiet_retries 3

echo "== 5/8 time to policy vs the environment count: SRAM megakernel and stock Isaac Lab"
for N in 1024 16384 65536 262144; do
    # shellcheck disable=SC2086
    "${PY}" scripts/sram/benchmark_cartpole.py run --out "${OUT}/envs_${N}" --num_envs "${N}" --configs sram stock_torch \
        --seeds ${SEEDS} --quiet_retries 3
done

echo "== 6/8 stock iteration times: ANYmal-C flat, G1 flat, DexSuite state + RGB64"
"${PY}" scripts/sram/benchmark_cartpole.py locomotion --out "${OUT}" --tasks anymal_c g1 dex_state dex_rgb64 \
    --quiet_retries 3

echo "== 7/8 PPO update cost: stock RSL-RL vs one CUDA graph"
"${PY}" scripts/sram/projection/measure_update_cost.py --tasks cartpole anymal_c g1 --repeats 20 --tf32 on \
    --out "${OUT}/update_cost.json" > "${OUT}/update_cost.log" 2>&1

echo "== 8/8 report artefacts: tables, projection, figures"
"${PY}" scripts/sram/projection/project_locomotion.py --stock "${OUT}/locomotion.json" \
    --update_cost "${OUT}/update_cost.json" --out "${OUT}/projection.json" > /dev/null
"${PY}" scripts/sram/summarize_benchmarks.py --bench "${OUT}" --out "${OUT}/summary.md" --export "${OUT}/results"
"${PY}" scripts/sram/make_report_figures.py --bench "${OUT}" --out "${OUT}/figures"
echo "done: ${OUT}"
