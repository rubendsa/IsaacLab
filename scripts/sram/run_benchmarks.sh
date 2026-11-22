#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Reproduce the Cartpole and scaling experiments, strictly sequentially.
# For native camera training and fused-kernel controls, use the separate commands
# in docs/sram/SRAM_TRAINING.md; this script does not benchmark that training path.
#
# Run on a QUIET host: no other GPU jobs and no other CPU-heavy processes (stock Isaac Lab training is
# host-latency bound, so background load inflates its iteration times).  ~30 minutes on an RTX PRO 6000.
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

echo "== 1/7 numerical verification"
"${PY}" scripts/sram/verify_megakernel.py --num_envs 4096 --seed 3 --out "${OUT}/verify_4096.json" > "${OUT}/verify_4096.log" 2>&1
"${PY}" scripts/sram/verify_megakernel.py --num_envs 4096 --seed 3 --warmup_iters 3 --out "${OUT}/verify_4096_warm.json" > "${OUT}/verify_4096_warm.log" 2>&1

echo "== 2/7 training runs (stock torch, stock warp, analytic ablation, SRAM) x seeds ${SEEDS}"
# shellcheck disable=SC2086
"${PY}" scripts/sram/benchmark_cartpole.py run --out "${OUT}" --seeds ${SEEDS}

echo "== 3/7 evaluation of every trained policy in the stock Newton env"
"${PY}" scripts/sram/benchmark_cartpole.py eval --out "${OUT}"

echo "== 4/7 scaling with the environment count: SRAM megakernel and stock Isaac Lab"
"${PY}" scripts/sram/benchmark_cartpole.py scaling --out "${OUT}"
"${PY}" scripts/sram/benchmark_cartpole.py stock_scaling --out "${OUT}"

echo "== 5/7 stock iteration times: ANYmal-C flat, G1 flat, DexSuite state + RGB64"
"${PY}" scripts/sram/benchmark_cartpole.py locomotion --out "${OUT}" --tasks anymal_c g1 dex_state dex_rgb64

echo "== 6/7 PPO update cost: stock RSL-RL vs one CUDA graph"
"${PY}" scripts/sram/projection/measure_update_cost.py --tasks cartpole anymal_c g1 --repeats 20 --tf32 on \
    --out "${OUT}/update_cost.json" > "${OUT}/update_cost.log" 2>&1

echo "== 7/7 report artefacts"
"${PY}" scripts/sram/benchmark_cartpole.py report --out "${OUT}" | tee "${OUT}/summary.md"
"${PY}" scripts/sram/projection/project_locomotion.py --stock "${OUT}/locomotion.json" \
    --update_cost "${OUT}/update_cost.json" --out "${OUT}/projection.json" | tee -a "${OUT}/summary.md"
"${PY}" scripts/sram/make_report_figures.py --bench "${OUT}" --out "${OUT}/figures"
echo "done: ${OUT}"
