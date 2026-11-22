# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
cuda_rt.py -- minimal, dependency-free (beyond what the Isaac Lab 3.0 venv already ships)
runtime for writing raw CUDA C++ megakernels from Python:

    NVRTC (cuda.bindings.nvrtc)  ->  cubin for sm_120  ->  CUDA driver API (cuda.bindings.driver)
    launched on torch's *current* stream, with torch tensors passed as raw device pointers.

No nvcc, no pip installs, no cooperative_groups.h (it needs CCCL's <nv/target>, which the
venv does not ship -- see test_01_headers.py).  Everything grid-wide is done with inline PTX
(device_lib.cuh, injected as an in-memory NVRTC header).

Verified on: RTX PRO 6000 Blackwell Server Edition (sm_120, 188 SMs), driver 580.178 (CUDA 13.0),
cuda-bindings 13.3.1, NVRTC 13.0.88, torch 2.11.0+cu130.

Typical use (see README section in the report / test_*.py for full examples):

    import torch, cuda_rt as rt
    mod = rt.compile_module(SRC, defines={"HIDDEN": 32})      # NVRTC -> cubin -> CUmodule
    k = mod.kernel("my_kernel")                                # extern "C" __global__ name
    k.set_max_dynamic_smem(96 * 1024)                          # >48 KB needs opt-in
    blocks_per_sm = k.occupancy(block=256, smem=96 * 1024)     # cuOccupancyMaxActiveBlocksPerMultiprocessor
    k.launch(grid=rt.num_sms() * blocks_per_sm, block=256, smem=96 * 1024,
             args=[x, y, rt.i32(n)], cooperative=True)        # on torch.cuda.current_stream()
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import pathlib
from collections.abc import Iterable, Sequence

import torch
from cuda.bindings import driver as cu
from cuda.bindings import nvrtc

__all__ = [
    "CudaError",
    "check",
    "ensure_context",
    "cuda_include_dirs",
    "compile_cubin",
    "compile_module",
    "Module",
    "Kernel",
    "i32",
    "u32",
    "i64",
    "u64",
    "f32",
    "ptr",
    "num_sms",
    "device_attr",
    "torch_stream",
    "time_cuda",
    "DEVICE_LIB",
]


# ----------------------------------------------------------------------------------------------
# Error handling: every cuda.bindings call returns (err, *values); unwrap + raise on failure.
# ----------------------------------------------------------------------------------------------
class CudaError(RuntimeError):
    pass


def check(ret):
    """Unwrap a cuda.bindings return tuple `(err, *values)`; raise CudaError if err != SUCCESS."""
    if isinstance(ret, tuple):
        err, vals = ret[0], ret[1:]
    else:
        err, vals = ret, ()
    if isinstance(err, cu.CUresult):
        if err != cu.CUresult.CUDA_SUCCESS:
            _, name = cu.cuGetErrorName(err)
            _, desc = cu.cuGetErrorString(err)
            raise CudaError(f"{name.decode() if name else err}: {desc.decode() if desc else ''}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise CudaError(f"NVRTC: {nvrtc.nvrtcGetErrorString(err)[1].decode()}")
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else vals


# ----------------------------------------------------------------------------------------------
# Context: share torch's (== Warp's == Newton's) *primary* context.  torch initialises the
# primary context lazily; we force it, then make it current on this thread for driver calls.
# ----------------------------------------------------------------------------------------------
_CTX = {}


def ensure_context(device: int | None = None):
    """Make the primary context of `device` (default: torch's current device) current on this thread."""
    if device is None:
        device = torch.cuda.current_device()
    if device not in _CTX:
        torch.cuda.init()
        torch.empty(1, device=f"cuda:{device}")  # force torch's lazy context creation
        check(cu.cuInit(0))
        dev = check(cu.cuDeviceGet(device))
        ctx = check(cu.cuDevicePrimaryCtxRetain(dev))  # same context torch/warp use
        _CTX[device] = (dev, ctx)
    check(cu.cuCtxSetCurrent(_CTX[device][1]))
    return _CTX[device]


def device_attr(name: str, device: int | None = None) -> int:
    """device_attr('MULTIPROCESSOR_COUNT') -> int (CU_DEVICE_ATTRIBUTE_<name>)."""
    dev, _ = ensure_context(device)
    return check(cu.cuDeviceGetAttribute(getattr(cu.CUdevice_attribute, "CU_DEVICE_ATTRIBUTE_" + name), dev))


def num_sms(device: int | None = None) -> int:
    return device_attr("MULTIPROCESSOR_COUNT", device)


def torch_stream() -> cu.CUstream:
    """Torch's current stream as a driver-API handle (int cudaStream_t -> CUstream)."""
    return cu.CUstream(torch.cuda.current_stream().cuda_stream)


# ----------------------------------------------------------------------------------------------
# Headers.  The venv ships the CUDA 13.0 runtime headers (cuda_fp16.h, cuda_bf16.h,
# cuda_runtime.h, cooperative_groups.h, mma.h stub) in site-packages/nvidia/cu13/include, but
# NOT CCCL (<nv/target>, <cuda/atomic>, <cuda/std/*>) nor crt/ (crt/mma.h) nor libc headers
# (<stdint.h>).  -> cuda_fp16.h / cuda_bf16.h work; cooperative_groups.h and mma.h (WMMA) do not.
# ----------------------------------------------------------------------------------------------
def cuda_include_dirs() -> list[str]:
    dirs = []
    try:
        import nvidia  # namespace package of the nvidia-* wheels

        for p in list(nvidia.__path__):
            d = os.path.join(p, "cu13", "include")
            if os.path.isdir(d):
                dirs.append(d)
    except ImportError:
        pass
    return dirs


# Small device-side library injected as an in-memory NVRTC header ("device_lib.cuh").
# Everything here is raw PTX: no CUDA headers needed.
DEVICE_LIB = r"""
#pragma once
// ---- fixed-width ints without <stdint.h> (NVRTC ships no libc headers) ----
typedef unsigned int       u32;
typedef unsigned long long u64;
typedef int                i32;
typedef long long          i64;

// ---- memory-model primitives (PTX ISA 6.0+, sm_70+) ----
__device__ __forceinline__ u32 ld_acquire_gpu(const u32* p) {
    u32 v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ u32 ld_relaxed_gpu(const u32* p) {
    u32 v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_release_gpu(u32* p, u32 v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_relaxed_gpu(u32* p, u32 v) {
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ u32 atom_add_release_gpu(u32* p, u32 v) {
    u32 old;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old; }
__device__ __forceinline__ u32 atom_add_acq_rel_gpu(u32* p, u32 v) {
    u32 old;
    asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old; }
__device__ __forceinline__ void fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ u64 globaltimer_ns() {
    u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u32 smid() { u32 r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ u32 nctas() { return gridDim.x * gridDim.y * gridDim.z; }
__device__ __forceinline__ u32 cta_rank() { return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z); }
__device__ __forceinline__ bool cta_master() { return (threadIdx.x | threadIdx.y | threadIdx.z) == 0; }

// ---------------------------------------------------------------------------------------------
// Grid-wide barrier #1 (RECOMMENDED): the same algorithm cooperative_groups::grid_group::sync()
// uses on sm_70+ (cooperative_groups/details/sync.h), re-implemented without headers.
// One u32 word in global memory, zero-initialised once (stays consistent across launches, see
// below).  Sense reversal is encoded in bit 31: CTA 0 adds 0x80000000-(N-1), every other CTA
// adds 1, so after all N CTAs arrived the word has grown by exactly 0x80000000 -> bit 31 flips
// (and bits 0..30 are back to their old value, i.e. the word is reusable forever, no reset).
//   * __syncthreads() before: all threads of this CTA finished their phase-k writes.
//   * atom.add.release.gpu by thread 0: release is cumulative -> covers the writes of the
//     other threads of the CTA that were ordered before it by bar.sync.
//   * spin with ld.acquire.gpu (no L1 caching; acquire also invalidates stale L1 lines so the
//     plain loads after the barrier see other CTAs' data).
//   * __syncthreads() after: propagate the acquire to the whole CTA.
// REQUIREMENT: all CTAs must be co-resident -> launch with cuLaunchCooperativeKernel /
// CU_LAUNCH_ATTRIBUTE_COOPERATIVE (the driver then refuses oversize grids instead of hanging).
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void grid_barrier(u32* bar) {
    __syncthreads();
    if (cta_master()) {
        const u32 nb = (cta_rank() == 0) ? (0x80000000u - (nctas() - 1u)) : 1u;
        const u32 old = atom_add_release_gpu(bar, nb);
        while (((old ^ ld_acquire_gpu(bar)) & 0x80000000u) == 0u) { /* spin */ }
    }
    __syncthreads();
}

// Split-phase variant: arrive early, do independent (CTA-local) work, then wait.
__device__ __forceinline__ u32 grid_barrier_arrive(u32* bar) {
    __syncthreads();
    u32 old = 0;
    if (cta_master()) {
        const u32 nb = (cta_rank() == 0) ? (0x80000000u - (nctas() - 1u)) : 1u;
        old = atom_add_release_gpu(bar, nb);
    }
    return old;  // only meaningful in thread 0
}
__device__ __forceinline__ void grid_barrier_wait(u32* bar, u32 old) {
    if (cta_master()) { while (((old ^ ld_acquire_gpu(bar)) & 0x80000000u) == 0u) { } }
    __syncthreads();
}
"""

_CACHE_DIR = pathlib.Path(
    os.environ.get("ISAACLAB_SRAM_CUBIN_CACHE", pathlib.Path.home() / ".cache" / "isaaclab_sram" / "cubin")
)


# cooperative_groups.h (CUDA 13.0, venv copy) unconditionally does `#include <nv/target>`
# (cooperative_groups/details/info.h:50) although NVRTC already defines the NV_IF_TARGET family
# implicitly (see the comment at cuda_fp16.h:4489-4492).  An EMPTY in-memory "nv/target" header +
# _CG_LIMIT_INCLUDED_DEPENDENCIES (use CG's own type traits instead of libcu++ <cuda/std/...>)
# makes grid.sync(), this_cluster(), map_shared_rank(), reduce(), memcpy_async() compile.
CG_SHIM_HEADERS = {"nv/target": "// empty: NVRTC provides NV_IF_TARGET & co. implicitly\n"}
CG_SHIM_DEFINES = {"_CG_LIMIT_INCLUDED_DEPENDENCIES": None}


def compile_cubin(
    src: str,
    *,
    name: str = "kernel.cu",
    arch: str | None = None,
    defines: dict | None = None,
    options: Sequence[str] = (),
    include_dirs: Iterable[str] | None = None,
    headers: dict | None = None,
    cooperative_groups: bool = False,
    verbose_ptxas: bool = False,
    cache: bool = True,
) -> tuple[bytes, str]:
    """NVRTC-compile CUDA C++ `src` straight to a cubin (SASS) for `arch` (default: the torch device).

    defines   : {"HIDDEN": 32, "USE_X": None} -> -DHIDDEN=32 -DUSE_X   (bake sizes in: big win for
                unrolling / register allocation of tiny MLPs)
    headers   : extra in-memory headers {"name.cuh": "source"}; DEVICE_LIB is always available as
                #include "device_lib.cuh"
    cooperative_groups : enable the 2-line shim that makes <cooperative_groups.h> usable
    options   : extra NVRTC flags, e.g. ["--use_fast_math", "-maxrregcount=128", "-lineinfo"]
    verbose_ptxas : add -Xptxas=-v (note: "-Xptxas=-v" form; "-Xptxas -v" as 2 args prints nothing)
    Returns (cubin_bytes, compile_log).
    """
    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
    opts = [
        f"--gpu-architecture={arch}",  # sm_XX => real cubin (no JIT at load); compute_XX => PTX only
        "-std=c++17",
        "--device-as-default-execution-space",
        "-DNVRTC=1",
    ]
    opts += [f"-I{d}" for d in (cuda_include_dirs() if include_dirs is None else include_dirs)]
    defines = dict(defines or {})
    headers = dict(headers or {})
    if cooperative_groups:
        defines.update(CG_SHIM_DEFINES)
        headers.update(CG_SHIM_HEADERS)
    for k, v in defines.items():
        opts.append(f"-D{k}" if v is None else f"-D{k}={v}")
    opts += list(options)
    if verbose_ptxas:
        opts.append("-Xptxas=-v")
    hdrs = {"device_lib.cuh": DEVICE_LIB, **headers}

    key = hashlib.sha256(repr((src, opts, sorted(hdrs.items()), nvrtc.nvrtcVersion())).encode()).hexdigest()[:24]
    cpath = _CACHE_DIR / f"{key}.cubin"
    if cache and not verbose_ptxas and cpath.exists():
        return cpath.read_bytes(), "(cached)"

    prog = check(
        nvrtc.nvrtcCreateProgram(
            src.encode(),
            name.encode(),
            len(hdrs),
            [v.encode() for v in hdrs.values()],
            [k.encode() for k in hdrs.keys()],
        )
    )
    try:
        bopts = [o.encode() for o in opts]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(bopts), bopts)
        logsz = check(nvrtc.nvrtcGetProgramLogSize(prog))
        log = b" " * logsz
        nvrtc.nvrtcGetProgramLog(prog, log)
        log = log.decode(errors="replace").rstrip("\x00").strip()
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise CudaError(f"NVRTC compile failed ({err.name}):\n{log}\nOptions: {opts}")
        size = check(nvrtc.nvrtcGetCUBINSize(prog))
        if size == 0:
            raise CudaError("no cubin produced (did you pass compute_XX instead of sm_XX?)")
        cubin = b" " * size
        check(nvrtc.nvrtcGetCUBIN(prog, cubin))
    finally:
        nvrtc.nvrtcDestroyProgram(prog)
    if cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cpath.write_bytes(cubin)
    return cubin, log


def compile_ptx(
    src: str, *, arch: str = "compute_120", defines=None, options=(), headers=None, cooperative_groups: bool = False
) -> str:
    """Same as compile_cubin but returns PTX text (useful to eyeball the generated code)."""
    defines, headers = dict(defines or {}), dict(headers or {})
    if cooperative_groups:
        defines.update(CG_SHIM_DEFINES)
        headers.update(CG_SHIM_HEADERS)
    opts = [f"--gpu-architecture={arch}", "-std=c++17", "--device-as-default-execution-space"]
    opts += [f"-I{d}" for d in cuda_include_dirs()]
    opts += [f"-D{k}" if v is None else f"-D{k}={v}" for k, v in defines.items()] + list(options)
    hdrs = {"device_lib.cuh": DEVICE_LIB, **headers}
    prog = check(
        nvrtc.nvrtcCreateProgram(
            src.encode(), b"kernel.cu", len(hdrs), [v.encode() for v in hdrs.values()], [k.encode() for k in hdrs]
        )
    )
    try:
        bopts = [o.encode() for o in opts]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(bopts), bopts)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            n = check(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise CudaError(log.decode(errors="replace"))
        n = check(nvrtc.nvrtcGetPTXSize(prog))
        ptx = b" " * n
        check(nvrtc.nvrtcGetPTX(prog, ptx))
        return ptx.decode().rstrip("\x00")
    finally:
        nvrtc.nvrtcDestroyProgram(prog)


# ----------------------------------------------------------------------------------------------
# Kernel arguments.  Kernel params are passed as an array of pointers to the argument values
# (exactly like the C driver API).  Use explicit ctypes wrappers for scalars so the size matches
# the C signature; torch tensors become their raw device pointer (void*).
# ----------------------------------------------------------------------------------------------
def i32(x):
    return ctypes.c_int32(int(x))


def u32(x):
    return ctypes.c_uint32(int(x))


def i64(x):
    return ctypes.c_int64(int(x))


def u64(x):
    return ctypes.c_uint64(int(x))


def f32(x):
    return ctypes.c_float(float(x))


def ptr(x):
    """torch.Tensor / int address -> void* argument."""
    if isinstance(x, torch.Tensor):
        assert x.is_cuda, "tensor must live on the GPU"
        return ctypes.c_void_p(x.data_ptr())
    return ctypes.c_void_p(int(x))


def _to_ctype(a):
    if isinstance(a, torch.Tensor):
        return ptr(a)
    if isinstance(a, (ctypes._SimpleCData, ctypes.Structure, ctypes.Array)):
        return a  # includes ctypes.Structure for "pass a whole param struct by value"
    if isinstance(a, bool):
        return ctypes.c_bool(a)
    if isinstance(a, int):
        if not -(2**31) <= a < 2**31:
            raise TypeError(f"int {a} does not fit int32: wrap it with i64()/u64()/ptr()")
        return ctypes.c_int32(a)
    if isinstance(a, float):
        return ctypes.c_float(a)
    raise TypeError(f"unsupported kernel arg type {type(a)}")


def pack_args(args: Sequence) -> tuple[int, tuple]:
    """Returns (address of void*[] array, keep-alive tuple).  Values are copied by the driver at
    launch time, so the keep-alive only has to survive the cuLaunch* call itself."""
    cargs = tuple(_to_ctype(a) for a in args)
    arr = (ctypes.c_void_p * max(1, len(cargs)))(*[ctypes.addressof(c) for c in cargs])
    return ctypes.addressof(arr), (arr, cargs)


# ----------------------------------------------------------------------------------------------
# Module / Kernel
# ----------------------------------------------------------------------------------------------
class Kernel:
    A = cu.CUfunction_attribute

    def __init__(self, func: cu.CUfunction, name: str):
        self.func, self.name = func, name

    # -- attributes ------------------------------------------------------------------------------
    def attr(self, which: str) -> int:
        return check(cu.cuFuncGetAttribute(getattr(self.A, "CU_FUNC_ATTRIBUTE_" + which), self.func))

    def info(self) -> dict:
        return {
            k: self.attr(k)
            for k in (
                "NUM_REGS",
                "LOCAL_SIZE_BYTES",
                "SHARED_SIZE_BYTES",
                "MAX_THREADS_PER_BLOCK",
                "MAX_DYNAMIC_SHARED_SIZE_BYTES",
                "BINARY_VERSION",
            )
        }

    def set_max_dynamic_smem(self, nbytes: int):
        """Opt in to > 48 KB dynamic shared memory (up to 99 KB = 101376 B per block on sm_120)."""
        check(cu.cuFuncSetAttribute(self.func, self.A.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(nbytes)))
        return self

    def set_attr(self, which: str, value: int):
        check(cu.cuFuncSetAttribute(self.func, getattr(self.A, "CU_FUNC_ATTRIBUTE_" + which), int(value)))
        return self

    def occupancy(self, block: int, smem: int = 0) -> int:
        """Max co-resident CTAs per SM for this (block, dynamic smem) -> grid size for a persistent
        kernel is occupancy(...) * num_sms()."""
        return check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(self.func, int(block), int(smem)))

    # -- launches --------------------------------------------------------------------------------
    @staticmethod
    def _dim3(d):
        if isinstance(d, int):
            return (d, 1, 1)
        d = tuple(d) + (1,) * (3 - len(d))
        return d

    def launch(
        self,
        grid,
        block,
        args: Sequence = (),
        *,
        smem: int = 0,
        stream=None,
        cooperative: bool = False,
        cluster=None,
        use_ex: bool | None = None,
        extra_attrs: Sequence = (),
    ):
        """Launch on torch's current stream (or `stream`: int cudaStream_t or CUstream).

        cooperative=True -> co-residency guaranteed (needed by grid_barrier) or the launch fails
                            with CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE (never a silent hang).
        cluster=(cx,cy,cz) -> thread-block clusters (sm_90+), via cuLaunchKernelEx.
        """
        g, b = self._dim3(grid), self._dim3(block)
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        hstream = stream if isinstance(stream, cu.CUstream) else cu.CUstream(int(stream))
        params, keep = pack_args(args)
        if use_ex is None:
            use_ex = cluster is not None or bool(extra_attrs)
        if not use_ex:
            if cooperative:
                check(cu.cuLaunchCooperativeKernel(self.func, *g, *b, int(smem), hstream, params))
            else:
                check(cu.cuLaunchKernel(self.func, *g, *b, int(smem), hstream, params, 0))
            return
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = g
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = b
        cfg.sharedMemBytes = int(smem)
        cfg.hStream = hstream
        attrs = []
        if cooperative:
            a = cu.CUlaunchAttribute()
            a.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_COOPERATIVE
            a.value.cooperative = 1
            attrs.append(a)
        if cluster is not None:
            cx, cy, cz = self._dim3(cluster)
            a = cu.CUlaunchAttribute()
            a.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            a.value.clusterDim.x, a.value.clusterDim.y, a.value.clusterDim.z = cx, cy, cz
            attrs.append(a)
        attrs.extend(extra_attrs)
        cfg.attrs = attrs
        cfg.numAttrs = len(attrs)
        check(cu.cuLaunchKernelEx(cfg, self.func, params, 0))

    def bind(self, grid, block, args: Sequence = (), *, smem: int = 0, stream=None, cooperative: bool = False):
        """Pre-pack everything once and return a zero-arg callable (~3 us/launch from Python instead
        of ~10 us for launch()).  Args are frozen: pointers must stay valid (keep tensors alive and
        don't reallocate them); scalars cannot change -> put per-call scalars in a device buffer."""
        g, b = self._dim3(grid), self._dim3(block)
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        hstream = stream if isinstance(stream, cu.CUstream) else cu.CUstream(int(stream))
        params, keep = pack_args(args)
        f, smem = self.func, int(smem)
        launch_fn = cu.cuLaunchCooperativeKernel if cooperative else cu.cuLaunchKernel
        extra = () if cooperative else (0,)

        def _go(_keep=keep):
            err = launch_fn(f, *g, *b, smem, hstream, params, *extra)[0]
            if err != cu.CUresult.CUDA_SUCCESS:
                check((err,))

        return _go

    def launch_config(self, grid, block, smem=0, cluster=None):
        """CUlaunchConfig for occupancy queries (cuOccupancyMaxActiveClusters etc.)."""
        g, b = self._dim3(grid), self._dim3(block)
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = g
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = b
        cfg.sharedMemBytes = int(smem)
        cfg.hStream = torch_stream()
        attrs = []
        if cluster is not None:
            a = cu.CUlaunchAttribute()
            a.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            a.value.clusterDim.x, a.value.clusterDim.y, a.value.clusterDim.z = self._dim3(cluster)
            attrs.append(a)
        cfg.attrs = attrs
        cfg.numAttrs = len(attrs)
        return cfg


class Module:
    def __init__(self, cubin: bytes, log: str = ""):
        ensure_context()
        self.cubin, self.log = cubin, log
        self.handle = check(cu.cuModuleLoadData(cubin))
        self._k = {}

    def kernel(self, name: str) -> Kernel:
        if name not in self._k:
            self._k[name] = Kernel(check(cu.cuModuleGetFunction(self.handle, name.encode())), name)
        return self._k[name]

    def global_ptr(self, name: str) -> tuple[int, int]:
        """(device address, size) of a __device__ global variable in the module."""
        dptr, size = check(cu.cuModuleGetGlobal(self.handle, name.encode()))
        return int(dptr), int(size)


def compile_module(src: str, **kw) -> Module:
    ensure_context()
    cubin, log = compile_cubin(src, **kw)
    return Module(cubin, log)


# ----------------------------------------------------------------------------------------------
# Timing helper (CUDA events on torch's current stream).  Returns per-call milliseconds list.
# ----------------------------------------------------------------------------------------------
def time_cuda(fn, *, iters: int = 10, warmup: int = 2) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e))
    return out
