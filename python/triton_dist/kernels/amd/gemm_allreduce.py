################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import os
import torch
import dataclasses
from typing import List
import triton
import triton.language as tl
import triton_dist
import triton_dist.tune
import triton_dist.language as dl
import pyrocshmem
from hip import hip
from triton_dist.language.extra.language_extra import st
from triton_dist.utils import HIP_CHECK

from triton.runtime.driver import driver
from triton_dist.kernels.amd.common_ops import barrier_all_ipc_kernel, barrier_all_ipc_kernel_v2


@dataclasses.dataclass
class GemmARContext:
    rank: int
    num_ranks: int
    comm_bufs: List[torch.Tensor]
    comm_buf_ptr: torch.Tensor
    symm_gemm_out_buf: torch.Tensor
    symm_gemm_out_buf_list: List[torch.Tensor]
    tile_completed_buf: torch.Tensor
    dma_staging_buf: torch.Tensor
    ar_stream: torch.cuda.Stream
    # Optional second symmetric buffer (the reduce-scatter scratch `z`) used
    # only by the two-shot all-reduce path. Allocated lazily via
    # create_gemm_ar_context(..., alloc_scratch=True) so the one-shot / dma /
    # fused paths keep their original (single symmetric buffer) heap footprint.
    symm_scratch_buf: torch.Tensor = None
    symm_scratch_buf_list: List[torch.Tensor] = None

    def get_gemm_out_buf(self, A: torch.Tensor, B: torch.Tensor):
        M, N = A.shape[0], B.shape[0]
        assert self.symm_gemm_out_buf.numel() >= M * N
        return self.symm_gemm_out_buf.reshape(-1)[:M * N].reshape(M, N)

    def get_peer_gemm_out_buf(self, peer_rank: int, A: torch.Tensor, B: torch.Tensor):
        M, N = A.shape[0], B.shape[0]
        peer_buf = self.symm_gemm_out_buf_list[peer_rank]
        assert peer_buf.numel() >= M * N
        return peer_buf.reshape(-1)[:M * N].reshape(M, N)

    def get_dma_staging_buf(self, A: torch.Tensor, B: torch.Tensor):
        M, N = A.shape[0], B.shape[0]
        assert self.dma_staging_buf.numel() >= M * N
        return self.dma_staging_buf.reshape(-1)[:M * N].reshape(M, N)


## TODO: get rid of rocshmem_create_tensor_list_intra_node
def create_gemm_ar_context(ar_stream: torch.cuda.Stream, rank, world_size, max_M, N, dtype, MIN_BLOCK_SIZE_M=64,
                           MIN_BLOCK_SIZE_N=64, alloc_scratch=False):
    comm_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([world_size], torch.int32)
    comm_buf_ptr = torch.tensor([t.data_ptr() for t in comm_bufs], device=torch.cuda.current_device(),
                                requires_grad=False)
    comm_bufs[rank].zero_()
    gemm_out_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([max_M, N], dtype)
    num_tiles = triton.cdiv(max_M, MIN_BLOCK_SIZE_M) * triton.cdiv(N, MIN_BLOCK_SIZE_N)
    tile_signal_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([num_tiles * world_size], torch.int32)
    tile_signal_bufs[rank].zero_()
    gemm_out_buf = gemm_out_bufs[rank]
    tile_completed_buf = tile_signal_bufs[rank]
    dma_staging_buf = torch.empty((max_M, N), dtype=dtype, device=torch.cuda.current_device())
    # Two-shot needs a second *symmetric* buffer (the reduce-scatter scratch
    # `z`), doubling the symmetric heap footprint, so only allocate on request.
    scratch_buf = None
    scratch_bufs = None
    if alloc_scratch:
        scratch_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([max_M, N], dtype)
        scratch_buf = scratch_bufs[rank]
    torch.cuda.synchronize()
    torch.distributed.barrier()
    return GemmARContext(rank=rank, num_ranks=world_size, comm_bufs=comm_bufs, comm_buf_ptr=comm_buf_ptr,
                         symm_gemm_out_buf=gemm_out_buf, symm_gemm_out_buf_list=gemm_out_bufs,
                         tile_completed_buf=tile_completed_buf, dma_staging_buf=dma_staging_buf, ar_stream=ar_stream,
                         symm_scratch_buf=scratch_buf, symm_scratch_buf_list=scratch_bufs)


@triton.jit(do_not_specialize=["rank"])
def reset_signal_and_barrier_all_kernel(
    rank,
    num_ranks,
    comm_buf_base_ptrs,
    tile_signal_ptr,
    num_tiles,
):
    sm_id = tl.program_id(axis=0)
    num_sms = tl.num_programs(axis=0)

    for i in range(sm_id, num_tiles, num_sms):
        tl.store(tile_signal_ptr + i, 0)

    if sm_id == 0:
        barrier_all_ipc_kernel(rank, num_ranks, comm_buf_base_ptrs)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton_dist.jit
def kernel_persistent_gemm_notify_ar(
    a_ptr,
    b_ptr,
    c_ptr,  # Input/Output pointers
    tile_signal_ptr,  # Tile completion signals
    M,
    N,
    K,  # Matrix dimensions
    stride_am,
    stride_ak,  # A strides
    stride_bn,
    stride_bk,  # B strides
    stride_cm,
    stride_cn,  # C strides
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_GEMM_SMS: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_GEMM_SMS):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
        signal_offset = tile_id * world_size + rank
        for remote in range(world_size):
            remote_signal_ptr = dl.symm_at(tile_signal_ptr, remote)
            st(remote_signal_ptr + signal_offset, 1, semantic="release", scope="system")


@triton_dist.jit
def consumer_all_reduce_kernel(symm_buf_ptr, tile_signal_ptr, M, N, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr,
                               BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, NUM_COMM_SMS: tl.constexpr):
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * tl.cdiv(N, BLOCK_SIZE_N)
    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        owner_rank = tile_id % world_size
        if rank == owner_rank:
            signal_base = tile_id * world_size
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

            final_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for i in range(world_size):
                target_rank = (i + rank) % world_size
                remote_c_ptr = dl.symm_at(symm_buf_ptr, target_rank)
                token = dl.wait(tile_signal_ptr + signal_base + target_rank, 1, "sys", "acquire", waitValue=1)
                remote_c_ptr = dl.consume_token(remote_c_ptr, token)
                remote_c_ptrs = remote_c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
                remote_data = tl.load(remote_c_ptrs, mask=c_mask, other=0.0)
                final_acc += remote_data

            c = final_acc.to(symm_buf_ptr.dtype.element_ty)
            for remote_rank in range(world_size):
                remote_buf_ptr = dl.symm_at(symm_buf_ptr, remote_rank)
                remote_buf_ptrs = remote_buf_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
                tl.store(remote_buf_ptrs, c, mask=c_mask)


DEFAULT_GEMM_CONFIG = triton.Config(
    kwargs={"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4, "waves_per_eu": 2},
    num_warps=8, num_stages=2)


def get_hip_autotune_config():
    return [
        DEFAULT_GEMM_CONFIG,
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 0,
                'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 0,
                'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 0,
                'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 0,
                'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=4, num_stages=2),
        ##
    ]


def key_fn(ctx: GemmARContext, A: torch.Tensor, B: torch.Tensor, *args, **kwargs):
    return triton_dist.tune.to_hashable(A), triton_dist.tune.to_hashable(B), ctx.num_ranks


def prune_fn_by_shared_memory(config, ctx: GemmARContext, A: torch.Tensor, *args, **kwargs):
    itemsize = A.itemsize
    gemm_config: triton.Config = config["gemm_config"]
    BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = gemm_config.kwargs["BLOCK_SIZE_K"]
    num_stages = max(0, gemm_config.num_stages - 1)
    shared_mem_size = num_stages * (BLOCK_SIZE_M * BLOCK_SIZE_K * itemsize + BLOCK_SIZE_N * BLOCK_SIZE_K * itemsize)
    device = torch.cuda.current_device()
    if shared_mem_size > driver.active.utils.get_device_properties(device)["max_shared_mem"]:
        return False
    return True


@triton_dist.tune.autotune(
    config_space=[{"gemm_config": c} for c in get_hip_autotune_config()],
    key_fn=key_fn,
    prune_fn=prune_fn_by_shared_memory,
)
def gemm_allreduce_op(ctx: GemmARContext, A: torch.Tensor, B: torch.Tensor, gemm_config: triton.Config):
    NUM_COMM_SMS = 32
    assert A.shape[1] == B.shape[1], "Incompatible dimensions"
    assert A.dtype == B.dtype, "Incompatible dtypes"

    symm_c = ctx.get_gemm_out_buf(A, B)
    tile_signal = ctx.tile_completed_buf

    current_stream = torch.cuda.current_stream()
    ar_stream = ctx.ar_stream
    ar_stream.wait_stream(current_stream)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = A.shape
    N, K = B.shape
    NUM_GEMM_SMS = num_sms - NUM_COMM_SMS
    assert NUM_GEMM_SMS > 0, "Not enough SMs to run GEMM kernel"
    BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
    GROUP_SIZE_M = gemm_config.kwargs["GROUP_SIZE_M"]
    kernel_persistent_gemm_notify_ar[(NUM_GEMM_SMS, )](A, B, symm_c, tile_signal, M, N, K, A.stride(0), A.stride(1),
                                                       B.stride(0), B.stride(1), symm_c.stride(0), symm_c.stride(1),
                                                       NUM_GEMM_SMS=NUM_GEMM_SMS, **gemm_config.all_kwargs())
    with torch.cuda.stream(ar_stream):
        consumer_all_reduce(symm_c, tile_signal, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                            GROUP_SIZE_M=GROUP_SIZE_M, NUM_COMM_SMS=NUM_COMM_SMS)
    current_stream.wait_stream(ar_stream)
    reset_signal_and_barrier_all_kernel[(num_sms, )](ctx.rank, ctx.num_ranks, ctx.comm_buf_ptr, tile_signal,
                                                     tile_signal.shape[0], num_warps=16)
    return symm_c


def consumer_all_reduce(symm_buf, tile_signal, BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, GROUP_SIZE_M=1, NUM_COMM_SMS=16):
    M, N = symm_buf.shape
    consumer_all_reduce_kernel[(NUM_COMM_SMS, )](symm_buf, tile_signal, M, N, symm_buf.stride(0), symm_buf.stride(1),
                                                 BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                 GROUP_SIZE_M=GROUP_SIZE_M, NUM_COMM_SMS=NUM_COMM_SMS, num_warps=16)


# -----------------------------------------------------------------------------
# DMA-based GEMM+AR path: GEMM uses all CUs; all-reduce uses the DMA/copy engine
# (hipMemcpyDeviceToDeviceNoCU) to transfer peer GEMM outputs, then a small
# local add kernel computes the reduction. This path is used to compare against
# the CU-based consumer_all_reduce_kernel path on intra-node configurations.
# -----------------------------------------------------------------------------


@triton.jit
def kernel_persistent_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def local_add_kernel(
    dst_ptr,
    src_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    a = tl.load(dst_ptr + offs, mask=mask)
    b = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, a + b, mask=mask)


def _barrier_all(ctx: GemmARContext, num_sms: int):
    # Reuse the existing reset-and-barrier kernel already proven-correct on AMD.
    # The tile_signal slice is zeroed as a harmless side-effect.
    reset_signal_and_barrier_all_kernel[(num_sms, )](
        ctx.rank, ctx.num_ranks, ctx.comm_buf_ptr, ctx.tile_completed_buf,
        ctx.tile_completed_buf.shape[0], num_warps=16)


def _barrier_all_v2(ctx: GemmARContext):
    """Trick B (W21): parallel-thread atomic-CAS team barrier as a 1-CTA launch.

    Equivalent semantically to ``_barrier_all`` (and to
    ``rocshmem_barrier_all_on_stream``) but cheaper per call: barrier_all_ipc_kernel_v2
    parallelizes the acquire/release handshake across ``world_size`` threads of
    one warp instead of the serial ``for i in range(num_ranks)`` loop in
    barrier_all_ipc_kernel. At WS=8 on MI355 this saves ~20% latency at medium
    payloads on the pull-based one_shot path and gains ~5.7% algo BW at 128 MB
    (see MI355_AR_W20_DeepDive_2026-05-15 §"W21 Trick B"). The trick is BW-
    neutral at WS=2 where the rocSHMEM team barrier was already cheap, and
    slightly worse at the smallest message floor — i.e. net positive at the
    scale that matters.

    Note: skips the tile_signal zero-out that ``_barrier_all`` does. Pure-AR
    ops do not consume tile_signal, so this is safe; the fused GEMM+AR paths
    must keep using ``_barrier_all`` (or zero the signal slice separately).
    """
    barrier_all_ipc_kernel_v2[(1, )](
        ctx.rank, ctx.num_ranks, ctx.comm_buf_ptr, num_warps=1)


@triton_dist.tune.autotune(
    config_space=[{"gemm_config": c} for c in get_hip_autotune_config()],
    key_fn=key_fn,
    prune_fn=prune_fn_by_shared_memory,
)
def gemm_allreduce_op_dma(ctx: GemmARContext, A: torch.Tensor, B: torch.Tensor, gemm_config: triton.Config):
    """GEMM + all-reduce where AR is transported via DMA/copy engine
    (hipMemcpyDeviceToDeviceNoCU) and reduction is done by a small local
    kernel. GEMM uses all SMs since no CUs are reserved for comm.
    """
    assert A.shape[1] == B.shape[1], "Incompatible dimensions"
    assert A.dtype == B.dtype, "Incompatible dtypes"

    symm_c = ctx.get_gemm_out_buf(A, B)
    staging = ctx.get_dma_staging_buf(A, B)

    current_stream = torch.cuda.current_stream()
    ar_stream = ctx.ar_stream
    ar_stream.wait_stream(current_stream)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = A.shape
    N, _ = B.shape

    kernel_persistent_gemm[(num_sms, )](
        A, B, symm_c, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), symm_c.stride(0), symm_c.stride(1),
        NUM_SMS=num_sms, **gemm_config.all_kwargs())

    _barrier_all(ctx, num_sms)

    nbytes = M * N * A.element_size()
    for peer in range(ctx.num_ranks):
        if peer == ctx.rank:
            continue
        peer_symm_c = ctx.get_peer_gemm_out_buf(peer, A, B)
        cp_res = hip.hipMemcpyAsync(
            staging.data_ptr(),
            peer_symm_c.data_ptr(),
            nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
            ar_stream.cuda_stream,
        )
        HIP_CHECK(cp_res)
        with torch.cuda.stream(ar_stream):
            numel = M * N
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(numel, BLOCK_SIZE), )
            local_add_kernel[grid](symm_c, staging, numel, BLOCK_SIZE=BLOCK_SIZE)

    current_stream.wait_stream(ar_stream)
    _barrier_all(ctx, num_sms)
    return symm_c


@triton_dist.tune.autotune(
    config_space=[{"gemm_config": c} for c in get_hip_autotune_config()],
    key_fn=key_fn,
    prune_fn=prune_fn_by_shared_memory,
)
def gemm_allreduce_op_two_shot(ctx: GemmARContext, A: torch.Tensor, B: torch.Tensor, gemm_config: triton.Config):
    """GEMM (all SMs) then the fast two-shot (reduce-scatter + push all-gather) AR.

    Unlike ``gemm_allreduce_op`` -- which reserves 32 CUs for comm and overlaps a
    per-tile *one-shot pull* all-reduce on a side stream -- this runs a plain
    full-occupancy GEMM into the symmetric output buffer and then the standalone
    two-shot all-reduce. The two-shot moves only ~2*numel bytes/rank (vs the
    consumer kernel's world_size*numel pull) and, at world>=4, all-gathers in the
    faster XGMI *write* direction (push-AG). Because the all-reduce dominates
    latency at these GEMM shapes, dropping the overlap but using the ~2x faster
    AR is a net win at WS>=4 (see the WS=8 GEMM+AR matrix: cu mode plateaus at
    0.2-0.4x torch while the two-shot AR alone reaches ~0.76x).

    Requires ``create_gemm_ar_context(..., alloc_scratch=True)`` for the
    reduce-scatter scratch buffer used by ``pure_allreduce_two_shot_op``.
    """
    assert A.shape[1] == B.shape[1], "Incompatible dimensions"
    assert A.dtype == B.dtype, "Incompatible dtypes"
    assert ctx.symm_scratch_buf is not None, (
        "gemm_allreduce_op_two_shot requires create_gemm_ar_context(..., alloc_scratch=True)")

    symm_c = ctx.get_gemm_out_buf(A, B)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = A.shape
    N, _ = B.shape

    kernel_persistent_gemm[(num_sms, )](
        A, B, symm_c, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), symm_c.stride(0), symm_c.stride(1),
        NUM_SMS=num_sms, **gemm_config.all_kwargs())

    # Two-shot AR over the freshly produced symmetric GEMM output. The op's
    # leading barrier publishes symm_c to peers; the self-copy of symm_c into
    # symm_gemm_out_buf is a no-op (same storage).
    return pure_allreduce_two_shot_op(ctx, symm_c)


# -----------------------------------------------------------------------------
# Pure AR ops (no fused GEMM). Used for apples-to-apples bandwidth comparison
# against torch.distributed.all_reduce (RCCL). The DMA variant exercises the
# same DMA transport + local_add_kernel path as gemm_allreduce_op_dma.
# -----------------------------------------------------------------------------


def pure_allreduce_dma_op(ctx: GemmARContext, x: torch.Tensor) -> torch.Tensor:
    """Pure all-reduce via DMA transport (no GEMM).

    Writes `x` into the symmetric GEMM output buffer, does a barrier so peers
    see the published data, then for each peer: hipMemcpyAsync peer's symmetric
    buffer to a local staging buffer (DMA copy engine via
    hipMemcpyDeviceToDeviceNoCU) and add it into the local symmetric buffer
    using local_add_kernel. Second barrier seals the op.

    Returns a view of the reduced tensor in the symmetric buffer (same shape
    as `x`). The symmetric buffer is sized by the caller when building `ctx`.
    """
    numel = x.numel()
    assert numel <= ctx.symm_gemm_out_buf.numel(), (
        f"symm_gemm_out_buf too small ({ctx.symm_gemm_out_buf.numel()} elems) "
        f"for payload ({numel} elems). Increase (max_M, N) in create_gemm_ar_context.")
    assert numel <= ctx.dma_staging_buf.numel(), (
        f"dma_staging_buf too small ({ctx.dma_staging_buf.numel()} elems) "
        f"for payload ({numel} elems).")
    assert x.dtype == ctx.symm_gemm_out_buf.dtype

    symm_flat = ctx.symm_gemm_out_buf.reshape(-1)[:numel]
    staging_flat = ctx.dma_staging_buf.reshape(-1)[:numel]

    symm_flat.copy_(x.reshape(-1))

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    # Keep the serial-loop barrier on the DMA hub-and-spoke path: the
    # ar_stream + current_stream cross-stream wait pattern below relies on the
    # bigger barrier kernel's launch to serialize the team handshake across
    # all participating ranks before the per-peer hipMemcpyAsync chain
    # starts. Swapping to barrier_all_kernel_v2 (1 CTA / 1 warp) breaks
    # correctness at WS>=4 (verified 2026-W21: rank 0 got 13 vs ref 10),
    # most likely because the 1-CTA barrier completes on this rank before
    # the cross-stream wait flushes the peer's symm_flat publish. Trick B
    # is kept on the one_shot pull path where the kernel reads peers
    # directly from a single stream.
    _barrier_all(ctx, num_sms)

    current_stream = torch.cuda.current_stream()
    ar_stream = ctx.ar_stream
    ar_stream.wait_stream(current_stream)

    nbytes = numel * x.element_size()
    for peer in range(ctx.num_ranks):
        if peer == ctx.rank:
            continue
        peer_symm_flat = ctx.symm_gemm_out_buf_list[peer].reshape(-1)[:numel]
        cp_res = hip.hipMemcpyAsync(
            staging_flat.data_ptr(),
            peer_symm_flat.data_ptr(),
            nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
            ar_stream.cuda_stream,
        )
        HIP_CHECK(cp_res)
        with torch.cuda.stream(ar_stream):
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(numel, BLOCK_SIZE), )
            local_add_kernel[grid](symm_flat, staging_flat, numel, BLOCK_SIZE=BLOCK_SIZE)

    current_stream.wait_stream(ar_stream)
    _barrier_all(ctx, num_sms)
    return symm_flat.reshape(x.shape)


# -----------------------------------------------------------------------------
# One-shot pure AR.
# Each rank gathers partials from every peer's symmetric buffer in parallel via
# dl.symm_at + tl.load and reduces locally. No remote stores, no atomics.
# Parallelism is across CTAs on the XGMI mesh instead of sequential DMA hops,
# so it scales with world_size unlike the hub-and-spoke pure_allreduce_dma_op.
# -----------------------------------------------------------------------------


@triton_dist.jit
def pure_allreduce_one_shot_kernel(
    input_ptr,  # symmetric input pointer (same symm offset on every rank)
    output_ptr,  # local output pointer
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(numel, BLOCK_SIZE)

    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel

        acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        # Stagger starting peer per rank so the first hop spreads across XGMI.
        for i in range(world_size):
            peer = (rank + i) % world_size
            peer_ptr = dl.symm_at(input_ptr, peer)
            partial = tl.load(peer_ptr + offs, mask=mask, other=0.0)
            acc += partial.to(tl.float32)

        tl.store(output_ptr + offs, acc.to(output_ptr.dtype.element_ty), mask=mask)


def pure_allreduce_one_shot_op(ctx: GemmARContext, x: torch.Tensor) -> torch.Tensor:
    """One-shot pure all-reduce via Triton.

    Copies `x` into the symmetric input buffer, barriers, launches a persistent
    kernel where every rank gathers partials from all peers in parallel, then
    barriers again. Result is written to `ctx.dma_staging_buf` (reused as output)
    and returned as a view shaped like `x`.
    """
    numel = x.numel()
    assert numel <= ctx.symm_gemm_out_buf.numel(), (
        f"symm_gemm_out_buf too small ({ctx.symm_gemm_out_buf.numel()} elems) "
        f"for payload ({numel} elems).")
    assert numel <= ctx.dma_staging_buf.numel(), (
        f"dma_staging_buf too small ({ctx.dma_staging_buf.numel()} elems) "
        f"for payload ({numel} elems).")
    assert x.dtype == ctx.symm_gemm_out_buf.dtype

    symm_in = ctx.symm_gemm_out_buf.reshape(-1)[:numel]
    out_buf = ctx.dma_staging_buf.reshape(-1)[:numel]
    symm_in.copy_(x.reshape(-1))

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    # W21 Trick B: parallel-thread atomic-CAS team barrier (1-CTA launch).
    _barrier_all_v2(ctx)

    BLOCK_SIZE = 2048

    pure_allreduce_one_shot_kernel[(num_sms, )](
        symm_in,
        out_buf,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_COMM_SMS=num_sms,
        num_warps=4,
    )

    _barrier_all_v2(ctx)
    return out_buf.reshape(x.shape)


# -----------------------------------------------------------------------------
# Two-shot pure AR (reduce-scatter + all-gather). The one-shot kernel above
# is a pure all-to-all read: every rank loads every peer's full payload, so
# per-rank traffic is O(world_size * numel) and busbw plateaus at the single
# XGMI-link cap (~80 GB/s on MI355) regardless of world size. Two-shot moves
# ~2*numel per rank regardless of world size (bandwidth-optimal traffic):
#
#   1. reduce-scatter: each rank owns the contiguous chunk
#      [rank*epb, +epb); it sums *that chunk only* across all peers' symmetric
#      input and writes the partial into its local symmetric scratch `z`. After
#      a barrier, `z` on rank `b` holds the fully-reduced chunk `b`.
#   2. all-gather: each rank reads every chunk `b` from peer `b`'s `z` into its
#      local output, materializing the full reduced array.
#
# This is a flat-1D specialization of the PR's 2D row-block kernels (the bench
# harness passes 1D tensors); the chunking math is identical with epb playing
# the role of rows_per_block * N. TD's dl.symm_at replaces the PR's
# symmetric_ptr(x_ptr, my_pe, peer, heap_bases).
# -----------------------------------------------------------------------------


@triton_dist.jit
def pure_allreduce_two_shot_rs_kernel(
    input_ptr,  # symmetric input pointer (same symm offset on every rank)
    scratch_ptr,  # symmetric scratch `z` (local store target)
    numel,
    elems_per_block,  # epb = cdiv(numel, world_size)
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)

    chunk_start = rank * elems_per_block
    chunk_end = tl.minimum(chunk_start + elems_per_block, numel)
    num_tiles = tl.cdiv(elems_per_block, BLOCK_SIZE)

    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        offs = chunk_start + tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_end

        acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        # Stagger starting peer per CTA+rank so first-hop traffic is spread
        # across peers at finer granularity than rank-only rotation.
        start_peer = (rank + pid) % world_size
        for i in range(world_size):
            peer = (start_peer + i) % world_size
            peer_ptr = dl.symm_at(input_ptr, peer)
            partial = tl.load(peer_ptr + offs, mask=mask, other=0.0)
            acc += partial.to(tl.float32)

        tl.store(scratch_ptr + offs, acc.to(scratch_ptr.dtype.element_ty), mask=mask)


@triton_dist.jit
def pure_allreduce_two_shot_ag_kernel(
    scratch_ptr,  # symmetric scratch `z` (read from peer `b`)
    output_ptr,  # local output pointer
    numel,
    elems_per_block,  # epb = cdiv(numel, world_size)
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    b = tl.program_id(1)  # owner chunk index == owner rank

    chunk_start = b * elems_per_block
    chunk_end = tl.minimum(chunk_start + elems_per_block, numel)
    num_tiles = tl.cdiv(elems_per_block, BLOCK_SIZE)

    peer_z = dl.symm_at(scratch_ptr, b)
    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        offs = chunk_start + tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_end
        data = tl.load(peer_z + offs, mask=mask, other=0.0)
        tl.store(output_ptr + offs, data, mask=mask)


@triton_dist.jit
def pure_allreduce_two_shot_rs_kernel_interleaved(
    input_ptr,  # symmetric input pointer (same symm offset on every rank)
    scratch_ptr,  # symmetric scratch `z` (local owner stores reduced tiles)
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    # Ownership policy: rank r owns tiles where tile_id % world_size == r.
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(numel, BLOCK_SIZE)
    first_tile = rank + pid * world_size
    tile_stride = NUM_COMM_SMS * world_size

    for tile_id in range(first_tile, num_tiles, tile_stride):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        start_peer = (rank + pid) % world_size
        for i in range(world_size):
            peer = (start_peer + i) % world_size
            peer_ptr = dl.symm_at(input_ptr, peer)
            partial = tl.load(peer_ptr + offs, mask=mask, other=0.0)
            acc += partial.to(tl.float32)
        tl.store(scratch_ptr + offs, acc.to(scratch_ptr.dtype.element_ty), mask=mask)


@triton_dist.jit
def pure_allreduce_two_shot_ag_kernel_interleaved(
    scratch_ptr,  # symmetric scratch `z` (owner rank decided by tile_id % world_size)
    output_ptr,  # local output pointer
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    world_size = dl.num_ranks()
    num_tiles = tl.cdiv(numel, BLOCK_SIZE)
    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        owner = tile_id % world_size
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        peer_z = dl.symm_at(scratch_ptr, owner)
        data = tl.load(peer_z + offs, mask=mask, other=0.0)
        tl.store(output_ptr + offs, data, mask=mask)


@triton_dist.jit
def pure_allreduce_two_shot_rs_push_kernel(
    input_ptr,  # symmetric input
    recv_ptr,  # symmetric recv buffer (W slots of epb); write to peer `b` slot `rank`
    numel,
    elems_per_block,  # epb = numel // world_size (push path requires exact division)
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    # Push reduce-scatter: rank `r` writes its chunk-`b` slice of the input into
    # owner `b`'s recv buffer at slot `r`. After a barrier, owner `b` holds all
    # W copies of chunk `b` and reduces them locally. Cross-fabric traffic is in
    # the (faster) write direction.
    rank = dl.rank()
    pid = tl.program_id(0)
    b = tl.program_id(1)  # owner rank of this chunk == target peer

    src_start = b * elems_per_block
    num_tiles = tl.cdiv(elems_per_block, BLOCK_SIZE)
    peer_recv = dl.symm_at(recv_ptr, b)
    dst_start = rank * elems_per_block  # my slot in owner b's recv buffer
    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        j = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = j < elems_per_block
        data = tl.load(input_ptr + src_start + j, mask=mask, other=0.0)
        tl.store(peer_recv + dst_start + j, data, mask=mask)


@triton_dist.jit
def pure_allreduce_two_shot_local_reduce_kernel(
    recv_ptr,  # symmetric recv buffer: W slots of epb, all holding this rank's chunk
    out_ptr,  # symmetric output: write reduced chunk to [rank*epb, ...)
    numel,
    elems_per_block,
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(elems_per_block, BLOCK_SIZE)
    my_chunk = rank * elems_per_block
    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        j = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = j < elems_per_block
        acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        for s in range(world_size):
            acc += tl.load(recv_ptr + s * elems_per_block + j, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + my_chunk + j, acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton_dist.jit(do_not_specialize=["send_chunk"])
def ring_rs_send_kernel(
    wbuf_ptr,  # symmetric working buffer (W chunks of epb); send chunk `send_chunk`
    recv_ptr,  # symmetric recv buffer (epb); write into next rank's recv
    elems_per_block,  # epb
    send_chunk,  # which chunk this rank forwards this step
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    # Ring reduce-scatter, push step: rank r writes its current partial for
    # `send_chunk` into the *next* rank's recv buffer (XGMI write direction).
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    nxt = (rank + 1) % world_size
    peer_recv = dl.symm_at(recv_ptr, nxt)
    src_start = send_chunk * elems_per_block
    num_tiles = tl.cdiv(elems_per_block, BLOCK_SIZE)
    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        j = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = j < elems_per_block
        data = tl.load(wbuf_ptr + src_start + j, mask=mask, other=0.0)
        tl.store(peer_recv + j, data, mask=mask)


@triton_dist.jit(do_not_specialize=["recv_chunk"])
def ring_rs_add_kernel(
    wbuf_ptr,  # symmetric working buffer; accumulate into chunk `recv_chunk`
    recv_ptr,  # local recv buffer holding neighbour's partial (epb)
    elems_per_block,  # epb
    recv_chunk,  # which chunk this rank reduces this step
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    # Local add (HBM): wbuf[recv_chunk] += recv. Both reads + the write are local.
    pid = tl.program_id(0)
    dst_start = recv_chunk * elems_per_block
    num_tiles = tl.cdiv(elems_per_block, BLOCK_SIZE)
    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        j = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = j < elems_per_block
        cur = tl.load(wbuf_ptr + dst_start + j, mask=mask, other=0.0).to(tl.float32)
        inc = tl.load(recv_ptr + j, mask=mask, other=0.0).to(tl.float32)
        tl.store(wbuf_ptr + dst_start + j, (cur + inc).to(wbuf_ptr.dtype.element_ty), mask=mask)


@triton_dist.jit
def pure_allreduce_two_shot_ag_push_kernel(
    scratch_ptr,  # symmetric scratch `z`: this rank owns chunk `rank` (local read)
    symm_out_ptr,  # symmetric output target (written on self + every peer)
    numel,
    elems_per_block,  # epb = cdiv(numel, world_size)
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    # Push all-gather: each rank owns
    # reduced chunk `rank` and *writes* it into every peer's output via the XGMI
    # write direction, which saturates the fabric better than dependent remote
    # loads at large world size. Persistent 1D grid; each tile is loaded once
    # and stored to self + every peer.
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)

    chunk_start = rank * elems_per_block
    chunk_end = tl.minimum(chunk_start + elems_per_block, numel)
    num_tiles = tl.cdiv(elems_per_block, BLOCK_SIZE)

    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        offs = chunk_start + tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_end
        data = tl.load(scratch_ptr + offs, mask=mask, other=0.0)
        # Self block (local store).
        tl.store(symm_out_ptr + offs, data, mask=mask)
        # Push the tile to every peer, rotating the start by `pid` for traffic
        # shaping. Rotate within the (world_size-1) non-self peers (mod W-1): the
        # naive (rank+1+pid+i)%W form would skip a real peer and break the AG.
        for i in range(world_size - 1):
            peer = (rank + 1 + (pid + i) % (world_size - 1)) % world_size
            peer_out = dl.symm_at(symm_out_ptr, peer)
            tl.store(peer_out + offs, data, mask=mask)


@triton_dist.jit
def pure_allreduce_two_shot_fused_kernel(
    symm_in_ptr,  # symmetric input (this rank's partial); chunk `rank` read from all peers
    symm_out_ptr,  # symmetric output (reduced chunk written self + pushed to peers)
    numel,
    elems_per_block,  # epb = cdiv(numel, world_size)
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    # Fused reduce-scatter + push all-gather (barrier-fusion, Lever A).
    #
    # Rank `r` owns chunk `r`. For each tile of its chunk it (1) pull-reduces
    # that tile across all peers' inputs *in registers*, then (2) immediately
    # writes the reduced tile to its own output and pushes it to every peer's
    # output. Because we read `symm_in` but write a *separate* symmetric
    # `symm_out`, there is no RS<->AG write-after-read hazard on the input, so
    # the mid barrier between the two phases AND the HBM scratch round-trip
    # (write reduced chunk to `z`, then reload it in the AG kernel) are both
    # removed. Net: 1 kernel + 2 barriers, vs 2 kernels + 3 barriers. Same XGMI
    # traffic as the split two-shot (2*(W-1)/W*numel).
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)

    chunk_start = rank * elems_per_block
    chunk_end = tl.minimum(chunk_start + elems_per_block, numel)
    num_tiles = tl.cdiv(elems_per_block, BLOCK_SIZE)

    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        offs = chunk_start + tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < chunk_end
        # (1) pull-reduce this tile across all peers (rotate start per CTA+rank).
        acc = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
        start_peer = (rank + pid) % world_size
        for i in range(world_size):
            peer = (start_peer + i) % world_size
            peer_in = dl.symm_at(symm_in_ptr, peer)
            partial = tl.load(peer_in + offs, mask=mask, other=0.0)
            acc += partial.to(tl.float32)
        data = acc.to(symm_out_ptr.dtype.element_ty)
        # (2) write reduced tile locally, then push to every peer's output
        # (rotate within the W-1 non-self peers, same as the push-AG kernel).
        tl.store(symm_out_ptr + offs, data, mask=mask)
        for i in range(world_size - 1):
            peer = (rank + 1 + (pid + i) % (world_size - 1)) % world_size
            peer_out = dl.symm_at(symm_out_ptr, peer)
            tl.store(peer_out + offs, data, mask=mask)


@triton_dist.jit
def pure_allreduce_two_shot_ag_push_kernel_interleaved(
    scratch_ptr,  # symmetric scratch `z`: owner rank determined by tile_id % world_size
    symm_out_ptr,  # symmetric output target (written on self + every peer)
    numel,
    BLOCK_SIZE: tl.constexpr,
    NUM_COMM_SMS: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(numel, BLOCK_SIZE)
    first_tile = rank + pid * world_size
    tile_stride = NUM_COMM_SMS * world_size
    for tile_id in range(first_tile, num_tiles, tile_stride):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        data = tl.load(scratch_ptr + offs, mask=mask, other=0.0)
        tl.store(symm_out_ptr + offs, data, mask=mask)
        # Rotate only across non-self peers; this preserves traffic shaping while
        # guaranteeing each remote peer gets the tile exactly once.
        for i in range(world_size - 1):
            peer = (rank + 1 + (pid + i) % (world_size - 1)) % world_size
            peer_out = dl.symm_at(symm_out_ptr, peer)
            tl.store(peer_out + offs, data, mask=mask)


def pure_allreduce_two_shot_op(ctx: GemmARContext, x: torch.Tensor) -> torch.Tensor:
    """Two-shot pure all-reduce (reduce-scatter + all-gather) via Triton.

    Moves ~2*numel bytes per rank
    regardless of world size (vs one-shot's world_size*numel), so busbw scales
    with world size instead of pinning to the single-link cap.

    Requires the context to have been built with ``alloc_scratch=True`` so the
    symmetric reduce-scatter scratch ``z`` is available.
    """
    assert ctx.symm_scratch_buf is not None, (
        "two-shot requires create_gemm_ar_context(..., alloc_scratch=True)")
    numel = x.numel()
    # Lever A (barrier/phase fusion): a single fused RS+AG kernel that drops the
    # mid barrier, the HBM scratch round-trip, and one kernel launch. It wins in
    # the latency-bound regime (e.g. +45% at 8 MB) but loses once bandwidth-bound
    # (crossover ~64 MB), so default it on only for payloads <= 64 MB.
    # TD_TWO_SHOT_FUSED=1/0 forces on/off regardless of size.
    _fused_env = os.environ.get("TD_TWO_SHOT_FUSED", "auto")
    _nbytes = numel * x.element_size()
    _use_fused = (ctx.num_ranks >= 4 and numel % ctx.num_ranks == 0
                  and (_nbytes <= (64 << 20) if _fused_env == "auto" else _fused_env == "1"))
    if _use_fused:
        return pure_allreduce_two_shot_fused_op(ctx, x)
    assert numel <= ctx.symm_gemm_out_buf.numel(), (
        f"symm_gemm_out_buf too small ({ctx.symm_gemm_out_buf.numel()} elems) "
        f"for payload ({numel} elems).")
    assert numel <= ctx.symm_scratch_buf.numel(), (
        f"symm_scratch_buf too small ({ctx.symm_scratch_buf.numel()} elems) "
        f"for payload ({numel} elems).")
    assert numel <= ctx.dma_staging_buf.numel(), (
        f"dma_staging_buf too small ({ctx.dma_staging_buf.numel()} elems) "
        f"for payload ({numel} elems).")
    assert x.dtype == ctx.symm_gemm_out_buf.dtype

    symm_in = ctx.symm_gemm_out_buf.reshape(-1)[:numel]
    symm_z = ctx.symm_scratch_buf.reshape(-1)[:numel]
    out_buf = ctx.dma_staging_buf.reshape(-1)[:numel]
    symm_in.copy_(x.reshape(-1))

    world_size = ctx.num_ranks
    dist_policy = os.environ.get("TD_TWO_SHOT_DIST", "block").lower()
    if dist_policy not in ("block", "interleaved"):
        raise ValueError(f"TD_TWO_SHOT_DIST must be block|interleaved, got {dist_policy}")
    elems_per_block = triton.cdiv(numel, world_size)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    BLOCK_SIZE = 2048

    # Publish `x` to peers before the reduce-scatter reads them.
    _barrier_all_v2(ctx)

    if dist_policy == "interleaved":
        pure_allreduce_two_shot_rs_kernel_interleaved[(num_sms, )](
            symm_in,
            symm_z,
            numel,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_COMM_SMS=num_sms,
            num_warps=4,
        )
    else:
        pure_allreduce_two_shot_rs_kernel[(num_sms, )](
            symm_in,
            symm_z,
            numel,
            elems_per_block,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_COMM_SMS=num_sms,
            num_warps=4,
        )

    # Reduced chunks must be globally visible before the all-gather.
    _barrier_all_v2(ctx)

    # All-gather direction is chosen by world size (evidence: on MI355 XGMI the
    # write direction scales better than read-pull once >2 peers contend for the
    # incoming links, so push-AG wins at world>=4; at world==2 a single link is
    # symmetric and pull-AG is marginally better). Override with TD_AG_PUSH=0/1.
    _push_env = os.environ.get("TD_AG_PUSH", "auto")
    use_push_ag = (world_size >= 4) if _push_env == "auto" else (_push_env == "1")

    if use_push_ag:
        # Push all-gather: write owned reduced chunk into every peer's symmetric
        # output (reuse symm_in, free after RS). Result lands in symm_in on all
        # ranks; return it directly (no extra device copy).
        if dist_policy == "interleaved":
            pure_allreduce_two_shot_ag_push_kernel_interleaved[(num_sms, )](
                symm_z,
                symm_in,
                numel,
                BLOCK_SIZE=BLOCK_SIZE,
                NUM_COMM_SMS=num_sms,
                num_warps=4,
            )
        elif numel % world_size == 0:
            pure_allreduce_two_shot_ag_push_kernel[(num_sms, )](
                symm_z,
                symm_in,
                numel,
                elems_per_block,
                BLOCK_SIZE=BLOCK_SIZE,
                NUM_COMM_SMS=num_sms,
                num_warps=4,
            )
        else:
            use_push_ag = False
    if use_push_ag:
        _barrier_all_v2(ctx)
        return symm_in.reshape(x.shape)

    if dist_policy == "interleaved":
        pure_allreduce_two_shot_ag_kernel_interleaved[(num_sms, )](
            symm_z,
            out_buf,
            numel,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_COMM_SMS=num_sms,
            num_warps=4,
        )
    else:
        pure_allreduce_two_shot_ag_kernel[(num_sms, world_size)](
            symm_z,
            out_buf,
            numel,
            elems_per_block,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_COMM_SMS=num_sms,
            num_warps=4,
        )
    # Ensure all peers finished reading `z` before it is reused next call.
    _barrier_all_v2(ctx)
    return out_buf.reshape(x.shape)


def pure_allreduce_two_shot_fused_op(ctx: GemmARContext, x: torch.Tensor) -> torch.Tensor:
    """Barrier-fused two-shot all-reduce (Lever A): one kernel, two barriers.

    Reads the (published) symmetric input and writes the reduced-and-gathered
    result into the *separate* symmetric scratch buffer, so the RS<->AG mid
    barrier and the HBM scratch round-trip of the split two-shot are both
    removed. Same XGMI traffic as the split two-shot, but 2 barriers + 1 kernel
    instead of 3 barriers + 2 kernels. Assumes the push regime (world>=4,
    numel divisible by world_size) and ``alloc_scratch=True``.
    """
    assert ctx.symm_scratch_buf is not None, (
        "two-shot fused requires create_gemm_ar_context(..., alloc_scratch=True)")
    numel = x.numel()
    world_size = ctx.num_ranks
    assert numel % world_size == 0, "two-shot fused requires numel divisible by world_size"
    assert numel <= ctx.symm_gemm_out_buf.numel()
    assert numel <= ctx.symm_scratch_buf.numel()
    assert x.dtype == ctx.symm_gemm_out_buf.dtype

    symm_in = ctx.symm_gemm_out_buf.reshape(-1)[:numel]
    symm_z = ctx.symm_scratch_buf.reshape(-1)[:numel]
    symm_in.copy_(x.reshape(-1))

    elems_per_block = triton.cdiv(numel, world_size)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    BLOCK_SIZE = 2048

    _barrier_all_v2(ctx)  # publish input to peers
    pure_allreduce_two_shot_fused_kernel[(num_sms, )](
        symm_in,
        symm_z,
        numel,
        elems_per_block,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_COMM_SMS=num_sms,
        num_warps=4,
    )
    _barrier_all_v2(ctx)  # all peer pushes landed before `z` is read/reused
    return symm_z.reshape(x.shape)


def pure_allreduce_two_shot_push_op(ctx: GemmARContext, x: torch.Tensor) -> torch.Tensor:
    """Fully push-based two-shot all-reduce (direct peer stores, no rocSHMEM put).

    Both phases move cross-fabric data in the XGMI *write* direction, which
    saturates the fabric better than dependent remote loads at large world size:

      1. RS-push:   rank r writes its chunk-b slice into owner b's recv slot r.
      2. local reduce: owner b sums its W recv slots -> reduced chunk b (HBM).
      3. AG-push:   rank r writes reduced chunk r into every peer's output.

    Requires ``alloc_scratch=True`` and ``numel % world_size == 0``.
    """
    assert ctx.symm_scratch_buf is not None, (
        "two-shot push requires create_gemm_ar_context(..., alloc_scratch=True)")
    numel = x.numel()
    world_size = ctx.num_ranks
    assert numel % world_size == 0, "two-shot push requires numel divisible by world_size"
    assert numel <= ctx.symm_gemm_out_buf.numel()
    assert numel <= ctx.symm_scratch_buf.numel()
    assert x.dtype == ctx.symm_gemm_out_buf.dtype

    symm_in = ctx.symm_gemm_out_buf.reshape(-1)[:numel]  # input, then reduced chunks
    symm_recv = ctx.symm_scratch_buf.reshape(-1)[:numel]  # recv slots, then AG output
    symm_in.copy_(x.reshape(-1))

    elems_per_block = numel // world_size
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    BLOCK_SIZE = 2048

    _barrier_all_v2(ctx)  # publish input
    pure_allreduce_two_shot_rs_push_kernel[(num_sms, world_size)](
        symm_in, symm_recv, numel, elems_per_block,
        BLOCK_SIZE=BLOCK_SIZE, NUM_COMM_SMS=num_sms, num_warps=4)
    _barrier_all_v2(ctx)  # all recv slots delivered
    pure_allreduce_two_shot_local_reduce_kernel[(num_sms, )](
        symm_recv, symm_in, numel, elems_per_block,
        BLOCK_SIZE=BLOCK_SIZE, NUM_COMM_SMS=num_sms, num_warps=4)
    _barrier_all_v2(ctx)  # reduced chunks ready; recv free to reuse as AG output
    pure_allreduce_two_shot_ag_push_kernel[(num_sms, )](
        symm_in, symm_recv, numel, elems_per_block,
        BLOCK_SIZE=BLOCK_SIZE, NUM_COMM_SMS=num_sms, num_warps=4)
    _barrier_all_v2(ctx)  # all gathers landed
    return symm_recv.reshape(x.shape)
