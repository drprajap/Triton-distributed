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
from triton_dist.kernels.amd.common_ops import barrier_all_ipc_kernel


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
                           MIN_BLOCK_SIZE_N=64):
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
    torch.cuda.synchronize()
    torch.distributed.barrier()
    return GemmARContext(rank=rank, num_ranks=world_size, comm_bufs=comm_bufs, comm_buf_ptr=comm_buf_ptr,
                         symm_gemm_out_buf=gemm_out_buf, symm_gemm_out_buf_list=gemm_out_bufs,
                         tile_completed_buf=tile_completed_buf, dma_staging_buf=dma_staging_buf, ar_stream=ar_stream)


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
# One-shot pure AR (Iris-style, iris.x.all_reduce_one_shot).
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
    """One-shot pure all-reduce via Triton, ported from iris.x.all_reduce_one_shot.

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
    _barrier_all(ctx, num_sms)

    BLOCK_SIZE = 2048

    pure_allreduce_one_shot_kernel[(num_sms, )](
        symm_in,
        out_buf,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_COMM_SMS=num_sms,
        num_warps=4,
    )

    _barrier_all(ctx, num_sms)
    return out_buf.reshape(x.shape)
