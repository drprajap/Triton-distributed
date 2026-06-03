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
"""Intra-node all-reduce via rocSHMEM (put + signal + ring reduce).

Mirrors the one-shot and two-shot push algorithms in ``kernels/nvidia/allreduce.py``,
adapted for HIP using ``libshmem_device`` (rocSHMEM): ``putmem_signal_nbi_wave``,
``fence``, ``signal_wait_until`` with ``ROCSHMEM_*`` constants, and ``barrier_all``.

Multimem, TMA, and double-tree variants are not supported on this path.
"""
import dataclasses
from typing import Optional

import pyrocshmem
import os
import torch
import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra.hip.librocshmem_device import set_rocshmem_ctx
from triton_dist.kernels.allreduce import AllReduceMethod
from triton_dist.kernels.amd.common_ops import barrier_all_kernel, barrier_all_kernel_v2, barrier_on_this_grid
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import __syncthreads, tid
from triton_dist.utils import (
    NVSHMEM_SIGNAL_DTYPE,
    get_device_property,
    launch_cooperative_grid_options,
    rocshmem_barrier_all_on_stream,
)

MAX_DOUBLE_TREE_BLOCKS = 1024


def workspace_bytes_per_in_byte(world_size, method: AllReduceMethod) -> int:
    if method in [AllReduceMethod.OneShot, AllReduceMethod.OneShot_TMA]:
        return world_size
    if method in [AllReduceMethod.TwoShot]:
        return 2
    if method in [
            AllReduceMethod.OneShot_Multimem,
            AllReduceMethod.TwoShot_Multimem,
    ]:
        raise NotImplementedError(f"AllReduce method {method} is not implemented for AMD/rocSHMEM")
    raise ValueError(f"Unknown allreduce method {method}")


def get_max_chunk_nbytes(workspace_nbytes, world_size, method: AllReduceMethod) -> int:
    return workspace_nbytes // workspace_bytes_per_in_byte(world_size, method)


@dataclasses.dataclass
class AllReduceContext:
    workspace_nbytes: int
    rank: int
    world_size: int
    local_world_size: int
    symm_scatter_buf: torch.Tensor
    symm_signal: torch.Tensor
    # Trick B: symmetric int32 buffer used by barrier_all_kernel_v2 for
    # device-side team barriers (no rocshmem_barrier_all_on_stream host call).
    barrier_comm_buf: torch.Tensor
    phase: int = 0
    grid_barrier: torch.Tensor = dataclasses.field(init=False)
    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        assert self.world_size % self.local_world_size == 0
        self.nnodes = self.world_size // self.local_world_size
        self.grid_barrier = torch.zeros((1024, ), dtype=torch.int32, device="cuda")

    def finalize(self):
        torch.cuda.synchronize()
        del self.symm_scatter_buf
        del self.symm_signal
        del self.barrier_comm_buf


def create_allreduce_ctx(workspace_nbytes, rank, world_size, local_world_size) -> AllReduceContext:
    symm_scatter_buf = pyrocshmem.rocshmem_create_tensor((workspace_nbytes, ), torch.int8)
    symm_signal = pyrocshmem.rocshmem_create_tensor((MAX_DOUBLE_TREE_BLOCKS * world_size, ), NVSHMEM_SIGNAL_DTYPE)
    symm_signal.fill_(0)
    # Trick B: symmetric int32[world_size] buffer for device-side team barriers
    # via barrier_all_kernel_v2 (atomic-CAS handshake; no rocshmem host call).
    barrier_comm_buf = pyrocshmem.rocshmem_create_tensor((world_size, ), torch.int32)
    barrier_comm_buf.zero_()
    rocshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    return AllReduceContext(
        workspace_nbytes=workspace_nbytes,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        symm_scatter_buf=symm_scatter_buf,
        symm_signal=symm_signal,
        barrier_comm_buf=barrier_comm_buf,
    )


def _run_straggler(ctx, straggler_option):
    if straggler_option:
        rank, cycles = straggler_option
        if rank == ctx.rank:
            torch.cuda._sleep(cycles)


@triton_dist.jit
def kernel_ring_reduce_non_tma(
    in_ptr,
    out_ptr,
    elems_per_rank,
    begin_idx,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    num_blocks = tl.cdiv(elems_per_rank, BLOCK_SIZE)
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    for n in range(pid, num_blocks, npid):
        segment = (begin_idx + 1) % NUM_SPLITS
        c_offs = elems_per_rank * segment + BLOCK_SIZE * n + tl.arange(0, BLOCK_SIZE)
        mask = c_offs < elems_per_rank * (segment + 1)
        accum = tl.load(in_ptr + c_offs, mask=mask)
        for i in range(1, NUM_SPLITS):
            segment = (i + begin_idx + 1) % NUM_SPLITS
            c_offs = elems_per_rank * segment + BLOCK_SIZE * n + tl.arange(0, BLOCK_SIZE)
            data = tl.load(in_ptr + c_offs, mask=mask)
            accum += data
        out_offs = BLOCK_SIZE * n + tl.arange(0, BLOCK_SIZE)
        tl.store(out_ptr + out_offs, accum, mask=mask)


@triton_dist.jit
def copy_continuous_kernel(src_ptr, dst_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    num_pid = tl.num_programs(axis=0)
    for n in range(pid, n_blocks, num_pid):
        offs = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        val = tl.load(src_ptr + offs, mask=mask)
        tl.store(dst_ptr + offs, val, mask=mask)


@triton_dist.jit(do_not_specialize=["rank"])
def allreduce_one_shot_push_intra_node_kernel(
    ctx,
    input_ptr,
    output_ptr,
    symm_signal_ptr,
    symm_buffer_ptr,
    grid_barrier_ptr,
    rank,
    world_size: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    use_cooperative: tl.constexpr,
    PUT_CHUNK_BYTES: tl.constexpr,
    USE_WG_PUT: tl.constexpr = False,
):
    set_rocshmem_ctx(ctx)
    thread_idx = tid(0)
    pid = tl.program_id(0)
    num_pid = tl.num_programs(axis=0)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    symm_buffer_ptr = tl.cast(symm_buffer_ptr, input_ptr.dtype)
    nbytes = tl.cast(n_elements * elem_size, tl.uint64)
    sig_one = tl.cast(1, tl.uint64)

    # Removed in-kernel barrier_all + signal zero: now done host-side per call.

    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    # FIX1 v3 (Option C): chunks across SMs, then grid barrier + fence,
    # then one CTA per peer issues the signal-setting LAST chunk.
    # Correctness rationale:
    #   - rocshmem_fence orders puts from this PE to a remote PE only across
    #     the fence boundary. Multiple puts issued *before* the same fence
    #     have no relative ordering at the remote.
    #   - So we must split into two phases: bulk puts (no signal) -> fence
    #     (orders them at every remote) -> signal-setting last put.
    chunk_elems: tl.constexpr = PUT_CHUNK_BYTES // (tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8)
    chunk_bytes_full = tl.cast(chunk_elems * (tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8), tl.uint64)
    nchunks = (n_elements + chunk_elems - 1) // chunk_elems
    elem_bytes_const = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    peer_dst_base = symm_buffer_ptr + n_elements * rank

    # ---------------- Phase A: bulk chunks (no signal), parallelized across SMs.
    bulk_tasks = world_size * (nchunks - 1)
    for tid_t in range(pid, bulk_tasks, num_pid):
        peer = tid_t // (nchunks - 1)
        k = tid_t - peer * (nchunks - 1)
        off_elems = k * chunk_elems
        if USE_WG_PUT:
            libshmem_device.putmem_nbi_wg(
                peer_dst_base + off_elems,
                input_ptr + off_elems,
                chunk_bytes_full,
                peer,
            )
        else:
            libshmem_device.putmem_nbi_wave(
                peer_dst_base + off_elems,
                input_ptr + off_elems,
                chunk_bytes_full,
                peer,
            )
    libshmem_device.fence()
    # Local grid barrier ensures phase-A submissions on this rank are complete
    # before any CTA emits a phase-B signal-setting put.
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    # ---------------- Phase B: one CTA per peer emits the signal-setting put.
    # Because of the fence above, the remote ordering guarantees this put
    # arrives at peer P only after all phase-A puts to P have arrived.
    if pid < world_size:
        peer = pid
        last_off_elems = (nchunks - 1) * chunk_elems
        last_bytes = nbytes - tl.cast(last_off_elems, tl.uint64) * tl.cast(elem_bytes_const, tl.uint64)
        if USE_WG_PUT:
            libshmem_device.putmem_signal_nbi_wg(
                peer_dst_base + last_off_elems,
                input_ptr + last_off_elems,
                last_bytes,
                symm_signal_ptr + rank,
                sig_one,
                libshmem_device.ROCSHMEM_SIGNAL_SET,
                peer,
            )
        else:
            libshmem_device.putmem_signal_nbi_wave(
                peer_dst_base + last_off_elems,
                input_ptr + last_off_elems,
                last_bytes,
                symm_signal_ptr + rank,
                sig_one,
                libshmem_device.ROCSHMEM_SIGNAL_SET,
                peer,
            )
    libshmem_device.fence()

    if thread_idx < world_size:
        libshmem_device.signal_wait_until(
            symm_signal_ptr + thread_idx,
            libshmem_device.ROCSHMEM_CMP_EQ,
            sig_one,
        )
    __syncthreads()

    kernel_ring_reduce_non_tma(
        symm_buffer_ptr,
        output_ptr,
        n_elements,
        rank,
        world_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton_dist.jit(do_not_specialize=["rank"])
def allreduce_two_shot_push_intra_node_kernel(
    ctx,
    input_ptr,
    symm_out_ptr,
    symm_signal_ptr,
    grid_barrier_ptr,
    out_ptr,
    rank,
    world_size: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    use_cooperative: tl.constexpr,
):
    set_rocshmem_ctx(ctx)
    thread_idx = tid(0)
    pid = tl.program_id(0)
    elem_size = tl.constexpr(input_ptr.dtype.element_ty.primitive_bitwidth) // 8
    elem_per_rank = tl.cdiv(n_elements, world_size)
    symm_out_ptr = tl.cast(symm_out_ptr, input_ptr.dtype)
    symm_recv_ptr = symm_out_ptr + n_elements
    nbytes_shard = tl.cast(elem_per_rank * elem_size, tl.uint64)
    sig_one = tl.cast(1, tl.uint64)

    # Removed in-kernel barrier_all + signal zero: now done host-side per call.

    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    if pid < world_size:
        peer = (rank + pid + 1) % world_size
        libshmem_device.putmem_signal_nbi_wave(
            symm_recv_ptr + rank * elem_per_rank,
            input_ptr + peer * elem_per_rank,
            nbytes_shard,
            symm_signal_ptr + rank,
            sig_one,
            libshmem_device.ROCSHMEM_SIGNAL_SET,
            peer,
        )
    libshmem_device.fence()

    if thread_idx < world_size:
        libshmem_device.signal_wait_until(
            symm_signal_ptr + thread_idx,
            libshmem_device.ROCSHMEM_CMP_EQ,
            sig_one,
        )
    __syncthreads()
    libshmem_device.fence()

    kernel_ring_reduce_non_tma(
        symm_recv_ptr,
        symm_out_ptr + elem_per_rank * rank,
        elem_per_rank,
        rank,
        world_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    barrier_on_this_grid(grid_barrier_ptr, use_cooperative)

    symm_signal_ptr += world_size
    if pid < world_size - 1:
        peer = (rank + pid + 1) % world_size
        libshmem_device.putmem_signal_nbi_wave(
            symm_out_ptr + rank * elem_per_rank,
            symm_out_ptr + rank * elem_per_rank,
            nbytes_shard,
            symm_signal_ptr + rank,
            sig_one,
            libshmem_device.ROCSHMEM_SIGNAL_SET,
            peer,
        )
    libshmem_device.fence()

    if thread_idx < world_size and thread_idx != rank:
        libshmem_device.signal_wait_until(
            symm_signal_ptr + thread_idx,
            libshmem_device.ROCSHMEM_CMP_EQ,
            sig_one,
        )
    __syncthreads()
    libshmem_device.fence()

    copy_continuous_kernel(symm_out_ptr, out_ptr, n_elements, BLOCK_SIZE)


def allreduce_one_shot_push_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    straggler_option=None,
    max_sm: int = -1,
    num_warps: int = 16,
):
    assert x.is_cuda and x.is_contiguous()
    if output is None:
        output = torch.empty_like(x)
    else:
        assert output.is_cuda and output.is_contiguous()
        assert x.dtype == output.dtype and x.shape == output.shape
    assert x.nbytes <= ctx.workspace_nbytes // ctx.world_size

    block_size = num_warps * 64 * 16 // x.itemsize
    num_elem = x.numel()
    num_tiles = triton.cdiv(num_elem, block_size)
    _run_straggler(ctx, straggler_option)
    if max_sm > 0:
        num_tiles = min(max_sm, num_tiles)
    num_tiles = min(max(get_device_property().multi_processor_count - 4, 1), num_tiles)
    # FIX1 v3 requires at least world_size CTAs for the phase-B signal-set step.
    num_tiles = max(num_tiles, ctx.world_size)
    # Host-side equivalent of the removed in-kernel barrier_all:
    # zero signals and synchronize the team before any rank issues puts.
    ctx.symm_signal[:ctx.world_size].zero_()
    rocshmem_barrier_all_on_stream(torch.cuda.current_stream())
    dev_ctx = pyrocshmem.rocshmem_get_device_ctx()
    allreduce_one_shot_push_intra_node_kernel[(num_tiles, )](
        dev_ctx,
        x,
        output,
        ctx.symm_signal,
        ctx.symm_scatter_buf,
        ctx.grid_barrier,
        ctx.rank,
        ctx.world_size,
        num_elem,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        use_cooperative=False,
        PUT_CHUNK_BYTES=int(os.environ.get("TD_PUT_CHUNK_KB", "512")) * 1024,  # FIX1 v3 pipelined chunks (env-tunable)
        USE_WG_PUT=(os.environ.get("TD_PUSH_WG", "0") == "1"),  # Lever B: WG-collective puts (16x more parallel waves per put)
        **launch_cooperative_grid_options(),
    )
    return output


def allreduce_two_shot_push_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    straggler_option=None,
    max_sm: int = -1,
    num_warps: int = 16,
):
    assert x.numel() % ctx.world_size == 0, "two_shot allreduce requires numel divisible by world_size"
    assert x.is_cuda and x.is_contiguous()
    if output is not None:
        assert output.is_cuda and output.is_contiguous()
        assert x.dtype == output.dtype and x.nbytes == output.nbytes
    else:
        output = torch.empty_like(x)
    assert x.nbytes <= ctx.workspace_nbytes // 2

    block_size = num_warps * 64 * 16 // x.itemsize
    num_elem = x.numel()
    num_tiles = triton.cdiv(num_elem, block_size)
    _run_straggler(ctx, straggler_option)
    if max_sm > 0:
        num_tiles = min(max_sm, num_tiles)
    num_tiles = max(ctx.world_size, min(get_device_property().multi_processor_count, num_tiles))
    # Host-side equivalent of the removed in-kernel barrier_all:
    # zero signals (2*world_size slots: phase 1 + phase 2) and synchronize.
    ctx.symm_signal[:2 * ctx.world_size].zero_()
    rocshmem_barrier_all_on_stream(torch.cuda.current_stream())
    dev_ctx = pyrocshmem.rocshmem_get_device_ctx()
    allreduce_two_shot_push_intra_node_kernel[(num_tiles, )](
        dev_ctx,
        x,
        ctx.symm_scatter_buf,
        ctx.symm_signal,
        ctx.grid_barrier,
        output,
        ctx.rank,
        ctx.world_size,
        num_elem,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        use_cooperative=False,
        **launch_cooperative_grid_options(),
    )
    return output


# -----------------------------------------------------------------------------
# FIX3: pull-based one-shot (no rocSHMEM put). Each rank publishes its input
# into symm_scatter_buf, then every CTA on every rank does a direct tl.load
# from each peer's mapped symmetric buffer (via dl.symm_at) and accumulates
# locally. This bypasses the rocSHMEM IPC put path and the per-put completion
# signal entirely. Equivalent algorithm to pure_allreduce_one_shot_kernel in
# gemm_allreduce.py, used by bench_pure_ar.py to reach ~30 GB/s at WS=2.
# -----------------------------------------------------------------------------


@triton_dist.jit(do_not_specialize=["rank"])
def allreduce_one_shot_pull_intra_node_kernel(
    symm_in_ptr,
    output_ptr,
    n_elements,
    rank,
    world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    for tile_id in range(pid, num_tiles, num_pid):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        # Stagger starting peer per rank so the first hop spreads across XGMI.
        for i in range(world_size):
            peer = (rank + i) % world_size
            peer_ptr = dl.symm_at(symm_in_ptr, peer)
            partial = tl.load(peer_ptr + offs, mask=mask, other=0.0)
            acc += partial.to(tl.float32)
        tl.store(output_ptr + offs, acc.to(output_ptr.dtype.element_ty), mask=mask)


def allreduce_one_shot_pull_intra_node(
    ctx: AllReduceContext,
    x: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    straggler_option=None,
    max_sm: int = -1,
    num_warps: int = 4,
):
    assert x.is_cuda and x.is_contiguous()
    if output is None:
        output = torch.empty_like(x)
    else:
        assert output.is_cuda and output.is_contiguous()
        assert x.dtype == output.dtype and x.shape == output.shape
    # We only need to stash one copy of x per rank (not world_size copies).
    assert x.nbytes <= ctx.workspace_nbytes // ctx.world_size

    num_elem = x.numel()
    _run_straggler(ctx, straggler_option)

    # Reinterpret symm_scatter_buf (int8) as the input dtype.
    symm_in = ctx.symm_scatter_buf.view(x.dtype)[:num_elem]
    symm_in.copy_(x.view(-1))

    # Trick B: device-side team barrier (atomic-CAS over symm comm_buf,
    # parallel-thread variant) replaces rocshmem_barrier_all_on_stream.
    # Cheaper per call than the host-enqueued rocSHMEM team barrier.
    barrier_all_kernel_v2[(1,)](ctx.rank, ctx.world_size, ctx.barrier_comm_buf, num_warps=1)

    num_sms = max(get_device_property().multi_processor_count - 4, 1)
    if max_sm > 0:
        num_sms = min(max_sm, num_sms)
    BLOCK_SIZE = 2048
    allreduce_one_shot_pull_intra_node_kernel[(num_sms, )](
        symm_in,
        output,
        num_elem,
        ctx.rank,
        ctx.world_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # Trick B back-edge barrier so symm_in is safe to overwrite by the next call.
    barrier_all_kernel_v2[(1,)](ctx.rank, ctx.world_size, ctx.barrier_comm_buf, num_warps=1)
    return output



def get_auto_allreduce_method_amd(nbytes: int, world_size: int) -> AllReduceMethod:
    """Prefer two-shot when symmetric memory for one-shot would dominate."""
    if nbytes * world_size <= 64 * 1024 * 1024:
        return AllReduceMethod.OneShot
    return AllReduceMethod.TwoShot


def all_reduce(
    x: torch.Tensor,
    method: Optional[AllReduceMethod],
    ctx: AllReduceContext,
    output: Optional[torch.Tensor] = None,
    max_sm: int = -1,
    straggler_option=None,
):
    method = method or get_auto_allreduce_method_amd(x.nbytes, ctx.world_size)
    if method not in (AllReduceMethod.OneShot, AllReduceMethod.TwoShot, AllReduceMethod.OneShot_TMA):
        raise NotImplementedError(
            f"AMD all_reduce supports only OneShot/TwoShot/OneShot_TMA(pull); got {method}. "
            "Use kernels/nvidia/allreduce.py for multimem/double-tree.")

    op_handle = {
        AllReduceMethod.OneShot: allreduce_one_shot_push_intra_node,
        AllReduceMethod.TwoShot: allreduce_two_shot_push_intra_node,
        # FIX3: reuse OneShot_TMA enum slot as the AMD pull-based one-shot.
        AllReduceMethod.OneShot_TMA: allreduce_one_shot_pull_intra_node,
    }[method]

    nbytes_per_chunk = ctx.workspace_nbytes // workspace_bytes_per_in_byte(ctx.world_size, method)
    nchunks = triton.cdiv(x.nbytes, nbytes_per_chunk)
    elems_per_chunk = nbytes_per_chunk // x.itemsize

    if nchunks == 1:
        return op_handle(
            ctx=ctx,
            x=x,
            output=output,
            max_sm=max_sm,
            num_warps=16,
            straggler_option=straggler_option,
        )
    if output is None:
        output = torch.empty_like(x)
    for n in range(nchunks):
        op_handle(
            ctx=ctx,
            x=x.flatten()[elems_per_chunk * n:elems_per_chunk * (n + 1)],
            output=output.flatten()[elems_per_chunk * n:elems_per_chunk * (n + 1)],
            max_sm=max_sm,
            num_warps=16,
            straggler_option=straggler_option,
        )
    return output
