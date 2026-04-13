import argparse
import datetime
import importlib.util
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import torch
import triton
import triton.language as tl

import pyrocshmem
from hip import hip
from triton_dist.utils import HIP_CHECK
from triton_dist.cu_masking import (
    CUMaskedStreamWrapper,
    create_cu_mask,
    create_stream_with_cu_mask,
    get_device_info,
    partition_cus,
    print_cu_distribution,
)


def load_tutorial09_module():
    tutorial09_path = Path(__file__).with_name("09-AMD-overlapping-allgather-gemm.py")
    spec = importlib.util.spec_from_file_location("tutorial09_ag_gemm", tutorial09_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {tutorial09_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


t09 = load_tutorial09_module()
RANK = 0
WORLD_SIZE = 1


@triton.jit
def cu_copy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, data, mask=mask)


def producer_ag_cu_kernel(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    one: torch.Tensor,
    M_PER_CHUNK: int,
    ag_stream_pool: List,
    barrier_buffers: List[torch.Tensor],
):
    M_per_rank, N = local_tensor.shape
    chunk_num_per_rank = M_per_rank // M_PER_CHUNK
    num_stream = len(ag_stream_pool)
    rank_orders = [(rank + i) % num_ranks for i in range(num_ranks)]

    barrier_elem_size = one.element_size()
    BLOCK_SIZE = 1024
    local_flat = local_tensor.reshape(-1)

    for idx, remote_rank in enumerate(rank_orders):
        if remote_rank == rank:
            continue
        for chunk_idx in range(chunk_num_per_rank):
            chunk_pos = rank * chunk_num_per_rank + chunk_idx
            stream_pos = idx % num_stream
            ag_stream = ag_stream_pool[stream_pos]

            M_dst_start = rank * M_per_rank + chunk_idx * M_PER_CHUNK
            M_src_start = chunk_idx * M_PER_CHUNK

            n_elements = M_PER_CHUNK * N
            src_view = local_flat[M_src_start * N: M_src_start * N + n_elements]
            remote_flat = remote_tensor_buffers[remote_rank].reshape(-1)
            dst_view = remote_flat[M_dst_start * N: M_dst_start * N + n_elements]

            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            cu_copy_kernel[grid](src_view, dst_view, n_elements, BLOCK_SIZE=BLOCK_SIZE)

            if isinstance(ag_stream, CUMaskedStreamWrapper):
                stream_ptr = ag_stream._hip_stream
            else:
                stream_ptr = ag_stream.cuda_stream if hasattr(ag_stream, 'cuda_stream') else 0

            cp_res = hip.hipMemcpyAsync(
                barrier_buffers[remote_rank].data_ptr() + chunk_pos * barrier_elem_size,
                one.data_ptr(),
                barrier_elem_size,
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                stream_ptr,
            )
            HIP_CHECK(cp_res)


@dataclass
class ModeResult:
    name: str
    median_ms: float
    mean_ms: float
    std_ms: float
    num_sms: int
    correct: bool
    comm_cus: Optional[int] = None
    compute_cus: Optional[int] = None


class AGGemmRunner:
    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        M: int,
        N: int,
        K: int,
        chunk_size: int,
        dtype: torch.dtype,
        num_sms: int,
        num_ag_streams: int,
        comm_engine: str = "copy-engine",
    ):
        self.tp_group = tp_group
        self.rank = tp_group.rank()
        self.num_ranks = tp_group.size()
        self.M = M
        self.N = N
        self.K = K
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.num_sms = num_sms
        self.num_ag_streams = num_ag_streams
        self.comm_engine = comm_engine

        self.ctx = t09.create_ag_gemm_intra_node_context(
            self.M,
            self.N,
            self.K,
            self.dtype,
            self.dtype,
            self.rank,
            self.num_ranks,
            self.tp_group,
            M_PER_CHUNK=self.chunk_size,
        )

    def run_one(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        ag_streams: List,
        compute_stream,
        gemm_iters: int = 1,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()

        for stream in ag_streams:
            stream.wait_stream(current_stream)
        compute_stream.wait_stream(current_stream)

        N_per_rank = weight.shape[0]
        output = torch.empty((self.M, N_per_rank), dtype=self.dtype, device=input_tensor.device)

        ag_producer = t09.producer_ag_push_mode if self.comm_engine == "copy-engine" else producer_ag_cu_kernel
        ag_producer(
            self.ctx.rank,
            self.ctx.num_ranks,
            input_tensor,
            self.ctx.workspace_tensors,
            self.ctx.one,
            self.ctx.M_PER_CHUNK,
            ag_streams,
            self.ctx.barrier_tensors,
        )

        full_input = self.ctx.workspace_tensors[self.ctx.rank][:self.M]
        with torch.cuda.stream(compute_stream):
            grid = lambda META: (
                min(
                    self.num_sms,
                    triton.cdiv(self.M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_per_rank, META["BLOCK_SIZE_N"]),
                ),
            )
            for _ in range(gemm_iters):
                t09.consumer_gemm_persistent_kernel[grid](
                    full_input,
                    input_tensor,
                    weight,
                    output,
                    self.M,
                    N_per_rank,
                    self.K,
                    full_input.stride(0),
                    full_input.stride(1),
                    weight.stride(1),
                    weight.stride(0),
                    output.stride(0),
                    output.stride(1),
                    self.ctx.rank,
                    self.ctx.num_ranks,
                    self.ctx.barrier_tensors[self.ctx.rank],
                    M_PER_CHUNK=self.ctx.M_PER_CHUNK,
                    NUM_SMS=self.num_sms,
                    NUM_XCDS=4,
                )

        return output


def parse_args():
    p = argparse.ArgumentParser(description="CU-masked AG+GEMM experiment using tutorial 09 kernel path")
    p.add_argument("--M", type=int, default=8192)
    p.add_argument("--N", type=int, default=11008)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--num-ag-streams", type=int, default=4)
    p.add_argument("--num-sms", type=int, default=272, help="Constrained NUM_SMS for GEMM kernel")
    p.add_argument("--gemm-iters", type=int, default=1,
                   help="Repeat GEMM kernel N times per timed iteration for longer compute phase")
    p.add_argument("--comm-engine", choices=["copy-engine", "cu-kernel"], default="copy-engine",
                   help="copy-engine: DMA via hipMemcpy (no CU usage); cu-kernel: Triton kernel copy (uses CUs)")
    p.add_argument("--modes", default="unmasked,cu-masked", help="Comma-separated: unmasked,cu-masked")
    p.add_argument("--strategy", choices=["interleaved", "sequential", "block", "ratio"], default="interleaved")
    p.add_argument("--comm-ratio", type=float, default=0.3)
    p.add_argument("--comm-cus-count", type=int, default=None)
    p.add_argument("--validate", action="store_true")
    return p.parse_args()


def benchmark_mode(
    runner: AGGemmRunner,
    mode_name: str,
    build_streams: Callable[[], tuple[List, object, Callable[[], None], Optional[dict]]],
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    warmup: int,
    repeats: int,
    ref_out: Optional[torch.Tensor],
    gemm_iters: int = 1,
) -> ModeResult:
    ag_streams, compute_stream, cleanup, metadata = build_streams()
    try:
        for _ in range(warmup):
            _ = runner.run_one(input_tensor, weight, ag_streams, compute_stream, gemm_iters)
            torch.cuda.synchronize()
            torch.distributed.barrier()

        latencies_ms = []
        last_out = None
        for _ in range(repeats):
            torch.distributed.barrier()
            start = datetime.datetime.now()
            last_out = runner.run_one(input_tensor, weight, ag_streams, compute_stream, gemm_iters)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            end = datetime.datetime.now()
            latencies_ms.append((end - start).total_seconds() * 1000.0)

        correct = True
        if ref_out is not None:
            correct = torch.allclose(last_out, ref_out, atol=1e-2, rtol=1e-2)

        return ModeResult(
            name=mode_name,
            median_ms=statistics.median(latencies_ms),
            mean_ms=statistics.mean(latencies_ms),
            std_ms=statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
            num_sms=runner.num_sms,
            correct=correct,
            comm_cus=None if metadata is None else metadata.get("comm_cus"),
            compute_cus=None if metadata is None else metadata.get("compute_cus"),
        )
    finally:
        cleanup()


def main():
    global RANK, WORLD_SIZE
    args = parse_args()

    RANK, _, WORLD_SIZE, TP_GROUP = t09.init()
    pyrocshmem.init_rocshmem_by_uniqueid(TP_GROUP)

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = t09.triton.runtime.driver.active.get_active_torch_device()
    local_M = args.M // WORLD_SIZE
    local_N = args.N // WORLD_SIZE

    scale = TP_GROUP.rank() + 1
    input_tensor = torch.randn((local_M, args.K), dtype=dtype, device=device) * (0.01 * scale)
    weight = torch.randn((local_N, args.K), dtype=dtype, device=device) * (0.01 * scale)

    device_info = get_device_info()
    constrained_sms = max(1, min(args.num_sms, device_info.total_cus))
    runner = AGGemmRunner(
        tp_group=TP_GROUP,
        M=args.M,
        N=args.N,
        K=args.K,
        chunk_size=args.chunk_size,
        dtype=dtype,
        num_sms=constrained_sms,
        num_ag_streams=args.num_ag_streams,
        comm_engine=args.comm_engine,
    )

    ref_out = None
    if args.validate:
        ref_out = t09.torch_ag_gemm(input_tensor, weight, False, None, TP_GROUP)
        torch.cuda.synchronize()
        torch.distributed.barrier()

    mode_list = [m.strip() for m in args.modes.split(",") if m.strip()]
    results: List[ModeResult] = []

    def build_unmasked_streams():
        ag_streams = [torch.cuda.Stream() for _ in range(args.num_ag_streams)]
        compute_stream = torch.cuda.Stream()
        return ag_streams, compute_stream, (lambda: None), None

    def build_cumasked_streams():
        comm_cus, compute_cus = partition_cus(
            total_cus=device_info.total_cus,
            strategy=args.strategy,
            comm_ratio=args.comm_ratio,
            comm_cus_count=args.comm_cus_count,
        )
        comm_mask = create_cu_mask(comm_cus, device_info.total_cus)
        compute_mask = create_cu_mask(compute_cus, device_info.total_cus)

        ag_streams = [
            CUMaskedStreamWrapper(create_stream_with_cu_mask(comm_mask))
            for _ in range(args.num_ag_streams)
        ]
        compute_stream = CUMaskedStreamWrapper(create_stream_with_cu_mask(compute_mask))

        def _cleanup():
            for s in ag_streams:
                s.destroy()
            compute_stream.destroy()

        metadata = {"comm_cus": len(comm_cus), "compute_cus": len(compute_cus)}
        return ag_streams, compute_stream, _cleanup, metadata

    if "unmasked" in mode_list:
        results.append(
            benchmark_mode(
                runner,
                "unmasked",
                build_unmasked_streams,
                input_tensor,
                weight,
                args.warmup,
                args.repeats,
                ref_out,
                gemm_iters=args.gemm_iters,
            ))

    if "cu-masked" in mode_list:
        if RANK == 0:
            comm_cus, compute_cus = partition_cus(
                total_cus=device_info.total_cus,
                strategy=args.strategy,
                comm_ratio=args.comm_ratio,
                comm_cus_count=args.comm_cus_count,
            )
            print_cu_distribution(
                comm_cus,
                compute_cus,
                device_info.total_cus,
                args.strategy,
                verbose=False,
            )
        results.append(
            benchmark_mode(
                runner,
                "cu-masked",
                build_cumasked_streams,
                input_tensor,
                weight,
                args.warmup,
                args.repeats,
                ref_out,
                gemm_iters=args.gemm_iters,
            ))

    if RANK == 0:
        print("\n=== CU-Masked AG+GEMM Results ===")
        print(
            f"Shape: M={args.M}, N={args.N}, K={args.K}, chunk={args.chunk_size}, "
            f"num_sms={constrained_sms}, gemm_iters={args.gemm_iters}, comm_engine={args.comm_engine}, "
            f"repeats={args.repeats}, modes={','.join(mode_list)}"
        )
        print(f"{'Mode':<12} {'Median(ms)':>12} {'Mean(ms)':>10} {'Std(ms)':>10} {'CommCUs':>8} {'CompCUs':>8} {'Correct':>8}")
        baseline = results[0].median_ms if results else 1.0
        for r in results:
            comm_cus = "-" if r.comm_cus is None else str(r.comm_cus)
            comp_cus = "-" if r.compute_cus is None else str(r.compute_cus)
            print(f"{r.name:<12} {r.median_ms:>12.3f} {r.mean_ms:>10.3f} {r.std_ms:>10.3f} {comm_cus:>8} {comp_cus:>8} {str(r.correct):>8}")
        print("\nSpeedups (vs first mode):")
        for r in results:
            print(f"  {r.name:<12}: {baseline / r.median_ms:.3f}x")

    # Ensure rocSHMEM-backed context tensors are released before finalize.
    del runner
    torch.cuda.synchronize()
    torch.distributed.barrier()
    pyrocshmem.rocshmem_finalize()
    t09.destroy()


if __name__ == "__main__":
    main()
