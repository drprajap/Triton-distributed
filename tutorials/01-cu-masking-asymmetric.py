#!/usr/bin/env python3
"""
CU Masking with Asymmetric Workloads
=====================================

Shows where CU masking actually helps: a small "comm" kernel running alongside
a large "compute" kernel. CU masking dedicates a few CUs to comm so it doesn't
starve or block the compute path.

Three measurements:
  1. Sequential          -- comm then compute on default stream
  2. Two regular streams -- overlap but comm/compute fight for CUs
  3. CU-masked streams   -- comm gets dedicated CUs, compute gets the rest

Usage:
    python tutorials/01-cu-masking-asymmetric.py

Profile:
    rocprofv3 --kernel-trace --hip-trace -f pftrace -d trace_asym \
        -- python tutorials/01-cu-masking-asymmetric.py
"""

import ctypes
import time

import torch
import triton
import triton.language as tl
from hip import hip


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@triton.jit
def comm_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Bandwidth-bound kernel simulating communication (load/store, minimal ALU).

    Duration limited by memory bandwidth, NOT CU count -- ideal for CU masking
    because it runs in ~same time whether it gets 30 CUs or 304 CUs.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x + 1.0, mask=mask)


@triton.jit
def compute_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """ALU-bound kernel simulating compute (heavy arithmetic, scales with CU count)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    for _ in range(200):
        x = 0.9 * x * x + 0.05 * x + 0.01
    tl.store(out_ptr + offs, x, mask=mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cu_masked_stream(cu_indices, total_cus):
    mask_size = (total_cus + 31) // 32
    cu_mask = [0] * mask_size
    for idx in cu_indices:
        cu_mask[idx // 32] |= 1 << (idx % 32)
    arr = (ctypes.c_uint32 * mask_size)(*cu_mask)
    err, stream = hip.hipExtStreamCreateWithCUMask(mask_size, arr)
    if err != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"hipExtStreamCreateWithCUMask failed: {err}")
    return stream


class CUStream:
    def __init__(self, hip_stream):
        self.cuda_stream = int(hip_stream)
        self._hip = hip_stream
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.device_index = torch.cuda.current_device()
        self.device_type = 1
        self.stream_id = self.cuda_stream

    def synchronize(self):
        hip.hipStreamSynchronize(self._hip)

    def destroy(self):
        hip.hipStreamDestroy(self._hip)


def bench(label, fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:40s} {ms:8.2f} ms")
    return ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    props = torch.cuda.get_device_properties(0)
    total_cus = props.multi_processor_count
    print(f"Device: {props.name}  |  CUs: {total_cus}\n")

    BLOCK_SIZE = 1024

    # Comm: bandwidth-bound, needs large data to take meaningful time on MI300X
    # Compute: ALU-bound, scales with CU count
    N_COMM = 262144 * BLOCK_SIZE     # ~268M elements (~1 GB), bandwidth-bound
    N_COMPUTE = 65536 * BLOCK_SIZE   # ~67M elements, ALU-bound
    grid_comm = (N_COMM // BLOCK_SIZE,)
    grid_compute = (N_COMPUTE // BLOCK_SIZE,)

    print(f"Comm kernel:    {N_COMM // BLOCK_SIZE:>6} blocks  ({N_COMM:,} elements)")
    print(f"Compute kernel: {N_COMPUTE // BLOCK_SIZE:>6} blocks  ({N_COMPUTE:,} elements)")

    # Data
    in_comm = torch.randn(N_COMM, device="cuda")
    out_comm = torch.empty_like(in_comm)
    in_comp = torch.randn(N_COMPUTE, device="cuda")
    out_comp = torch.empty_like(in_comp)

    # Regular streams
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    # CU-masked streams: 25% for comm, 75% for compute
    n_comm_cus = max(1, round(total_cus * 0.25))
    n_comp_cus = total_cus - n_comm_cus
    cus_comm = list(range(0, n_comm_cus))
    cus_comp = list(range(n_comm_cus, total_cus))
    ms_comm = CUStream(make_cu_masked_stream(cus_comm, total_cus))
    ms_comp = CUStream(make_cu_masked_stream(cus_comp, total_cus))

    print(f"\nCU-masked split: comm={n_comm_cus} CUs [{cus_comm[0]}..{cus_comm[-1]}], "
          f"compute={n_comp_cus} CUs [{cus_comp[0]}..{cus_comp[-1]}]")
    print()

    # -- Benchmarks --

    def run_sequential():
        comm_kernel[grid_comm](in_comm, out_comm, N_COMM, BLOCK_SIZE=BLOCK_SIZE)
        compute_kernel[grid_compute](in_comp, out_comp, N_COMPUTE, BLOCK_SIZE=BLOCK_SIZE)

    def run_two_streams():
        with torch.cuda.stream(s1):
            comm_kernel[grid_comm](in_comm, out_comm, N_COMM, BLOCK_SIZE=BLOCK_SIZE)
        with torch.cuda.stream(s2):
            compute_kernel[grid_compute](in_comp, out_comp, N_COMPUTE, BLOCK_SIZE=BLOCK_SIZE)

    def run_cu_masked():
        with torch.cuda.stream(ms_comm):
            comm_kernel[grid_comm](in_comm, out_comm, N_COMM, BLOCK_SIZE=BLOCK_SIZE)
        with torch.cuda.stream(ms_comp):
            compute_kernel[grid_compute](in_comp, out_comp, N_COMPUTE, BLOCK_SIZE=BLOCK_SIZE)

    # Also measure individual kernels
    def run_comm_only():
        comm_kernel[grid_comm](in_comm, out_comm, N_COMM, BLOCK_SIZE=BLOCK_SIZE)

    def run_compute_only():
        compute_kernel[grid_compute](in_comp, out_comp, N_COMPUTE, BLOCK_SIZE=BLOCK_SIZE)

    t_comm = bench("Comm kernel alone (all CUs)", run_comm_only)
    t_comp = bench("Compute kernel alone (all CUs)", run_compute_only)
    print()
    t_seq = bench("1. Sequential (same stream)", run_sequential)
    t_str = bench("2. Two regular streams", run_two_streams)
    t_cu  = bench("3. CU-masked (25% comm / 75% compute)", run_cu_masked)

    print(f"\n  Analysis:")
    print(f"    Comm alone:      {t_comm:.2f} ms")
    print(f"    Compute alone:   {t_comp:.2f} ms")
    print(f"    Sequential:      {t_seq:.2f} ms  (comm + compute)")
    print(f"    Two streams:     {t_str:.2f} ms  (speedup {t_seq/t_str:.2f}x vs sequential)")
    print(f"    CU-masked:       {t_cu:.2f} ms  (speedup {t_seq/t_cu:.2f}x vs sequential)")
    print(f"    CU-mask vs streams: {t_str/t_cu:.2f}x")

    ms_comm.destroy()
    ms_comp.destroy()


if __name__ == "__main__":
    main()
