#!/usr/bin/env python3
"""
Minimal CU Masking Example
===========================

Shows that CU-masked streams enable true concurrent execution on AMD GPUs,
where regular streams may serialize due to CU contention.

Three measurements:
  1. Same stream      -- sequential baseline (one kernel after another)
  2. Two streams      -- no CU masking (may still serialize on AMD)
  3. CU-masked streams -- explicit CU isolation, guaranteed overlap

Usage:
    python tutorials/00-cu-masking-benchmark.py

Profile:
    rocprofv3 --plugin perfetto python tutorials/00-cu-masking-benchmark.py
"""

import ctypes
import time

import torch
import triton
import triton.language as tl
from hip import hip


@triton.jit
def work_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    for _ in range(200):
        x = 0.9 * x * x + 0.05 * x + 0.01
    tl.store(out_ptr + offs, x, mask=mask)


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
    """Thin wrapper so torch.cuda.stream() accepts our HIP stream."""
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
    print(f"  {label:30s} {ms:8.2f} ms")
    return ms


def main():
    props = torch.cuda.get_device_properties(0)
    total_cus = props.multi_processor_count
    print(f"Device: {props.name}  |  CUs: {total_cus}\n")

    BLOCK_SIZE = 1024
    N = 65536 * BLOCK_SIZE
    grid = (N // BLOCK_SIZE,)

    in_a = torch.randn(N, device="cuda")
    out_a = torch.empty_like(in_a)
    in_b = torch.randn(N, device="cuda")
    out_b = torch.empty_like(in_b)

    # Regular (non-masked) streams
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    # CU-masked streams (50/50 split)
    half = total_cus // 2
    cus_a = list(range(0, half))
    cus_b = list(range(half, total_cus))
    ms_a = CUStream(make_cu_masked_stream(cus_a, total_cus))
    ms_b = CUStream(make_cu_masked_stream(cus_b, total_cus))
    print(f"CU split: A=[0..{half-1}] ({len(cus_a)}), B=[{half}..{total_cus-1}] ({len(cus_b)})\n")

    def run_single():
        work_kernel[grid](in_a, out_a, N, BLOCK_SIZE=BLOCK_SIZE)

    def run_sequential():
        work_kernel[grid](in_a, out_a, N, BLOCK_SIZE=BLOCK_SIZE)
        work_kernel[grid](in_b, out_b, N, BLOCK_SIZE=BLOCK_SIZE)

    def run_two_streams():
        with torch.cuda.stream(s1):
            work_kernel[grid](in_a, out_a, N, BLOCK_SIZE=BLOCK_SIZE)
        with torch.cuda.stream(s2):
            work_kernel[grid](in_b, out_b, N, BLOCK_SIZE=BLOCK_SIZE)

    def run_cu_masked():
        with torch.cuda.stream(ms_a):
            work_kernel[grid](in_a, out_a, N, BLOCK_SIZE=BLOCK_SIZE)
        with torch.cuda.stream(ms_b):
            work_kernel[grid](in_b, out_b, N, BLOCK_SIZE=BLOCK_SIZE)

    t_one = bench("1. Single kernel (304 CUs)", run_single)
    t_seq = bench("2. Two kernels, same stream", run_sequential)
    t_str = bench("3. Two kernels, two streams", run_two_streams)
    t_cu  = bench("4. Two kernels, CU-masked", run_cu_masked)

    print(f"\n  Analysis:")
    print(f"    Single kernel time (T):         {t_one:.2f} ms")
    print(f"    Sequential = 2T:                {t_seq:.2f} ms  (expected ~{2*t_one:.2f})")
    print(f"    Two streams (no mask):          {t_str:.2f} ms  ({'overlapping' if t_str < 1.5*t_one else 'serializing'})")
    print(f"    CU-masked (each on {half} CUs): {t_cu:.2f} ms  ({'overlapping' if t_cu < 3*t_one else 'serializing'})")
    print(f"    If CU-masked serialized, expect: ~{4*t_one:.2f} ms (2 x 2T)")

    ms_a.destroy()
    ms_b.destroy()


if __name__ == "__main__":
    main()
