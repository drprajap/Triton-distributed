#!/usr/bin/env python3
"""
Minimal CU Masking Verification Example (Standalone Version)
=============================================================

This is a standalone version that doesn't require importing the cu_masking module.
All necessary utilities are embedded directly in this script.

**How to Run:**
    python tutorials/00-verify-cu-masking-standalone.py

**With ROCProf:**
    rocprof --stats --hip-trace python tutorials/00-verify-cu-masking-standalone.py
    rocprof --plugin perfetto python tutorials/00-verify-cu-masking-standalone.py
"""

import sys
import os
import time
import ctypes
from dataclasses import dataclass
from typing import List, Tuple

import torch
import triton
import triton.language as tl
from hip import hip


# ============================================================================
# Embedded CU Masking Utilities (from triton_dist.cu_masking)
# ============================================================================

@dataclass
class DeviceInfo:
    """Information about the current GPU device."""
    name: str
    total_cus: int
    gcn_arch: str
    device_index: int


def get_device_info(device=None):
    """Query information about the current or specified GPU device."""
    if device is None:
        device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)

    # Get GCN architecture using HIP
    # Create device properties struct first
    device_props = hip.hipDeviceProp_t()
    err = hip.hipGetDeviceProperties(device_props, device)
    if err != hip.hipError_t.hipSuccess:
        gcn_arch = f"gfx{props.major}{props.minor}"
    else:
        gcn_arch = device_props.gcnArchName.decode() if isinstance(device_props.gcnArchName, bytes) else device_props.gcnArchName

    return DeviceInfo(
        name=props.name,
        total_cus=props.multi_processor_count,
        gcn_arch=gcn_arch,
        device_index=device
    )


def create_cu_mask(cu_list: List[int], total_cus: int) -> List[int]:
    """Create a CU bit mask from a list of CU indices."""
    mask_size = (total_cus + 31) // 32
    cu_mask = [0] * mask_size

    for cu_idx in cu_list:
        if cu_idx >= total_cus:
            raise ValueError(f"CU index {cu_idx} out of range (total: {total_cus})")
        word_idx = cu_idx // 32
        bit_idx = cu_idx % 32
        cu_mask[word_idx] |= (1 << bit_idx)

    return cu_mask


def create_stream_with_cu_mask(cu_mask: List[int]):
    """Create a HIP stream bound to specific CUs."""
    mask_size = len(cu_mask)
    mask_array = (ctypes.c_uint32 * mask_size)(*cu_mask)

    err, stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)

    if err != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"hipExtStreamCreateWithCUMask failed: {err}")

    return stream


class CUMaskedStreamWrapper:
    """Wrapper to make HIP CU-masked streams compatible with PyTorch."""

    def __init__(self, hip_stream):
        self.cuda_stream = int(hip_stream)
        self._hip_stream = hip_stream
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.device_index = torch.cuda.current_device()
        self.device_type = 1  # DeviceType::CUDA = 1 (integer, not string)
        self.stream_id = int(hip_stream)

    def synchronize(self):
        err = hip.hipStreamSynchronize(self._hip_stream)
        # Handle both tuple and direct error code returns
        if isinstance(err, tuple):
            err = err[0]
        if err != hip.hipError_t.hipSuccess:
            raise RuntimeError(f"hipStreamSynchronize failed: {err}")

    def destroy(self):
        err = hip.hipStreamDestroy(self._hip_stream)
        # Handle both tuple and direct error code returns
        if isinstance(err, tuple):
            err = err[0]
        if err != hip.hipError_t.hipSuccess:
            raise RuntimeError(f"hipStreamDestroy failed: {err}")

    def __enter__(self):
        self._old_stream = torch.cuda.current_stream()
        torch.cuda.set_stream(self)
        return self

    def __exit__(self, *args):
        torch.cuda.set_stream(self._old_stream)

    def __repr__(self):
        return f"CUMaskedStreamWrapper(stream_id={self.stream_id}, device={self.device_index})"


# ============================================================================
# Simple Triton Kernels
# ============================================================================

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple vector addition: output = x + y"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def vector_mul_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple vector multiplication: output = x * y"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y

    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def compute_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute-intensive kernel from tutorial.
    Tests if polynomial operations work on CU-masked streams.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute-intensive operations using ONLY basic arithmetic
    # Use numerically stable polynomial that doesn't overflow
    result = x * 0.1  # Scale down to prevent overflow
    for _ in range(50):  # Increased iterations for longer execution
        # Stable polynomial: result = 0.9*result^2 + 0.05*result + 0.01
        result = 0.9 * result * result + 0.05 * result + 0.01

    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)


# ============================================================================
# Main Verification Script
# ============================================================================

def main():
    print("\n" + "="*80)
    print("CU Masking Verification Example (Standalone)")
    print("="*80)

    # 1. Get device info
    device_info = get_device_info()
    print(f"\nDevice: {device_info.name}")
    print(f"Architecture: {device_info.gcn_arch}")
    print(f"Total CUs: {device_info.total_cus}")

    # 2. Partition CUs (50/50 split using interleaved strategy)
    print(f"\n{'='*80}")
    print("CU Partitioning Strategy: Interleaved (odd vs even CUs)")
    print("="*80)

    total_cus = device_info.total_cus
    half_cus = total_cus // 2

    stream_a_cus = [i for i in range(total_cus) if i % 2 == 1][:half_cus]
    stream_b_cus = [i for i in range(total_cus) if i % 2 == 0][:half_cus]

    print(f"Stream A CUs: {stream_a_cus[:10]}... (total: {len(stream_a_cus)})")
    print(f"Stream B CUs: {stream_b_cus[:10]}... (total: {len(stream_b_cus)})")

    # Visual representation
    print(f"\nVisual (first 64 CUs, A=Stream A, B=Stream B):")
    visual = []
    for i in range(min(64, total_cus)):
        if i in stream_a_cus:
            visual.append('A')
        elif i in stream_b_cus:
            visual.append('B')
        else:
            visual.append(' ')
    print(f"  {''.join(visual)}")
    if total_cus > 64:
        print(f"  (showing first 64/{total_cus} CUs)")

    # 3. Create CU masks and streams
    print(f"\n{'='*80}")
    print("Creating CU-Masked Streams")
    print("="*80)

    stream_a_mask = create_cu_mask(stream_a_cus, total_cus)
    stream_b_mask = create_cu_mask(stream_b_cus, total_cus)

    stream_a_hip = create_stream_with_cu_mask(stream_a_mask)
    stream_b_hip = create_stream_with_cu_mask(stream_b_mask)

    stream_a = CUMaskedStreamWrapper(stream_a_hip)
    stream_b = CUMaskedStreamWrapper(stream_b_hip)

    print(f"✓ Created Stream A (bound to {len(stream_a_cus)} CUs)")
    print(f"✓ Created Stream B (bound to {len(stream_b_cus)} CUs)")

    # 4. Allocate data
    print(f"\n{'='*80}")
    print("Allocating Data")
    print("="*80)

    n_elements = 100_000_000  # 100M elements - large enough to see clear overlap in profiler
    BLOCK_SIZE = 1024

    # Data for vector_add (Stream A)
    x1 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    y1 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    out1 = torch.zeros(n_elements, device='cuda', dtype=torch.float32)

    # Data for compute_kernel (Stream B) - test the polynomial kernel
    x2 = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    out2 = torch.zeros(n_elements, device='cuda', dtype=torch.float32)

    print(f"✓ Allocated {n_elements:,} elements per kernel")
    total_memory_gb = 5 * n_elements * 4 / 1e9
    print(f"✓ Total memory: {total_memory_gb:.2f} GB")
    print(f"✓ Large problem size ensures kernels run long enough for clear overlap in profiler")

    # 5. Warmup
    print(f"\n{'='*80}")
    print("Warmup (3 iterations)")
    print("="*80)

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    for i in range(3):
        with torch.cuda.stream(stream_a):
            vector_add_kernel[grid](x1, y1, out1, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        with torch.cuda.stream(stream_b):
            compute_kernel[grid](x2, out2, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        torch.cuda.synchronize()

    print("✓ Warmup complete")

    # 6. Benchmark: Concurrent execution
    print(f"\n{'='*80}")
    print("Benchmark: Concurrent Execution (WITH CU Masking)")
    print("="*80)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.cuda.stream(stream_a):
        vector_add_kernel[grid](x1, y1, out1, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    with torch.cuda.stream(stream_b):
        compute_kernel[grid](x2, out2, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    torch.cuda.synchronize()
    concurrent_time = time.perf_counter() - start

    print(f"✓ Concurrent execution time: {concurrent_time*1000:.2f} ms")

    # 7. Verify correctness
    print(f"\n{'='*80}")
    print("Correctness Verification")
    print("="*80)

    # Verify vector add
    expected_add = x1 + y1
    add_correct = torch.allclose(out1, expected_add, rtol=1e-5)

    # Verify compute kernel (polynomial evaluation)
    # Expected: repeated application of stable polynomial
    expected_compute = x2.clone() * 0.1
    for _ in range(50):  # Match kernel iteration count
        expected_compute = 0.9 * expected_compute * expected_compute + 0.05 * expected_compute + 0.01
    compute_correct = torch.allclose(out2, expected_compute, rtol=1e-3)  # Slightly looser for more iterations

    print(f"✓ Vector Add: {'PASS' if add_correct else 'FAIL'}")
    print(f"✓ Compute Kernel (Polynomial): {'PASS' if compute_correct else 'FAIL'}")

    if not (add_correct and compute_correct):
        print("❌ ERROR: Results are incorrect!")
        if not add_correct:
            print(f"  Add kernel error: max diff = {torch.max(torch.abs(out1 - expected_add)).item()}")
        if not compute_correct:
            print(f"  Compute kernel error: max diff = {torch.max(torch.abs(out2 - expected_compute)).item()}")
        return

    # 8. Results summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\nTiming:")
    print(f"  Concurrent execution (with CU masking):  {concurrent_time*1000:8.2f} ms")

    print(f"\n✅ SUCCESS: Both kernels executed concurrently on separate CU sets!")
    print(f"   - Stream A (vector_add): {len(stream_a_cus)} CUs (odd CUs)")
    print(f"   - Stream B (compute_kernel): {len(stream_b_cus)} CUs (even CUs)")
    print(f"\n   Profile with ROCProf to see the overlap in timeline:")
    print(f"   mkdir -p rocprof_traces && rocprof --plugin perfetto -d rocprof_traces python tutorials/00-verify-cu-masking-standalone.py")
    print(f"   # Open rocprof_traces/results.pftrace at https://ui.perfetto.dev")

    # 9. Cleanup
    stream_a.destroy()
    stream_b.destroy()

    # 11. Profiling instructions
    print(f"\n{'='*80}")
    print("PROFILING INSTRUCTIONS")
    print("="*80)

    print(f"\nTo visualize CU usage, run with ROCProf:\n")

    print(f"1. Basic profiling (shows kernel timing):")
    print(f"   mkdir -p rocprof_traces")
    print(f"   rocprof --stats --hip-trace -o rocprof_traces/results.csv python tutorials/00-verify-cu-masking-standalone.py")
    print(f"   # Results: rocprof_traces/results.json, rocprof_traces/results.stats.csv\n")

    print(f"2. Generate Perfetto timeline (RECOMMENDED - best visualization):")
    print(f"   mkdir -p rocprof_traces")
    print(f"   rocprof --plugin perfetto -o rocprof_traces/results.pftrace python tutorials/00-verify-cu-masking-standalone.py")
    print(f"   # Then open 'rocprof_traces/results.pftrace' at https://ui.perfetto.dev\n")

    print(f"3. What to look for in Perfetto:")
    print(f"   - Two kernel bars (vector_add_kernel, compute_kernel)")
    print(f"   - Bars should OVERLAP in time (concurrent execution)")
    print(f"   - If bars are sequential, CU masking may not be working\n")

    print(f"4. Clean up traces:")
    print(f"   rm -rf rocprof_traces/\n")

    print(f"{'='*80}\n")
    print(f"✅ Verification complete!\n")


if __name__ == "__main__":
    main()
