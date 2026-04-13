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
"""
CU Masking Utilities for AMD GPUs
==================================

This module provides utilities for explicit Compute Unit (CU) partitioning
on AMD GPUs using hipExtStreamCreateWithCUMask. This enables fine-grained
control over which CUs execute which kernels, allowing for efficient
overlap of communication and computation workloads.

Key Features:
    - CU mask creation and management
    - Multiple partitioning strategies (sequential, interleaved, block, ratio)
    - PyTorch-compatible stream wrapper for CU-masked streams
    - Device information querying

Example Usage:
    >>> from triton_dist.cu_masking import *
    >>>
    >>> # Get device info
    >>> info = get_device_info()
    >>> print(f"Device: {info.name}, CUs: {info.total_cus}")
    >>>
    >>> # Partition CUs (30% comm, 70% compute)
    >>> comm_cus, compute_cus = partition_cus(
    ...     total_cus=info.total_cus,
    ...     strategy='interleaved',
    ...     comm_ratio=0.3
    ... )
    >>>
    >>> # Create CU-masked streams
    >>> comm_stream = create_stream_with_cu_mask(create_cu_mask(comm_cus, info.total_cus))
    >>> compute_stream = create_stream_with_cu_mask(create_cu_mask(compute_cus, info.total_cus))
    >>>
    >>> # Use with PyTorch
    >>> comm_wrapper = CUMaskedStreamWrapper(comm_stream)
    >>> with torch.cuda.stream(comm_wrapper):
    ...     # Launch communication kernels
    ...     pass
    >>>
    >>> # Cleanup
    >>> comm_wrapper.destroy()
    >>> compute_wrapper.destroy()

References:
    - HIP documentation: https://rocmdocs.amd.com/en/latest/
    - hipExtStreamCreateWithCUMask API
"""

import ctypes
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
from hip import hip


@dataclass
class DeviceInfo:
    """Information about the current GPU device."""
    name: str
    total_cus: int
    gcn_arch: str
    device_index: int


def get_device_info(device: Optional[int] = None) -> DeviceInfo:
    """
    Query information about the current or specified GPU device.

    Args:
        device: Optional device index. If None, uses current device.

    Returns:
        DeviceInfo object containing device properties

    Example:
        >>> info = get_device_info()
        >>> print(f"Device: {info.name}")
        >>> print(f"Total CUs: {info.total_cus}")
        >>> print(f"Architecture: {info.gcn_arch}")
    """
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
    """
    Create a CU bit mask from a list of CU indices.

    The mask is represented as a list of uint32 values, where each bit
    corresponds to a CU. Bit i in word j represents CU (j*32 + i).

    Args:
        cu_list: List of CU indices to enable (e.g., [0, 1, 2, 3])
        total_cus: Total number of CUs on the device

    Returns:
        List of uint32 representing the bit mask

    Raises:
        ValueError: If any CU index exceeds total_cus

    Example:
        >>> mask = create_cu_mask([0, 1, 2, 3], total_cus=110)
        >>> # mask[0] will have bits 0-3 set (value: 0b1111 = 15)
    """
    # Calculate mask size (each uint32 holds 32 bits)
    mask_size = (total_cus + 31) // 32
    cu_mask = [0] * mask_size

    # Set bits for specified CUs
    for cu_idx in cu_list:
        if cu_idx >= total_cus:
            raise ValueError(f"CU index {cu_idx} out of range (total: {total_cus})")
        word_idx = cu_idx // 32
        bit_idx = cu_idx % 32
        cu_mask[word_idx] |= (1 << bit_idx)

    return cu_mask


def decode_cu_mask(cu_mask: List[int], total_cus: int) -> List[int]:
    """
    Decode a CU bit mask back to a list of enabled CU indices.

    Args:
        cu_mask: List of uint32 representing the CU bit mask
        total_cus: Total number of CUs on the device

    Returns:
        List of enabled CU indices

    Example:
        >>> mask = create_cu_mask([0, 2, 4], total_cus=110)
        >>> enabled = decode_cu_mask(mask, total_cus=110)
        >>> print(enabled)  # [0, 2, 4]
    """
    enabled_cus = []
    for word_idx, word in enumerate(cu_mask):
        for bit_idx in range(32):
            cu_idx = word_idx * 32 + bit_idx
            if cu_idx >= total_cus:
                break
            if word & (1 << bit_idx):
                enabled_cus.append(cu_idx)
    return enabled_cus


def partition_cus(
    total_cus: int,
    strategy: str = 'interleaved',
    comm_ratio: float = 0.3,
    comm_cus_count: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """
    Partition CUs between communication and computation using various strategies.

    Args:
        total_cus: Total number of CUs on the device
        strategy: Partitioning strategy. Options:
            - 'sequential': CUs 0 to N-1 for comm, N to end for compute
            - 'interleaved': Odd CUs for comm, even for compute (better bandwidth)
            - 'block': First half for comm, second half for compute (NUMA-aware)
            - 'ratio': Dynamic allocation based on comm_ratio
        comm_ratio: Fraction of CUs for communication (0.0-1.0, default: 0.3)
        comm_cus_count: Fixed number of CUs for communication (overrides comm_ratio)

    Returns:
        Tuple of (comm_cu_list, compute_cu_list)

    Raises:
        ValueError: If strategy is invalid or comm_ratio out of range

    Example:
        >>> # Interleaved strategy with 30% for communication
        >>> comm_cus, compute_cus = partition_cus(304, 'interleaved', 0.3)
        >>> print(f"Comm CUs: {len(comm_cus)}, Compute CUs: {len(compute_cus)}")

        >>> # Fixed count strategy
        >>> comm_cus, compute_cus = partition_cus(304, 'sequential', comm_cus_count=32)
    """
    if not 0.0 < comm_ratio < 1.0:
        raise ValueError(f"comm_ratio must be between 0 and 1, got {comm_ratio}")

    if strategy not in ['sequential', 'interleaved', 'block', 'ratio']:
        raise ValueError(f"Invalid strategy '{strategy}'. Must be one of: sequential, interleaved, block, ratio")

    # Determine number of communication CUs
    if comm_cus_count is not None:
        num_comm_cus = min(max(1, comm_cus_count), total_cus - 1)
    else:
        num_comm_cus = max(1, int(total_cus * comm_ratio))

    all_cus = list(range(total_cus))

    if strategy == 'sequential':
        # Sequential: First N CUs for comm, rest for compute
        comm_cus = all_cus[:num_comm_cus]
        compute_cus = all_cus[num_comm_cus:]

    elif strategy == 'interleaved':
        # Interleaved: Odd CUs for comm, even for compute
        # Better memory bandwidth distribution
        odd_cus = [i for i in all_cus if i % 2 == 1]
        even_cus = [i for i in all_cus if i % 2 == 0]

        # Assign first N odd CUs to communication
        comm_cus = odd_cus[:num_comm_cus]

        # If we need more comm CUs than available odd indices, take from even
        if len(comm_cus) < num_comm_cus:
            remaining = num_comm_cus - len(comm_cus)
            comm_cus.extend(even_cus[:remaining])

        # Compute gets ALL remaining CUs
        compute_cus = [i for i in all_cus if i not in comm_cus]

    elif strategy == 'block':
        # Block: First half for comm (up to num_comm_cus), second half for compute
        # NUMA-aware for GPUs with multiple GCDs (e.g., MI250X has 2 GCDs)
        half = total_cus // 2
        first_half = all_cus[:half]
        second_half = all_cus[half:]

        if num_comm_cus <= half:
            comm_cus = first_half[:num_comm_cus]
            compute_cus = first_half[num_comm_cus:] + second_half
        else:
            comm_cus = first_half + second_half[:num_comm_cus - half]
            compute_cus = second_half[num_comm_cus - half:]

    elif strategy == 'ratio':
        # Ratio-based: Simple split based on ratio, taking first N CUs
        comm_cus = all_cus[:num_comm_cus]
        compute_cus = all_cus[num_comm_cus:]

    return comm_cus, compute_cus


def create_stream_with_cu_mask(cu_mask: List[int]):
    """
    Create a HIP stream bound to specific CUs using hipExtStreamCreateWithCUMask.

    Args:
        cu_mask: List of uint32 representing the CU bit mask

    Returns:
        hip.hip.ihipStream_t object (HIP stream handle)

    Raises:
        RuntimeError: If hipExtStreamCreateWithCUMask fails

    Example:
        >>> # Create mask for CUs 0-31
        >>> mask = create_cu_mask(list(range(32)), total_cus=304)
        >>> stream = create_stream_with_cu_mask(mask)
        >>> # Use stream for kernel launches
        >>> hip.hipStreamDestroy(stream)  # Cleanup when done

    Note:
        The Python binding for hipExtStreamCreateWithCUMask returns a tuple
        (hipError_t, ihipStream_t), unlike the C API which takes a stream pointer.
    """
    # Prepare the mask array
    mask_size = len(cu_mask)
    mask_array = (ctypes.c_uint32 * mask_size)(*cu_mask)

    # Call hipExtStreamCreateWithCUMask
    # Python binding returns: (hipError_t, hipStream_t)
    err, stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)

    if err != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"hipExtStreamCreateWithCUMask failed: {err}")

    # stream is a hip.hip.ihipStream_t object
    return stream


class CUMaskedStreamWrapper:
    """
    Wrapper to make HIP streams created with hipExtStreamCreateWithCUMask
    compatible with PyTorch's stream interface.

    PyTorch expects stream objects to have specific attributes:
        - cuda_stream: The underlying HIP stream handle (as integer)
        - device: The torch.device object
        - device_index: The device index (0, 1, 2, ...)
        - device_type: The device type ('cuda')
        - stream_id: Unique identifier for the stream

    Example:
        >>> # Create CU-masked stream
        >>> mask = create_cu_mask([0, 1, 2, 3], total_cus=304)
        >>> hip_stream = create_stream_with_cu_mask(mask)
        >>> stream = CUMaskedStreamWrapper(hip_stream)
        >>>
        >>> # Use with PyTorch context manager
        >>> with torch.cuda.stream(stream):
        ...     # Kernels launched here run on specified CUs
        ...     my_kernel[grid](...)
        >>>
        >>> # Or use directly
        >>> with stream:
        ...     my_kernel[grid](...)
        >>>
        >>> # Cleanup
        >>> stream.destroy()
    """

    def __init__(self, hip_stream):
        """
        Initialize the stream wrapper.

        Args:
            hip_stream: hip.hip.ihipStream_t object from hipExtStreamCreateWithCUMask
        """
        # PyTorch expects cuda_stream to be an integer (stream pointer)
        self.cuda_stream = int(hip_stream)  # Convert ihipStream_t to int
        self._hip_stream = hip_stream       # Keep reference for cleanup

        # PyTorch requires these attributes
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.device_index = torch.cuda.current_device()
        self.device_type = 1  # DeviceType::CUDA = 1 (integer, not string)
        self.stream_id = int(hip_stream)  # Use stream pointer as ID

    def synchronize(self):
        """
        Synchronize this stream (wait for all work to complete).

        Example:
            >>> stream.synchronize()  # Wait for all kernels to finish
        """
        err = hip.hipStreamSynchronize(self._hip_stream)
        # Handle both tuple and direct error code returns
        if isinstance(err, tuple):
            err = err[0]
        if err != hip.hipError_t.hipSuccess:
            raise RuntimeError(f"hipStreamSynchronize failed: {err}")

    def wait_stream(self, other_stream):
        """
        Make this stream wait for another stream to complete.

        Args:
            other_stream: Another stream (can be CUMaskedStreamWrapper or torch.cuda.Stream)
        """
        if hasattr(other_stream, 'cuda_stream'):
            other = other_stream.cuda_stream
        else:
            other = other_stream

        # Create event and synchronize
        err, event = hip.hipEventCreate()
        if err != hip.hipError_t.hipSuccess:
            raise RuntimeError(f"hipEventCreate failed: {err}")

        err = hip.hipEventRecord(event, other)
        # Handle both tuple and direct error code returns
        if isinstance(err, tuple):
            err = err[0]
        if err != hip.hipError_t.hipSuccess:
            raise RuntimeError(f"hipEventRecord failed: {err}")

        err = hip.hipStreamWaitEvent(self._hip_stream, event, 0)
        # Handle both tuple and direct error code returns
        if isinstance(err, tuple):
            err = err[0]
        if err != hip.hipError_t.hipSuccess:
            raise RuntimeError(f"hipStreamWaitEvent failed: {err}")

        hip.hipEventDestroy(event)

    def destroy(self):
        """
        Destroy the underlying HIP stream.

        Warning:
            After calling destroy(), the stream object should not be used.
        """
        err = hip.hipStreamDestroy(self._hip_stream)
        # Handle both tuple and direct error code returns
        if isinstance(err, tuple):
            err = err[0]
        if err != hip.hipError_t.hipSuccess:
            raise RuntimeError(f"hipStreamDestroy failed: {err}")

    def __enter__(self):
        """Context manager entry: Set this stream as current."""
        self._old_stream = torch.cuda.current_stream()
        torch.cuda.set_stream(self)
        return self

    def __exit__(self, *args):
        """Context manager exit: Restore previous stream."""
        torch.cuda.set_stream(self._old_stream)

    def __repr__(self):
        return f"CUMaskedStreamWrapper(stream_id={self.stream_id}, device={self.device_index})"


def print_cu_distribution(
    comm_cus: List[int],
    compute_cus: List[int],
    total_cus: int,
    strategy: str = '',
    verbose: bool = True
):
    """
    Print a visual representation of CU distribution between communication and computation.

    Args:
        comm_cus: List of CU indices assigned to communication
        compute_cus: List of CU indices assigned to computation
        total_cus: Total number of CUs on the device
        strategy: Optional string describing the partitioning strategy
        verbose: If True, print detailed CU assignments; otherwise just summary

    Example:
        >>> comm_cus, compute_cus = partition_cus(304, 'interleaved', 0.3)
        >>> print_cu_distribution(comm_cus, compute_cus, 304, 'interleaved')
    """
    print(f"\n{'='*80}")
    print(f"CU Distribution{f' ({strategy})' if strategy else ''}")
    print(f"{'='*80}")
    print(f"Total CUs: {total_cus}")
    print(f"Communication CUs: {len(comm_cus)} ({100*len(comm_cus)/total_cus:.1f}%)")
    print(f"Computation CUs: {len(compute_cus)} ({100*len(compute_cus)/total_cus:.1f}%)")

    if verbose:
        # Print CU assignment visualization
        print(f"\nCU Assignment:")
        print(f"  Comm CUs: {comm_cus[:10]}{'...' if len(comm_cus) > 10 else ''}")
        print(f"  Comp CUs: {compute_cus[:10]}{'...' if len(compute_cus) > 10 else ''}")

        # Visual bar chart (simplified for first 64 CUs)
        max_visual = min(total_cus, 64)
        visual = []
        for i in range(max_visual):
            if i in comm_cus:
                visual.append('C')
            elif i in compute_cus:
                visual.append('.')
            else:
                visual.append(' ')

        print(f"\nVisual (first {max_visual} CUs, C=Comm, .=Compute):")
        print(f"  {''.join(visual)}")
        if total_cus > 64:
            print(f"  (showing first {max_visual}/{total_cus} CUs)")

    print(f"{'='*80}\n")


# Convenience function for quick setup
def setup_cu_partitioned_streams(
    strategy: str = 'interleaved',
    comm_ratio: float = 0.3,
    device: Optional[int] = None
) -> Tuple[CUMaskedStreamWrapper, CUMaskedStreamWrapper, DeviceInfo]:
    """
    One-liner to set up CU-partitioned streams for communication and computation.

    Args:
        strategy: Partitioning strategy ('sequential', 'interleaved', 'block', 'ratio')
        comm_ratio: Fraction of CUs for communication (0.0-1.0)
        device: Optional device index. If None, uses current device.

    Returns:
        Tuple of (comm_stream_wrapper, compute_stream_wrapper, device_info)

    Example:
        >>> comm_stream, compute_stream, info = setup_cu_partitioned_streams('interleaved', 0.3)
        >>> print(f"Device: {info.name}, Total CUs: {info.total_cus}")
        >>>
        >>> # Use streams
        >>> with torch.cuda.stream(comm_stream):
        ...     launch_communication(...)
        >>> with torch.cuda.stream(compute_stream):
        ...     launch_computation(...)
        >>>
        >>> # Cleanup
        >>> comm_stream.destroy()
        >>> compute_stream.destroy()
    """
    # Get device info
    device_info = get_device_info(device)

    # Partition CUs
    comm_cus, compute_cus = partition_cus(
        total_cus=device_info.total_cus,
        strategy=strategy,
        comm_ratio=comm_ratio
    )

    # Create CU masks
    comm_mask = create_cu_mask(comm_cus, device_info.total_cus)
    compute_mask = create_cu_mask(compute_cus, device_info.total_cus)

    # Create streams
    comm_hip_stream = create_stream_with_cu_mask(comm_mask)
    compute_hip_stream = create_stream_with_cu_mask(compute_mask)

    # Wrap for PyTorch compatibility
    comm_stream_wrapper = CUMaskedStreamWrapper(comm_hip_stream)
    compute_stream_wrapper = CUMaskedStreamWrapper(compute_hip_stream)

    # Print distribution
    print_cu_distribution(comm_cus, compute_cus, device_info.total_cus, strategy, verbose=False)

    return comm_stream_wrapper, compute_stream_wrapper, device_info
