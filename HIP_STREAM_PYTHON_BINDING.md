# hipExtStreamCreateWithCUMask Python Binding - Correct Usage

## Issue Summary

The Python `hip.hip` module's binding for `hipExtStreamCreateWithCUMask` differs from the C API in both **signature** and **return type**.

## C API vs Python Binding

### ❌ C API (doesn't work in Python)
```c
hipError_t hipExtStreamCreateWithCUMask(
    hipStream_t* stream,       // [out] Pointer to stream
    uint32_t cuMaskSize,       // [in] Mask size
    const uint32_t* cuMask     // [in] Mask array
);
```

### ✅ Python Binding (correct)
```python
# Signature: hipExtStreamCreateWithCUMask(cuMaskSize, cuMask)
# Returns: tuple(hipError_t, hipStream_t)

err, stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)
```

## Key Differences

| Aspect | C API | Python Binding |
|--------|-------|---------------|
| **Arguments** | 3: `(stream*, size, mask*)` | 2: `(size, mask)` |
| **Return Type** | `hipError_t` | `tuple(hipError_t, ihipStream_t)` |
| **Stream Output** | Via pointer argument | Via return tuple |
| **Stream Type** | `hipStream_t` (pointer) | `hip.hip.ihipStream_t` (object) |

## Correct Implementation

### Complete Example

```python
import ctypes
from hip import hip
import torch

def create_stream_with_cu_mask(cu_mask: list):
    """
    Create a HIP stream bound to specific CUs.
    
    Args:
        cu_mask: List of uint32 representing CU bit mask
        
    Returns:
        hip.hip.ihipStream_t object
    """
    # Prepare mask array
    mask_size = len(cu_mask)
    mask_array = (ctypes.c_uint32 * mask_size)(*cu_mask)
    
    # Call Python binding (2 args, returns tuple)
    err, stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)
    
    if err != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"hipExtStreamCreateWithCUMask failed: {err}")
    
    # stream is a hip.hip.ihipStream_t object (NOT an integer!)
    return stream


# Usage
total_cus = torch.cuda.get_device_properties(0).multi_processor_count
mask_size = (total_cus + 31) // 32
cu_mask = [0x0000000F] + [0] * (mask_size - 1)  # Enable CUs 0-3

# Create stream
comm_stream = create_stream_with_cu_mask(cu_mask)
print(f"Stream type: {type(comm_stream)}")  # hip.hip.ihipStream_t

# Use with hipMemcpyAsync (pass stream object directly)
hip.hipMemcpyAsync(
    dst_ptr, src_ptr, nbytes,
    hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
    comm_stream  # ✅ Pass ihipStream_t object
)

# Use with PyTorch/Triton (wrap in class with cuda_stream attribute)
class StreamWrapper:
    def __init__(self, hip_stream):
        self.cuda_stream = hip_stream  # Store ihipStream_t object

with torch.cuda.stream(StreamWrapper(comm_stream)):
    triton_kernel[grid](...)

# Cleanup (pass stream object to destroy)
hip.hipStreamDestroy(comm_stream)
```

## Common Errors and Fixes

### Error 1: TypeError - Too Many Arguments

**Error:**
```
TypeError: hipExtStreamCreateWithCUMask() takes exactly 2 positional arguments (3 given)
```

**Cause:** Using C API signature with 3 arguments

**Fix:**
```python
# ❌ Wrong (C-style)
stream_ptr = ctypes.c_void_p()
err = hip.hipExtStreamCreateWithCUMask(
    ctypes.byref(stream_ptr),  # ❌ Don't pass stream pointer
    mask_size,
    mask_array
)

# ✅ Correct (Python binding)
err, stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)
```

### Error 2: TypeError - Cannot Interpret as Integer

**Error:**
```
TypeError: 'hip.hip.ihipStream_t' object cannot be interpreted as an integer
```

**Cause:** Trying to use `hex()`, format as int, or pass where int is expected

**Fix:**
```python
# ❌ Wrong - trying to convert to int
print(f"Stream: {hex(stream)}")  # ❌ Fails!

# ✅ Correct - treat as object
print(f"Stream type: {type(stream).__name__}")  # ✅ Works!
print(f"Stream: {stream}")  # ✅ Shows object representation

# ❌ Wrong - storing as int
comm_stream_handle = int(stream)  # ❌ Fails!

# ✅ Correct - store as object
comm_stream = stream  # ✅ Works!
```

### Error 3: AttributeError - No 'device' or 'stream_id' Attribute

**Error:**
```
AttributeError: 'CUMaskedStream' object has no attribute 'device'
AttributeError: 'CUMaskedStream' object has no attribute 'stream_id'
```

**Cause:** PyTorch's `torch.cuda.stream()` checks device compatibility and tracks streams using these attributes

**Fix:**
```python
# ❌ Wrong - passing ihipStream_t object directly
class StreamWrapper:
    def __init__(self, hip_stream):
        self.cuda_stream = hip_stream  # ❌ Wrong type!

# ✅ Correct - convert ihipStream_t to integer pointer
class StreamWrapper:
    def __init__(self, hip_stream):
        # PyTorch expects an integer (stream pointer)
        self.cuda_stream = int(hip_stream)  # ✅ Convert to int!
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.device_index = torch.cuda.current_device()
        self.device_type = torch.device('cuda').type
        self.stream_id = int(hip_stream)

with torch.cuda.stream(StreamWrapper(comm_stream)):  # ✅ Works!
    triton_kernel[grid](...)
```

**Key Point:** The `ihipStream_t` object has a `__int__()` method that returns the stream pointer as an integer!

## Stream Type Handling

### The ihipStream_t Object

```python
err, stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)

# What is stream?
type(stream)  # <class 'hip.hip.ihipStream_t'>

# How to use it?
# 1. Pass directly to HIP APIs
hip.hipMemcpyAsync(..., stream)  # ✅
hip.hipStreamSynchronize(stream)  # ✅
hip.hipStreamDestroy(stream)      # ✅

# 2. Wrap for PyTorch
wrapper = StreamWrapper(stream)
with torch.cuda.stream(wrapper):  # ✅
    kernel[grid](...)

# 3. Cannot convert to int
int(stream)        # ❌ TypeError
hex(stream)        # ❌ TypeError
stream_id = stream # ✅ But keeps as object, not int
```

## Pattern: PyTorch Stream Wrapper

```python
class CUMaskedStreamWrapper:
    """
    Wrapper to make hipExtStreamCreateWithCUMask stream compatible with PyTorch.
    
    PyTorch expects stream objects to have:
    - `cuda_stream`: The underlying HIP stream handle
    - `device()`: Method that returns the torch.device
    - `device_index`: The device index (0, 1, 2, ...)
    - `device_type`: The device type ('cuda')
    - `stream_id`: Unique identifier for the stream
    """
    def __init__(self, hip_stream):
        """
        Args:
            hip_stream: hip.hip.ihipStream_t object from hipExtStreamCreateWithCUMask
        """
        # PyTorch expects cuda_stream to be an integer (stream pointer)
        self.cuda_stream = int(hip_stream)  # Convert ihipStream_t to int
        self._hip_stream = hip_stream       # Keep reference for cleanup
        
        # PyTorch requires these attributes
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.device_index = torch.cuda.current_device()
        self.device_type = torch.device('cuda').type
        self.stream_id = int(hip_stream)  # Use stream pointer as ID
    
    def synchronize(self):
        """Synchronize this stream"""
        hip.hipStreamSynchronize(self._hip_stream)
    
    def wait_stream(self, other_stream):
        """Make this stream wait for another stream"""
        if hasattr(other_stream, 'cuda_stream'):
            other = other_stream.cuda_stream
        else:
            other = other_stream
        hip.hipStreamWaitEvent(self._hip_stream, other, 0)
    
    def destroy(self):
        """Destroy the underlying HIP stream"""
        hip.hipStreamDestroy(self._hip_stream)
    
    def __enter__(self):
        """Context manager support"""
        self._old_stream = torch.cuda.current_stream()
        torch.cuda.set_stream(self)
        return self
    
    def __exit__(self, *args):
        """Context manager cleanup"""
        torch.cuda.set_stream(self._old_stream)


# Usage
err, hip_stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)
stream = CUMaskedStreamWrapper(hip_stream)

# Use with context manager
with stream:
    triton_kernel[grid](...)

# Or explicit stream specification
with torch.cuda.stream(stream):
    triton_kernel[grid](...)

# Cleanup
stream.destroy()
```

## Complete Working Example

```python
import torch
import ctypes
from hip import hip
import triton
import triton.language as tl

# Get CU count
total_cus = torch.cuda.get_device_properties(0).multi_processor_count
print(f"Total CUs: {total_cus}")

# Create CU mask (enable first 10 CUs)
mask_size = (total_cus + 31) // 32
cu_mask = [0] * mask_size
for i in range(10):
    cu_mask[i // 32] |= (1 << (i % 32))

mask_array = (ctypes.c_uint32 * mask_size)(*cu_mask)

# Create CU-masked stream
err, stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)
assert err == hip.hipError_t.hipSuccess

print(f"✅ Created stream (type: {type(stream).__name__})")

# Test 1: HIP memory copy
src = torch.randn(1000, device='cuda')
dst = torch.zeros(1000, device='cuda')

hip.hipMemcpyAsync(
    dst.data_ptr(),
    src.data_ptr(),
    src.numel() * src.element_size(),
    hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
    stream
)
hip.hipStreamSynchronize(stream)
print(f"✅ HIP memcpy succeeded")

# Test 2: Triton kernel
@triton.jit
def add_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x + 1.0, mask=mask)

class StreamWrapper:
    def __init__(self, s):
        self.cuda_stream = s

output = torch.zeros_like(src)
with torch.cuda.stream(StreamWrapper(stream)):
    grid = lambda meta: (triton.cdiv(src.numel(), meta['BLOCK']),)
    add_kernel[grid](src, output, src.numel(), BLOCK=256)

torch.cuda.synchronize()
print(f"✅ Triton kernel succeeded")

# Cleanup
hip.hipStreamDestroy(stream)
print(f"✅ Stream destroyed")
```

## Testing Your Setup

Run the verification script:

```bash
cd /dev/data/diprajap/workspace/rocm7/Triton-distributed
python test_hip_cu_mask_binding.py
```

Expected output:
```
Testing hipExtStreamCreateWithCUMask Python binding...
Total CUs: 110
Mask size: 4 uint32s
Mask: ['0xf', '0x0', '0x0', '0x0']

Calling hipExtStreamCreateWithCUMask...
  Arguments: mask_size=4, mask_array=<...>
  Result type: <class 'tuple'>
  Result: (<hipError_t.hipSuccess: 0>, <hip.hip.ihipStream_t object at 0x...>)

✅ Success!
  Error code: hipSuccess (True)
  Stream type: <class 'hip.hip.ihipStream_t'>
  Stream: <hip.hip.ihipStream_t object at 0x...>
  Stream class: ihipStream_t

✅ Stream destroyed successfully (err: hipSuccess)

============================================================
✅ hipExtStreamCreateWithCUMask binding verified!
============================================================
```

## Summary

| ✅ Do This | ❌ Not This |
|-----------|------------|
| `err, stream = hip.hipExtStreamCreateWithCUMask(size, mask)` | `hip.hipExtStreamCreateWithCUMask(ptr, size, mask)` |
| `comm_stream = stream` (store as object) | `comm_stream = int(stream)` |
| `hip.hipMemcpyAsync(..., stream)` | `hip.hipMemcpyAsync(..., stream.handle)` |
| `class W: self.cuda_stream = stream` | `with torch.cuda.stream(stream)` directly |
| `print(f"Type: {type(stream)}")` | `print(f"Handle: {hex(stream)}")` |

**Key Takeaway:** The stream is a **`hip.hip.ihipStream_t` object**, not an integer. Treat it as an opaque object that you pass to HIP APIs and wrap for PyTorch usage.

