# CU Partitioning Test - Complete Debugging Journey & Fixes

This document summarizes all the issues encountered and fixed while implementing explicit CU partitioning with `hipExtStreamCreateWithCUMask`.

## 🎯 Goal
Enable explicit Compute Unit (CU) partitioning to run communication and computation kernels on separate CUs using `hipExtStreamCreateWithCUMask`.

## 🐛 Issues Encountered & Fixes

### Issue 1: Wrong Python Binding Signature

**Error:**
```
TypeError: hipExtStreamCreateWithCUMask() takes exactly 2 positional arguments (3 given)
```

**Root Cause:** Used C API signature instead of Python binding.

**Fix:**
```python
# ❌ Wrong (C API style)
stream_ptr = ctypes.c_void_p()
err = hip.hipExtStreamCreateWithCUMask(ctypes.byref(stream_ptr), mask_size, mask_array)

# ✅ Correct (Python binding)
err, stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)
```

---

### Issue 2: Trying to Convert Stream to Integer with `hex()`

**Error:**
```
TypeError: 'hip.hip.ihipStream_t' object cannot be interpreted as an integer
```

**Root Cause:** Attempted to format stream as hex/int for logging.

**Fix:**
```python
# ❌ Wrong
print(f"Stream: {hex(stream)}")  # Fails!

# ✅ Correct
print(f"Stream type: {type(stream).__name__}")
```

---

### Issue 3: Buffer Size Mismatch in AllGather

**Error:**
```
RuntimeError: The size of tensor a (4194304) must match the size of tensor b (524288)
```

**Root Cause:** Tried to copy entire local buffer to entire symmetric buffer without proper offset.

**Fix:**
```python
# ❌ Wrong
remote_bufs[rank].copy_(local_buf)  # Sizes don't match!

# ✅ Correct
dst_slice = remote_bufs[rank][rank * nelems:(rank + 1) * nelems]
dst_slice.copy_(local_buf)  # Copy to correct slice
```

---

### Issue 4: Missing `device` Attribute

**Error:**
```
AttributeError: 'CUMaskedStream' object has no attribute 'device'
```

**Root Cause:** PyTorch's stream context manager checks for `device` attribute.

**Fix:**
```python
# ❌ Wrong
class StreamWrapper:
    def __init__(self, stream):
        self.cuda_stream = stream  # Missing device!

# ✅ Correct
class StreamWrapper:
    def __init__(self, stream):
        self.cuda_stream = stream
        self.device = torch.device('cuda', torch.cuda.current_device())
```

---

### Issue 5: Missing `stream_id` Attribute

**Error:**
```
AttributeError: 'CUMaskedStream' object has no attribute 'stream_id'
```

**Root Cause:** PyTorch tracks streams using `stream_id`.

**Fix:**
```python
class StreamWrapper:
    def __init__(self, stream):
        self.cuda_stream = stream
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.stream_id = id(stream)  # ✅ Add unique ID
```

---

### Issue 6: `device` Must Be a Property (Not Just Attribute)

**Error:**
```
RuntimeError: Expected stream.device().is_cuda() to be true, but got false.
```

**Root Cause:** PyTorch's internal code may call `device()` or access `device` - using `@property` handles both cases.

**Fix:**
```python
# ❌ Wrong - direct attribute
class StreamWrapper:
    def __init__(self, stream):
        self.device = torch.device('cuda', ...)

# ✅ Correct - use @property
class StreamWrapper:
    def __init__(self, stream):
        self._device = torch.device('cuda', ...)
    
    @property
    def device(self):
        return self._device
```

---

### Issue 7: CRITICAL - `cuda_stream` Must Be Integer, Not Object

**Error:**
```
RuntimeError: Expected stream.device().is_cuda() to be true, but got false.
```

**Root Cause:** PyTorch expects `cuda_stream` to be an **integer pointer**, not the `ihipStream_t` object!

**Verification:**
```python
# Real PyTorch stream
s = torch.cuda.Stream()
type(s.cuda_stream)  # <class 'int'>
s.cuda_stream        # 94343620248368 (integer!)

# Our HIP stream
err, stream = hip.hipExtStreamCreateWithCUMask(...)
type(stream)         # <class 'hip.hip.ihipStream_t'>  # ❌ Wrong type!
```

**Fix: Use `int()` to Convert**
```python
# The ihipStream_t object has __int__() method!
print(int(stream))   # 94559452578192 (integer pointer!)

class StreamWrapper:
    def __init__(self, hip_stream):
        # ✅ Convert ihipStream_t to integer pointer
        self.cuda_stream = int(hip_stream)
        self._device = torch.device('cuda', torch.cuda.current_device())
        self.device_index = torch.cuda.current_device()
        self.device_type = 'cuda'
        self.stream_id = int(hip_stream)
    
    @property
    def device(self):
        return self._device
```

---

## ✅ Complete Working Solution

### Final Stream Wrapper Class

```python
import torch

class CUMaskedStreamWrapper:
    """
    Complete PyTorch-compatible wrapper for hipExtStreamCreateWithCUMask streams
    """
    def __init__(self, hip_stream):
        """
        Args:
            hip_stream: hip.hip.ihipStream_t object from hipExtStreamCreateWithCUMask
        """
        # CRITICAL: Convert ihipStream_t to integer pointer
        self.cuda_stream = int(hip_stream)
        
        # Required PyTorch attributes
        self._device = torch.device('cuda', torch.cuda.current_device())
        self.device_index = torch.cuda.current_device()
        self.device_type = 'cuda'
        self.stream_id = int(hip_stream)
        
        # Keep reference for cleanup
        self._hip_stream = hip_stream
    
    @property
    def device(self):
        """PyTorch accesses this as a property"""
        return self._device
    
    def synchronize(self):
        """Synchronize this stream"""
        from hip import hip
        hip.hipStreamSynchronize(self._hip_stream)
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            from hip import hip
            hip.hipStreamDestroy(self._hip_stream)
        except:
            pass
```

### Usage Example

```python
import torch
import ctypes
from hip import hip

# 1. Create CU mask
total_cus = torch.cuda.get_device_properties(0).multi_processor_count
mask_size = (total_cus + 31) // 32
cu_mask = [0x0000000F] + [0] * (mask_size - 1)  # Enable CUs 0-3

# 2. Create CU-masked stream
mask_array = (ctypes.c_uint32 * mask_size)(*cu_mask)
err, hip_stream = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)

# 3. Wrap for PyTorch
stream_wrapper = CUMaskedStreamWrapper(hip_stream)

# 4. Use with PyTorch!
with torch.cuda.stream(stream_wrapper):
    # Kernels run on specified CUs only
    result = my_triton_kernel[grid](...)

torch.cuda.synchronize()
```

## 📊 Key Learnings

| Component | Expected Type | What We Had | Solution |
|-----------|--------------|-------------|----------|
| `cuda_stream` | `int` | `ihipStream_t` object | Use `int(stream)` |
| `device` | `torch.device` (property) | Missing | Add with `@property` |
| `device_index` | `int` | Missing | `torch.cuda.current_device()` |
| `device_type` | `str` | Wrong value | `'cuda'` (not `'cuda:0'`) |
| `stream_id` | `int` | Missing | `int(stream)` |

## 🎯 The Critical Insight

**The `ihipStream_t` object is a Cython wrapper around the C `hipStream_t` pointer.**

- It has a `__int__()` method that extracts the actual pointer value
- PyTorch's C++ code expects the raw integer pointer, not the Python object
- Always use `int(hip_stream)` when passing to PyTorch!

## ✅ Verification

Run the test:
```bash
cd /dev/data/diprajap/workspace/rocm7/Triton-distributed

# Quick verification
python test_stream_wrapper_final.py

# Full test (requires 2+ GPUs)
WORLD_SIZE=2 LOCAL_WORLD_SIZE=2 \
python python/triton_dist/test/amd/test_cu_partitioning.py
```

## 📚 Files Updated

1. `test_cu_partitioning.py` - Full test with all fixes
2. `HIP_STREAM_PYTHON_BINDING.md` - Updated documentation
3. `test_stream_wrapper_final.py` - Standalone verification
4. `CU_PARTITIONING_GUIDE.md` - Complete guide
5. This file - `CU_PARTITIONING_DEBUG_JOURNEY.md`

---

**Success Criteria:** ✅ All tests pass, streams work with PyTorch, CU partitioning verified!

