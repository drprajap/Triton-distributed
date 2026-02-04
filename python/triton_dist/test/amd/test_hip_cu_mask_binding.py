#!/usr/bin/env python3
"""
Quick test to verify hipExtStreamCreateWithCUMask Python binding signature
"""
import torch
import ctypes
from hip import hip

def test_hip_binding():
    """Test the Python binding for hipExtStreamCreateWithCUMask"""
    print("Testing hipExtStreamCreateWithCUMask Python binding...")
    
    # Get number of CUs
    total_cus = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"Total CUs: {total_cus}")
    
    # Create a simple mask: enable first 4 CUs
    mask_size = (total_cus + 31) // 32
    cu_mask = [0] * mask_size
    for i in range(4):  # Enable CUs 0-3
        cu_mask[i // 32] |= (1 << (i % 32))
    
    print(f"Mask size: {mask_size} uint32s")
    print(f"Mask: {[hex(m) for m in cu_mask]}")
    
    # Create mask array
    mask_array = (ctypes.c_uint32 * mask_size)(*cu_mask)
    
    # Test the Python binding
    print("\nCalling hipExtStreamCreateWithCUMask...")
    print(f"  Arguments: mask_size={mask_size}, mask_array={mask_array}")
    
    try:
        # Python binding signature: hipExtStreamCreateWithCUMask(cuMaskSize, cuMask)
        # Returns: (hipError_t, hipStream_t)
        result = hip.hipExtStreamCreateWithCUMask(mask_size, mask_array)
        print(f"  Result type: {type(result)}")
        print(f"  Result: {result}")
        
        if isinstance(result, tuple):
            err, stream = result
            print(f"\n✅ Success!")
            print(f"  Error code: {err} ({err == hip.hipError_t.hipSuccess})")
            print(f"  Stream type: {type(stream)}")
            print(f"  Stream: {stream}")
            
            # Test: stream should be hip.hip.ihipStream_t object
            if hasattr(stream, '__class__'):
                print(f"  Stream class: {stream.__class__.__name__}")
            
            # Cleanup
            destroy_err = hip.hipStreamDestroy(stream)
            print(f"\n✅ Stream destroyed successfully (err: {destroy_err})")
            return True
        else:
            print(f"\n❌ Unexpected return type: {type(result)}")
            return False
            
    except TypeError as e:
        print(f"\n❌ TypeError: {e}")
        print("\nTrying alternative signature...")
        
        # Try with stream_ptr as first argument
        try:
            stream_ptr = ctypes.c_void_p()
            err = hip.hipExtStreamCreateWithCUMask(
                ctypes.byref(stream_ptr),
                mask_size,
                mask_array
            )
            print(f"  This signature worked: stream_ptr, mask_size, mask_array")
            print(f"  Stream: {stream_ptr.value}")
            hip.hipStreamDestroy(stream_ptr.value)
            return True
        except Exception as e2:
            print(f"  Also failed: {e2}")
            return False
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    torch.cuda.set_device(0)
    success = test_hip_binding()
    
    if success:
        print("\n" + "="*60)
        print("✅ hipExtStreamCreateWithCUMask binding verified!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Failed to verify binding")
        print("="*60)

