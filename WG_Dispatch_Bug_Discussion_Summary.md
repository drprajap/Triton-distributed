# WG Dispatch & CU Masking Bug — Discussion Summary

**Date:** April 15, 2026  
**Participants:** Dimple Prajapati, Muhammad Osama  
**Topic:** Why CU masking causes 2x+ kernel slowdown on MI300X  
**Source:** Meeting recording transcript

---

## 1. Root Cause: CU Masking Is Broken on MI300X

The 2.18x on-device kernel slowdown observed in our experiments is caused by a **hardware WG dispatch counter bug** specific to GPUs with asymmetric CU harvesting (MI300X, and any GPU where not all physical CUs are active).

### MI300X CU Layout

| Level           | Count                                        |
|:--|:--|
| XCDs            | 8                                            |
| Shader Engines (SE) per XCD | 4                                |
| Physical CUs per SE | 10                                       |
| Disabled (harvested) CUs per XCD | 2 (from 40 → 38 active)    |
| Total active CUs | 304 (8 × 38)                                |

The 2 disabled CUs per XCD are physically defective and their positions vary per GPU (could be in any combination of SEs).

### How Normal WG Dispatch Works

1. WGs dispatch to SEs via **round-robin** within each XCD.
2. Each SE has a **hardware counter** tracking occupancy.
3. When round-robin hits a **disabled (harvested) CU**, a **dummy work group** is scheduled that updates the counter to zero, allowing the dispatch baton to pass to the next SE.
4. This works correctly — the dummy WG mechanism compensates for asymmetric harvesting.

### How CU Masking Breaks Dispatch

When `hipExtStreamCreateWithCUMask` disables additional CUs:

1. The **hardware counter still reflects physical CU count** (e.g., reads 9 or 10 per SE), NOT the masked CU count.
2. When round-robin dispatch hits a **CU-masked (but physically present) CU**, no dummy WG is scheduled — the hardware doesn't know this CU is masked.
3. The counter **never reaches zero** for that SE, so the dispatch baton **never passes** to the next SE.
4. Result: **deadlock** — the dispatcher is stuck trying to schedule on a SE with no available CUs, while other SEs have free resources.
5. The deadlock only resolves when **all work on other CUs completes**, freeing resources so the counter can eventually zero out and the baton moves on.
6. This can happen **repeatedly during kernel execution** — mid-scheduling deadlocks create an arbitrary tail latency effect.

### Why This Causes 2x Slowdown (Not Complete Deadlock)

The deadlock is **temporary** — once WGs on other CUs finish, space opens up and dispatch continues. But the repeated stalls during dispatch create:

- **Bimodal execution patterns** (alternating ~1.26ms and ~1.80ms iterations, as we observed)
- **Effective 2.18x on-device slowdown** — the kernel binary and resources are identical, only the stream dispatch mechanism differs

### Why MI355X (256 CUs) Won't Have This Problem

MI355X has 256 CUs = 4 XCDs × 64 CUs = symmetric harvesting. No disabled CUs means no counter mismatch, no dispatch deadlock.

---

## 2. Wave Quantization / Tail Latency Analysis

For our GEMM kernel (BLOCK_SIZE_M=256, BLOCK_SIZE_N=256):

- M=8192, N=11008 → 32 × 22 = **704 output tiles**

| Grid Size | Waves         | Last Wave Utilization | Quality |
|:-:|:--|:-:|:--|
| 304       | 2.32 (304+304+96) | 31.6%             | Poor    |
| 213       | 3.31 (213×3+65)    | 30.5%             | Poor    |
| 176       | 4.00 (176×4)       | **100%**          | **Optimal** |
| 152       | 4.63 (152×4+96)    | 63.2%             | OK      |

**Recommendation:** Use grid sizes that evenly divide the tile count. 704/176 = 4 complete waves with zero tail latency.

This explains why 152 CUs showed the least relative slowdown — it has better last-wave utilization than 213 or 304.

---

## 3. Hierarchical Scheduler Problem

Beyond CU masking, there's a fundamental **hierarchical scheduling problem** when running GEMM + communication concurrently:

### Two-Level Round-Robin

1. **XCD scheduler** — distributes WGs across XCDs via round-robin
2. **SE scheduler** — distributes WGs across SEs within an XCD via round-robin

### Queue Independence Problem

- GEMM queue and communication queue dispatch **independently**
- Neither queue knows about the other's WG count or scheduling
- No **work stealing across XCDs** — even if one XCD has free resources, another overloaded XCD can't offload to it
- The independent round-robin distributions create **probabilistic imbalances**: some SEs get more GEMM WGs + comm WGs than they have CUs, while other SEs sit idle
- This creates a **tail latency effect from dispatch imbalance**, separate from the CU masking bug

### Impact

Even without CU masking, concurrent GEMM + comm kernels experience some slowdown due to this scheduling imbalance. The work-stealing GEMM approach (atomic tile counters) partially mitigates this because it doesn't rely on static grid assignment.

---

## 4. Recommended Experiments

### Experiment A: Software CU Masking (No `hipExtStreamCreateWithCUMask`)

Instead of hardware CU masking, modify the GEMM kernel to **return early** for masked WGs:

```python
@triton.jit
def gemm_kernel_with_soft_mask(
    ...,
    USE_MASK: tl.constexpr,
    CU_MASK_THRESHOLD: tl.constexpr,
):
    pid = tl.program_id(0)
    if USE_MASK:
        if pid >= CU_MASK_THRESHOLD:
            return  # This WG does nothing — simulates CU masking
    # ... rest of GEMM kernel ...
```

- Launch on a **regular (unmasked) stream** with full grid (e.g., 304 WGs)
- Set `CU_MASK_THRESHOLD = 213` to simulate 213 available CUs
- The WGs with pid >= 213 dispatch but return immediately
- Avoids the hardware dispatch counter bug entirely
- Compare this timing with actual CU-masked result — the difference = dispatch bug cost

### Experiment B: Persistent Occupancy Kernel

For HipBLASLt or other kernels you can't modify:

1. Launch a persistent `while(True)` kernel on N CUs (e.g., 91) with enough LDS/registers to fully occupy each CU
2. Wait for it to be scheduled (sleep briefly)
3. Launch GEMM on the remaining CUs (304 - 91 = 213)
4. Measure GEMM time — this simulates CU-masked GEMM without the dispatch bug

### Experiment C: Reproduce Dispatch Bug in Comm Kernel

Current comm-only benchmark shows <1% CU masking impact because:
- Copy kernel uses very few resources (few VGPRs, zero LDS)
- Multiple WGs fit per CU → high occupancy
- Dispatch counter never deadlocks (enough concurrent WGs absorb the stalls)

To reproduce the 2x bug in comm:
- Increase **register pressure** and **LDS usage** in the copy kernel
- Force **1 WG per CU** occupancy
- Use **many WGs** (more waves) to stress the round-robin dispatcher
- See Section 5 below for specifics

---

## 5. Why Comm-Only Benchmark Didn't Show 2x Slowdown

The copy kernel (`copy_kernel`, `persistent_copy_kernel`) uses BLOCK_SIZE=1024 with simple `tl.load` + `tl.store`:

- **~8-10 VGPRs** per wavefront (trivial register usage)
- **0 LDS** bytes
- **Occupancy:** ~32 WGs per CU (register-limited max = 512 VGPRs/wave ÷ ~10 VGPRs ≈ 50 waves, LDS-limited = unlimited)
- Even with CU masking to 91 CUs: 91 × 32 = 2,912 concurrent WGs → 16,384 / 2,912 ≈ 6 waves

With such high per-CU occupancy, the dispatch counter stalls are absorbed by the concurrent WGs already running. The deadlock effect only manifests when CUs are **fully occupied** (1 WG per CU) and the dispatcher can't find free slots.

To hit the bug: need a **resource-heavy copy kernel** that fully occupies each CU.

---

## 6. Other Performance Factors (for reference)

| Factor                    | Impact                                   | Mitigation                                |
|:--|:--|:--|
| Tail latency (wave quantization) | Depends on tiles / grid ratio    | Choose grid sizes that evenly divide tiles |
| Hierarchical scheduler imbalance | Creates probabilistic stalls     | Work-stealing GEMM (atomic tile counters) |
| HBM / NOC (XGMI) contention     | Bandwidth sharing between GEMM and comm | Inherent to concurrent execution    |
| L2 / MALL thrashing              | Cache eviction from comm traffic | Uncached access for comm, rotating buffers for GEMM |

---

## 7. Action Items

1. **Implement software CU masking** (early return in kernel) to isolate dispatch bug cost
2. **Implement persistent occupancy kernel** as alternative CU consumption method
3. **Modify comm benchmark** to use resource-heavy copy kernel to confirm dispatch bug is kernel-agnostic
4. **Re-run GEMM experiments** with software masking to get "true" CU reduction cost without dispatch bug
5. **Try grid=176** (perfect wave utilization for 704 tiles) to eliminate tail latency from measurements
6. **Report dispatch bug** to ROCm team — `hipExtStreamCreateWithCUMask` counter mismatch with harvested CUs
