# CU Masking & Work-Stealing GEMM: Consolidated Experiment Results

**Platform:** AMD MI300X (304 CUs, 4 XCDs), ROCm 7.1, Triton 3.4  
**Branch:** `cu_mask_partitioning`  
**Container:** `diprajap-pytorch-rocm7_1-triton`  
**GPU pinning:** `HIP_VISIBLE_DEVICES=4,5`

---

## 1. Summary of Key Findings

### CU masking consistently hurts performance — the root cause is on-device kernel slowdown

Across all experiments — different problem sizes, comm-ratios, comm engines, and GEMM durations — CU masking via `hipExtStreamCreateWithCUMask` never improved wall-clock latency. The penalty ranges from 1.2x to 2.2x slower depending on configuration.

**Root cause identified via profiling:**

The `hipExtStreamCreateWithCUMask` mechanism causes a **2.18x on-device kernel execution slowdown** for the same kernel binary, same grid size, same resources. This was proven by:

1. **rocprofv3 kernel trace**: The identical GEMM kernel (120 VGPRs, 128 AccVGPRs, 213 WGs, 512 threads/WG) takes 0.764 ms/iter on a regular stream vs 1.664 ms/iter on a CU-masked stream — a 2.18x slowdown.
2. **No host-side overhead**: `hipModuleLaunchKernel` takes ~6-7 us on both stream types.
3. **No inter-kernel gaps**: Kernels dispatch back-to-back (0 us gap) on both streams.
4. **Bimodal execution pattern**: CU-masked kernels show alternating ~1.26 ms and ~1.80 ms iterations, suggesting workgroup scheduling instability with CU masks.

**Confirmed root cause: HW dispatch counter bug** (discussion with Muhammad Osama):
- The SE workgroup dispatch counters read `CC_GC_SHADER_ARRAY_CONFIG` which reflects **physical CUs, not masked CUs**. When round-robin dispatch hits a CU-masked (but physically present) CU, no dummy WG is scheduled — the counter never zeros — the dispatch "baton" never passes to the next SE. This creates intermittent deadlocks during WG scheduling.
- The slowdown is **proportional to compute density**: BW-limited kernels show 1.00x (stalls hidden under memory latency), 64 FMAs/element → 1.28x, 1024 FMAs → 1.57x, GEMM → 2.18x.
- The bug triggers even within a single dispatch wave (grid = CU count) — it is not a wave-transition effect.

Not a contributing factor:
- ~~Wave quantization~~ — same grid size, same tile count
- ~~Host dispatch overhead~~ — identical launch latency
- ~~Communication contention~~ — comm-only benchmarks show < 1% CU masking impact

### Communication is XGMI-limited, not CU-limited

P2P comm benchmarks prove that CU masking has **< 1% impact on communication bandwidth**. Whether using 30, 91, or 304 CUs, the XGMI link (~47 GB/s) saturates the same. The entire AG+GEMM penalty comes from the CU-masked **compute** stream.

### Work-stealing GEMM shows better overlap tolerance

When using atomic tile counters instead of static grid assignment, the GEMM kernel naturally adapts to available CUs without CU masking. Reducing the work-stealing grid size monotonically decreases communication overhead percentage.

---

## 2. Bug Fix: `cu_copy_kernel` Stream Dispatch

A critical bug was found and fixed in `tutorials/12-cu-masked-ag-gemm.py`: the `cu_copy_kernel` in `producer_ag_cu_kernel` was dispatching on the **default (null) stream** instead of the intended `ag_stream`. This meant "cu-kernel" experiments were never actually CU-masking the communication — they were running unmasked comm + CU-masked GEMM.

**Fix:** Wrapped the kernel launch with `with torch.cuda.stream(ag_stream):` (line 92).

All experiments below use the corrected code.

---

## 3. GEMM-Only CU Masking (Tutorial 16 — No Communication)

Standalone persistent GEMM kernel (256x256x64 tiles, 8 warps, 2 stages) on a single GPU, no AG, no distributed setup. Isolates CU mask overhead cleanly.

**Setup:** M=8192, N=5504, K=4096, gemm-iters=8, 10 repeats, single GPU

| Grid (NUM_SMS) | CU mask               | Regular Unmasked (ms) | CU-masked (ms) | Slowdown  |
|:-:|:--|:-:|:-:|:-:|
| 213            | 213 compute CUs (70%) | 6.07                  | 13.44          | **2.21x** |
| 304            | 213 compute CUs (70%) | 5.48                  | 11.36          | **2.07x** |
| 152            | 213 compute CUs (70%) | 6.90                  | 11.89          | **1.72x** |

5x stability check at grid=213:

| Run | Regular Unmasked (ms) | CU-masked (ms) | Slowdown |
|:-:|:-:|:-:|:-:|
| 1   | 6.07                  | 13.50           | 2.23x    |
| 2   | 6.09                  | 13.65           | 2.24x    |
| 3   | 6.11                  | 13.50           | 2.21x    |
| 4   | 6.16                  | 13.53           | 2.20x    |
| 5   | 6.06                  | 13.43           | 2.22x    |

**Finding:** CU masking causes a stable ~2.2x on-device slowdown for the same GEMM kernel, independent of communication. This is the dominant cost in all AG+GEMM experiments.

---

## 4. rocprofv3 Kernel Trace Analysis

Profiled the GEMM-only benchmark (grid=213) with `rocprofv3 --kernel-trace --hip-trace -f csv`.

### Kernel Resource Usage (identical for both streams)

| Resource       | Value                          |
|:--|:--|
| VGPRs          | 120                            |
| AccVGPRs       | 128                            |
| SGPRs          | 64                             |
| LDS            | 0 bytes                        |
| Workgroup size | 512 threads (8 wavefronts)     |
| Grid           | 213 workgroups (109,056 threads) |
| Occupancy      | ~1 WG/CU (register-limited)    |

### Per-Kernel Execution Time (measured 8 iterations)

**Regular stream:**

| Dispatch  | Duration (ms) | Gap (us)          |
|:--|:-:|:-:|
| 11        | 0.761         | —                 |
| 12        | 0.715         | 0                 |
| 13        | 0.743         | 0                 |
| 14        | 0.773         | 0                 |
| 15        | 0.796         | 0                 |
| 16        | 0.774         | 0                 |
| 17        | 0.773         | 0                 |
| 18        | 0.773         | 0                 |
| **Total** | **6.109 ms**  | **Avg: 0.764 ms** |

**CU-masked stream:**

| Dispatch  | Duration (ms) | Gap (us)           |
|:--|:-:|:-:|
| 27        | 1.259         | —                  |
| 28        | 1.816         | 0                  |
| 29        | 1.797         | 0                  |
| 30        | 1.810         | 0                  |
| 31        | 1.255         | 0                  |
| 32        | 1.832         | 0                  |
| 33        | 1.752         | 0                  |
| 34        | 1.789         | 0                  |
| **Total** | **13.310 ms** | **Avg: 1.664 ms**  |

### Summary

| Metric                             | Regular      | CU-masked     | Ratio     |
|:--|:-:|:-:|:-:|
| Avg kernel duration                | 0.764 ms     | 1.664 ms      | **2.18x** |
| Wall time (8 iters)                | 6.109 ms     | 13.310 ms     | **2.18x** |
| Host launch overhead               | ~6.8 us      | ~6.8 us       | 1.00x     |
| Inter-kernel gaps                  | 0 us         | 0 us          | —         |
| `hipExtStreamCreateWithCUMask`     | —            | 9.3 ms        | one-time  |

**The entire 2.18x slowdown is on-device kernel execution time. Zero host-side or dispatch overhead.**

---

## 5. AG+GEMM Experiments — Post Bug Fix (Tutorial 12)

### Experiment: Comm Engine + CU Ratio Comparison

**Setup:** M=8192, N=11008, K=4096, gemm-iters=8, num-sms=304, 7 repeats, 2 ranks

| Comm Engine | ratio | Comm CUs | Comp CUs | Unmasked (ms) | CU-masked (ms) | Speedup | Old CU-masked |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| copy-engine | 0.3   | 91       | 213      | 6.26          | 10.40          | 0.60x   | 10.7          |
| copy-engine | 0.1   | 30       | 274      | 6.65          | 7.88           | 0.84x   | 10.2          |
| cu-kernel   | 0.3   | 91       | 213      | 7.13          | 11.75          | 0.61x   | **25.8**      |
| cu-kernel   | 0.1   | 30       | 274      | 8.23          | 9.31           | 0.88x   | 9.7           |

**Key changes after bug fix:**
- **cu-kernel 0.3**: 25.8 ms → 11.75 ms — the old 25.8 ms was because the copy kernel ran on the null stream (competing with all CUs + null stream serialization), not the CU-masked comm stream.
- **copy-engine results**: Essentially unchanged (fix only affects cu-kernel path).
- **CU masking still hurts in all cases** — penalty comes from the compute stream, not comm.
- **High variance on CU-masked**: std 4.5–12.2 ms, confirming `hipExtStreamCreateWithCUMask` instability with occasional latency spikes.

### Experiment D: Long-Running Overlapped Workloads (Tutorial 12)

**Setup:** M=16384, N=11008, K=4096, 2 ranks, 7 repeats, copy-engine comm

| gemm-iters | comm-ratio | Unmasked (ms) | CU-masked (ms) | Speedup |
|:-:|:-:|:-:|:-:|:-:|
| 8          | 0.3        | 11.7          | 35.5           | 0.33x   |
| 8          | 0.1        | 12.4          | 17.8           | 0.70x   |
| 16         | 0.1        | 23.2          | 46.9           | 0.50x   |

**Finding:** CU masking penalty *worsens* with longer compute phases. The on-device kernel slowdown compounds over multiple GEMM iterations.

### Experiment 3: Grid-based 70/30 split (no CU masking)

**Setup:** M=8192, N=11008, K=4096, gemm-iters=8, cu-kernel comm, unmasked streams

| GEMM Grid (CUs) | Median (ms) | vs Baseline  |
|:-:|:-:|:-:|
| 304 (100%)       | 7.409       | baseline     |
| 213 (70%)        | 8.645       | 1.17x slower |
| 152 (50%)        | 9.129       | 1.23x slower |

**Finding:** Reducing grid without CU masking costs only 17-23% — far less than CU masking's 2.2x penalty. Grid-based partitioning is a viable alternative.

---

## 6. Communication-Only Benchmarks (Tutorial 15)

Pure p2p transfers between 2 GPUs via rocshmem IPC memory (XGMI). No compute kernel. Isolates CU masking effect on communication.

### Contiguous Transfers: 32 MB x 16 iters = 512 MB total

| Mode      | Stream                  | Grid             | Median (ms) | BW (GB/s) | vs baseline |
|:--|:--|:--|:-:|:-:|:-:|
| DMA       | regular                 | —                | 10.82       | 46.2      | **1.000x**  |
| DMA       | CU-masked 0.3 (91 CUs) | —                | 10.81       | 46.2      | 1.001x      |
| DMA       | CU-masked 0.1 (30 CUs) | —                | 10.80       | 46.3      | 1.001x      |
| CU-kernel | regular                 | full (16384 WGs) | 11.32       | 44.2      | 0.956x      |
| CU-kernel | CU-masked 0.3           | full             | 11.36       | 44.0      | 0.952x      |
| CU-kernel | CU-masked 0.1           | full             | 11.41       | 43.8      | 0.948x      |
| CU-kernel | regular                 | 70% (212 WGs)   | 11.68       | 42.8      | 0.926x      |
| CU-kernel | CU-masked 0.3           | 70%              | 11.65       | 42.9      | 0.928x      |
| CU-kernel | CU-masked 0.1           | 70%              | 11.62       | 43.0      | 0.931x      |

### Chunked Transfers: 16 x 2 MB chunks x 16 iters = 512 MB total

Matches tutorial 12 AG pattern (M_PER_CHUNK=256, K=4096 → 2 MB per chunk, 16 chunks per rank).

| Mode               | Stream                  | Grid           | Median (ms) | BW (GB/s) | vs baseline |
|:--|:--|:--|:-:|:-:|:-:|
| DMA chunked        | regular                 | —              | 13.31       | 37.6      | **1.000x**  |
| DMA chunked        | CU-masked 0.3 (91 CUs) | —              | 13.24       | 37.8      | 1.006x      |
| DMA chunked        | CU-masked 0.1 (30 CUs) | —              | 13.35       | 37.5      | 0.997x      |
| CU-kernel chunked  | regular                 | full           | 11.92       | 41.9      | 1.117x      |
| CU-kernel chunked  | CU-masked 0.3           | full           | 11.95       | 41.9      | 1.114x      |
| CU-kernel chunked  | CU-masked 0.1           | full           | 11.95       | 41.8      | 1.114x      |
| CU-kernel chunked  | regular                 | 70% (212 WGs)  | 12.00       | 41.7      | 1.109x      |
| CU-kernel chunked  | CU-masked 0.3           | 70%            | 12.03       | 41.6      | 1.107x      |
| CU-kernel chunked  | CU-masked 0.1           | 70%            | 12.05       | 41.5      | 1.105x      |

### Contiguous vs Chunked Comparison (same 512 MB total)

| Mode              | Contiguous BW | Chunked BW | Chunking overhead |
|:--|:-:|:-:|:-:|
| DMA regular       | 46.2 GB/s     | 37.6 GB/s  | **23% slower**    |
| CU-kernel regular | 44.2 GB/s     | 41.9 GB/s  | **5% slower**     |

DMA takes a bigger hit from chunking (256 small `hipMemcpyAsync` calls accumulate per-call overhead). CU-kernel handles chunking better since kernel launch overhead is smaller.

### Communication Findings

1. **DMA is completely unaffected by CU masking** — stream type doesn't matter for copy-engine.
2. **CU-kernel comm is XGMI-limited** — 30 CUs vs 304 CUs gives <1% bandwidth difference (~47 GB/s link ceiling).
3. **CU masking has < 1% impact on communication** regardless of chunk pattern, grid size, or CU ratio.
4. **All AG+GEMM CU masking penalty comes from the compute stream**, not communication.

---

## 7. Compute Intensity vs CU Masking Slowdown

Tests whether the WG dispatch counter bug can be triggered in non-GEMM kernels by adding ALU work. Uses a persistent copy kernel (`BLOCK_SIZE=8192`, `num_warps=8`) with a configurable FMA loop after each load: `for _ in range(COMPUTE_ITERS): acc = acc * 0.999 + data * 0.001`. Single GPU, CU-masked stream at 0.3 (91 CUs).

### Varying compute intensity (grid=304 WGs, 32 MB × 16 iters)

| Kernel Type          | COMPUTE_ITERS | Regular (ms) | Masked (ms) | Slowdown  |
|:--|:-:|:-:|:-:|:-:|
| BW-only (load/store) | 0             | 0.47         | 0.47        | **1.00x** |
| Light compute        | 64            | 0.55         | 0.72        | **1.28x** |
| Medium compute       | 256           | 1.51         | 2.22        | **1.47x** |
| Heavy compute        | 1024          | 5.21         | 8.23        | **1.57x** |
| GEMM (Section 4)     | —             | 6.11         | 13.31       | **2.18x** |

### Findings

1. **The WG dispatch bug is proportional to compute density.** Pure BW-limited kernels show zero impact (1.00x) even at 53.9 waves. Adding 64 FMAs/element triggers 1.28x; 1024 FMAs gives 1.57x; GEMM (extreme ALU density) hits 2.18x.

2. **Grid size (wave count) does not affect the slowdown.** The ratio stays constant from 1 wave to 8 waves. The bug triggers even within a single dispatch wave — it is not solely a wave-transition problem. This suggests the dispatch counter mismatch causes stalls within the initial WG scheduling phase itself.

3. **Bandwidth-limited kernels are immune** because WGs spend most time waiting on memory. Dispatch stalls are hidden under memory latency and don't extend the critical path.

4. **The dispatch bug is kernel-agnostic** — it is not specific to GEMM. Any compute-bound kernel dispatched on a CU-masked stream will experience a slowdown proportional to its ALU intensity.

### Grid-Matched Experiment: grid=213, CU mask=213 (Apples to Apples)

**File:** `tutorials/test_grid_matched_intensity.py`

The experiments above used CU mask = 91 CUs (comm CUs from 0.3 ratio), with grid=213 exceeding the mask. This subsection uses CU mask = **213 compute CUs** from the same 0.3 ratio, so grid = mask exactly. Both regular and CU-masked streams launch 213 WGs — the only difference is `hipExtStreamCreateWithCUMask`. This is the definitive apples-to-apples test.

**Setup:** grid=213, CU mask=213 compute CUs (interleaved, 32 MB × 16 iters, single GPU, 3 runs averaged

| Kernel Type          | COMPUTE_ITERS | Regular (ms) | Masked (ms) | Slowdown  |
|:--|:-:|:-:|:-:|:-:|
| BW-only (load/store) | 0             | 0.73         | 0.70        | **0.97x** |
| Light compute        | 64            | 0.71         | 1.16        | **1.62x** |
| Medium compute       | 256           | 1.85         | 3.76        | **2.03x** |
| Heavy compute        | 1024          | 6.65         | 14.25       | **2.15x** |
| GEMM (Section 3)     | —             | 6.07         | 13.44       | **2.21x** |

**Key finding:** This is a single-GPU, single-stream test with no communication. The only difference between "Regular" and "Masked" is whether `hipExtStreamCreateWithCUMask` is used — both launch exactly 213 WGs. On the regular stream, the GPU scheduler distributes 213 WGs across all 304 CUs. On the masked stream, 213 WGs are restricted to 213 specific CUs (interleaved compute CUs from 0.3 ratio). If CU masking worked correctly, these should be **identical** since 213 WGs fit perfectly on 213 CUs.

Instead, the dispatch bug causes **2.15x slowdown** for compute-bound kernels even with a perfect grid-to-CU match. This proves:

1. **The bug triggers regardless of grid vs mask relationship.** Even with grid = mask (213 WGs on 213 CUs), the interleaved CU mask pattern leaves masked-but-physical CU slots within each Shader Engine. The round-robin dispatcher visits all SE slots and stalls when it hits a masked-but-physical CU where no dummy WG is scheduled.

2. **BW-limited kernels remain immune** — dispatch stalls are fully hidden under memory latency.

3. **This is the definitive apples-to-apples evidence** that `hipExtStreamCreateWithCUMask` is broken for compute-bound kernels on MI300X. Same grid size, same kernel, same data — only the stream type differs.

### Varying mask size: Slowdown vs number of masked CUs (Heavy compute, 1024 FMA)

**File:** `tutorials/test_grid_matched_intensity.py`

Sweeps different CU mask sizes and grid configurations to understand how many masked-out CUs affect the dispatch bug severity. All use the same persistent copy kernel with `BLOCK_SIZE=8192`, `num_warps=8`, 32 MB × 16 iters, single GPU.

| Config                         | BW-only | Light (64 FMA) | Medium (256 FMA) | Heavy (1024 FMA) |
|:--|:-:|:-:|:-:|:-:|
| grid=304, mask=91 CUs (30%)   | 0.96x   | 1.10x          | 1.50x            | 1.61x            |
| grid=304, mask=213 CUs (70%)  | 0.97x   | 1.39x          | 2.07x            | 2.17x            |
| grid=304, mask=270 CUs (89%)  | 0.97x   | 1.34x          | 1.46x            | 1.58x            |
| grid=213, mask=213 CUs (70%)  | 0.99x   | 1.60x          | 2.04x            | 2.14x            |
| grid=270, mask=270 CUs (89%)  | 0.98x   | 1.17x          | 1.57x            | 1.64x            |

**Key findings:**

1. **Slowdown is not monotonic with mask size.** Mask=213 (70%, 91 CUs excluded) produces the worst slowdown (~2.17x), worse than both mask=91 (30%, 213 CUs excluded, ~1.61x) and mask=270 (89%, 34 CUs excluded, ~1.58x).

2. **Mask=270 (excluding only 34 CUs) still causes 1.58x slowdown.** Even masking out just 11% of CUs triggers a significant dispatch bug for compute-bound kernels.

3. **Grid-matched vs grid-oversubscribed barely matters.** grid=304/mask=270 (1.58x) vs grid=270/mask=270 (1.64x) are within noise. Similarly grid=304/mask=213 (2.17x) vs grid=213/mask=213 (2.14x). The dispatch bug severity is driven by the **mask pattern**, not grid-to-mask ratio.

4. **The worst case is the interleaved 213-CU mask** — likely because interleaving 91 "holes" across all 8 XCDs × 4 SEs creates the maximum number of masked-but-physical CU slots for the round-robin dispatcher to stall on.

---

## 8. GEMM Kernel Resource Analysis

From rocprofv3 trace data and kernel configuration:

| Parameter            | Value                             |
|:--|:--|
| Tile size            | 256 x 256 x 64 (BLOCK_SIZE_M/N/K) |
| num_warps            | 8 (512 threads/WG)                |
| num_stages           | 2 (double-buffered)               |
| VGPRs/wave           | 120                               |
| AccVGPRs/wave        | 128                               |
| SGPRs/wave           | 64                                |
| LDS                  | 0 bytes (compiler optimized)      |
| Occupancy            | ~1 WG/CU (register-limited)       |
| Total tiles (M=8192) | 32 x 22 = 704                     |

### Tile Wave Analysis

| Available CUs | Waves              | Last wave utilization |
|:-:|:-:|:-:|
| 304 (all)     | 3 (304+304+96)     | 31.6%                 |
| 213 (70%)     | 4 (213+213+213+65) | 30.5%                 |
| 152 (50%)     | 5 (152x4+96)       | 63.2%                 |

### Communication Workload

| Parameter              | Value                          |
|:--|:--|
| Per chunk              | 256 x 4096 x 2 bytes = 2 MB    |
| Chunks per rank        | 16                             |
| Total per remote rank  | 32 MB                          |
| At ~47 GB/s XGMI      | ~0.68 ms for all p2p comm      |
| Copy kernel grid/chunk | 1024 WGs (trivial workgroups)  |

**Optimal split (theory):** ~95% compute / ~5% comm (290 comp / 14 comm). Comm saturates XGMI with even 16 CUs. But **CU masking's 2.2x on-device penalty makes any ratio counterproductive.**

---

## 9. Experiment B: Work-Stealing GEMM + Communication Overlap (Tutorial 13)

Implements work-stealing persistent GEMM using per-XCD atomic tile counters. Runs GEMM at various CU counts with concurrent Triton copy kernel on a separate stream (no CU masking — just grid size control).

**Setup:** gemm-iters=4, comm=128MB x32 iterations, 7 repeats

#### M=8192, N=8192, K=8192

| Mode       |  Grid | CUs | GEMM-only (ms) | w/Comm (ms) | Overhead% |
|:--|--:|--:|--:|--:|--:|
| static     |  4096 | 304 | 12.3           | 14.7        | **19.6%** |
| ws-full    |   304 | 304 | 14.7           | 17.6        | 20.0%     |
| ws-3/4     |   228 | 228 | 16.5           | 18.7        | **13.5%** |
| ws-half    |   152 | 152 | 22.8           | 24.8        | **8.7%**  |
| ws-quarter |    76 |  76 | 38.9           | 41.8        | **7.6%**  |

#### M=16384, N=16384, K=8192

| Mode       |  Grid | CUs | GEMM-only (ms) | w/Comm (ms) | Overhead% |
|:--|--:|--:|--:|--:|--:|
| static     | 16384 | 304 | 51.5           | 54.5        | **5.9%**  |
| ws-full    |   304 | 304 | 59.1           | 63.4        | 7.1%      |
| ws-3/4     |   228 | 228 | 70.4           | 73.3        | **4.1%**  |
| ws-half    |   152 | 152 | 93.9           | 96.0        | **2.2%**  |
| ws-quarter |    76 |  76 | 162.0          | 164.3       | **1.4%**  |

**Findings:**
- Overhead% drops monotonically as fewer CUs are given to GEMM — freeing CUs for concurrent comm reduces contention.
- At ws-quarter: only 1.4% overhead vs 5.9% for static (16K) — a 4x reduction in comm interference.
- Trade-off: absolute GEMM latency increases with fewer CUs (GEMM alone takes longer).
- The static grid cannot adapt; work-stealing naturally handles variable CU availability.

---

## 10. Earlier Results (Tutorials 11, 09)

### Tutorial 11: CU Masking Overlap (Synthetic Kernels)

| Config                                 | Standard | Two Streams | CU-masked  |
|:--|--:|--:|--:|
| comm+compute, block=1024, ratio=0.3    | 0.95ms   | 0.87ms      | 1.51ms     |
| compute+compute, block=1024, ratio=0.3 | 1.43ms   | 1.37ms      | 1.63ms     |
| compute+compute, block=1024, ratio=0.4 | 1.39ms   | 1.37ms      | **4.17ms** |
| compute+compute, block=1024, ratio=0.5 | 1.40ms   | 1.34ms      | 1.49ms     |

Two unmasked streams achieve ~93% overlap naturally. CU masking always adds overhead.


## 11. Conclusions

| Approach                                        | Does it help overlap?                | Why / Why not                                                                    |
|:--|:--|:--|
| **CU masking** (`hipExtStreamCreateWithCUMask`) | **No** — consistently 1.6-2.2x worse | On-device kernel execution is 2.18x slower; NOT dispatch overhead, NOT host-side |
| **Copy-engine comm + CU masking**               | **No** — wasted CUs                  | DMA bypasses CUs; masked "comm CUs" sit idle                                    |
| **CU-kernel comm + CU masking**                 | **No** — still worse                 | Comm is XGMI-limited (<1% CU masking impact); compute stream penalty dominates  |
| **Static NUM_SMS reduction**                    | **No** — marginal                    | Not adaptive; same dispatch constraints                                          |
| **Work-stealing GEMM** (fewer CUs)             | **Yes** — lower overhead%            | Atomic counters adapt to available CUs; freed CUs absorb comm load              |
| **Two unmasked streams** (HW scheduler)         | **Yes** — good overlap               | GPU's native round-robin WG dispatch achieves ~93% overlap for free             |
| **Grid-based partitioning** (no CU mask)        | **Partial** — 17% cost               | Acceptable overhead vs CU masking's 120%+ penalty                               |

### Practical Recommendations

1. **Don't use CU masking for comm/compute overlap on MI300X** — `hipExtStreamCreateWithCUMask` causes a fundamental 2.18x on-device kernel slowdown that no CU ratio can overcome.
2. **Use work-stealing GEMM** when you need deterministic CU partitioning — it achieves the goal of "use fewer CUs for compute" without triggering the CU mask penalty.
3. **Two unmasked streams** are sufficient for most overlap scenarios — the HW scheduler handles CU sharing well.
4. **Grid-based partitioning** (reducing NUM_SMS without CU masking) costs only 17-23% vs CU masking's 120%+ penalty — a practical alternative if work-stealing is too complex.
5. **Communication needs very few CUs** — XGMI saturates with ~16 CUs. Even 10% ratio (30 CUs) gives identical bandwidth to 304 CUs.

### Remaining Experiments

- **Experiment C (rocshmem device-side comm + GEMM):** Test with Iris `store`/`load` operations that execute on CUs via rocshmem. This is the real-world use case for Triton-distributed and may reveal different contention patterns than the synthetic Triton copy kernel.
- **ROCm :** The 2.18x on-device slowdown with `hipExtStreamCreateWithCUMask` should be reported as a potential ROCm runtime bug — the kernel binary and resources are identical, only the stream differs.

---

## 12. Files Created

| File                                   | Description                                                                                |
|:--|:--|
| `tutorials/12-cu-masked-ag-gemm.py`    | AG+GEMM with CU masking (bug fixed: cu_copy_kernel now uses ag_stream)                     |
| `tutorials/15-comm-only-benchmark.py`  | Comm-only p2p benchmark: DMA vs CU-kernel, contiguous vs chunked, with/without CU masking  |
| `tutorials/16-gemm-only-cu-mask.py`    | Standalone GEMM-only CU mask benchmark — no communication, single GPU                      |
| `tutorials/test_grid_matched_intensity.py` | Grid-matched compute intensity sweep: grid=213, mask=213 (apples-to-apples CU masking test) |

---

## 13. Git Log

```
677aea3 Experiment B: work-stealing GEMM shows better overlap tolerance
7d51654 Add CU-consuming communication mode (cu-kernel) to tutorial 12
5f8a7f4 Experiment D: long-running overlapped workloads with CU masking
51503db Add CU masking benchmarks and AG+GEMM overlap experiments
```
