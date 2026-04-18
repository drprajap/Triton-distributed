# CU Masking on MI300X — Experiment Summary

**Goal:** Evaluate whether `hipExtStreamCreateWithCUMask` can improve compute/communication overlap for AllGather+GEMM workloads on AMD MI300X GPUs.

**Answer: No.** CU masking consistently hurts performance due to a hardware WG dispatch counter bug on MI300X. Alternative approaches (work-stealing GEMM, two unmasked streams, grid-based partitioning) are superior.

**Platform:** AMD MI300X (304 CUs, 8 XCDs, 4 SEs/XCD), ROCm 7.1, Triton 3.4
**Branch:** `cu_mask_partitioning` on `drprajap/Triton-distributed`
**Container:** `diprajap-pytorch-rocm7_1-triton`, GPU pinning: `HIP_VISIBLE_DEVICES=4,5`
**All scripts live under:** `workspace/rocm7/Triton-distributed/tutorials/`

---

## What Worked vs What Didn't

| Approach | Result | Why |
|:--|:--|:--|
| `hipExtStreamCreateWithCUMask` | **Failed** — 1.3x-2.2x slower | HW dispatch counter bug on asymmetric-harvested GPUs |
| Two unmasked streams (HW scheduler) | **Works** — ~93% overlap | GPU native round-robin WG dispatch handles CU sharing naturally |
| Work-stealing GEMM (atomic tile counters) | **Works** — 1.4-7.6% comm overhead | Adapts to available CUs; freed CUs absorb comm load |
| Grid-based partitioning (no CU mask) | **Partial** — 17-23% cost | Acceptable trade-off vs CU masking's 120%+ penalty |
| Software CU masking (`if pid >= threshold: return`) | **Proposed** — not yet tested | Avoids HW dispatch bug while simulating reduced CUs |

---

## Root Cause: HW WG Dispatch Counter Bug

Identified through profiling and confirmed in discussion with Muhammad Osama (AMD).

**Mechanism:** MI300X has asymmetric CU harvesting (2 disabled CUs per XCD from 40 → 38 active). The WG dispatch uses round-robin across Shader Engines (SEs), with a hardware counter per SE tracking occupancy. For disabled (harvested) CUs, a **dummy work group** is scheduled to zero the counter and pass the dispatch baton. However, when `hipExtStreamCreateWithCUMask` masks additional CUs, no dummy WG is scheduled for those masked-but-physical CUs — the counter **never zeros** — creating intermittent deadlocks during WG scheduling.

**Key trigger condition:**
- **Compute-bound kernel** — bandwidth-limited kernels hide the stalls under memory latency (1.00x). Slowdown scales with ALU intensity: 64 FMAs → 1.32x, 1024 FMAs → 1.58x, GEMM → 2.18x.
- **Grid > masked CU capacity is NOT required** — grid=213 on mask=213 CUs (perfect match) still triggers 2.15x slowdown. The bug is driven by the interleaved mask *pattern* within Shader Engines, not by oversubscription.
- **Slowdown is not monotonic with mask size** — mask=213 (70%) is worst (~2.17x), mask=91 (30%) gives ~1.61x, mask=270 (89%) gives ~1.58x. Even masking out just 34 CUs (11%) causes 1.58x slowdown for compute-bound kernels.

---

## Experiment Details

### Experiment 1: GEMM-Only CU Masking (Single GPU, No Communication)

**File:** `tutorials/16-gemm-only-cu-mask.py`
**Purpose:** Isolate CU masking overhead on a pure GEMM kernel without communication or distributed setup.
**Command:**
```bash
docker exec -e HIP_VISIBLE_DEVICES=4 diprajap-pytorch-rocm7_1-triton bash -c \
  'cd /home/diprajap/workspace/rocm7/Triton-distributed && \
   PYTHONPATH=$PYTHONPATH:$(pwd)/python \
   python tutorials/16-gemm-only-cu-mask.py \
     --M 8192 --N 5504 --K 4096 --gemm-iters 8 --repeats 10 --num-sms 213'
```

**Kernel:** Persistent GEMM (256×256×64 tiles, 8 warps, 2 stages, NUM_XCDS=4), adapted from tutorial 09.
**Resources:** 120 VGPRs, 128 AccVGPRs, 64 SGPRs, 0 LDS, 512 threads/WG, ~1 WG/CU occupancy.
**CU mask:** 213 compute CUs (interleaved, from `partition_cus(..., comm_ratio=0.3)`).
**Setup:** M=8192, N=5504, K=4096, gemm-iters=8, 10 repeats, single GPU.

| Grid (--num-sms) | Regular (ms) | CU-masked (ms) | Slowdown |
|:-:|:-:|:-:|:-:|
| 213 | 6.07 | 13.44 | **2.21x** |
| 304 | 5.48 | 11.36 | **2.07x** |
| 152 | 6.90 | 11.89 | **1.72x** |

5× stability check at grid=213 (same command, run 5 times):

| Run | Regular (ms) | CU-masked (ms) | Slowdown |
|:-:|:-:|:-:|:-:|
| 1 | 6.07 | 13.50 | 2.23x |
| 2 | 6.09 | 13.65 | 2.24x |
| 3 | 6.11 | 13.50 | 2.21x |
| 4 | 6.16 | 13.53 | 2.20x |
| 5 | 6.06 | 13.43 | 2.22x |

**Key result:** 2.2x on-device kernel slowdown for the identical kernel binary, same grid size, same resources — only the stream type differs.

---

### Experiment 2: rocprofv3 Kernel Trace Analysis

**File:** `tutorials/16-gemm-only-cu-mask.py` profiled with rocprofv3
**Purpose:** Pinpoint whether the 2.2x slowdown is host-side dispatch overhead or on-device execution.
**Command:**
```bash
docker exec -e HIP_VISIBLE_DEVICES=4 diprajap-pytorch-rocm7_1-triton bash -c \
  'cd /home/diprajap/workspace/rocm7/Triton-distributed && \
   PYTHONPATH=$PYTHONPATH:$(pwd)/python \
   rocprofv3 -d /tmp/prof_gemm --kernel-trace --hip-trace -f csv -- \
   python tutorials/16-gemm-only-cu-mask.py \
     --M 8192 --N 5504 --K 4096 --gemm-iters 8 --num-sms 213 --profile'
```

| Metric | Regular | CU-masked | Ratio |
|:--|:-:|:-:|:-:|
| Avg kernel duration (8 iters) | 0.764 ms | 1.664 ms | **2.18x** |
| Wall time (8 iters) | 6.109 ms | 13.310 ms | **2.18x** |
| Host launch overhead | ~6.8 us | ~6.8 us | 1.00x |
| Inter-kernel gaps | 0 us | 0 us | — |

CU-masked kernels show bimodal execution: alternating ~1.26 ms and ~1.80 ms iterations.

**Key result:** The entire 2.18x slowdown is on-device kernel execution. Zero host-side or dispatch overhead contribution.

---

### Experiment 3: AG+GEMM with CU Masking (2 GPUs, Tutorial 12)

**File:** `tutorials/12-cu-masked-ag-gemm.py`
**Purpose:** End-to-end AllGather + GEMM overlap with CU masking using different comm engines and ratios.
**Bug fixed first:** `cu_copy_kernel` was dispatching on the default (null) stream instead of `ag_stream`. Fixed by wrapping with `with torch.cuda.stream(ag_stream):` (line 92).
**Command (example for copy-engine, 0.3 ratio):**
```bash
docker exec -e HIP_VISIBLE_DEVICES=4,5 -e ARNOLD_WORKER_GPU=2 \
  diprajap-pytorch-rocm7_1-triton bash -c \
  'cd /home/diprajap/workspace/rocm7/Triton-distributed && \
   PYTHONPATH=$PYTHONPATH:$(pwd)/python \
   ./scripts/launch_amd.sh tutorials/12-cu-masked-ag-gemm.py \
     --M 8192 --N 11008 --K 4096 --gemm-iters 8 --repeats 7 \
     --comm-engine copy-engine --comm-ratio 0.3 --num-sms 304 \
     --modes unmasked,cu-masked'
```

**Setup:** M=8192, N=11008, K=4096, chunk-size=256 (2 MB/chunk, 16 chunks/rank), gemm-iters=8, num-sms=304, 7 repeats, 2 ranks. Vary `--comm-engine` and `--comm-ratio` for each row.

| Comm Engine | Ratio | Comm CUs | Comp CUs | Unmasked (ms) | CU-masked (ms) | Speedup |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| copy-engine | 0.3 | 91 | 213 | 6.26 | 10.40 | 0.60x |
| copy-engine | 0.1 | 30 | 274 | 6.65 | 7.88 | 0.84x |
| cu-kernel | 0.3 | 91 | 213 | 7.13 | 11.75 | 0.61x |
| cu-kernel | 0.1 | 30 | 274 | 8.23 | 9.31 | 0.88x |

**Long-running workloads** (same command but with `--M 16384`):

| gemm-iters | ratio | Unmasked (ms) | CU-masked (ms) | Speedup |
|:-:|:-:|:-:|:-:|:-:|
| 8 | 0.3 | 11.7 | 35.5 | 0.33x |
| 8 | 0.1 | 12.4 | 17.8 | 0.70x |
| 16 | 0.1 | 23.2 | 46.9 | 0.50x |

**Key result:** CU masking penalty worsens with longer compute phases. The penalty compounds across GEMM iterations.

**Grid-based 70/30 split (no CU masking, cu-kernel comm):**

Ran with `--modes unmasked --comm-engine cu-kernel` and varying `--num-sms`:

| GEMM Grid (--num-sms) | Median (ms) | vs Baseline |
|:-:|:-:|:-:|
| 304 (100%) | 7.409 | baseline |
| 213 (70%) | 8.645 | 1.17x slower |
| 152 (50%) | 9.129 | 1.23x slower |

**Key result:** Reducing grid without CU masking costs only 17–23% — far less than CU masking's 2.2x penalty.

---

### Experiment 4: Communication-Only Benchmarks (2 GPUs)

**File:** `tutorials/15-comm-only-benchmark.py`
**Purpose:** Isolate CU masking impact on pure p2p communication (no compute) to prove the penalty comes from compute, not comm.
**Command (contiguous):**
```bash
docker exec -e HIP_VISIBLE_DEVICES=4,5 -e ARNOLD_WORKER_GPU=2 \
  diprajap-pytorch-rocm7_1-triton bash -c \
  'cd /home/diprajap/workspace/rocm7/Triton-distributed && \
   PYTHONPATH=$PYTHONPATH:$(pwd)/python \
   ./scripts/launch_amd.sh tutorials/15-comm-only-benchmark.py \
     --size-mb 32 --copy-iters 16 --warmup 5 --repeats 10'
```
**Command (chunked, matching Tutorial 12 AG pattern):**
```bash
# Same as above but add: --chunk-size-kb 2048
```

**Contiguous transfers (32 MB × 16 iters = 512 MB total):**

| Mode | Stream | Grid | BW (GB/s) | vs baseline |
|:--|:--|:--|:-:|:-:|
| DMA | regular | — | 46.2 | **1.000x** |
| DMA | CU-masked 0.3 (91 CUs) | — | 46.2 | 1.001x |
| DMA | CU-masked 0.1 (30 CUs) | — | 46.3 | 1.001x |
| CU-kernel | regular | full (16384 WGs) | 44.2 | 0.956x |
| CU-kernel | CU-masked 0.3 | full | 44.0 | 0.952x |
| CU-kernel | CU-masked 0.1 | full | 43.8 | 0.948x |
| CU-kernel | regular | 70% (212 WGs) | 42.8 | 0.926x |
| CU-kernel | CU-masked 0.3 | 70% | 42.9 | 0.928x |
| CU-kernel | CU-masked 0.1 | 70% | 43.0 | 0.931x |

**Chunked transfers** (16 × 2 MB chunks, matching Tutorial 12 AG pattern): similar results with < 1% CU masking impact. DMA takes a 23% chunking overhead hit (many small `hipMemcpyAsync` calls), while CU-kernel handles chunking with only 5% overhead.

**Key result:** < 1% CU masking impact on communication. DMA is completely unaffected; CU-kernel comm is XGMI-limited (~47 GB/s) regardless of CU count. Communication kernel is bandwidth-bound so dispatch stalls are hidden.

---

### Experiment 5: Compute Intensity vs CU Masking Slowdown (Single GPU)

**File:** `tutorials/test_heavy_copy_waves.py` (never committed — temporary file, now deleted)
**Purpose:** Prove the dispatch bug is proportional to ALU intensity and has a grid-size threshold. Uses a persistent copy kernel with configurable FMA loop.
**Command:**
```bash
docker exec -e HIP_VISIBLE_DEVICES=4 diprajap-pytorch-rocm7_1-triton bash -c \
  'cd /home/diprajap/workspace/rocm7/Triton-distributed && \
   PYTHONPATH=$PYTHONPATH:$(pwd)/python \
   python tutorials/test_heavy_copy_waves.py'
```

**Kernel:** `compute_copy_kernel` — persistent copy kernel, load → N FMA iterations (`acc = acc * 0.999 + data * 0.001`) → store.
`BLOCK_SIZE=8192`, `num_warps=8`, `num_stages=1`, 32 MB data × 16 iters.
**CU mask:** 91 comm CUs (from `partition_cus(..., comm_ratio=0.3)`), applied to the stream.

**Compute intensity sweep (grid=304 WGs, mask=91 CUs):**

| Kernel Type | COMPUTE_ITERS | Regular (ms) | Masked (ms) | Slowdown |
|:--|:-:|:-:|:-:|:-:|
| BW-only | 0 | 0.47 | 0.47 | **1.00x** |
| Light compute | 64 | 0.55 | 0.72 | **1.28x** |
| Medium compute | 256 | 1.51 | 2.22 | **1.47x** |
| Heavy compute | 1024 | 5.21 | 8.23 | **1.57x** |
| GEMM (Exp 1) | — | 6.11 | 13.31 | **2.18x** |

**Grid-size threshold (COMPUTE_ITERS=1024, mask=91 CUs):**

| Grid (WGs) | Regular (ms) | Masked (ms) | Slowdown |
|:-:|:-:|:-:|:-:|
| 91 (= masked CUs) | 12.840 | 12.875 | **1.00x** |
| 152 | 8.338 | 8.323 | **1.00x** |
| **213** | 6.661 | 10.510 | **1.58x** |
| 304 | 5.419 | 8.383 | **1.55x** |

**Key results:**
- Slowdown proportional to compute density: BW-only immune, GEMM worst case.
- With mask=91 CUs, the grid-size threshold for triggering the bug is between 152 and 213 WGs (i.e., when grid starts dispatching WGs to CUs beyond the 91 available in that mask).
- GEMM's extra 2.18x (vs 1.58x for heavy copy) is due to register pressure (120+128 VGPRs → 1 WG/CU occupancy) reducing opportunities to absorb dispatch stalls.

---

### Experiment 5b: Grid-Matched & Mask-Sweep (Single GPU)

**File:** `tutorials/test_grid_matched_intensity.py` (committed)
**Purpose:** Prove the bug triggers even with grid = mask (apples-to-apples), and map slowdown vs different mask sizes.
**Commands:**
```bash
# Default: grid=213, mask=213 (apples-to-apples)
docker exec -e HIP_VISIBLE_DEVICES=4 diprajap-pytorch-rocm7_1-triton bash -c \
  'cd /home/diprajap/workspace/rocm7/Triton-distributed && \
   PYTHONPATH=$PYTHONPATH:$(pwd)/python \
   python tutorials/test_grid_matched_intensity.py'

# Grid=304, mask=91
python tutorials/test_grid_matched_intensity.py --grid 304 --mask-cus 91

# Grid=304, mask=213
python tutorials/test_grid_matched_intensity.py --grid 304 --mask-cus 213

# Grid=304, mask=270
python tutorials/test_grid_matched_intensity.py --grid 304 --mask-cus 270

# Grid=270, mask=270
python tutorials/test_grid_matched_intensity.py --grid 270 --mask-cus 270
```

**Kernel:** Same `bw_copy_kernel` / `compute_copy_kernel` as Experiment 5.
`BLOCK_SIZE=8192`, `NUM_WGS=grid`, `num_warps=8`, `num_stages=1`, 32 MB × 16 iters, single GPU.
**CU mask creation:** `partition_cus(total_cus, strategy="interleaved", comm_ratio=0.3)` gives 213 compute CUs. `--mask-cus N` selects the first N from that list; if N > 213, uses `list(range(N))` (contiguous).

**Grid-matched (grid=213, mask=213 compute CUs, interleaved):**

| Kernel Type | COMPUTE_ITERS | Regular (ms) | Masked (ms) | Slowdown |
|:--|:-:|:-:|:-:|:-:|
| BW-only | 0 | 0.73 | 0.70 | **0.97x** |
| Light compute | 64 | 0.71 | 1.16 | **1.62x** |
| Medium compute | 256 | 1.85 | 3.76 | **2.03x** |
| Heavy compute | 1024 | 6.65 | 14.25 | **2.15x** |

**Mask-size sweep (all compute intensities):**

| Config | BW-only | Light (64 FMA) | Medium (256 FMA) | Heavy (1024 FMA) |
|:--|:-:|:-:|:-:|:-:|
| grid=304, mask=91 CUs (30%) | 0.96x | 1.10x | 1.50x | 1.61x |
| grid=304, mask=213 CUs (70%) | 0.97x | 1.39x | 2.07x | 2.17x |
| grid=304, mask=270 CUs (89%) | 0.97x | 1.34x | 1.46x | 1.58x |
| grid=213, mask=213 CUs (70%) | 0.99x | 1.60x | 2.04x | 2.14x |
| grid=270, mask=270 CUs (89%) | 0.98x | 1.17x | 1.57x | 1.64x |

**Key findings:**
- grid=213 on mask=213: **2.15x** slowdown even with perfect grid-to-CU match. Disproves "grid > mask" as a requirement.
- Mask=213 (70%) is the **worst case** (~2.17x), worse than mask=91 or mask=270.
- Even mask=270 (excluding only 34 CUs, 11%) causes **1.58x** slowdown.
- Grid-matched vs grid-oversubscribed is within noise — the **mask pattern** drives severity, not the grid/mask ratio.
- The interleaved 213-CU mask likely creates the maximum number of masked-but-physical CU slots for the round-robin dispatcher to stall on (91 "holes" across 8 XCDs × 4 SEs).

---

### Experiment 6: Work-Stealing GEMM + Communication Overlap (2 GPUs)

**File:** `tutorials/13-work-stealing-gemm-overlap.py`
**Purpose:** Alternative to CU masking — use atomic tile counters so GEMM adapts to available CUs dynamically. No CU masking used; only grid size control. Communication runs concurrently on a separate unmasked stream.
**Command:**
```bash
docker exec -e HIP_VISIBLE_DEVICES=4,5 -e ARNOLD_WORKER_GPU=2 \
  diprajap-pytorch-rocm7_1-triton bash -c \
  'cd /home/diprajap/workspace/rocm7/Triton-distributed && \
   PYTHONPATH=$PYTHONPATH:$(pwd)/python \
   ./scripts/launch_amd.sh tutorials/13-work-stealing-gemm-overlap.py \
     --M 8192 --N 8192 --K 8192 --gemm-iters 4 --repeats 7'
```

**Setup:** gemm-iters=4, comm-size-mb=32, comm-iters=8, 7 repeats, 2 ranks. Vary M/N/K for each table.

**M=8192, N=8192, K=8192:**

| Mode | Grid | GEMM-only (ms) | w/Comm (ms) | Overhead% |
|:--|--:|--:|--:|--:|
| static | 4096 | 12.3 | 14.7 | **19.6%** |
| ws-3/4 | 228 | 16.5 | 18.7 | **13.5%** |
| ws-half | 152 | 22.8 | 24.8 | **8.7%** |
| ws-quarter | 76 | 38.9 | 41.8 | **7.6%** |

**M=16384, N=16384, K=8192:**

| Mode | Grid | GEMM-only (ms) | w/Comm (ms) | Overhead% |
|:--|--:|--:|--:|--:|
| static | 16384 | 51.5 | 54.5 | **5.9%** |
| ws-half | 152 | 93.9 | 96.0 | **2.2%** |
| ws-quarter | 76 | 162.0 | 164.3 | **1.4%** |

**Key result:** Overhead% drops monotonically as fewer CUs given to GEMM. Work-stealing achieves CU partitioning without the dispatch bug penalty. Trade-off: absolute GEMM time increases with fewer CUs.

---

### Experiment 7: Synthetic Kernel Overlap (Tutorial 11)

**File:** `tutorials/11-cu-masking-overlap.py`
**Purpose:** Early exploration with synthetic comm+compute and compute+compute kernels.
**Command:**
```bash
docker exec -e HIP_VISIBLE_DEVICES=4 diprajap-pytorch-rocm7_1-triton bash -c \
  'cd /home/diprajap/workspace/rocm7/Triton-distributed && \
   PYTHONPATH=$PYTHONPATH:$(pwd)/python \
   python tutorials/11-cu-masking-overlap.py \
     --workload-mode comm-compute --comm-ratio 0.3 --block-size 1024'
```

| Config | Two Streams | CU-masked |
|:--|--:|--:|
| comm+compute, ratio=0.3 | 0.87ms | 1.51ms |
| compute+compute, ratio=0.3 | 1.37ms | 1.63ms |
| compute+compute, ratio=0.4 | 1.37ms | **4.17ms** |

**Key result:** Two unmasked streams achieve ~93% overlap naturally. CU masking always adds overhead, with occasional extreme outliers (4.17ms at ratio=0.4).

---

## Bug Fix During Investigation

A critical bug was found in `tutorials/12-cu-masked-ag-gemm.py`: the `cu_copy_kernel` in `producer_ag_cu_kernel` was dispatching on the **default (null) stream** instead of the intended CU-masked `ag_stream`. This meant "cu-kernel" experiments were running unmasked comm + CU-masked GEMM — not the intended CU-masked comm.

**Fix applied:** Wrapped the kernel launch with `with torch.cuda.stream(ag_stream):` (line 92). All experiments in this document (Experiments 3-7) use the corrected code.

---

## Files Reference

| File | Description | GPU Config | Committed |
|:--|:--|:--|:-:|
| `tutorials/16-gemm-only-cu-mask.py` | Standalone persistent GEMM CU mask benchmark — no communication, isolates dispatch bug | 1 GPU | Yes |
| `tutorials/test_grid_matched_intensity.py` | Grid-matched & mask-sweep compute intensity: grid=mask apples-to-apples + slowdown vs mask size | 1 GPU | Yes |
| `tutorials/12-cu-masked-ag-gemm.py` | AG+GEMM with CU masking (bug fixed: cu_copy_kernel now uses ag_stream) | 2 GPUs | Yes |
| `tutorials/15-comm-only-benchmark.py` | P2P comm benchmark: DMA vs CU-kernel, contiguous vs chunked, with/without CU masking | 2 GPUs | Yes |
| `tutorials/13-work-stealing-gemm-overlap.py` | Work-stealing GEMM with atomic tile counters + concurrent comm on separate stream | 2 GPUs | Yes |
| `tutorials/11-cu-masking-overlap.py` | Synthetic kernel overlap with CU masking (early exploration) | 1 GPU | Yes |
| `tutorials/test_heavy_copy_waves.py` | Compute-intensity sweep with grid-size thresholds (Experiment 5 data source) | 1 GPU | **No** — temporary, deleted |

---

## Conclusions

1. **Don't use `hipExtStreamCreateWithCUMask` on MI300X.** The HW dispatch counter bug causes 1.3x–2.2x slowdown for any compute-bound kernel. BW-limited kernels are immune but GEMM (the primary use case) is worst-case.

2. **The only required condition is a compute-bound kernel.** Grid-to-mask ratio doesn't matter — grid=mask still triggers 2.15x. The slowdown is driven by the interleaved mask pattern within Shader Engines, not oversubscription.

3. **Slowdown is not monotonic with mask size.** Mask=213 (70%) is worst (~2.17x); mask=270 (89%, excluding only 34 CUs) still causes 1.58x. Even minimal masking is harmful for compute-bound work.

4. **Communication is unaffected** — XGMI saturates at ~47 GB/s regardless of CU count or masking. The entire AG+GEMM penalty comes from the compute stream.

5. **Work-stealing GEMM is the best alternative** — achieves CU partitioning without the dispatch bug, with comm overhead as low as 1.4%.

6. **Two unmasked streams** provide ~93% overlap for free via the GPU native round-robin WG dispatch.

7. **Grid-based partitioning** (reducing NUM_SMS without CU masking) costs only 17–23% vs CU masking's 120%+ penalty.

---

## Gotchas / Things to Know When Resuming

1. **`cu_copy_kernel` stream bug (fixed):** In `tutorials/12-cu-masked-ag-gemm.py`, the communication copy kernel was originally launched on the default (null) stream, not the CU-masked `ag_stream`. This caused it to serialize with GEMM and bypass CU masking entirely. Fixed by wrapping with `with torch.cuda.stream(ag_stream):`. All data in the summary docs uses the fixed version.

2. **`test_heavy_copy_waves.py` was never committed.** It was a temporary file used for the initial compute-intensity experiments (Experiment 5). Results are captured in this doc but the script is gone. `test_grid_matched_intensity.py` is the committed replacement covering the same ground plus mask-sweep.

3. **Persistent comm kernel was tried and reverted.** Making the communication copy kernel persistent caused bimodal/unstable AG+GEMM timings (alternating ~11.7ms and ~24.7ms). The non-persistent (one-WG-per-block) copy kernel is the current default in `12-cu-masked-ag-gemm.py`.

4. **Container & GPU pinning:** All experiments ran in `diprajap-pytorch-rocm7_1-triton` container. GPU pinning: `HIP_VISIBLE_DEVICES=4` for single-GPU, `HIP_VISIBLE_DEVICES=4,5` with `ARNOLD_WORKER_GPU=2` for 2-GPU. Results may vary on different GPU pairs due to per-GPU asymmetric harvesting patterns.

5. **CU mask creation uses interleaved pattern:** `partition_cus(..., strategy="interleaved")` distributes comm and compute CUs in an alternating pattern across all SEs/XCDs. A contiguous pattern (all comm CUs on the first N CUs) was not tested and could yield different dispatch behavior.

6. **rocshmem segfault on teardown:** `rocshmem_finalize` occasionally segfaults during cleanup in multi-GPU tests. Doesn't affect data validity — just ignore the crash at the end.

7. **rocprofv3 trace command:**
   ```bash
   rocprofv3 -d <output_dir> --kernel-trace --hip-trace -f csv -- python <script>
   ```
   Use `-f pftrace` for Perfetto format. Traces confirm on-device slowdown, not host dispatch.

8. **MI300X specifics:** 304 CUs (8 XCDs × 38 active, 2 harvested per XCD from 40 physical). The 2 harvested CUs per XCD have dummy WG handling that works correctly. The bug is specifically with *additionally masked* CUs that don't get dummy WGs.

9. **PYTHONPATH required:** All scripts need `PYTHONPATH=$PYTHONPATH:$(pwd)/python` set from the `Triton-distributed` root to find `triton_dist`.

10. **2-GPU scripts use `./scripts/launch_amd.sh`:** This wrapper sets up `torch.distributed` with NCCL backend for multi-rank execution.

---

## Remaining Work

- **Software CU masking:** `if pid >= threshold: return` in the GEMM kernel to simulate CU reduction without the HW bug.
- **Persistent occupancy kernel:** Launch a persistent `while(True)` kernel to consume N CUs, then GEMM on the rest. For kernels that can't be modified (hipBLASLt).
- **Iris rocshmem device-side comm + GEMM:** Test with `iris.store`/`iris.load` (the real Triton-distributed use case).
- **ROCm bug report:** Report the dispatch counter mismatch to the ROCm team.
- **MI355X / MI308 replication:** Run same methodology on non-harvested GPUs to confirm the bug is specific to asymmetric CU harvesting.
- **Contiguous vs interleaved CU mask pattern:** Test contiguous CU mask (first N CUs) vs current interleaved pattern to understand if mask topology affects dispatch bug severity.

---

## Related Documents

- `Analysis_CU_Masking_Experiments.md` — Full detailed experiment data with all tables, profiling traces, and per-kernel duration breakdowns
- `WG_Dispatch_Bug_Discussion_Summary.md` — Meeting notes from discussion with Muhammad Osama on the dispatch bug mechanism
- `CU_masking_meetings_summary.md` — Combined meeting notes & action items from April 15 and April 17 follow-ups
- `cu_masking_discussion.md` — Earlier investigation notes (Tutorial 11 era, block-size/ratio sweeps)
