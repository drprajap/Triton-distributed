# CU Masking Methodology (Stage 1)

This document defines the canonical workflow for CU masking experiments in this branch.

## Canonical entrypoint

Use a single runner:

- `scripts/cu_experiments.py`

Do not run stage-1 analysis directly from ad-hoc shell scripts. Existing tutorial scripts remain as implementation backends and compatibility references, but execution should be routed through the canonical runner so grid, masks, message sizes, and repeats stay consistent.

## Scope

Stage-1 includes:

1. Compute-only characterization (`tutorials/16-gemm-only-cu-mask.py`)
2. Comm-only characterization (`tutorials/15-comm-only-benchmark.py`)
3. AG+GEMM characterization (`tutorials/12-cu-masked-ag-gemm.py`)
4. Grid-matched intensity sweep (`tutorials/test_grid_matched_intensity.py`)

Stage-2 backlog (deferred):

- Software masking (kernel early-return based masking)
- Persistent occupancy baseline
- Additional exploratory mask-pattern studies

## Fixed configuration

Fixed presets and methodology defaults are captured in:

- `configs/cu_experiments.json`

The config stores:

- baseline warmup/repeat policy
- fixed grid and mask intent
- fixed message/chunk sizes
- environment snapshot commands for reproducibility
- experiment matrix IDs used in stakeholder reporting

## Running experiments

List configured presets:

```bash
python scripts/cu_experiments.py list
```

Run entire stage-1 matrix:

```bash
python scripts/cu_experiments.py run-all
```

Run a single scenario:

```bash
python scripts/cu_experiments.py ag-gemm
python scripts/cu_experiments.py comm-only
python scripts/cu_experiments.py compute-only
python scripts/cu_experiments.py intensity-sweep
```

Run specific preset IDs:

```bash
python scripts/cu_experiments.py run-all --experiment-id ag_gemm_cukernel_fixed --experiment-id compute_only_gemm
```

## Profiling mode (default off)

Profiling is optional and disabled by default:

- `--profile none` (default)
- `--profile kernel-trace` (rocprofv3 kernel and HIP traces)
- `--profile perfetto` (rocprofv3 perfetto plugin timeline)

Examples:

```bash
python scripts/cu_experiments.py ag-gemm --profile kernel-trace
python scripts/cu_experiments.py run-all --profile perfetto
```

For profiled runs, artifacts are written under:

- `results/<run_id>/experiments/<experiment_id>/profiles/`

The runner also writes a `profile_summary.json` with:

- average kernel execution time (when profile CSV columns are available)
- simple overlap ratio estimate between two busiest streams (when start/end/stream columns are available)

## Output schema

Each run writes:

- `manifest.json` (command + config + metadata + status)
- `env_snapshot.json` (machine/runtime capture)
- `results.jsonl` (normalized rows)
- `results.csv` (tabular report input)
- `summary.md` (slide-friendly summary table)

Optional plots:

```bash
python scripts/plot_cu_experiments.py results/<run_id>/results.csv
```

Generated figures include:

- latency comparison bars
- comm bandwidth comparison bars
- compute intensity slowdown curve
