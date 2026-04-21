#!/usr/bin/env python3
"""Generate static comparison plots from consolidated CU experiment results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CU masking experiment results.")
    parser.add_argument("results_csv", type=Path, help="Path to consolidated results.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: alongside CSV)")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def to_float(value: str) -> float | None:
    if value in ("", "-", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.strip().split())


def median_for(rows: list[dict[str, Any]], experiment_id: str, mode: str) -> float | None:
    mode_norm = normalize_text(mode)
    for row in rows:
        if row.get("experiment_id") != experiment_id:
            continue
        if normalize_text(row.get("mode")) != mode_norm:
            continue
        return to_float(row.get("median_ms"))
    return None


def save_latency_bars(rows: list[dict[str, Any]], output_dir: Path) -> None:
    latency_rows = [
        row for row in rows
        if row.get("median_ms") not in ("", "-", None)
        and row.get("scenario") in {"ag-gemm", "compute-only", "comm-only"}
    ]
    if not latency_rows:
        return

    labels = [f"{row.get('experiment_id')}::{row.get('mode')}" for row in latency_rows]
    values = [to_float(row.get("median_ms")) or 0.0 for row in latency_rows]
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.ylabel("Median latency (ms)")
    plt.title("CU masking stage-1 latency comparison")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.png", dpi=200)
    plt.savefig(output_dir / "latency_comparison.svg")
    plt.close()


def save_headline_latency(rows: list[dict[str, Any]], output_dir: Path) -> None:
    scenarios = [
        ("AG+GEMM copy-engine", "ag_gemm_copy_engine_fixed", "unmasked", "cu-masked"),
        ("AG+GEMM cu-kernel", "ag_gemm_cukernel_fixed", "unmasked", "cu-masked"),
        ("Compute-only GEMM", "compute_only_gemm", "Regular stream", "CU-masked stream (213 CUs)"),
    ]

    labels: list[str] = []
    unmasked_vals: list[float] = []
    masked_vals: list[float] = []
    for label, experiment_id, unmasked_mode, masked_mode in scenarios:
        unmasked = median_for(rows, experiment_id, unmasked_mode)
        masked = median_for(rows, experiment_id, masked_mode)
        if unmasked is None or masked is None:
            continue
        labels.append(label)
        unmasked_vals.append(unmasked)
        masked_vals.append(masked)

    if not labels:
        return

    x = list(range(len(labels)))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar([xi - width / 2 for xi in x], unmasked_vals, width=width, label="Unmasked")
    plt.bar([xi + width / 2 for xi in x], masked_vals, width=width, label="CU-masked")
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Median latency (ms)")
    plt.title("Headline experiment comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "headline_latency_comparison.png", dpi=200)
    plt.savefig(output_dir / "headline_latency_comparison.svg")
    plt.close()


def save_comm_stability(rows: list[dict[str, Any]], output_dir: Path) -> None:
    dma_regular = median_for(rows, "comm_only_contiguous", "DMA | regular stream")
    dma_masked = median_for(rows, "comm_only_contiguous", "DMA | CU-masked 0.3 (91 CUs)")
    cuk_regular = median_for(rows, "comm_only_contiguous", "CU-k | regular, full grid (16384 WGs)")
    cuk_masked = median_for(rows, "comm_only_contiguous", "CU-k | CU-masked 0.3, full grid")

    if None in (dma_regular, dma_masked, cuk_regular, cuk_masked):
        return

    ratios = [
        dma_masked / dma_regular if dma_regular else 0.0,
        cuk_masked / cuk_regular if cuk_regular else 0.0,
    ]
    labels = ["DMA masked/regular", "CU-k masked/regular"]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(ratios)), ratios)
    plt.axhline(1.0, linestyle="--")
    plt.ylim(0.9, 1.1)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("Latency ratio")
    plt.title("Comm-only stability (contiguous)")
    plt.tight_layout()
    plt.savefig(output_dir / "comm_stability_ratio.png", dpi=200)
    plt.savefig(output_dir / "comm_stability_ratio.svg")
    plt.close()


def save_bandwidth_bars(rows: list[dict[str, Any]], output_dir: Path) -> None:
    bw_rows = [row for row in rows if row.get("bandwidth_gbps") not in ("", "-", None)]
    if not bw_rows:
        return
    labels = [f"{row.get('experiment_id')}::{row.get('mode')}" for row in bw_rows]
    values = [to_float(row.get("bandwidth_gbps")) or 0.0 for row in bw_rows]
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.ylabel("Bandwidth (GB/s)")
    plt.title("Communication throughput comparison (DMA vs CU modes)")
    plt.tight_layout()
    plt.savefig(output_dir / "bandwidth_comparison.png", dpi=200)
    plt.savefig(output_dir / "bandwidth_comparison.svg")
    plt.close()


def save_intensity_curve(rows: list[dict[str, Any]], output_dir: Path) -> None:
    intensity_rows = [row for row in rows if row.get("scenario") == "intensity-sweep"]
    if not intensity_rows:
        return

    pairs: list[tuple[str, float]] = []
    for row in intensity_rows:
        slowdown = to_float(row.get("slowdown_x"))
        if slowdown is None:
            continue
        pairs.append((row.get("mode", "unknown"), slowdown))
    if not pairs:
        return

    labels = [label for label, _ in pairs]
    values = [value for _, value in pairs]
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(values)), values, marker="o")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("Masked / Regular slowdown (x)")
    plt.title("Compute intensity vs CU masking slowdown")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "intensity_slowdown.png", dpi=200)
    plt.savefig(output_dir / "intensity_slowdown.svg")
    plt.close()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.results_csv.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.results_csv)
    save_headline_latency(rows, output_dir)
    save_comm_stability(rows, output_dir)
    save_latency_bars(rows, output_dir)
    save_bandwidth_bars(rows, output_dir)
    save_intensity_curve(rows, output_dir)
    print(f"Wrote plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
