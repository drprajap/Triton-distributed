#!/usr/bin/env python3
"""Unified entrypoint for CU masking experiment implementations.

This script centralizes experiment launching for:
  - ag-gemm
  - comm-only
  - compute-only
  - intensity-sweep

Legacy tutorial entrypoints remain for compatibility, but the canonical
workflow should execute this file through scripts/cu_experiments.py.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BACKEND_FILES = {
    "ag-gemm": "12-cu-masked-ag-gemm.py",
    "comm-only": "15-comm-only-benchmark.py",
    "compute-only": "16-gemm-only-cu-mask.py",
    "intensity-sweep": "test_grid_matched_intensity.py",
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load backend module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_backend(scenario: str, passthrough_args: list[str]) -> int:
    backend_name = BACKEND_FILES[scenario]
    backend_path = THIS_DIR / backend_name
    module = load_module(backend_path)
    if not hasattr(module, "main"):
        raise RuntimeError(f"{backend_name} has no main() entrypoint")

    args = list(passthrough_args)
    if args and args[0] == "--":
        args = args[1:]

    saved_argv = sys.argv
    prev_flag = os.environ.get("CU_MASK_SUITE_CALLER")
    try:
        # Preserve backend argparse behavior as if launched directly.
        os.environ["CU_MASK_SUITE_CALLER"] = "1"
        sys.argv = [str(backend_path), *args]
        module.main()
    finally:
        sys.argv = saved_argv
        if prev_flag is None:
            os.environ.pop("CU_MASK_SUITE_CALLER", None)
        else:
            os.environ["CU_MASK_SUITE_CALLER"] = prev_flag
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        valid = ", ".join(BACKEND_FILES.keys())
        raise SystemExit(f"Usage: {Path(sys.argv[0]).name} <{valid}> [backend args...]")

    scenario = sys.argv[1]
    if scenario not in BACKEND_FILES:
        valid = ", ".join(BACKEND_FILES.keys())
        raise SystemExit(f"Unknown scenario '{scenario}'. Expected one of: {valid}")

    return run_backend(scenario, sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
