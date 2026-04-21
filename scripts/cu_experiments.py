#!/usr/bin/env python3
"""Canonical stage-1 runner for CU masking experiments.

This script is the single entrypoint for the consolidated experiment flow.
It orchestrates selected experiment presets, captures reproducible artifacts,
and emits normalized results for side-by-side comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cu_experiments.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "cu_mask_stage1"

AG_SHAPE_RE = re.compile(r"^Shape:\s*(?P<shape>.+)$")
AG_ROW_RE = re.compile(
    r"^\s*(?P<mode>[a-zA-Z\-]+)\s+"
    r"(?P<median_ms>[-+]?\d*\.?\d+)\s+"
    r"(?P<mean_ms>[-+]?\d*\.?\d+)\s+"
    r"(?P<std_ms>[-+]?\d*\.?\d+)\s+"
    r"(?P<comm_cus>\-|\d+)\s+"
    r"(?P<compute_cus>\-|\d+)\s+"
    r"(?P<correct>\w+)\s*$"
)
COMM_ROW_RE = re.compile(
    r"^\s{2}(?P<label>.+?)\s+"
    r"(?P<median_ms>\d+\.\d+)\s+"
    r"(?P<mean_ms>\d+\.\d+)\s+"
    r"(?P<std_ms>\d+\.\d+)\s+"
    r"(?P<bandwidth_gbps>\d+\.\d+)\s*$"
)
GEMM_ROW_RE = re.compile(
    r"^(?P<label>Regular stream|CU-masked stream.*)\s+"
    r"(?P<median_ms>\d+\.\d+)\s+"
    r"(?P<mean_ms>\d+\.\d+)\s+"
    r"(?P<std_ms>\d+\.\d+)\s*$"
)
INTENSITY_ROW_RE = re.compile(
    r"^(?P<label>.+?)\s+"
    r"(?P<regular_ms>\d+\.\d+)\s+"
    r"(?P<masked_ms>\d+\.\d+)\s+"
    r"(?P<slowdown>\d+\.\d+x)\s*$"
)


@dataclass
class Experiment:
    id: str
    scenario: str
    description: str
    script: Path
    distributed: bool
    args: list[str]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run consolidated CU masking experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("run-all", "ag-gemm", "comm-only", "compute-only", "intensity-sweep"):
        p = subparsers.add_parser(command, help=f"Run {command} experiments")
        add_common_args(p)

    p_list = subparsers.add_parser("list", help="List configured experiment presets")
    p_list.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--profile",
        choices=("none", "kernel-trace", "perfetto"),
        default="none",
        help="Optional profiling mode (default: none).",
    )
    parser.add_argument(
        "--experiment-id",
        action="append",
        default=[],
        help="Filter to one or more specific experiment IDs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and selected experiments without executing.",
    )


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "experiments" not in data or not isinstance(data["experiments"], list):
        raise ValueError("Config must define an experiments list.")
    return data


def build_experiments(config: dict[str, Any]) -> list[Experiment]:
    experiments: list[Experiment] = []
    for raw in config["experiments"]:
        experiments.append(
            Experiment(
                id=raw["id"],
                scenario=raw["scenario"],
                description=raw.get("description", ""),
                script=REPO_ROOT / raw["script"],
                distributed=bool(raw.get("distributed", False)),
                args=[str(a) for a in raw.get("args", [])],
                metadata=dict(raw.get("metadata", {})),
            )
        )
    return experiments


def select_experiments(
    command: str, experiments: list[Experiment], wanted_ids: list[str]
) -> list[Experiment]:
    if command == "run-all":
        selected = experiments
    elif command == "list":
        selected = experiments
    else:
        selected = [e for e in experiments if e.scenario == command]

    if wanted_ids:
        wanted = set(wanted_ids)
        selected = [e for e in selected if e.id in wanted]
    return selected


def run_shell_capture(command: str, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def capture_env_snapshot(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    commands = config.get("fixed_methodology", {}).get("env_snapshot_commands", [])
    for command in commands:
        key = command.split()[0]
        snapshot[key] = run_shell_capture(command, REPO_ROOT)
    (output_dir / "env_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    return snapshot


def build_command(exp: Experiment, profile_mode: str, profile_dir: Path) -> list[str]:
    if exp.distributed:
        base = [str(REPO_ROOT / "scripts" / "launch_amd.sh"), str(exp.script), *exp.args]
    else:
        base = [sys.executable, str(exp.script), *exp.args]

    if profile_mode == "none":
        return base
    if profile_mode == "kernel-trace":
        return [
            "rocprofv3",
            "--kernel-trace",
            "--hip-trace",
            "-f",
            "csv",
            "-d",
            str(profile_dir),
            *base,
        ]
    return [
        "rocprofv3",
        "--kernel-trace",
        "--hip-trace",
        "--plugin",
        "perfetto",
        "-d",
        str(profile_dir),
        *base,
    ]


def parse_kv_shape(shape: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for part in shape.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().replace("-", "_")
        value = value.strip()
        try:
            parsed[key] = int(value)
        except ValueError:
            parsed[key] = value
    return parsed


def parse_ag_gemm(stdout: str, exp: Experiment) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    shape_meta: dict[str, Any] = {}
    for line in stdout.splitlines():
        shape_match = AG_SHAPE_RE.match(line.strip())
        if shape_match:
            shape_meta = parse_kv_shape(shape_match.group("shape"))
            continue
        row_match = AG_ROW_RE.match(line)
        if not row_match:
            continue
        row = row_match.groupdict()
        row.update(
            {
                "metric_type": "latency_ms",
                "scenario": exp.scenario,
                "experiment_id": exp.id,
                "mode": row["mode"],
                "median_ms": float(row["median_ms"]),
                "mean_ms": float(row["mean_ms"]),
                "std_ms": float(row["std_ms"]),
                "comm_cus": None if row["comm_cus"] == "-" else int(row["comm_cus"]),
                "compute_cus": None if row["compute_cus"] == "-" else int(row["compute_cus"]),
                "correct": row["correct"],
            }
        )
        row.update(shape_meta)
        row.update(exp.metadata)
        rows.append(row)
    return rows


def parse_comm_only(stdout: str, exp: Experiment) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        match = COMM_ROW_RE.match(line)
        if not match:
            continue
        row = match.groupdict()
        row.update(
            {
                "metric_type": "latency_and_bw",
                "scenario": exp.scenario,
                "experiment_id": exp.id,
                "mode": row["label"].strip(),
                "median_ms": float(row["median_ms"]),
                "mean_ms": float(row["mean_ms"]),
                "std_ms": float(row["std_ms"]),
                "bandwidth_gbps": float(row["bandwidth_gbps"]),
            }
        )
        row.update(exp.metadata)
        rows.append(row)
    return rows


def parse_compute_only(stdout: str, exp: Experiment) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        match = GEMM_ROW_RE.match(line.strip())
        if not match:
            continue
        row = match.groupdict()
        row.update(
            {
                "metric_type": "latency_ms",
                "scenario": exp.scenario,
                "experiment_id": exp.id,
                "mode": row["label"].strip(),
                "median_ms": float(row["median_ms"]),
                "mean_ms": float(row["mean_ms"]),
                "std_ms": float(row["std_ms"]),
            }
        )
        row.update(exp.metadata)
        rows.append(row)
    return rows


def parse_intensity(stdout: str, exp: Experiment) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        match = INTENSITY_ROW_RE.match(line.strip())
        if not match:
            continue
        row = match.groupdict()
        slowdown = float(row["slowdown"].rstrip("x"))
        row.update(
            {
                "metric_type": "intensity_slowdown",
                "scenario": exp.scenario,
                "experiment_id": exp.id,
                "mode": row["label"].strip(),
                "regular_ms": float(row["regular_ms"]),
                "masked_ms": float(row["masked_ms"]),
                "slowdown_x": slowdown,
            }
        )
        row.update(exp.metadata)
        rows.append(row)
    return rows


def parse_rows(stdout: str, exp: Experiment) -> list[dict[str, Any]]:
    if exp.scenario == "ag-gemm":
        return parse_ag_gemm(stdout, exp)
    if exp.scenario == "comm-only":
        return parse_comm_only(stdout, exp)
    if exp.scenario == "compute-only":
        return parse_compute_only(stdout, exp)
    if exp.scenario == "intensity-sweep":
        return parse_intensity(stdout, exp)
    return []


def try_float(value: str) -> float | None:
    value = value.strip().replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


def profile_summary(profile_dir: Path, exp: Experiment) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scenario": exp.scenario,
        "experiment_id": exp.id,
        "csv_files": [],
        "kernel_avg_ms": [],
        "overlap_ratio_stream01": None,
    }
    csv_files = sorted(profile_dir.rglob("*.csv"))
    summary["csv_files"] = [str(path.relative_to(profile_dir)) for path in csv_files]
    if not csv_files:
        return summary

    durations_by_kernel: dict[str, list[float]] = {}
    stream_intervals: dict[str, list[tuple[float, float]]] = {}
    for csv_file in csv_files:
        try:
            with csv_file.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                fields = {field.lower(): field for field in reader.fieldnames}
                kernel_col = next(
                    (
                        fields[name]
                        for name in ("kernel_name", "kernelname", "kernel", "name", "dispatchname")
                        if name in fields
                    ),
                    None,
                )
                dur_col = next(
                    (
                        fields[name]
                        for name in (
                            "durationns",
                            "duration(ns)",
                            "duration_ns",
                            "duration",
                            "time(ns)",
                        )
                        if name in fields
                    ),
                    None,
                )
                start_col = next(
                    (
                        fields[name]
                        for name in ("beginns", "startns", "start_ns", "start")
                        if name in fields
                    ),
                    None,
                )
                end_col = next(
                    (fields[name] for name in ("endns", "end_ns", "end") if name in fields),
                    None,
                )
                stream_col = next(
                    (
                        fields[name]
                        for name in ("stream_id", "streamid", "stream", "queueid")
                        if name in fields
                    ),
                    None,
                )

                for row in reader:
                    kernel_name = row.get(kernel_col, "unknown") if kernel_col else "unknown"
                    if dur_col:
                        duration = try_float(row.get(dur_col, ""))
                        if duration is not None:
                            durations_by_kernel.setdefault(kernel_name, []).append(duration / 1e6)
                    if start_col and end_col and stream_col:
                        start = try_float(row.get(start_col, ""))
                        end = try_float(row.get(end_col, ""))
                        stream_id = row.get(stream_col, "")
                        if start is not None and end is not None and stream_id:
                            stream_intervals.setdefault(stream_id, []).append((start, end))
        except OSError:
            continue

    for kernel, values in sorted(durations_by_kernel.items(), key=lambda item: len(item[1]), reverse=True):
        avg_ms = sum(values) / len(values)
        summary["kernel_avg_ms"].append({"kernel": kernel, "avg_ms": avg_ms, "samples": len(values)})

    # Simple overlap ratio between two busiest streams when timestamps exist.
    if len(stream_intervals) >= 2:
        busiest = sorted(stream_intervals.items(), key=lambda item: len(item[1]), reverse=True)[:2]
        first = busiest[0][1]
        second = busiest[1][1]
        intersection = 0.0
        for s1, e1 in first:
            for s2, e2 in second:
                left = max(s1, s2)
                right = min(e1, e2)
                if right > left:
                    intersection += right - left
        total = sum((e - s) for s, e in first)
        if total > 0:
            summary["overlap_ratio_stream01"] = intersection / total
    return summary


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    lines = ["# CU Masking Stage-1 Summary", ""]
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row.get("scenario", "unknown"), []).append(row)

    for scenario, scenario_rows in sorted(by_scenario.items()):
        lines.append(f"## {scenario}")
        lines.append("")
        lines.append("| experiment_id | mode | median_ms | bandwidth_gbps | slowdown_x | grid_size | comm_engine |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for row in scenario_rows:
            lines.append(
                "| {experiment_id} | {mode} | {median_ms} | {bandwidth_gbps} | {slowdown_x} | {grid_size} | {comm_engine} |".format(
                    experiment_id=row.get("experiment_id", "-"),
                    mode=row.get("mode", "-"),
                    median_ms=_fmt(row.get("median_ms")),
                    bandwidth_gbps=_fmt(row.get("bandwidth_gbps")),
                    slowdown_x=_fmt(row.get("slowdown_x")),
                    grid_size=row.get("grid_size", "-"),
                    comm_engine=row.get("comm_engine", "-"),
                )
            )
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    experiments = build_experiments(config)
    selected = select_experiments(args.command, experiments, getattr(args, "experiment_id", []))

    if args.command == "list":
        print("Configured experiment presets:")
        for exp in selected:
            print(f"- {exp.id:<30} scenario={exp.scenario:<16} distributed={exp.distributed} :: {exp.description}")
        return 0

    if not selected:
        raise SystemExit("No experiments selected. Check scenario filter or --experiment-id.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config.get('suite_name', 'cu_mask_stage1')}_{timestamp}"
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "suite_name": config.get("suite_name", "cu_mask_stage1"),
        "started_utc": timestamp,
        "profile_mode": args.profile,
        "selected_experiments": [exp.id for exp in selected],
        "fixed_methodology": config.get("fixed_methodology", {}),
        "experiments": [],
    }

    print(f"Run ID: {run_id}")
    print(f"Output dir: {output_dir}")
    env_snapshot = capture_env_snapshot(config, output_dir)
    manifest["env_snapshot"] = env_snapshot

    all_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for exp in selected:
        exp_dir = output_dir / "experiments" / exp.id
        exp_dir.mkdir(parents=True, exist_ok=True)
        profile_dir = exp_dir / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(exp, args.profile, profile_dir)

        exp_record: dict[str, Any] = {
            "experiment_id": exp.id,
            "scenario": exp.scenario,
            "description": exp.description,
            "distributed": exp.distributed,
            "script": str(exp.script.relative_to(REPO_ROOT)),
            "args": exp.args,
            "metadata": exp.metadata,
            "command": command,
        }
        manifest["experiments"].append(exp_record)

        (exp_dir / "command.json").write_text(json.dumps({"command": command}, indent=2))
        print(f"\n[{exp.id}] {' '.join(command)}")
        if args.dry_run:
            exp_record["returncode"] = None
            continue

        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        (exp_dir / "stdout.log").write_text(proc.stdout)
        (exp_dir / "stderr.log").write_text(proc.stderr)
        exp_record["returncode"] = proc.returncode
        exp_record["stdout_log"] = str((exp_dir / "stdout.log").relative_to(output_dir))
        exp_record["stderr_log"] = str((exp_dir / "stderr.log").relative_to(output_dir))
        rows = parse_rows(proc.stdout, exp)
        for row in rows:
            row["run_id"] = run_id
            row["profile_mode"] = args.profile
            row["exit_code"] = proc.returncode
        all_rows.extend(rows)

        if proc.returncode != 0:
            print(f"[{exp.id}] FAILED (exit={proc.returncode})")
        else:
            print(f"[{exp.id}] OK")

        if args.profile != "none":
            profile = profile_summary(profile_dir, exp)
            (exp_dir / "profile_summary.json").write_text(json.dumps(profile, indent=2))
            profile_rows.append(profile)

    manifest["completed_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest["row_count"] = len(all_rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if all_rows:
        write_jsonl(all_rows, output_dir / "results.jsonl")
        write_csv(all_rows, output_dir / "results.csv")
        write_summary(all_rows, output_dir)

    if profile_rows:
        (output_dir / "profile_summary.json").write_text(json.dumps(profile_rows, indent=2))

    print(f"\nFinished. Parsed rows: {len(all_rows)}")
    print(f"Artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
