#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


SUMMARY_MARKER = "=== AG+GEMM Benchmark Summary ==="
ROW_RE = re.compile(
    r"^\s*(?P<mode>\w+)\s+"
    r"(?P<num_sms>\d+)\s+"
    r"(?P<median_ms>[-+]?\d*\.?\d+)\s+"
    r"(?P<mean_ms>[-+]?\d*\.?\d+)\s+"
    r"(?P<std_ms>[-+]?\d*\.?\d+)\s+"
    r"(?P<correct>\w+)\s*$"
)
SHAPE_RE = re.compile(
    r"Shape:\s*M=(?P<M>\d+),\s*N=(?P<N>\d+),\s*K=(?P<K>\d+),\s*chunk=(?P<chunk>\d+),\s*repeats=(?P<repeats>\d+)"
)


def parse_log(path: Path):
    lines = path.read_text().splitlines()
    rows = []
    i = 0
    while i < len(lines):
        if SUMMARY_MARKER not in lines[i]:
            i += 1
            continue

        shape = {}
        mode_rows = []
        j = i + 1
        while j < len(lines):
            line = lines[j].strip()
            if not line:
                j += 1
                continue
            if SUMMARY_MARKER in line:
                break
            m_shape = SHAPE_RE.search(lines[j])
            if m_shape:
                shape = m_shape.groupdict()
                j += 1
                continue
            if line.startswith("Mode"):
                j += 1
                continue
            m_row = ROW_RE.match(lines[j])
            if m_row:
                row = m_row.groupdict()
                row.update(shape)
                row["source_log"] = str(path)
                row["median_ms"] = float(row["median_ms"])
                row["mean_ms"] = float(row["mean_ms"])
                row["std_ms"] = float(row["std_ms"])
                row["num_sms"] = int(row["num_sms"])
                for key in ("M", "N", "K", "chunk", "repeats"):
                    if key in row and row[key] != "":
                        row[key] = int(row[key])
                mode_rows.append(row)
            j += 1

        if mode_rows:
            rows.extend(mode_rows)
        i = j
    return rows


def add_speedup(rows):
    by_group = {}
    for r in rows:
        key = (r.get("source_log"), r.get("M"), r.get("N"), r.get("K"), r.get("chunk"), r.get("repeats"))
        by_group.setdefault(key, []).append(r)

    for group_rows in by_group.values():
        static_median = None
        dynamic_median = None
        for r in group_rows:
            if r["mode"] == "static":
                static_median = r["median_ms"]
            if r["mode"] == "dynamic":
                dynamic_median = r["median_ms"]
        speedup = None
        if static_median and dynamic_median and dynamic_median > 0:
            speedup = static_median / dynamic_median
        for r in group_rows:
            r["dynamic_vs_static_speedup"] = speedup


def write_csv(rows, out_path: Path):
    fields = [
        "source_log",
        "M",
        "N",
        "K",
        "chunk",
        "repeats",
        "mode",
        "num_sms",
        "median_ms",
        "mean_ms",
        "std_ms",
        "correct",
        "dynamic_vs_static_speedup",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_md(rows, out_path: Path):
    md = []
    md.append("| source_log | M | N | K | chunk | repeats | mode | NUM_SMS | median_ms | std_ms | correct | dynamic_vs-static speedup |")
    md.append("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|")
    for r in rows:
        speedup = r.get("dynamic_vs_static_speedup")
        speedup_str = f"{speedup:.3f}x" if isinstance(speedup, float) else "-"
        md.append(
            f"| {Path(r['source_log']).name} | {r.get('M','-')} | {r.get('N','-')} | {r.get('K','-')} | "
            f"{r.get('chunk','-')} | {r.get('repeats','-')} | {r['mode']} | {r['num_sms']} | "
            f"{r['median_ms']:.3f} | {r['std_ms']:.3f} | {r['correct']} | {speedup_str} |"
        )
    out_path.write_text("\n".join(md) + "\n")


def main():
    p = argparse.ArgumentParser(description="Parse AG+GEMM benchmark summary blocks from logs")
    p.add_argument("logs", nargs="+", help="Log files to parse")
    p.add_argument("--csv-out", default="ag_gemm_summary.csv")
    p.add_argument("--md-out", default="ag_gemm_summary.md")
    args = p.parse_args()

    all_rows = []
    for log in args.logs:
        path = Path(log)
        if not path.exists():
            continue
        all_rows.extend(parse_log(path))

    if not all_rows:
        raise SystemExit("No AG+GEMM benchmark summary blocks found.")

    add_speedup(all_rows)
    write_csv(all_rows, Path(args.csv_out))
    write_md(all_rows, Path(args.md_out))
    print(f"Wrote {args.csv_out} and {args.md_out} with {len(all_rows)} rows")


if __name__ == "__main__":
    main()
