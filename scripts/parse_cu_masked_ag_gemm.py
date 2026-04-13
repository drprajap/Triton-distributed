#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


SUMMARY_MARKER = "=== CU-Masked AG+GEMM Results ==="
SHAPE_RE = re.compile(
    r"Shape:\s*M=(?P<M>\d+),\s*N=(?P<N>\d+),\s*K=(?P<K>\d+),\s*chunk=(?P<chunk>\d+),\s*"
    r"num_sms=(?P<num_sms>\d+),\s*repeats=(?P<repeats>\d+),\s*modes=(?P<modes>.+)$"
)
ROW_RE = re.compile(
    r"^\s*(?P<mode>[a-zA-Z\-]+)\s+"
    r"(?P<median_ms>[-+]?\d*\.?\d+)\s+"
    r"(?P<mean_ms>[-+]?\d*\.?\d+)\s+"
    r"(?P<std_ms>[-+]?\d*\.?\d+)\s+"
    r"(?P<comm_cus>\-|\d+)\s+"
    r"(?P<compute_cus>\-|\d+)\s+"
    r"(?P<correct>\w+)\s*$"
)


def parse_one(path: Path):
    rows = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if SUMMARY_MARKER not in lines[i]:
            i += 1
            continue
        shape = {}
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
            if line.startswith("Mode") or line.startswith("Speedups"):
                j += 1
                continue
            if line.startswith("unmasked") or line.startswith("cu-masked"):
                m_row = ROW_RE.match(lines[j])
                if m_row:
                    r = m_row.groupdict()
                    r.update(shape)
                    r["source_log"] = str(path)
                    for k in ("M", "N", "K", "chunk", "num_sms", "repeats"):
                        r[k] = int(r[k])
                    for k in ("median_ms", "mean_ms", "std_ms"):
                        r[k] = float(r[k])
                    r["comm_cus"] = None if r["comm_cus"] == "-" else int(r["comm_cus"])
                    r["compute_cus"] = None if r["compute_cus"] == "-" else int(r["compute_cus"])
                    rows.append(r)
            j += 1
        i = j
    return rows


def add_speedup(rows):
    grouped = {}
    for r in rows:
        key = (r["source_log"], r["M"], r["N"], r["K"], r["chunk"], r["num_sms"], r["repeats"])
        grouped.setdefault(key, []).append(r)

    for rs in grouped.values():
        baseline = None
        masked = None
        for r in rs:
            if r["mode"] == "unmasked":
                baseline = r["median_ms"]
            elif r["mode"] == "cu-masked":
                masked = r["median_ms"]
        speedup = None
        if baseline and masked and masked > 0:
            speedup = baseline / masked
        for r in rs:
            r["unmasked_vs_masked_speedup"] = speedup


def write_csv(rows, out_path: Path):
    fields = [
        "source_log",
        "M",
        "N",
        "K",
        "chunk",
        "num_sms",
        "repeats",
        "modes",
        "mode",
        "median_ms",
        "mean_ms",
        "std_ms",
        "comm_cus",
        "compute_cus",
        "correct",
        "unmasked_vs_masked_speedup",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows, out_path: Path):
    lines = []
    lines.append("| source_log | M | N | K | num_sms | repeats | mode | median_ms | std_ms | comm_cus | compute_cus | correct | unmasked/masked |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|")
    for r in rows:
        ratio = r["unmasked_vs_masked_speedup"]
        ratio_str = f"{ratio:.3f}x" if isinstance(ratio, float) else "-"
        comm = "-" if r["comm_cus"] is None else str(r["comm_cus"])
        comp = "-" if r["compute_cus"] is None else str(r["compute_cus"])
        lines.append(
            f"| {Path(r['source_log']).name} | {r['M']} | {r['N']} | {r['K']} | {r['num_sms']} | "
            f"{r['repeats']} | {r['mode']} | {r['median_ms']:.3f} | {r['std_ms']:.3f} | "
            f"{comm} | {comp} | {r['correct']} | {ratio_str} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description="Parse tutorial 12 benchmark logs")
    p.add_argument("logs", nargs="+")
    p.add_argument("--csv-out", default="cu_masked_ag_gemm_summary.csv")
    p.add_argument("--md-out", default="cu_masked_ag_gemm_summary.md")
    args = p.parse_args()

    rows = []
    for log in args.logs:
        path = Path(log)
        if path.exists():
            rows.extend(parse_one(path))

    if not rows:
        raise SystemExit("No summary blocks found in provided logs.")

    add_speedup(rows)
    write_csv(rows, Path(args.csv_out))
    write_md(rows, Path(args.md_out))
    print(f"Wrote {args.csv_out} and {args.md_out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
