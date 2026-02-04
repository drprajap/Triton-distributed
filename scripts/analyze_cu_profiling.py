#!/usr/bin/env python3
"""
Analyze ROCProf traces for CU partitioning test
"""
import sys
import os
import csv
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def parse_hip_trace(csv_file: Path) -> Dict:
    """Parse HIP API trace CSV"""
    print(f"\n📊 Analyzing HIP Trace: {csv_file.name}")
    
    stats = {
        'hipMemcpyAsync': [],
        'hipExtStreamCreateWithCUMask': [],
        'hipStreamDestroy': [],
        'hipStreamSynchronize': [],
        'total_api_calls': 0
    }
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats['total_api_calls'] += 1
                api_name = row.get('Name', '')
                
                if api_name in stats:
                    duration_ns = float(row.get('DurationNs', 0))
                    stats[api_name].append({
                        'duration_ns': duration_ns,
                        'duration_ms': duration_ns / 1e6,
                        'start': float(row.get('BeginNs', 0)),
                        'end': float(row.get('EndNs', 0))
                    })
        
        # Print summary
        print(f"  Total API calls: {stats['total_api_calls']}")
        
        for api_name, calls in stats.items():
            if api_name == 'total_api_calls':
                continue
            if calls:
                total_time = sum(c['duration_ms'] for c in calls)
                avg_time = total_time / len(calls)
                print(f"  {api_name}:")
                print(f"    Count: {len(calls)}")
                print(f"    Total: {total_time:.3f} ms")
                print(f"    Average: {avg_time:.3f} ms")
                print(f"    Min: {min(c['duration_ms'] for c in calls):.3f} ms")
                print(f"    Max: {max(c['duration_ms'] for c in calls):.3f} ms")
        
        return stats
    
    except Exception as e:
        print(f"  ❌ Error parsing {csv_file}: {e}")
        return stats


def parse_stats(csv_file: Path) -> Dict:
    """Parse ROCProf statistics CSV"""
    print(f"\n📈 Analyzing Statistics: {csv_file.name}")
    
    try:
        with open(csv_file, 'r') as f:
            content = f.read()
            print(content)
        return {}
    except Exception as e:
        print(f"  ❌ Error parsing {csv_file}: {e}")
        return {}


def parse_log(log_file: Path) -> Dict:
    """Extract performance metrics from test log"""
    print(f"\n📋 Analyzing Log: {log_file.name}")
    
    metrics = {
        'cu_count': None,
        'comm_cus': None,
        'compute_cus': None,
        'standard_time_ms': [],
        'partitioned_time_ms': [],
        'standard_bandwidth_gbps': [],
        'partitioned_bandwidth_gbps': [],
        'speedup': [],
        'world_size': None
    }
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Extract CU information
                if 'GPU has' in line and 'Compute Units' in line:
                    match = re.search(r'(\d+) Compute Units', line)
                    if match:
                        metrics['cu_count'] = int(match.group(1))
                
                if 'CU Partition:' in line:
                    match = re.search(r'(\d+) comm, (\d+) compute', line)
                    if match:
                        metrics['comm_cus'] = int(match.group(1))
                        metrics['compute_cus'] = int(match.group(2))
                
                # Extract timing information
                if 'Standard streams time:' in line:
                    match = re.search(r'([\d.]+) ms', line)
                    if match:
                        metrics['standard_time_ms'].append(float(match.group(1)))
                
                if 'CU-partitioned time:' in line:
                    match = re.search(r'([\d.]+) ms', line)
                    if match:
                        metrics['partitioned_time_ms'].append(float(match.group(1)))
                
                # Extract bandwidth
                if 'Standard streams bandwidth:' in line:
                    match = re.search(r'([\d.]+) GB/s', line)
                    if match:
                        metrics['standard_bandwidth_gbps'].append(float(match.group(1)))
                
                if 'CU-partitioned bandwidth:' in line:
                    match = re.search(r'([\d.]+) GB/s', line)
                    if match:
                        metrics['partitioned_bandwidth_gbps'].append(float(match.group(1)))
                
                # Extract speedup
                if 'Speedup:' in line:
                    match = re.search(r'([\d.]+)x', line)
                    if match:
                        metrics['speedup'].append(float(match.group(1)))
        
        # Print summary
        if metrics['cu_count']:
            print(f"  GPU: {metrics['cu_count']} CUs total")
            if metrics['comm_cus'] and metrics['compute_cus']:
                print(f"  Partitioning: {metrics['comm_cus']} comm + {metrics['compute_cus']} compute")
        
        if metrics['standard_time_ms']:
            avg_std = sum(metrics['standard_time_ms']) / len(metrics['standard_time_ms'])
            print(f"  Standard streams: {avg_std:.2f} ms (avg)")
        
        if metrics['partitioned_time_ms']:
            avg_part = sum(metrics['partitioned_time_ms']) / len(metrics['partitioned_time_ms'])
            print(f"  CU-partitioned: {avg_part:.2f} ms (avg)")
        
        if metrics['standard_bandwidth_gbps']:
            avg_bw_std = sum(metrics['standard_bandwidth_gbps']) / len(metrics['standard_bandwidth_gbps'])
            print(f"  Standard bandwidth: {avg_bw_std:.2f} GB/s (avg)")
        
        if metrics['partitioned_bandwidth_gbps']:
            avg_bw_part = sum(metrics['partitioned_bandwidth_gbps']) / len(metrics['partitioned_bandwidth_gbps'])
            print(f"  CU-partitioned bandwidth: {avg_bw_part:.2f} GB/s (avg)")
        
        if metrics['speedup']:
            avg_speedup = sum(metrics['speedup']) / len(metrics['speedup'])
            print(f"  Average speedup: {avg_speedup:.2f}x")
        
        return metrics
    
    except Exception as e:
        print(f"  ❌ Error parsing {log_file}: {e}")
        return metrics


def generate_summary(profile_dir: Path):
    """Generate overall summary of profiling results"""
    print(f"\n{'='*80}")
    print(f"CU Partitioning Profiling Analysis")
    print(f"{'='*80}")
    print(f"Directory: {profile_dir}")
    
    # Find all relevant files
    hip_traces = list(profile_dir.glob('*_hip_trace.csv'))
    hsa_traces = list(profile_dir.glob('*_hsa_trace.csv'))
    stats_files = list(profile_dir.glob('*_stats.csv'))
    log_files = list(profile_dir.glob('*.log'))
    
    print(f"\nFiles found:")
    print(f"  HIP traces: {len(hip_traces)}")
    print(f"  HSA traces: {len(hsa_traces)}")
    print(f"  Stats files: {len(stats_files)}")
    print(f"  Log files: {len(log_files)}")
    
    # Parse all files
    all_metrics = []
    
    for hip_trace in hip_traces:
        parse_hip_trace(hip_trace)
    
    for stats_file in stats_files:
        parse_stats(stats_file)
    
    for log_file in log_files:
        metrics = parse_log(log_file)
        all_metrics.append(metrics)
    
    # Generate final summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    
    if all_metrics:
        # Aggregate metrics
        all_speedups = [s for m in all_metrics for s in m.get('speedup', [])]
        all_std_times = [t for m in all_metrics for t in m.get('standard_time_ms', [])]
        all_part_times = [t for m in all_metrics for t in m.get('partitioned_time_ms', [])]
        all_std_bw = [b for m in all_metrics for b in m.get('standard_bandwidth_gbps', [])]
        all_part_bw = [b for m in all_metrics for b in m.get('partitioned_bandwidth_gbps', [])]
        
        if all_speedups:
            print(f"Average Speedup: {sum(all_speedups)/len(all_speedups):.2f}x")
        
        if all_std_times and all_part_times:
            print(f"Time Reduction: {sum(all_std_times)/len(all_std_times):.2f} ms → {sum(all_part_times)/len(all_part_times):.2f} ms")
        
        if all_std_bw and all_part_bw:
            print(f"Bandwidth: {sum(all_std_bw)/len(all_std_bw):.2f} GB/s → {sum(all_part_bw)/len(all_part_bw):.2f} GB/s")
            bw_improvement = ((sum(all_part_bw)/len(all_part_bw)) - (sum(all_std_bw)/len(all_std_bw))) / (sum(all_std_bw)/len(all_std_bw)) * 100
            print(f"Bandwidth Improvement: {bw_improvement:.1f}%")
    
    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_cu_profiling.py <profiling_results_directory>")
        sys.exit(1)
    
    profile_dir = Path(sys.argv[1])
    
    if not profile_dir.exists():
        print(f"❌ Error: Directory not found: {profile_dir}")
        sys.exit(1)
    
    generate_summary(profile_dir)

