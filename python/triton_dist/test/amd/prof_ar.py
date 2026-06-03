"""Per-phase profiler for the pure all-reduce paths (one-shot / two-shot).

Splits each op into its constituent phases (barriers vs reduce-scatter vs
all-gather kernels) and times each with CUDA events, so we get evidence of
*where* the time goes rather than only the end-to-end number from
bench_pure_ar.py.

Run under launch_amd.sh, e.g.::

    ARNOLD_WORKER_GPU=8 HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        bash scripts/launch_amd.sh python/triton_dist/test/amd/prof_ar.py \
        --transport two_shot --bytes 268435456 --iters 50 --warmup 10

Can also be wrapped with rocprofv3 (kernel-trace) for authoritative per-kernel
durations; the kernel names below appear directly in the trace.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import triton
import torch.distributed as dist
from hip import hip
from triton_dist.utils import HIP_CHECK

from triton_dist.utils import initialize_distributed
from triton_dist.kernels.amd.gemm_allreduce import (
    create_gemm_ar_context,
    pure_allreduce_one_shot_kernel,
    pure_allreduce_one_shot_op,
    pure_allreduce_two_shot_rs_kernel,
    pure_allreduce_two_shot_ag_kernel,
    pure_allreduce_two_shot_ag_push_kernel,
    pure_allreduce_two_shot_rs_push_kernel,
    pure_allreduce_two_shot_local_reduce_kernel,
    pure_allreduce_two_shot_fused_kernel,
    pure_allreduce_two_shot_op,
    pure_allreduce_two_shot_fused_op,
    pure_allreduce_two_shot_push_op,
    ring_rs_send_kernel,
    ring_rs_add_kernel,
    _barrier_all_v2,
)

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _events(n):
    return [torch.cuda.Event(enable_timing=True) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", default="two_shot",
                    choices=["one_shot", "two_shot", "two_shot_fused", "overlap_probe", "ring_rs", "dma_probe",
                             "full_push"])
    ap.add_argument("--dtype", default="float16", choices=list(DTYPES))
    ap.add_argument("--bytes", type=int, default=256 << 20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--ctx_N", type=int, default=8192)
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--num_warps", type=int, default=4)
    ap.add_argument("--num_comm_sms", type=int, default=0, help="0 = all CUs")
    ap.add_argument("--ag", default="push", choices=["push", "pull"],
                    help="all-gather direction for two_shot profiling")
    args = ap.parse_args()

    initialize_distributed()
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dtype = DTYPES[args.dtype]
    elem = torch.tensor([], dtype=dtype).element_size()

    numel = args.bytes // elem
    N = args.ctx_N
    max_M = (numel + N - 1) // N
    alloc_scratch = args.transport in ("two_shot", "two_shot_fused", "overlap_probe", "ring_rs", "dma_probe",
                                       "full_push")
    ar_stream = torch.cuda.Stream()
    ctx = create_gemm_ar_context(ar_stream, rank, world, max_M=max_M, N=N, dtype=dtype,
                                 alloc_scratch=alloc_scratch)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    NUM_COMM_SMS = args.num_comm_sms or num_sms
    BLOCK_SIZE = args.block_size

    symm_in = ctx.symm_gemm_out_buf.reshape(-1)[:numel]
    out_buf = ctx.dma_staging_buf.reshape(-1)[:numel]
    x = torch.full((numel, ), 1.0, dtype=dtype, device="cuda")
    symm_in.copy_(x)

    def run_one_shot(ev):
        ev[0].record()
        _barrier_all_v2(ctx)
        ev[1].record()
        pure_allreduce_one_shot_kernel[(NUM_COMM_SMS, )](symm_in, out_buf, numel, BLOCK_SIZE=BLOCK_SIZE,
                                                         NUM_COMM_SMS=NUM_COMM_SMS, num_warps=args.num_warps)
        ev[2].record()
        _barrier_all_v2(ctx)
        ev[3].record()

    symm_z = ctx.symm_scratch_buf.reshape(-1)[:numel] if alloc_scratch else None
    epb = triton.cdiv(numel, world)

    def run_two_shot(ev):
        ev[0].record()
        _barrier_all_v2(ctx)
        ev[1].record()
        pure_allreduce_two_shot_rs_kernel[(NUM_COMM_SMS, )](symm_in, symm_z, numel, epb, BLOCK_SIZE=BLOCK_SIZE,
                                                            NUM_COMM_SMS=NUM_COMM_SMS, num_warps=args.num_warps)
        ev[2].record()
        _barrier_all_v2(ctx)
        ev[3].record()
        if args.ag == "push":
            pure_allreduce_two_shot_ag_push_kernel[(NUM_COMM_SMS, )](symm_z, symm_in, numel, epb, BLOCK_SIZE=BLOCK_SIZE,
                                                                     NUM_COMM_SMS=NUM_COMM_SMS, num_warps=args.num_warps)
        else:
            pure_allreduce_two_shot_ag_kernel[(NUM_COMM_SMS, world)](symm_z, out_buf, numel, epb, BLOCK_SIZE=BLOCK_SIZE,
                                                                     NUM_COMM_SMS=NUM_COMM_SMS, num_warps=args.num_warps)
        ev[4].record()
        _barrier_all_v2(ctx)
        ev[5].record()

    def run_two_shot_fused(ev):
        ev[0].record()
        _barrier_all_v2(ctx)
        ev[1].record()
        pure_allreduce_two_shot_fused_kernel[(NUM_COMM_SMS, )](symm_in, symm_z, numel, epb, BLOCK_SIZE=BLOCK_SIZE,
                                                               NUM_COMM_SMS=NUM_COMM_SMS, num_warps=args.num_warps)
        ev[2].record()
        _barrier_all_v2(ctx)
        ev[3].record()

    if args.transport == "dma_probe":
        # Answer two questions with raw SDMA copies (timing only; no reduce):
        #  (a) can CONCURRENT multi-stream SDMA peer-reads beat pull-RS (264 GB/s)?
        #  (b) does SDMA-read (copy engines) overlap with CU push-AG (writes)?
        elem_sz = elem
        chunk_bytes = epb * elem_sz
        staging = ctx.dma_staging_buf.reshape(-1)[:numel]
        n_peers = world - 1
        streams = [torch.cuda.Stream() for _ in range(world)]
        cu_stream = torch.cuda.Stream()

        def sdma_gather():
            # Gather this rank's chunk (chunk == rank) from every peer into a
            # separate staging slot, each on its own stream (one SDMA queue each).
            for p in range(world):
                if p == rank:
                    continue
                peer = ctx.symm_gemm_out_buf_list[p].reshape(-1)[:numel]
                src = peer.data_ptr() + rank * chunk_bytes
                dst = staging.data_ptr() + p * chunk_bytes
                HIP_CHECK(hip.hipMemcpyAsync(dst, src, chunk_bytes,
                                             hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                                             streams[p].cuda_stream))

        def sync_streams():
            for p in range(world):
                if p != rank:
                    streams[p].synchronize()

        def ag_push(stream):
            with torch.cuda.stream(stream):
                pure_allreduce_two_shot_ag_push_kernel[(NUM_COMM_SMS, )](symm_z, symm_in, numel, epb,
                                                                         BLOCK_SIZE=BLOCK_SIZE,
                                                                         NUM_COMM_SMS=NUM_COMM_SMS,
                                                                         num_warps=args.num_warps)

        for _ in range(args.warmup):
            sdma_gather()
            ag_push(cu_stream)
        torch.cuda.synchronize()
        dist.barrier()

        def _wall(fn, it):
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(it):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / it * 1e3

        t_sdma = _wall(sdma_gather, args.iters)
        t_ag = _wall(lambda: ag_push(cu_stream), args.iters)
        t_both = _wall(lambda: (sdma_gather(), ag_push(cu_stream)), args.iters)

        remote = args.bytes * (world - 1) / world
        if rank == 0:
            seq = t_sdma + t_ag
            print(f"\n# dma_probe bytes={args.bytes>>20}MB world={world} (chunk={chunk_bytes>>20}MB x{n_peers})", flush=True)
            print(f"#   SDMA gather alone  {t_sdma:.4f} ms  ({remote/(t_sdma*1e-3)/1e9:.0f} GB/s remote read)  [pull-RS was ~264]", flush=True)
            print(f"#   push-AG alone      {t_ag:.4f} ms  ({remote/(t_ag*1e-3)/1e9:.0f} GB/s write)", flush=True)
            print(f"#   SDMA + CU-AG concur {t_both:.4f} ms ; seq sum {seq:.4f} ms ; overlap {seq/t_both:.2f}x", flush=True)
        torch.cuda.synchronize()
        os._exit(0)

    if args.transport == "ring_rs":
        # Write-based ring reduce-scatter: W-1 steps of (push partial to next
        # neighbour -> sync -> local add). Times it standalone against the
        # pull-RS baseline (one kernel, all peers read + on-the-fly reduce).
        recv = symm_z  # reuse scratch (only first epb used as recv slot)
        wbuf = symm_in

        def run_ring_rs():
            for s in range(world - 1):
                send_chunk = (rank - s) % world
                recv_chunk = (rank - s - 1) % world
                ring_rs_send_kernel[(NUM_COMM_SMS, )](wbuf, recv, epb, send_chunk,
                                                      BLOCK_SIZE=BLOCK_SIZE, NUM_COMM_SMS=NUM_COMM_SMS,
                                                      num_warps=args.num_warps)
                _barrier_all_v2(ctx)
                ring_rs_add_kernel[(NUM_COMM_SMS, )](wbuf, recv, epb, recv_chunk,
                                                     BLOCK_SIZE=BLOCK_SIZE, NUM_COMM_SMS=NUM_COMM_SMS,
                                                     num_warps=args.num_warps)
                _barrier_all_v2(ctx)

        def run_pull_rs():
            pure_allreduce_two_shot_rs_kernel[(NUM_COMM_SMS, )](symm_in, symm_z, numel, epb,
                                                               BLOCK_SIZE=BLOCK_SIZE, NUM_COMM_SMS=NUM_COMM_SMS,
                                                               num_warps=args.num_warps)

        for _ in range(args.warmup):
            run_ring_rs()
            run_pull_rs()
        torch.cuda.synchronize()
        dist.barrier()

        def _time(fn, it):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            a.record()
            for _ in range(it):
                fn()
            b.record()
            b.synchronize()
            return a.elapsed_time(b) / it

        t_ring = _time(run_ring_rs, args.iters)
        t_pull = _time(run_pull_rs, args.iters)
        remote = args.bytes * (world - 1) / world
        if rank == 0:
            print(f"\n# ring_rs vs pull_rs  bytes={args.bytes>>20}MB world={world}", flush=True)
            print(f"#   pull-RS (read, on-the-fly reduce) {t_pull:.4f} ms  ({remote/(t_pull*1e-3)/1e9:.0f} GB/s remote)", flush=True)
            print(f"#   ring-RS (write + local add, {world-1} steps) {t_ring:.4f} ms  ({remote/(t_ring*1e-3)/1e9:.0f} GB/s remote)", flush=True)
            print(f"#   ring/pull = {t_ring/t_pull:.2f}x  ({'ring WINS' if t_ring < t_pull else 'pull wins'})", flush=True)
        torch.cuda.synchronize()
        os._exit(0)

    if args.transport == "overlap_probe":
        # Direction-overlap capacity probe: can XGMI reads (RS) and writes
        # (push-AG) run concurrently? Times RS alone, push-AG alone, then both
        # on separate streams started together. Output is garbage (shared
        # buffers) — we only measure timing of read+write concurrency.
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        def _rs(stream):
            with torch.cuda.stream(stream):
                pure_allreduce_two_shot_rs_kernel[(NUM_COMM_SMS, )](symm_in, symm_z, numel, epb,
                                                                    BLOCK_SIZE=BLOCK_SIZE, NUM_COMM_SMS=NUM_COMM_SMS,
                                                                    num_warps=args.num_warps)

        def _ag(stream):
            with torch.cuda.stream(stream):
                pure_allreduce_two_shot_ag_push_kernel[(NUM_COMM_SMS, )](symm_z, symm_in, numel, epb,
                                                                         BLOCK_SIZE=BLOCK_SIZE,
                                                                         NUM_COMM_SMS=NUM_COMM_SMS,
                                                                         num_warps=args.num_warps)

        for _ in range(args.warmup):
            _rs(torch.cuda.current_stream())
            _ag(torch.cuda.current_stream())
        torch.cuda.synchronize()
        dist.barrier()

        def _time(fn, it):
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            a.record()
            for _ in range(it):
                fn()
            b.record()
            b.synchronize()
            return a.elapsed_time(b) / it

        t_rs = _time(lambda: _rs(torch.cuda.current_stream()), args.iters)
        t_ag = _time(lambda: _ag(torch.cuda.current_stream()), args.iters)

        # Concurrent: RS on s1, AG on s2, both gated on a common start event.
        torch.cuda.synchronize()
        dist.barrier()
        a = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        a.record()
        s1.wait_event(a)
        s2.wait_event(a)
        for _ in range(args.iters):
            _rs(s1)
            _ag(s2)
        e1.record(s1)
        e2.record(s2)
        torch.cuda.synchronize()
        wall = max(a.elapsed_time(e1), a.elapsed_time(e2)) / args.iters

        if rank == 0:
            seq = t_rs + t_ag
            ideal = max(t_rs, t_ag)
            speedup = seq / wall
            print(f"\n# overlap_probe bytes={args.bytes>>20}MB world={world}", flush=True)
            print(f"#   RS alone       {t_rs:.4f} ms  (read  {args.bytes*(world-1)/world/(t_rs*1e-3)/1e9:.0f} GB/s)", flush=True)
            print(f"#   push-AG alone  {t_ag:.4f} ms  (write {args.bytes*(world-1)/world/(t_ag*1e-3)/1e9:.0f} GB/s)", flush=True)
            print(f"#   concurrent     {wall:.4f} ms", flush=True)
            print(f"#   sequential sum {seq:.4f} ms ; ideal(max) {ideal:.4f} ms ; overlap speedup {speedup:.2f}x", flush=True)
        torch.cuda.synchronize()
        os._exit(0)

    def run_full_push(ev):
        ev[0].record()
        _barrier_all_v2(ctx)
        ev[1].record()
        pure_allreduce_two_shot_rs_push_kernel[(NUM_COMM_SMS, world)](symm_in, symm_z, numel, epb,
                                                                      BLOCK_SIZE=BLOCK_SIZE, NUM_COMM_SMS=NUM_COMM_SMS,
                                                                      num_warps=args.num_warps)
        ev[2].record()
        _barrier_all_v2(ctx)
        ev[3].record()
        pure_allreduce_two_shot_local_reduce_kernel[(NUM_COMM_SMS, )](symm_z, symm_in, numel, epb,
                                                                      BLOCK_SIZE=BLOCK_SIZE, NUM_COMM_SMS=NUM_COMM_SMS,
                                                                      num_warps=args.num_warps)
        ev[4].record()
        _barrier_all_v2(ctx)
        ev[5].record()
        pure_allreduce_two_shot_ag_push_kernel[(NUM_COMM_SMS, )](symm_in, symm_z, numel, epb,
                                                                 BLOCK_SIZE=BLOCK_SIZE, NUM_COMM_SMS=NUM_COMM_SMS,
                                                                 num_warps=args.num_warps)
        ev[6].record()
        _barrier_all_v2(ctx)
        ev[7].record()

    if args.transport == "one_shot":
        nseg, runner, labels = 3, run_one_shot, ["pre_barrier", "kernel", "post_barrier"]
    elif args.transport == "two_shot_fused":
        nseg, runner, labels = 3, run_two_shot_fused, ["pre_barrier", "fused_kernel", "post_barrier"]
    elif args.transport == "full_push":
        nseg, runner, labels = 7, run_full_push, ["pre_barrier", "rs_push", "bar1", "local_reduce",
                                                  "bar2", "ag_push", "post_barrier"]
    else:
        nseg, runner, labels = 5, run_two_shot, ["pre_barrier", "rs_kernel", "mid_barrier", "ag_kernel", "post_barrier"]

    for _ in range(args.warmup):
        runner(_events(nseg + 1))
    torch.cuda.synchronize()
    dist.barrier()

    sums = [0.0] * nseg
    total = 0.0
    for _ in range(args.iters):
        ev = _events(nseg + 1)
        runner(ev)
        ev[nseg].synchronize()
        for s in range(nseg):
            sums[s] += ev[s].elapsed_time(ev[s + 1])
        total += ev[0].elapsed_time(ev[nseg])

    avg = [s / args.iters for s in sums]
    tot = total / args.iters

    # Effective busbw using the same convention as bench_pure_ar.py.
    busbw = (args.bytes / (tot * 1e-3)) * 2.0 * (world - 1) / world / 1e9

    # --- Reconciliation against bench_pure_ar.py ---------------------------
    # (1) the real op end-to-end, measured back-to-back exactly like the bench
    #     (single start/end event around `iters` calls, no inter-iter sync);
    # (2) the isolated symm_in.copy_(x) cost that the op pays per call but a
    #     fused GEMM+AR path would not (GEMM writes straight into the symm buf).
    real_op = {"two_shot": pure_allreduce_two_shot_op,
               "two_shot_fused": pure_allreduce_two_shot_fused_op,
               "full_push": pure_allreduce_two_shot_push_op}.get(args.transport, pure_allreduce_one_shot_op)
    for _ in range(args.warmup):
        real_op(ctx, x)
    torch.cuda.synchronize()
    dist.barrier()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(args.iters):
        real_op(ctx, x)
    e1.record()
    e1.synchronize()
    fullop_ms = e0.elapsed_time(e1) / args.iters
    fullop_busbw = (args.bytes / (fullop_ms * 1e-3)) * 2.0 * (world - 1) / world / 1e9

    # Isolated copy cost.
    torch.cuda.synchronize()
    dist.barrier()
    e0.record()
    for _ in range(args.iters):
        symm_in.copy_(x.reshape(-1))
    e1.record()
    e1.synchronize()
    copy_ms = e0.elapsed_time(e1) / args.iters

    if rank == 0:
        print(f"\n# prof transport={args.transport} dtype={args.dtype} "
              f"bytes={args.bytes} ({args.bytes>>20}MB) world={world} "
              f"BLOCK_SIZE={BLOCK_SIZE} num_warps={args.num_warps} NUM_COMM_SMS={NUM_COMM_SMS}", flush=True)
        print(f"# per-phase total {tot:.4f} ms/iter  busbw {busbw:.2f} GB/s "
              f"(per-iter sync, NO copy)", flush=True)
        for lab, a in zip(labels, avg):
            print(f"#   {lab:<14} {a:.4f} ms  ({100*a/tot:5.1f}%)", flush=True)
        print(f"# real-op back-to-back {fullop_ms:.4f} ms/iter  busbw {fullop_busbw:.2f} GB/s "
              f"(matches bench method)", flush=True)
        print(f"#   symm_in.copy_ {copy_ms:.4f} ms ({100*copy_ms/fullop_ms:.1f}% of real-op)", flush=True)

    torch.cuda.synchronize()
    os._exit(0)


if __name__ == "__main__":
    main()
