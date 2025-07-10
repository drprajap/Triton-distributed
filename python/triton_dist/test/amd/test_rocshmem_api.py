################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
# from triton.language.extra import libshmem_device

import argparse
import os
from typing import Optional
import datetime

from mpi4py import MPI
import numpy as np

from functools import partial

# from hip import hip
import triton
import torch
import triton.language as tl
import torch.distributed as dist
import triton_dist.language as dl
from triton.language.extra import libdevice
from triton.language.extra.hip import libdevice  # noqa: F811
from triton.language.extra import libshmem_device
import time
import pyrocshmem
import random

def test_rocshmem_basic():
    @triton.jit
    def _rocshmem_basic(comm_buf, ctx):
        
        libshmem_device.set_rocshmem_ctx(ctx)

        # dl_my_pe = dl.rank()
        # dl_num_ranks = dl.num_ranks()

        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        peer = (mype + 1) % npes

        # ipcBase = libshmem_device.get_device_ctx_ipc_base(mype)

        # rptr = libshmem_device.remote_ptr(ipcBase, peer)

        # tl.store(comm_buf, dl_my_pe)
        # comm_buf+=1
        # tl.store(comm_buf, dl_num_ranks)
        # comm_buf+=1
        tl.store(comm_buf, mype)
        comm_buf+=1
        tl.store(comm_buf, npes)


    @triton.jit
    def _rocshmem_put(ptr,ctx):
        libshmem_device.set_rocshmem_ctx(ctx)

        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        peer = (mype + 1) % npes

        libshmem_device.int_p(ptr, mype, peer)

    @triton.jit
    def _rocshmem_put_symm_at(ptr,ctx, comm_buf):
        libshmem_device.set_rocshmem_ctx(ctx)

        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        peer = (mype + 1) % npes
        num_blocks = tl.num_programs(axis=0)
        start_id = tl.program_id(axis=0)
        # libshmem_device.int_p(ptr, mype, peer)
        #remote_ptr = dl.symm_at(ptr, peer)
        remote_ptr = libshmem_device.remote_ptr(ptr, peer)
        for i in range (1, npes):
            src_rank = (mype + i) % npes
            rank_offset = src_rank * 4
            # for pid in range(start_id, 1, num_blocks):
            boffset = start_id + tl.arange(0, 4)
            val = tl.load(remote_ptr + rank_offset+ boffset)
            tl.store(ptr +rank_offset + boffset, val)


    print("**rocshmem basic start!")
    pyrocshmem.rocshmem_init()

    my_pe = pyrocshmem.rocshmem_my_pe()
    
    npes =  pyrocshmem.rocshmem_n_pes()
    peer = (my_pe + 1) % npes

    print('mype: {} -- num_pes: {}'.format(my_pe, npes))

    ctx = pyrocshmem.rocshmem_get_device_ctx()
    # print("ctx - {}".format(hex(ctx)))

    # ipcbase = pyrocshmem.rocshmem_ptr(peer)
    # print("ipcbase - {}".format(hex(ipcbase)))

    # M = 16
    # N=16
    # K=8
    comm_buffs = pyrocshmem.rocshmem_create_tensor_list_intra_node([npes],torch.int32)

    comm_buffs[rank].fill_(0)
    comm_buf_ptr = torch.tensor([t.data_ptr() for t in comm_buffs], device=torch.cuda.current_device(),
                                requires_grad=False)
    peer = (my_pe + 1) % npes
    print(f"mype#: {rank} peer# {peer} ptr[{rank}]: {hex(comm_buf_ptr[rank])} ptr[{peer}]: {hex(comm_buf_ptr[peer])}")

    # workspace_tensors = pyrocshmem.rocshmem_create_tensor_list_intra_node([M, K],torch.int32)

    # local_barrier_buff = pyrocshmem.rocshmem_create_tensor([M],torch.int32)

    comm_buf = pyrocshmem.rocshmem_create_tensor((2,), torch.int32)

    _rocshmem_basic[(1, )](comm_buf, ctx)
    print(f"_rocshmem_basic [dl.rank , dl.num_ranks] from pe#{my_pe}: {comm_buf}")
    
    pyrocshmem.rocshmem_barrier_all()

    # try:
    #     torch.testing.assert_close(
    #         comm_buf,
    #         torch.tensor([my_pe, npes], dtype=torch.int32,
    #                      device="cuda")), comm_buf
    # except Exception as e:
    #     print(" _rocshmem_basic failed")
    #     raise (e)
    # else:
    #     print("✅ _rocshmem_basic pass")
    
    comm_buf.zero_()
    put_buf = pyrocshmem.rocshmem_create_tensor((1,), torch.int32)

    _rocshmem_put[(1, )](put_buf, ctx)
    pyrocshmem.rocshmem_barrier_all()

    # print(f"put_buf from pe#{my_pe}: {put_buf}")
    nelems_per_rank = 4
    n_elements = npes*nelems_per_rank
    dtype = torch.int32

    put_bufs = pyrocshmem.rocshmem_create_tensor((n_elements,), torch.int32)
    ref_tensor = torch.arange(n_elements, dtype=dtype).cuda()
    put_bufs[nelems_per_rank * my_pe : nelems_per_rank *(my_pe+1)].copy_(ref_tensor[nelems_per_rank * my_pe : nelems_per_rank *(my_pe+1)])
    pyrocshmem.rocshmem_barrier_all()
    _rocshmem_put_symm_at[(1, )](put_bufs, ctx,comm_buf)
    pyrocshmem.rocshmem_barrier_all()

    print(f"put_buf remote_ptr from pe#{my_pe}: {put_bufs}")

    try:
        torch.testing.assert_close(put_bufs, ref_tensor, atol=0, rtol=0)
    except Exception as e:
        print(f"❌ RANK[{my_pe}] check failed")
        raise e
    else:
        print(f"✅ RANK[{my_pe}] check passed")  

    pyrocshmem.rocshmem_finalize()


def test_rocshmem_getmem():
    @triton.jit
    def _rocshmem_getmem(ctx):
        libshmem_device.set_rocshmem_ctx(ctx)

        my_pe = libshmem_device.my_pe()
        num_pes = libshmem_device.n_pes()
        


if __name__ == "__main__":
    # init
    # args = parse_args()
 
    ## Keep this for correlating run with torch
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    os.environ["RANK"]  = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)


    torch.distributed.init_process_group(
            backend="nccl", init_method="env://")

    # TP_GROUP = torch.distributed.new_group(ranks=list(range(torch.distributed.get_world_size())), backend="mpi")
    # torch.distributed.barrier(TP_GROUP)

    # torch.cuda.synchronize()
    # torch.distributed.barrier()
    test_rocshmem_basic()


