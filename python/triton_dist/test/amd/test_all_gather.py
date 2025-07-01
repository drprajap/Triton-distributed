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

# from mpi4py import MPI
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
    def all_gather_kernel(ptr, ctx):
        libshmem_device.set_rocshmem_ctx(ctx)
        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()

        

    print("rocshmem basic start!")
    my_pe = pyrocshmem.rocshmem_my_pe()
    
    npes =  pyrocshmem.rocshmem_n_pes()
    peer = (my_pe + 1) % npes

    print('mype: {} -- num_pes: {}'.format(my_pe, npes))
    pyrocshmem.rocshmem_init()

    ctx = pyrocshmem.rocshmem_get_device_ctx()
    # print("ctx - {}".format(hex(ctx)))

    # M = 16
    # N=16
    # K=8

    # workspace_tensors = pyrocshmem.rocshmem_create_tensor_list_intra_node([M, K],torch.int32)

    # local_barrier_buff = pyrocshmem.rocshmem_create_tensor([M],torch.int32)

    comm_buf = pyrocshmem.rocshmem_create_tensor((2,), torch.int32)

    _rocshmem_basic[(1, )](comm_buf, ctx)
    print(f"_rocshmem_basic [dl.rank , dl.num_ranks] from pe#{my_pe}: {comm_buf}")
    
    pyrocshmem.rocshmem_barrier_all()

    try:
        torch.testing.assert_close(
            comm_buf,
            torch.tensor([my_pe, npes], dtype=torch.int32,
                         device="cuda")), comm_buf
    except Exception as e:
        print(" _rocshmem_basic failed")
        raise (e)
    else:
        print("✅ _rocshmem_basic pass")
    
    comm_buf.zero_()
    put_buf = pyrocshmem.rocshmem_create_tensor((1,), torch.int32)

    _rocshmem_put[(1, )](put_buf, ctx)
    pyrocshmem.rocshmem_barrier_all()

    print(f"put_buf from pe#{my_pe}: {put_buf}")

    pyrocshmem.rocshmem_finalize()


if __name__ == "__main__":
    # init
    # args = parse_args()
 
    ## Keep this for correlating run with torch
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # world_size = comm.Get_size()
    # torch.distributed.init_process_group(
    #         backend="mpi")
    # print('Hello from process {} (out of {})!'.format(torch.distributed.get_rank(), torch.distributed.get_world_size()))

    # TP_GROUP = torch.distributed.new_group(ranks=list(range(torch.distributed.get_world_size())), backend="mpi")
    # torch.distributed.barrier(TP_GROUP)

    # torch.cuda.synchronize()
    # torch.distributed.barrier()
    test_rocshmem_basic()


