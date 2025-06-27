/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "rocshmem_wrapper.h"

#include <hip/hip_runtime.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

extern "C" {

__device__ int __attribute__((visibility("default"))) rocshmem_my_pe_kernel() {
  return rocshmem_my_pe();
}


__device__ void __attribute__((visibility("default"))) rocshmem_set_rocshmem_ctx(
  void *ctx) {
  ROCSHMEM_CTX_DEFAULT.ctx_opaque = ctx;
}

__device__ int __attribute__((used)) rocshmem_n_pes_kernel() {
  return rocshmem_n_pes();
}

__device__ void __attribute__((used)) rocshmem_ptr_kernel(void *src,
                                                                     void *dest,
                                                                     int pe) {
  dest = rocshmem_ptr(src, pe);
}

__device__ void __attribute__((used)) rocshmem_int_p_kernel(
    int *dest, int value, int pe) {
  rocshmem_int_p(dest, value, pe);
}

__device__ void __attribute__((used)) rocshmem_get_next_pe_kernel(
    int *dest) {
  int mype = rocshmem_my_pe();
  int npes = rocshmem_n_pes();
  int peer = (mype + 1) % npes;

  rocshmem_int_p(dest, mype, peer);
}

__device__ void testing_wrapper(int *sym_buf) {
  int tid = threadIdx.x;
  int mype = rocshmem_my_pe();

  if (tid < 4) {
    sym_buf[tid] = rocshmem_my_pe() + 100;
    printf("\n testing_wrapper_kernel >> device ptr: %p mype: %d sym_buf[%d]: %d ", sym_buf, mype, tid,
           sym_buf[tid]);
  } else {
    sym_buf[tid] = 501;
    printf("\n testing_wrapper_kernel >> device ptr: %p mype: %d sym_buf[%d]: %d ", sym_buf, mype, tid,
           sym_buf[tid]);
  }
}

}
extern "C" {

__device__ int rocshmem_my_pe_wrapper() { return rocshmem_my_pe(); }

__device__ int rocshmem_n_pes_wrapper() { return rocshmem_n_pes(); }

__device__ void *rocshmem_ptr_wrapper(void *dest, int pe) {
  return rocshmem_ptr(dest, pe);
}
}
