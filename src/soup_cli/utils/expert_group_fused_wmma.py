"""Grouped packed NF4 WMMA kernel, compiled lazily with NVRTC."""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

_CACHE = {}


def launch_grouped(
    table: Any, shape: tuple[int, int], *, groups: int, max_rows: int,
    blocksize: int, nested: bool, nested_blocksize: int, transpose: bool,
) -> None:
    """Launch independent expert CTAs, using a shared-tile WMMA mainloop."""
    import torch

    driver, _module, function, _ptx, _log = compile_kernel(
        torch.bfloat16, blocksize, nested_blocksize)
    n, k = shape
    columns = k if transpose else n
    pointer = ctypes.c_void_p(table.data_ptr())
    scalars = [ctypes.c_int(x) for x in
               (groups, n, k, blocksize, int(nested), nested_blocksize, int(transpose))]
    parameters = (ctypes.c_void_p * 8)(
        *(ctypes.cast(ctypes.byref(arg), ctypes.c_void_p) for arg in [pointer, *scalars]))
    check(driver, "cuLaunchKernel", function, (columns + 31) // 32,
          (max_rows + 31) // 32, groups, 128, 1, 1, 0,
          ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), parameters, None)

SOURCE = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
using namespace nvcuda;
using scalar = SCALAR;
extern "C" __global__ void fused(const long long* table, int groups,
    int N, int K, int blocksize, int nested, int block2, int transpose) {
  int expert=blockIdx.z;
  int M=(int)table[8*groups+expert];
  if ((int)blockIdx.y*TILE_SIZE>=M) return;
  const scalar* x=(const scalar*)table[expert];
  const unsigned char* packed=(const unsigned char*)table[groups+expert];
  const void* absmax=(const void*)table[2*groups+expert];
  const float* code=(const float*)table[3*groups+expert];
  const float* absmax2=(const float*)table[4*groups+expert];
  const float* code2=(const float*)table[5*groups+expert];
  const float* offset=(const float*)table[6*groups+expert];
  scalar* output=(scalar*)table[7*groups+expert];
  __shared__ __align__(32) scalar A[TILE_SIZE*REDUCE_TILE];
  __shared__ __align__(32) scalar B[TILE_SIZE*REDUCE_TILE];
  __shared__ __align__(32) float C[TILE_SIZE*TILE_SIZE];
  int outsize = transpose ? K : N;
  int reduce = transpose ? N : K;
  int outbase = blockIdx.x * TILE_SIZE, mbase = blockIdx.y * TILE_SIZE;
  int warp = threadIdx.x / 32;
  int warp_out = warp % (TILE_SIZE/16), warp_m = warp / (TILE_SIZE/16);
  wmma::fragment<wmma::matrix_a,16,16,16,scalar,wmma::row_major> a;
  wmma::fragment<wmma::matrix_b,16,16,16,scalar,wmma::col_major> b;
  wmma::fragment<wmma::accumulator,16,16,16,float> c;
  wmma::fill_fragment(c,0.0f);
  for (int start=0; start<reduce; start+=REDUCE_TILE) {
    for (int q=threadIdx.x; q<TILE_SIZE*REDUCE_TILE; q+=blockDim.x) {
      // Decode contiguous original W rows, then transpose in shared memory.
      int i=transpose ? q%TILE_SIZE : q/REDUCE_TILE;
      int j=transpose ? q/TILE_SIZE : q%REDUCE_TILE;
      int out=outbase+i, inner=start+j;
      float weight=0.0f;
      if (out<outsize && inner<reduce) {
        long long pos=transpose ? (long long)inner*K+out : (long long)out*K+inner;
        int byte=packed[pos/2];
        int nibble=(pos%2==0) ? byte>>4 : byte&15;
        long long scale=pos/BLOCKSIZE;
        float magnitude;
        if (nested) {
          magnitude=code2[((const unsigned char*)absmax)[scale]]*absmax2[scale/BLOCK2];
          magnitude=magnitude+offset[0];
        } else magnitude=((const float*)absmax)[scale];
        weight=code[nibble]*magnitude;
      }
      A[i*REDUCE_TILE+j]=CONVERT(weight);
      int row=mbase+q/REDUCE_TILE, red=start+q%REDUCE_TILE;
      B[q]=(row<M && red<reduce) ? x[(long long)row*reduce+red] : CONVERT(0.0f);
    }
    __syncthreads();
    for (int inner=0; inner<REDUCE_TILE; inner+=16) {
      wmma::load_matrix_sync(a,A+warp_out*16*REDUCE_TILE+inner,REDUCE_TILE);
      wmma::load_matrix_sync(b,B+warp_m*16*REDUCE_TILE+inner,REDUCE_TILE);
      wmma::mma_sync(c,a,b,c);
    }
    __syncthreads();
  }
  wmma::store_matrix_sync(C+warp_m*16*TILE_SIZE+warp_out*16,c,TILE_SIZE,wmma::mem_col_major);
  __syncthreads();
  for (int q=threadIdx.x; q<TILE_SIZE*TILE_SIZE; q+=blockDim.x) {
    int row=mbase+q/TILE_SIZE, col=outbase+q%TILE_SIZE;
    if (row<M && col<outsize) output[(long long)row*outsize+col]=CONVERT(C[q]);
  }
}
"""


def check(library: ctypes.CDLL, function: str, *args: Any) -> None:
    code = getattr(library, function)(*args)
    if code:
        raise RuntimeError(f"{function} failed: {code}")


def compile_kernel(
    dtype: Any,
    blocksize: int = 64,
    block2: int = 256,
) -> tuple[Any, ...]:
    import torch

    if blocksize <= 0 or block2 < 0:
        raise ValueError("invalid NF4 scale block sizes")
    key = (torch.cuda.current_device(), dtype, blocksize, block2)
    if key in _CACHE:
        return _CACHE[key]
    packages = Path(torch.__file__).parents[1]
    nvrtc_paths = list(packages.glob("nvidia/cu13/lib/libnvrtc.so*"))
    nvrtc_paths += list(packages.glob("nvidia/cuda_nvrtc/lib/libnvrtc.so*"))
    if not nvrtc_paths:
        raise RuntimeError("CUDA NVRTC library not installed")
    nvrtc_path = nvrtc_paths[0]
    nvrtc = ctypes.CDLL(str(nvrtc_path))
    driver = ctypes.CDLL("libcuda.so.1")
    program = ctypes.c_void_p()
    source = (
        SOURCE.replace("SCALAR", "__nv_bfloat16")
        .replace("CONVERT", "__float2bfloat16_rn")
        .replace("BLOCKSIZE", str(blocksize))
        .replace("BLOCK2", str(max(1, block2)))
        .replace("TILE_SIZE", "32")
        .replace("REDUCE_TILE", "64")
        .encode()
    )
    check(
        nvrtc,
        "nvrtcCreateProgram",
        ctypes.byref(program),
        ctypes.c_char_p(source),
        ctypes.c_char_p(b"fused.cu"),
        0,
        None,
        None,
    )
    major, minor = torch.cuda.get_device_capability()
    headers = [
        packages / "nvidia/cu13/include",
        packages / "nvidia/cuda_runtime/include",
        packages / "triton/backends/nvidia/include",
        packages / "nvidia/cuda_cccl/include",
    ]
    options = [
        b"--std=c++17",
        b"--fmad=false",
        f"--gpu-architecture=compute_{major}{minor}".encode(),
    ]
    options += [("--include-path=" + str(p)).encode() for p in headers if p.exists()]
    arguments = (ctypes.c_char_p * len(options))(*options)
    status = nvrtc.nvrtcCompileProgram(program, len(options), arguments)
    size = ctypes.c_size_t()
    check(nvrtc, "nvrtcGetProgramLogSize", program, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    check(nvrtc, "nvrtcGetProgramLog", program, log)
    if status:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
        raise RuntimeError("NVRTC compile failed: " + log.value.decode())
    check(nvrtc, "nvrtcGetPTXSize", program, ctypes.byref(size))
    ptx = ctypes.create_string_buffer(size.value)
    check(nvrtc, "nvrtcGetPTX", program, ptx)
    check(nvrtc, "nvrtcDestroyProgram", ctypes.byref(program))
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    check(driver, "cuModuleLoadDataEx", ctypes.byref(module), ptx, 0, None, None)
    check(driver, "cuModuleGetFunction", ctypes.byref(function), module, ctypes.c_char_p(b"fused"))
    _CACHE[key] = (driver, module, function, ptx.raw, log.value.decode())
    return _CACHE[key]
