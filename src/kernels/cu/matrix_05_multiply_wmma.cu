#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
// Если строка "using namespace nvcuda;" не компилируется, добавьте в CMake options: -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    constexpr int tile_size = 16;
    __shared__ __align__(32) half a_tile[tile_size * tile_size];
    __shared__ __align__(32) half b_tile[tile_size * tile_size];

    const unsigned int row_tile = blockIdx.y;
    const unsigned int col_tile = blockIdx.x;
    const unsigned int lane = threadIdx.x;

    wmma::fragment<wmma::accumulator, tile_size, tile_size, tile_size, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (unsigned int k0 = 0; k0 < k; k0 += tile_size) {
        for (unsigned int index = lane; index < tile_size * tile_size; index += blockDim.x) {
            const unsigned int row = index / tile_size;
            const unsigned int col = index % tile_size;
            a_tile[index] = __float2half(a[(row_tile * tile_size + row) * k + k0 + col]);
            b_tile[index] = __float2half(b[(k0 + row) * w + col_tile * tile_size + col]);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, tile_size, tile_size, tile_size, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, tile_size, tile_size, tile_size, half, wmma::row_major> b_fragment;
        wmma::load_matrix_sync(a_fragment, a_tile, tile_size);
        wmma::load_matrix_sync(b_fragment, b_tile, tile_size);
        wmma::mma_sync(acc, a_fragment, b_fragment, acc);
        __syncthreads();
    }

    wmma::store_matrix_sync(c + (row_tile * tile_size) * w + col_tile * tile_size,
                            acc, w, wmma::mem_row_major);
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
