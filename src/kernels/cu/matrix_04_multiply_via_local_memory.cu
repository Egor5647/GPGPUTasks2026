#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float a_tile[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float b_tile[GROUP_SIZE_Y][GROUP_SIZE_X];
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    float acc = 0.0f;

    for (unsigned int k0 = 0; k0 < k; k0 += GROUP_SIZE_X) {
        a_tile[threadIdx.y][threadIdx.x] = a[j * k + k0 + threadIdx.x];
        b_tile[threadIdx.y][threadIdx.x] = b[(k0 + threadIdx.y) * w + i];
        __syncthreads();

        for (unsigned int ki = 0; ki < GROUP_SIZE_X; ++ki) {
            acc += a_tile[threadIdx.y][ki] * b_tile[ki][threadIdx.x];
        }
        __syncthreads();
    }

    c[j * w + i] = acc;
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
