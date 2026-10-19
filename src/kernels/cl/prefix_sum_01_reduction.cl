#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(SCAN_GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_01_reduction(
    __global const uint* input,
    __global uint* block_sums,
    unsigned int n)
{
    const uint lid = get_local_id(0);
    const size_t base = get_group_id(0) * SCAN_BLOCK_SIZE;
    uint sum = 0;
    #pragma unroll
    for (uint j = 0; j < SCAN_ITEMS_PER_THREAD; ++j) {
        const size_t index = base + lid + j * SCAN_GROUP_SIZE;
        if (index < n)
            sum += input[index];
    }

    __local uint sums[SCAN_GROUP_SIZE];
    sums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = SCAN_GROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (lid < stride)
            sums[lid] += sums[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        block_sums[get_group_id(0)] = sums[0];
}
