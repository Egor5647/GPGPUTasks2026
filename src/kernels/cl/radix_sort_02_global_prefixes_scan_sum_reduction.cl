#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(SCAN_GROUP_SIZE, 1, 1)))
__kernel void radix_sort_02_global_prefixes_scan_sum_reduction(
    __global const uint* values,
    __global uint* block_sums,
    uint n)
{
    const uint lid = get_local_id(0);
    const uint base = get_group_id(0) * SCAN_BLOCK_SIZE;
    uint sum = 0;
    #pragma unroll
    for (uint j = 0; j < SCAN_ITEMS_PER_THREAD; ++j) {
        const uint index = base + lid + j * SCAN_GROUP_SIZE;
        if (index < n)
            sum += values[index];
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
