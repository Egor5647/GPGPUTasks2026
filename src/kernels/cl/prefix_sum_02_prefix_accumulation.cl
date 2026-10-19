#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(SCAN_GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* input,
    __global const uint* block_prefix,
    __global uint* output,
    unsigned int n)
{
    const uint lid = get_local_id(0);
    const size_t group = get_group_id(0);
    const size_t base = group * SCAN_BLOCK_SIZE + lid * SCAN_ITEMS_PER_THREAD;
    uint values[SCAN_ITEMS_PER_THREAD];
    if (base + SCAN_ITEMS_PER_THREAD <= n) {
        vstore16(vload16(0, input + base), 0, values);
    } else {
        #pragma unroll
        for (uint j = 0; j < SCAN_ITEMS_PER_THREAD; ++j)
            values[j] = base + j < n ? input[base + j] : 0;
    }

    uint sum = 0;
    #pragma unroll
    for (uint j = 0; j < SCAN_ITEMS_PER_THREAD; ++j) {
        sum += values[j];
        values[j] = sum;
    }

    // Blelloch exclusive scan of the per-thread totals.
    __local uint sums[SCAN_GROUP_SIZE];
    sums[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 1; stride < SCAN_GROUP_SIZE; stride *= 2) {
        const uint right = (lid + 1) * 2 * stride - 1;
        if (right < SCAN_GROUP_SIZE)
            sums[right] += sums[right - stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        sums[SCAN_GROUP_SIZE - 1] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = SCAN_GROUP_SIZE / 2; stride > 0; stride /= 2) {
        const uint right = (lid + 1) * 2 * stride - 1;
        if (right < SCAN_GROUP_SIZE) {
            const uint left_sum = sums[right - stride];
            sums[right - stride] = sums[right];
            sums[right] += left_sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint offset = sums[lid] + (group == 0 ? 0 : block_prefix[group - 1]);
    #pragma unroll
    for (uint j = 0; j < SCAN_ITEMS_PER_THREAD; ++j)
        values[j] += offset;

    if (base + SCAN_ITEMS_PER_THREAD <= n) {
        vstore16(vload16(0, values), 0, output + base);
    } else {
        #pragma unroll
        for (uint j = 0; j < SCAN_ITEMS_PER_THREAD; ++j) {
            if (base + j < n)
                output[base + j] = values[j];
        }
    }
}
