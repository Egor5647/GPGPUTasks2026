#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(SCAN_GROUP_SIZE, 1, 1)))
__kernel void radix_sort_03_global_prefixes_scan_accumulation(
    __global uint* values,
    __global const uint* block_prefixes,
    uint n,
    uint has_parent)
{
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    const uint base = group * SCAN_BLOCK_SIZE + lid * SCAN_ITEMS_PER_THREAD;
    uint items[SCAN_ITEMS_PER_THREAD];
    uint sum = 0;
    #pragma unroll
    for (uint j = 0; j < SCAN_ITEMS_PER_THREAD; ++j) {
        const uint value = base + j < n ? values[base + j] : 0;
        items[j] = sum;
        sum += value;
    }

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
            const uint left = sums[right - stride];
            sums[right - stride] = sums[right];
            sums[right] += left;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint offset = sums[lid] + (has_parent ? block_prefixes[group] : 0);
    #pragma unroll
    for (uint j = 0; j < SCAN_ITEMS_PER_THREAD; ++j) {
        if (base + j < n)
            values[base + j] = offset + items[j];
    }
}
