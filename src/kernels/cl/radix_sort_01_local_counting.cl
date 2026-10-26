#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(RADIX_GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    __global const uint* values,
    __global uint* counts,
    uint n,
    uint shift)
{
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    const uint groups = get_num_groups(0);
    __local uint histogram[RADIX_GROUP_SIZE / 32][RADIX_BINS];
    for (uint i = lid; i < RADIX_GROUP_SIZE / 32 * RADIX_BINS; i += RADIX_GROUP_SIZE)
        histogram[i / RADIX_BINS][i % RADIX_BINS] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (uint j = 0; j < RADIX_ITEMS_PER_THREAD; ++j) {
        const uint index = group * RADIX_BLOCK_SIZE + lid + j * RADIX_GROUP_SIZE;
        if (index < n) {
            const uint digit = (values[index] >> shift) & (RADIX_BINS - 1);
            atomic_inc(&histogram[lid / 32][digit]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < RADIX_BINS) {
        uint sum = 0;
        for (uint i = 0; i < RADIX_GROUP_SIZE / 32; ++i)
            sum += histogram[i][lid];
        counts[lid * groups + group] = sum;
    }
}
