#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(RADIX_GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    __global const uint* values,
    __global const uint* prefixes,
    __global uint* output,
    uint n,
    uint shift)
{
    const uint lid = get_local_id(0);
    const uint group = get_group_id(0);
    const uint groups = get_num_groups(0);
    const uint base = group * RADIX_BLOCK_SIZE;
    __local uint tile[RADIX_BLOCK_SIZE + RADIX_BLOCK_SIZE / 32];
    __local uint sums[2][RADIX_GROUP_SIZE];
    __local uint starts[RADIX_BINS];

    #pragma unroll
    for (uint j = 0; j < RADIX_ITEMS_PER_THREAD; ++j) {
        const uint index = lid + j * RADIX_GROUP_SIZE;
        tile[index + index / 32] = base + index < n ? values[base + index] : 0xffffffffu;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Stable binary partitions sort the current digit inside the group.
    for (uint bit = shift; bit < shift + RADIX_BITS; ++bit) {
        uint items[RADIX_ITEMS_PER_THREAD];
        uint zeros = 0;
        #pragma unroll
        for (uint j = 0; j < RADIX_ITEMS_PER_THREAD; ++j) {
            const uint index = lid * RADIX_ITEMS_PER_THREAD + j;
            items[j] = tile[index + index / 32];
            zeros += 1 - ((items[j] >> bit) & 1);
        }
        sums[0][lid] = zeros;
        barrier(CLK_LOCAL_MEM_FENCE);
        uint current = 0;
        for (uint stride = 1; stride < RADIX_GROUP_SIZE; stride *= 2) {
            uint sum = sums[current][lid];
            if (lid >= stride)
                sum += sums[current][lid - stride];
            sums[1 - current][lid] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);
            current = 1 - current;
        }
        const uint total_zeros = sums[current][RADIX_GROUP_SIZE - 1];
        uint zero_position = sums[current][lid] - zeros;
        uint one_position = total_zeros + lid * RADIX_ITEMS_PER_THREAD - zero_position;
        #pragma unroll
        for (uint j = 0; j < RADIX_ITEMS_PER_THREAD; ++j) {
            const uint position = ((items[j] >> bit) & 1) ? one_position++ : zero_position++;
            tile[position + position / 32] = items[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (uint j = 0; j < RADIX_ITEMS_PER_THREAD; ++j) {
        const uint index = lid + j * RADIX_GROUP_SIZE;
        const uint digit = (tile[index + index / 32] >> shift) & (RADIX_BINS - 1);
        if (index == 0 || ((tile[index - 1 + (index - 1) / 32] >> shift) & (RADIX_BINS - 1)) != digit)
            starts[digit] = index;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (uint j = 0; j < RADIX_ITEMS_PER_THREAD; ++j) {
        const uint index = lid + j * RADIX_GROUP_SIZE;
        if (index < min((uint)RADIX_BLOCK_SIZE, n - base)) {
            const uint value = tile[index + index / 32];
            const uint digit = (value >> shift) & (RADIX_BINS - 1);
            output[prefixes[digit * groups + group] + index - starts[digit]] = value;
        }
    }
}
