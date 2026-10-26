#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define RADIX_BITS 4
#define RADIX_BINS (1 << RADIX_BITS)
#define RADIX_GROUP_SIZE 128
#define RADIX_ITEMS_PER_THREAD 8
#define RADIX_BLOCK_SIZE (RADIX_GROUP_SIZE * RADIX_ITEMS_PER_THREAD)
#define SCAN_GROUP_SIZE 128
#define SCAN_ITEMS_PER_THREAD 16
#define SCAN_BLOCK_SIZE (SCAN_GROUP_SIZE * SCAN_ITEMS_PER_THREAD)

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
