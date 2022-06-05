#include <arm_neon.h>

#include "poly_utils.h"

void transpose8x8x16(uint16_t polys[8 * 8 * 16]) {
    uint16x8_t  v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,
                v8,  v9, v10, v11, v12, v13, v14, v15,
               v16, v17, v18, v19;

    for (int i = 0; i != 8; ++i) {
        v0 = vld1q_u16(polys + 0 * 16);                                 // [0, 1, 2, 3, 4, 5, 6, 7]
        v1 = vld1q_u16(polys + 1 * 16);
        v2 = vld1q_u16(polys + 2 * 16);
        v3 = vld1q_u16(polys + 3 * 16);
        v4 = vld1q_u16(polys + 4 * 16);
        v5 = vld1q_u16(polys + 5 * 16);
        v6 = vld1q_u16(polys + 6 * 16);
        v7 = vld1q_u16(polys + 7 * 16);

        v8 = vtrn1q_u16(v0, v1);                                        // [0, 0, 2, 2, 4, 4, 6, 6]
        v9 = vtrn2q_u16(v0, v1);                                        // [1, 1, 3, 3, 5, 5, 7, 7]
        v10 = vtrn1q_u16(v2, v3);
        v11 = vtrn2q_u16(v2, v3);
        v12 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [0, 0, 0, 0, 4, 4, 4, 4]
        v13 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [2, 2, 2, 2, 6, 6, 6, 6]
        v14 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [1, 1, 1, 1, 5, 5, 5, 5]
        v15 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [3, 3, 3, 3, 7, 7, 7, 7]

        v8 = vtrn1q_u16(v4, v5);
        v9 = vtrn2q_u16(v4, v5);
        v10 = vtrn1q_u16(v6, v7);
        v11 = vtrn2q_u16(v6, v7);
        v16 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v17 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v18 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);
        v19 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);

        v0 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v4 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v2 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v6 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v1 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v5 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v3 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v15, (uint64x2_t)v19);
        v7 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v15, (uint64x2_t)v19);

        vst1q_u16(polys + 0 * 16, v0);
        vst1q_u16(polys + 1 * 16, v1);
        vst1q_u16(polys + 2 * 16, v2);
        vst1q_u16(polys + 3 * 16, v3);
        vst1q_u16(polys + 4 * 16, v4);
        vst1q_u16(polys + 5 * 16, v5);
        vst1q_u16(polys + 6 * 16, v6);
        vst1q_u16(polys + 7 * 16, v7);

        v0 = vld1q_u16(polys + 0 * 16 + 8);                              // [0, 1, 2, 3, 4, 5, 6, 7]
        v1 = vld1q_u16(polys + 1 * 16 + 8);
        v2 = vld1q_u16(polys + 2 * 16 + 8);
        v3 = vld1q_u16(polys + 3 * 16 + 8);
        v4 = vld1q_u16(polys + 4 * 16 + 8);
        v5 = vld1q_u16(polys + 5 * 16 + 8);
        v6 = vld1q_u16(polys + 6 * 16 + 8);
        v7 = vld1q_u16(polys + 7 * 16 + 8);

        v8 = vtrn1q_u16(v0, v1);                                        // [0, 0, 2, 2, 4, 4, 6, 6]
        v9 = vtrn2q_u16(v0, v1);                                        // [1, 1, 3, 3, 5, 5, 7, 7]
        v10 = vtrn1q_u16(v2, v3);
        v11 = vtrn2q_u16(v2, v3);
        v12 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [0, 0, 0, 0, 4, 4, 4, 4]
        v13 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [2, 2, 2, 2, 6, 6, 6, 6]
        v14 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [1, 1, 1, 1, 5, 5, 5, 5]
        v15 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [3, 3, 3, 3, 7, 7, 7, 7]

        v8 = vtrn1q_u16(v4, v5);
        v9 = vtrn2q_u16(v4, v5);
        v10 = vtrn1q_u16(v6, v7);
        v11 = vtrn2q_u16(v6, v7);
        v16 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v17 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v18 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);
        v19 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);

        v0 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v4 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v2 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v6 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v1 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v5 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v3 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v15, (uint64x2_t)v19);
        v7 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v15, (uint64x2_t)v19);

        vst1q_u16(polys + 0 * 16 + 8, v0);
        vst1q_u16(polys + 1 * 16 + 8, v1);
        vst1q_u16(polys + 2 * 16 + 8, v2);
        vst1q_u16(polys + 3 * 16 + 8, v3);
        vst1q_u16(polys + 4 * 16 + 8, v4);
        vst1q_u16(polys + 5 * 16 + 8, v5);
        vst1q_u16(polys + 6 * 16 + 8, v6);
        vst1q_u16(polys + 7 * 16 + 8, v7);

        polys += 128;
    }
}

void transpose8x8x32(uint16_t polys[8 * 8 * 32]) {
    uint16x8_t  v0,  v1,  v2,  v3,  v4,  v5,  v6,  v7,
                v8,  v9, v10, v11, v12, v13, v14, v15,
               v16, v17, v18, v19;

    for (int i = 0; i != 8; ++i) {
        v0 = vld1q_u16(polys + 0 * 32);                                 // [0, 1, 2, 3, 4, 5, 6, 7]
        v1 = vld1q_u16(polys + 1 * 32);
        v2 = vld1q_u16(polys + 2 * 32);
        v3 = vld1q_u16(polys + 3 * 32);
        v4 = vld1q_u16(polys + 4 * 32);
        v5 = vld1q_u16(polys + 5 * 32);
        v6 = vld1q_u16(polys + 6 * 32);
        v7 = vld1q_u16(polys + 7 * 32);

        v8 = vtrn1q_u16(v0, v1);                                        // [0, 0, 2, 2, 4, 4, 6, 6]
        v9 = vtrn2q_u16(v0, v1);                                        // [1, 1, 3, 3, 5, 5, 7, 7]
        v10 = vtrn1q_u16(v2, v3);
        v11 = vtrn2q_u16(v2, v3);
        v12 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [0, 0, 0, 0, 4, 4, 4, 4]
        v13 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [2, 2, 2, 2, 6, 6, 6, 6]
        v14 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [1, 1, 1, 1, 5, 5, 5, 5]
        v15 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [3, 3, 3, 3, 7, 7, 7, 7]

        v8 = vtrn1q_u16(v4, v5);
        v9 = vtrn2q_u16(v4, v5);
        v10 = vtrn1q_u16(v6, v7);
        v11 = vtrn2q_u16(v6, v7);
        v16 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v17 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v18 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);
        v19 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);

        v0 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v4 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v2 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v6 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v1 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v5 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v3 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v15, (uint64x2_t)v19);
        v7 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v15, (uint64x2_t)v19);

        vst1q_u16(polys + 0 * 32, v0);
        vst1q_u16(polys + 1 * 32, v1);
        vst1q_u16(polys + 2 * 32, v2);
        vst1q_u16(polys + 3 * 32, v3);
        vst1q_u16(polys + 4 * 32, v4);
        vst1q_u16(polys + 5 * 32, v5);
        vst1q_u16(polys + 6 * 32, v6);
        vst1q_u16(polys + 7 * 32, v7);

        v0 = vld1q_u16(polys + 0 * 32 + 8);                              // [0, 1, 2, 3, 4, 5, 6, 7]
        v1 = vld1q_u16(polys + 1 * 32 + 8);
        v2 = vld1q_u16(polys + 2 * 32 + 8);
        v3 = vld1q_u16(polys + 3 * 32 + 8);
        v4 = vld1q_u16(polys + 4 * 32 + 8);
        v5 = vld1q_u16(polys + 5 * 32 + 8);
        v6 = vld1q_u16(polys + 6 * 32 + 8);
        v7 = vld1q_u16(polys + 7 * 32 + 8);

        v8 = vtrn1q_u16(v0, v1);                                        // [0, 0, 2, 2, 4, 4, 6, 6]
        v9 = vtrn2q_u16(v0, v1);                                        // [1, 1, 3, 3, 5, 5, 7, 7]
        v10 = vtrn1q_u16(v2, v3);
        v11 = vtrn2q_u16(v2, v3);
        v12 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [0, 0, 0, 0, 4, 4, 4, 4]
        v13 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [2, 2, 2, 2, 6, 6, 6, 6]
        v14 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [1, 1, 1, 1, 5, 5, 5, 5]
        v15 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [3, 3, 3, 3, 7, 7, 7, 7]

        v8 = vtrn1q_u16(v4, v5);
        v9 = vtrn2q_u16(v4, v5);
        v10 = vtrn1q_u16(v6, v7);
        v11 = vtrn2q_u16(v6, v7);
        v16 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v17 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v18 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);
        v19 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);

        v0 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v4 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v2 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v6 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v1 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v5 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v3 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v15, (uint64x2_t)v19);
        v7 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v15, (uint64x2_t)v19);

        vst1q_u16(polys + 0 * 32 + 8, v0);
        vst1q_u16(polys + 1 * 32 + 8, v1);
        vst1q_u16(polys + 2 * 32 + 8, v2);
        vst1q_u16(polys + 3 * 32 + 8, v3);
        vst1q_u16(polys + 4 * 32 + 8, v4);
        vst1q_u16(polys + 5 * 32 + 8, v5);
        vst1q_u16(polys + 6 * 32 + 8, v6);
        vst1q_u16(polys + 7 * 32 + 8, v7);

        v0 = vld1q_u16(polys + 0 * 32 + 16);                            // [0, 1, 2, 3, 4, 5, 6, 7]
        v1 = vld1q_u16(polys + 1 * 32 + 16);
        v2 = vld1q_u16(polys + 2 * 32 + 16);
        v3 = vld1q_u16(polys + 3 * 32 + 16);
        v4 = vld1q_u16(polys + 4 * 32 + 16);
        v5 = vld1q_u16(polys + 5 * 32 + 16);
        v6 = vld1q_u16(polys + 6 * 32 + 16);
        v7 = vld1q_u16(polys + 7 * 32 + 16);

        v8 = vtrn1q_u16(v0, v1);                                        // [0, 0, 2, 2, 4, 4, 6, 6]
        v9 = vtrn2q_u16(v0, v1);                                        // [1, 1, 3, 3, 5, 5, 7, 7]
        v10 = vtrn1q_u16(v2, v3);
        v11 = vtrn2q_u16(v2, v3);
        v12 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [0, 0, 0, 0, 4, 4, 4, 4]
        v13 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [2, 2, 2, 2, 6, 6, 6, 6]
        v14 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [1, 1, 1, 1, 5, 5, 5, 5]
        v15 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [3, 3, 3, 3, 7, 7, 7, 7]

        v8 = vtrn1q_u16(v4, v5);
        v9 = vtrn2q_u16(v4, v5);
        v10 = vtrn1q_u16(v6, v7);
        v11 = vtrn2q_u16(v6, v7);
        v16 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v17 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v18 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);
        v19 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);

        v0 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v4 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v2 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v6 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v1 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v5 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v3 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v15, (uint64x2_t)v19);
        v7 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v15, (uint64x2_t)v19);

        vst1q_u16(polys + 0 * 32 + 16, v0);
        vst1q_u16(polys + 1 * 32 + 16, v1);
        vst1q_u16(polys + 2 * 32 + 16, v2);
        vst1q_u16(polys + 3 * 32 + 16, v3);
        vst1q_u16(polys + 4 * 32 + 16, v4);
        vst1q_u16(polys + 5 * 32 + 16, v5);
        vst1q_u16(polys + 6 * 32 + 16, v6);
        vst1q_u16(polys + 7 * 32 + 16, v7);

        v0 = vld1q_u16(polys + 0 * 32 + 24);                             // [0, 1, 2, 3, 4, 5, 6, 7]
        v1 = vld1q_u16(polys + 1 * 32 + 24);
        v2 = vld1q_u16(polys + 2 * 32 + 24);
        v3 = vld1q_u16(polys + 3 * 32 + 24);
        v4 = vld1q_u16(polys + 4 * 32 + 24);
        v5 = vld1q_u16(polys + 5 * 32 + 24);
        v6 = vld1q_u16(polys + 6 * 32 + 24);
        v7 = vld1q_u16(polys + 7 * 32 + 24);

        v8 = vtrn1q_u16(v0, v1);                                        // [0, 0, 2, 2, 4, 4, 6, 6]
        v9 = vtrn2q_u16(v0, v1);                                        // [1, 1, 3, 3, 5, 5, 7, 7]
        v10 = vtrn1q_u16(v2, v3);
        v11 = vtrn2q_u16(v2, v3);
        v12 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [0, 0, 0, 0, 4, 4, 4, 4]
        v13 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);  // [2, 2, 2, 2, 6, 6, 6, 6]
        v14 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [1, 1, 1, 1, 5, 5, 5, 5]
        v15 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);  // [3, 3, 3, 3, 7, 7, 7, 7]

        v8 = vtrn1q_u16(v4, v5);
        v9 = vtrn2q_u16(v4, v5);
        v10 = vtrn1q_u16(v6, v7);
        v11 = vtrn2q_u16(v6, v7);
        v16 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v17 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v8, (uint32x4_t)v10);
        v18 = (uint16x8_t)vtrn1q_u32((uint32x4_t)v9, (uint32x4_t)v11);
        v19 = (uint16x8_t)vtrn2q_u32((uint32x4_t)v9, (uint32x4_t)v11);

        v0 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v4 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v12, (uint64x2_t)v16);
        v2 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v6 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v13, (uint64x2_t)v17);
        v1 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v5 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v14, (uint64x2_t)v18);
        v3 = (uint16x8_t)vtrn1q_u64((uint64x2_t)v15, (uint64x2_t)v19);
        v7 = (uint16x8_t)vtrn2q_u64((uint64x2_t)v15, (uint64x2_t)v19);

        vst1q_u16(polys + 0 * 32 + 24, v0);
        vst1q_u16(polys + 1 * 32 + 24, v1);
        vst1q_u16(polys + 2 * 32 + 24, v2);
        vst1q_u16(polys + 3 * 32 + 24, v3);
        vst1q_u16(polys + 4 * 32 + 24, v4);
        vst1q_u16(polys + 5 * 32 + 24, v5);
        vst1q_u16(polys + 6 * 32 + 24, v6);
        vst1q_u16(polys + 7 * 32 + 24, v7);

        polys += 256;
    }
}