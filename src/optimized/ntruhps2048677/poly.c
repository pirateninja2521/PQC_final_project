#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include "poly.h"
#include "poly_utils.h"

/* Polynomial multiplication using     */
/* Toom-3, Toom-4 and two layers of Karatsuba. */

/* L -> L/3 -> L/12 -> L/48 */
#define PAD48(X) ((((X) + 47)/48)*48)
#define PAD16(X) ((((X) + 15)/16)*16)
#define L PAD48(NTRU_N)
#define N (L/3)
#define M (L/12)
#define K (L/48)
#define K16 PAD16(K)

static void toom3_toom4_k2x2_mul(uint16_t ab[2 * L], const uint16_t a[L], const uint16_t b[L]);

static void toom3_toom4_k2x2_eval_0(uint16_t r[64 * K16], const uint16_t a[L]);
static void toom3_toom4_k2x2_eval_p1(uint16_t r[64 * K16], const uint16_t a[L]);
static void toom3_toom4_k2x2_eval_m1(uint16_t r[64 * K16], const uint16_t a[L]);
static void toom3_toom4_k2x2_eval_m2(uint16_t r[64 * K16], const uint16_t a[L]);
static void toom3_toom4_k2x2_eval_inf(uint16_t r[64 * K16], const uint16_t a[L]);
static inline void toom4_k2x2_eval(uint16_t r[64 * K16], const uint16_t a[N]);

static inline void toom4_k2x2_eval_0(uint16_t r[9 * K16], const uint16_t a[N]);
static inline void toom4_k2x2_eval_p1(uint16_t r[9 * K16], const uint16_t a[N]);
static inline void toom4_k2x2_eval_m1(uint16_t r[9 * K16], const uint16_t a[N]);
static inline void toom4_k2x2_eval_p2(uint16_t r[9 * K16], const uint16_t a[N]);
static inline void toom4_k2x2_eval_m2(uint16_t r[9 * K16], const uint16_t a[N]);
static inline void toom4_k2x2_eval_p3(uint16_t r[9 * K16], const uint16_t a[N]);
static inline void toom4_k2x2_eval_inf(uint16_t r[9 * K16], const uint16_t a[N]);
static inline void k2x2_eval(uint16_t r[9 * K16]);

static void toom3_toom4_k2x2_basemul(uint16_t r[2 * 64 * K16], const uint16_t a[64 * K16], const uint16_t b[64 * K16]);
static void toom3_toom4_k2x2_basemul_neon(uint16_t r[2 * 64 * K16], uint16_t a[64 * K16], uint16_t b[64 * K16]);
static inline void schoolbook_KxK(uint16_t r[2 * K16], const uint16_t a[K16], const uint16_t b[K16]);
static inline void schoolbook_KxK_neon(uint16_t r[16 * K16], const uint16_t a[8 * K16], const uint16_t b[8 * K16]);

static void toom3_toom4_k2x2_interpolate(uint16_t r[2 * L], const uint16_t a[5 * 2 * 63 * K]);
static void toom4_k2x2_interpolate(uint16_t r[2 * N], const uint16_t a[63 * 2 * K]);
static inline void k2x2_interpolate(uint16_t r[2 * M], const uint16_t a[18 * K]);

void poly_Rq_mul_small(poly *r, const poly *a, const poly *b) {
    size_t i;
    uint16_t ab[2 * L];

    for (i = 0; i < NTRU_N; i++) {
        ab[i] = a->coeffs[i];
        ab[L + i] = b->coeffs[i];
    }
    for (i = NTRU_N; i < L; i++) {
        ab[i] = 0;
        ab[L + i] = 0;
    }

    toom3_toom4_k2x2_mul(ab, ab, ab + L);

    for (i = 0; i < NTRU_N; i++) {
        r->coeffs[i] = ab[i] + ab[NTRU_N + i];
    }
}

static void toom3_toom4_k2x2_mul(uint16_t ab[2 * L], const uint16_t a[L], const uint16_t b[L]) {
    uint16_t tmpA[64 * K16];
    uint16_t tmpB[64 * K16];
    uint16_t eC[5 * 2 * 64 * K16];

    toom3_toom4_k2x2_eval_0(tmpA, a);
    toom3_toom4_k2x2_eval_0(tmpB, b);
    toom3_toom4_k2x2_basemul_neon(eC + 0 * 2 * 64 * K16, tmpA, tmpB);

    toom3_toom4_k2x2_eval_p1(tmpA, a);
    toom3_toom4_k2x2_eval_p1(tmpB, b);
    toom3_toom4_k2x2_basemul_neon(eC + 1 * 2 * 64 * K16, tmpA, tmpB);

    toom3_toom4_k2x2_eval_m1(tmpA, a);
    toom3_toom4_k2x2_eval_m1(tmpB, b);
    toom3_toom4_k2x2_basemul_neon(eC + 2 * 2 * 64 * K16, tmpA, tmpB);

    toom3_toom4_k2x2_eval_m2(tmpA, a);
    toom3_toom4_k2x2_eval_m2(tmpB, b);
    toom3_toom4_k2x2_basemul_neon(eC + 3 * 2 * 64 * K16, tmpA, tmpB);

    toom3_toom4_k2x2_eval_inf(tmpA, a);
    toom3_toom4_k2x2_eval_inf(tmpB, b);
    toom3_toom4_k2x2_basemul_neon(eC + 4 * 2 * 64 * K16, tmpA, tmpB);

    toom3_toom4_k2x2_interpolate(ab, eC);
}

static uint16_t buf[N];
static void toom3_toom4_k2x2_eval_0(uint16_t r[64 * K16], const uint16_t a[L]) {
    // uint16_t buf[N];
    for (size_t i = 0; i < N; ++i) {
        buf[i] = a[i];
    }
    toom4_k2x2_eval(r, buf);
}

static void toom3_toom4_k2x2_eval_p1(uint16_t r[64 * K16], const uint16_t a[L]) {
    // uint16_t buf[N];
    for (size_t i = 0; i < N; ++i) {
        buf[i]  = a[0 * N + i];
        buf[i] += a[1 * N + i];
        buf[i] += a[2 * N + i];
    }
    toom4_k2x2_eval(r, buf);
}

static void toom3_toom4_k2x2_eval_m1(uint16_t r[64 * K16], const uint16_t a[L]) {
    // uint16_t buf[N];
    for (size_t i = 0; i < N; ++i) {
        buf[i]  = a[0 * N + i];
        buf[i] -= a[1 * N + i];
        buf[i] += a[2 * N + i];
    }
    toom4_k2x2_eval(r, buf);
}

static void toom3_toom4_k2x2_eval_m2(uint16_t r[64 * K16], const uint16_t a[L]) {
    // uint16_t buf[N];
    for (size_t i = 0; i < N; ++i) {
        buf[i]  = a[0 * N + i];
        buf[i] -= 2 * a[1 * N + i];
        buf[i] += 4 * a[2 * N + i];
    }
    toom4_k2x2_eval(r, buf);
}

static void toom3_toom4_k2x2_eval_inf(uint16_t r[64 * K16], const uint16_t a[L]) {
    // uint16_t buf[N];
    for (size_t i = 0; i < N; ++i) {
        buf[i] = a[2 * N + i];
    }
    toom4_k2x2_eval(r, buf);
}

static inline void toom4_k2x2_eval(uint16_t r[64 * K16], const uint16_t a[N]) {
    toom4_k2x2_eval_0(r + 0 * 9 * K16, a);
    toom4_k2x2_eval_p1(r + 1 * 9 * K16, a);
    toom4_k2x2_eval_m1(r + 2 * 9 * K16, a);
    toom4_k2x2_eval_p2(r + 3 * 9 * K16, a);
    toom4_k2x2_eval_m2(r + 4 * 9 * K16, a);
    toom4_k2x2_eval_p3(r + 5 * 9 * K16, a);
    toom4_k2x2_eval_inf(r + 6 * 9 * K16, a);
}

static void toom3_toom4_k2x2_basemul(uint16_t r[2 * 64 * K16], const uint16_t a[64 * K16], const uint16_t b[64 * K16]) {
    for (size_t i = 0; i < 63; ++i)
        schoolbook_KxK(r + i * 2 * K16, a + i * K16, b + i * K16);
}

static void toom3_toom4_k2x2_basemul_neon(uint16_t r[2 * 64 * K16], uint16_t a[64 * K16], uint16_t b[64 * K16]) {
    transpose8x8x16(a);
    transpose8x8x16(b);
    schoolbook_KxK_neon(r + 0 * 2 * K16, a + 0 * K16, b + 0 * K16);
    schoolbook_KxK_neon(r + 8 * 2 * K16, a + 8 * K16, b + 8 * K16);
    schoolbook_KxK_neon(r + 16 * 2 * K16, a + 16 * K16, b + 16 * K16);
    schoolbook_KxK_neon(r + 24 * 2 * K16, a + 24 * K16, b + 24 * K16);
    schoolbook_KxK_neon(r + 32 * 2 * K16, a + 32 * K16, b + 32 * K16);
    schoolbook_KxK_neon(r + 40 * 2 * K16, a + 40 * K16, b + 40 * K16);
    schoolbook_KxK_neon(r + 48 * 2 * K16, a + 48 * K16, b + 48 * K16);
    schoolbook_KxK_neon(r + 56 * 2 * K16, a + 56 * K16, b + 56 * K16);
    transpose8x8x32(r);
}

static void toom4_k2x2_eval_0(uint16_t r[9 * K16], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i] = a[i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_p1(uint16_t r[9 * K16], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i]  = a[0 * M + i];
        r[i] += a[1 * M + i];
        r[i] += a[2 * M + i];
        r[i] += a[3 * M + i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_m1(uint16_t r[9 * K16], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i]  = a[0 * M + i];
        r[i] -= a[1 * M + i];
        r[i] += a[2 * M + i];
        r[i] -= a[3 * M + i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_p2(uint16_t r[9 * K16], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i]  = a[0 * M + i];
        r[i] += 2 * a[1 * M + i];
        r[i] += 4 * a[2 * M + i];
        r[i] += 8 * a[3 * M + i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_m2(uint16_t r[9 * K16], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i]  = a[0 * M + i];
        r[i] -= 2 * a[1 * M + i];
        r[i] += 4 * a[2 * M + i];
        r[i] -= 8 * a[3 * M + i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_p3(uint16_t r[9 * K16], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i]  = a[0 * M + i];
        r[i] += 3 * a[1 * M + i];
        r[i] += 9 * a[2 * M + i];
        r[i] += 27 * a[3 * M + i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_inf(uint16_t r[9 * K16], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i] = a[3 * M + i];
    }
    k2x2_eval(r);
}

static inline void k2x2_eval(uint16_t r[9 * K16]) {
    /* Input:  e + f.Y + g.Y^2 + h.Y^3                              */
    /* Output: [ e | f | g | h | e+f | f+h | g+e | h+g | e+f+g+h ]  */

    int i;
    for (i = K - 1; i >= 0; --i)
        r[3 * K16 + i] = r[3 * K + i];
    for (i = K - 1; i >= 0; --i)
        r[2 * K16 + i] = r[2 * K + i];
    for (i = K - 1; i >= 0; --i)
        r[1 * K16 + i] = r[1 * K + i];
    for (i = K - 1; i >= 0; --i)
        r[0 * K16 + i] = r[0 * K + i];

    for (i = 0; i < K; i++) {
        r[4 * K16 + i] = r[0 * K16 + i];
        r[5 * K16 + i] = r[1 * K16 + i];
        r[6 * K16 + i] = r[2 * K16 + i];
        r[7 * K16 + i] = r[3 * K16 + i];
    }
    for (i = 0; i < K; i++) {
        r[4 * K16 + i] += r[1 * K16 + i];
        r[5 * K16 + i] += r[3 * K16 + i];
        r[6 * K16 + i] += r[0 * K16 + i];
        r[7 * K16 + i] += r[2 * K16 + i];
        r[8 * K16 + i] = r[5 * K16 + i];
        r[8 * K16 + i] += r[6 * K16 + i];
    }
}

static inline void schoolbook_KxK(uint16_t r[2 * K], const uint16_t a[K16], const uint16_t b[K16]) {
    size_t i, j;
    for (j = 0; j < K; j++) {
        r[j] = a[0] * (uint32_t)b[j];
    }
    for (i = 1; i < K; i++) {
        for (j = 0; j < K - 1; j++) {
            r[i + j] += a[i] * (uint32_t)b[j];
        }
        r[i + K - 1] = a[i] * (uint32_t)b[K - 1];
    }
    r[2 * K - 1] = 0;
}

static inline void schoolbook_KxK_neon(uint16_t r[16 * K16], const uint16_t a[8 * K16], const uint16_t b[8 * K16]) {
    uint16x8_t va[K], vb[K], vr;

    va[0] = vld1q_u16(a);
    vb[0] = vld1q_u16(b);
    vr = vmulq_u16(va[0], vb[0]);
    vst1q_u16(r, vr);

    va[1] = vld1q_u16(a + 16 * 1);
    vb[1] = vld1q_u16(b + 16 * 1);
    vr = vmulq_u16(va[0], vb[1]);
    vr = vmlaq_u16(vr, va[1], vb[0]);
    vst1q_u16(r + 32 * 1, vr);

    va[2] = vld1q_u16(a + 16 * 2);
    vb[2] = vld1q_u16(b + 16 * 2);
    vr = vmulq_u16(va[0], vb[2]);
    vr = vmlaq_u16(vr, va[1], vb[1]);
    vr = vmlaq_u16(vr, va[2], vb[0]);
    vst1q_u16(r + 32 * 2, vr);

    va[3] = vld1q_u16(a + 16 * 3);
    vb[3] = vld1q_u16(b + 16 * 3);
    vr = vmulq_u16(va[0], vb[3]);
    vr = vmlaq_u16(vr, va[1], vb[2]);
    vr = vmlaq_u16(vr, va[2], vb[1]);
    vr = vmlaq_u16(vr, va[3], vb[0]);
    vst1q_u16(r + 32 * 3, vr);

    va[4] = vld1q_u16(a + 16 * 4);
    vb[4] = vld1q_u16(b + 16 * 4);
    vr = vmulq_u16(va[0], vb[4]);
    vr = vmlaq_u16(vr, va[1], vb[3]);
    vr = vmlaq_u16(vr, va[2], vb[2]);
    vr = vmlaq_u16(vr, va[3], vb[1]);
    vr = vmlaq_u16(vr, va[4], vb[0]);
    vst1q_u16(r + 32 * 4, vr);

    va[5] = vld1q_u16(a + 16 * 5);
    vb[5] = vld1q_u16(b + 16 * 5);
    vr = vmulq_u16(va[0], vb[5]);
    vr = vmlaq_u16(vr, va[1], vb[4]);
    vr = vmlaq_u16(vr, va[2], vb[3]);
    vr = vmlaq_u16(vr, va[3], vb[2]);
    vr = vmlaq_u16(vr, va[4], vb[1]);
    vr = vmlaq_u16(vr, va[5], vb[0]);
    vst1q_u16(r + 32 * 5, vr);

    va[6] = vld1q_u16(a + 16 * 6);
    vb[6] = vld1q_u16(b + 16 * 6);
    vr = vmulq_u16(va[0], vb[6]);
    vr = vmlaq_u16(vr, va[1], vb[5]);
    vr = vmlaq_u16(vr, va[2], vb[4]);
    vr = vmlaq_u16(vr, va[3], vb[3]);
    vr = vmlaq_u16(vr, va[4], vb[2]);
    vr = vmlaq_u16(vr, va[5], vb[1]);
    vr = vmlaq_u16(vr, va[6], vb[0]);
    vst1q_u16(r + 32 * 6, vr);

    va[7] = vld1q_u16(a + 16 * 7);
    vb[7] = vld1q_u16(b + 16 * 7);
    vr = vmulq_u16(va[0], vb[7]);
    vr = vmlaq_u16(vr, va[1], vb[6]);
    vr = vmlaq_u16(vr, va[2], vb[5]);
    vr = vmlaq_u16(vr, va[3], vb[4]);
    vr = vmlaq_u16(vr, va[4], vb[3]);
    vr = vmlaq_u16(vr, va[5], vb[2]);
    vr = vmlaq_u16(vr, va[6], vb[1]);
    vr = vmlaq_u16(vr, va[7], vb[0]);
    vst1q_u16(r + 32 * 7, vr);

    va[8] = vld1q_u16(a + 16 * 0 + 8);
    vb[8] = vld1q_u16(b + 16 * 0 + 8);
    vr = vmulq_u16(va[0], vb[8]);
    vr = vmlaq_u16(vr, va[1], vb[7]);
    vr = vmlaq_u16(vr, va[2], vb[6]);
    vr = vmlaq_u16(vr, va[3], vb[5]);
    vr = vmlaq_u16(vr, va[4], vb[4]);
    vr = vmlaq_u16(vr, va[5], vb[3]);
    vr = vmlaq_u16(vr, va[6], vb[2]);
    vr = vmlaq_u16(vr, va[7], vb[1]);
    vr = vmlaq_u16(vr, va[8], vb[0]);
    vst1q_u16(r + 32 * 0 + 8, vr);

    va[9] = vld1q_u16(a + 16 * 1 + 8);
    vb[9] = vld1q_u16(b + 16 * 1 + 8);
    vr = vmulq_u16(va[0], vb[9]);
    vr = vmlaq_u16(vr, va[1], vb[8]);
    vr = vmlaq_u16(vr, va[2], vb[7]);
    vr = vmlaq_u16(vr, va[3], vb[6]);
    vr = vmlaq_u16(vr, va[4], vb[5]);
    vr = vmlaq_u16(vr, va[5], vb[4]);
    vr = vmlaq_u16(vr, va[6], vb[3]);
    vr = vmlaq_u16(vr, va[7], vb[2]);
    vr = vmlaq_u16(vr, va[8], vb[1]);
    vr = vmlaq_u16(vr, va[9], vb[0]);
    vst1q_u16(r + 32 * 1 + 8, vr);

    va[10] = vld1q_u16(a + 16 * 2 + 8);
    vb[10] = vld1q_u16(b + 16 * 2 + 8);
    vr = vmulq_u16(va[0], vb[10]);
    vr = vmlaq_u16(vr, va[1], vb[9]);
    vr = vmlaq_u16(vr, va[2], vb[8]);
    vr = vmlaq_u16(vr, va[3], vb[7]);
    vr = vmlaq_u16(vr, va[4], vb[6]);
    vr = vmlaq_u16(vr, va[5], vb[5]);
    vr = vmlaq_u16(vr, va[6], vb[4]);
    vr = vmlaq_u16(vr, va[7], vb[3]);
    vr = vmlaq_u16(vr, va[8], vb[2]);
    vr = vmlaq_u16(vr, va[9], vb[1]);
    vr = vmlaq_u16(vr, va[10], vb[0]);
    vst1q_u16(r + 32 * 2 + 8, vr);

    va[11] = vld1q_u16(a + 16 * 3 + 8);
    vb[11] = vld1q_u16(b + 16 * 3 + 8);
    vr = vmulq_u16(va[0], vb[11]);
    vr = vmlaq_u16(vr, va[1], vb[10]);
    vr = vmlaq_u16(vr, va[2], vb[9]);
    vr = vmlaq_u16(vr, va[3], vb[8]);
    vr = vmlaq_u16(vr, va[4], vb[7]);
    vr = vmlaq_u16(vr, va[5], vb[6]);
    vr = vmlaq_u16(vr, va[6], vb[5]);
    vr = vmlaq_u16(vr, va[7], vb[4]);
    vr = vmlaq_u16(vr, va[8], vb[3]);
    vr = vmlaq_u16(vr, va[9], vb[2]);
    vr = vmlaq_u16(vr, va[10], vb[1]);
    vr = vmlaq_u16(vr, va[11], vb[0]);
    vst1q_u16(r + 32 * 3 + 8, vr);

    va[12] = vld1q_u16(a + 16 * 4 + 8);
    vb[12] = vld1q_u16(b + 16 * 4 + 8);
    vr = vmulq_u16(va[0], vb[12]);
    vr = vmlaq_u16(vr, va[1], vb[11]);
    vr = vmlaq_u16(vr, va[2], vb[10]);
    vr = vmlaq_u16(vr, va[3], vb[9]);
    vr = vmlaq_u16(vr, va[4], vb[8]);
    vr = vmlaq_u16(vr, va[5], vb[7]);
    vr = vmlaq_u16(vr, va[6], vb[6]);
    vr = vmlaq_u16(vr, va[7], vb[5]);
    vr = vmlaq_u16(vr, va[8], vb[4]);
    vr = vmlaq_u16(vr, va[9], vb[3]);
    vr = vmlaq_u16(vr, va[10], vb[2]);
    vr = vmlaq_u16(vr, va[11], vb[1]);
    vr = vmlaq_u16(vr, va[12], vb[0]);
    vst1q_u16(r + 32 * 4 + 8, vr);

    va[13] = vld1q_u16(a + 16 * 5 + 8);
    vb[13] = vld1q_u16(b + 16 * 5 + 8);
    vr = vmulq_u16(va[0], vb[13]);
    vr = vmlaq_u16(vr, va[1], vb[12]);
    vr = vmlaq_u16(vr, va[2], vb[11]);
    vr = vmlaq_u16(vr, va[3], vb[10]);
    vr = vmlaq_u16(vr, va[4], vb[9]);
    vr = vmlaq_u16(vr, va[5], vb[8]);
    vr = vmlaq_u16(vr, va[6], vb[7]);
    vr = vmlaq_u16(vr, va[7], vb[6]);
    vr = vmlaq_u16(vr, va[8], vb[5]);
    vr = vmlaq_u16(vr, va[9], vb[4]);
    vr = vmlaq_u16(vr, va[10], vb[3]);
    vr = vmlaq_u16(vr, va[11], vb[2]);
    vr = vmlaq_u16(vr, va[12], vb[1]);
    vr = vmlaq_u16(vr, va[13], vb[0]);
    vst1q_u16(r + 32 * 5 + 8, vr);

    va[14] = vld1q_u16(a + 16 * 6 + 8);
    vb[14] = vld1q_u16(b + 16 * 6 + 8);
    vr = vmulq_u16(va[0], vb[14]);
    vr = vmlaq_u16(vr, va[1], vb[13]);
    vr = vmlaq_u16(vr, va[2], vb[12]);
    vr = vmlaq_u16(vr, va[3], vb[11]);
    vr = vmlaq_u16(vr, va[4], vb[10]);
    vr = vmlaq_u16(vr, va[5], vb[9]);
    vr = vmlaq_u16(vr, va[6], vb[8]);
    vr = vmlaq_u16(vr, va[7], vb[7]);
    vr = vmlaq_u16(vr, va[8], vb[6]);
    vr = vmlaq_u16(vr, va[9], vb[5]);
    vr = vmlaq_u16(vr, va[10], vb[4]);
    vr = vmlaq_u16(vr, va[11], vb[3]);
    vr = vmlaq_u16(vr, va[12], vb[2]);
    vr = vmlaq_u16(vr, va[13], vb[1]);
    vr = vmlaq_u16(vr, va[14], vb[0]);
    vst1q_u16(r + 32 * 6 + 8, vr);

    vr = vmulq_u16(va[1], vb[14]);
    vr = vmlaq_u16(vr, va[2], vb[13]);
    vr = vmlaq_u16(vr, va[3], vb[12]);
    vr = vmlaq_u16(vr, va[4], vb[11]);
    vr = vmlaq_u16(vr, va[5], vb[10]);
    vr = vmlaq_u16(vr, va[6], vb[9]);
    vr = vmlaq_u16(vr, va[7], vb[8]);
    vr = vmlaq_u16(vr, va[8], vb[7]);
    vr = vmlaq_u16(vr, va[9], vb[6]);
    vr = vmlaq_u16(vr, va[10], vb[5]);
    vr = vmlaq_u16(vr, va[11], vb[4]);
    vr = vmlaq_u16(vr, va[12], vb[3]);
    vr = vmlaq_u16(vr, va[13], vb[2]);
    vr = vmlaq_u16(vr, va[14], vb[1]);
    vst1q_u16(r + 32 * 7 + 8, vr);

    vr = vmulq_u16(va[2], vb[14]);
    vr = vmlaq_u16(vr, va[3], vb[13]);
    vr = vmlaq_u16(vr, va[4], vb[12]);
    vr = vmlaq_u16(vr, va[5], vb[11]);
    vr = vmlaq_u16(vr, va[6], vb[10]);
    vr = vmlaq_u16(vr, va[7], vb[9]);
    vr = vmlaq_u16(vr, va[8], vb[8]);
    vr = vmlaq_u16(vr, va[9], vb[7]);
    vr = vmlaq_u16(vr, va[10], vb[6]);
    vr = vmlaq_u16(vr, va[11], vb[5]);
    vr = vmlaq_u16(vr, va[12], vb[4]);
    vr = vmlaq_u16(vr, va[13], vb[3]);
    vr = vmlaq_u16(vr, va[14], vb[2]);
    vst1q_u16(r + 32 * 0 + 16, vr);

    vr = vmulq_u16(va[3], vb[14]);
    vr = vmlaq_u16(vr, va[4], vb[13]);
    vr = vmlaq_u16(vr, va[5], vb[12]);
    vr = vmlaq_u16(vr, va[6], vb[11]);
    vr = vmlaq_u16(vr, va[7], vb[10]);
    vr = vmlaq_u16(vr, va[8], vb[9]);
    vr = vmlaq_u16(vr, va[9], vb[8]);
    vr = vmlaq_u16(vr, va[10], vb[7]);
    vr = vmlaq_u16(vr, va[11], vb[6]);
    vr = vmlaq_u16(vr, va[12], vb[5]);
    vr = vmlaq_u16(vr, va[13], vb[4]);
    vr = vmlaq_u16(vr, va[14], vb[3]);
    vst1q_u16(r + 32 * 1 + 16, vr);

    vr = vmulq_u16(va[4], vb[14]);
    vr = vmlaq_u16(vr, va[5], vb[13]);
    vr = vmlaq_u16(vr, va[6], vb[12]);
    vr = vmlaq_u16(vr, va[7], vb[11]);
    vr = vmlaq_u16(vr, va[8], vb[10]);
    vr = vmlaq_u16(vr, va[9], vb[9]);
    vr = vmlaq_u16(vr, va[10], vb[8]);
    vr = vmlaq_u16(vr, va[11], vb[7]);
    vr = vmlaq_u16(vr, va[12], vb[6]);
    vr = vmlaq_u16(vr, va[13], vb[5]);
    vr = vmlaq_u16(vr, va[14], vb[4]);
    vst1q_u16(r + 32 * 2 + 16, vr);

    vr = vmulq_u16(va[5], vb[14]);
    vr = vmlaq_u16(vr, va[6], vb[13]);
    vr = vmlaq_u16(vr, va[7], vb[12]);
    vr = vmlaq_u16(vr, va[8], vb[11]);
    vr = vmlaq_u16(vr, va[9], vb[10]);
    vr = vmlaq_u16(vr, va[10], vb[9]);
    vr = vmlaq_u16(vr, va[11], vb[8]);
    vr = vmlaq_u16(vr, va[12], vb[7]);
    vr = vmlaq_u16(vr, va[13], vb[6]);
    vr = vmlaq_u16(vr, va[14], vb[5]);
    vst1q_u16(r + 32 * 3 + 16, vr);

    vr = vmulq_u16(va[6], vb[14]);
    vr = vmlaq_u16(vr, va[7], vb[13]);
    vr = vmlaq_u16(vr, va[8], vb[12]);
    vr = vmlaq_u16(vr, va[9], vb[11]);
    vr = vmlaq_u16(vr, va[10], vb[10]);
    vr = vmlaq_u16(vr, va[11], vb[9]);
    vr = vmlaq_u16(vr, va[12], vb[8]);
    vr = vmlaq_u16(vr, va[13], vb[7]);
    vr = vmlaq_u16(vr, va[14], vb[6]);
    vst1q_u16(r + 32 * 4 + 16, vr);

    vr = vmulq_u16(va[7], vb[14]);
    vr = vmlaq_u16(vr, va[8], vb[13]);
    vr = vmlaq_u16(vr, va[9], vb[12]);
    vr = vmlaq_u16(vr, va[10], vb[11]);
    vr = vmlaq_u16(vr, va[11], vb[10]);
    vr = vmlaq_u16(vr, va[12], vb[9]);
    vr = vmlaq_u16(vr, va[13], vb[8]);
    vr = vmlaq_u16(vr, va[14], vb[7]);
    vst1q_u16(r + 32 * 5 + 16, vr);

    vr = vmulq_u16(va[8], vb[14]);
    vr = vmlaq_u16(vr, va[9], vb[13]);
    vr = vmlaq_u16(vr, va[10], vb[12]);
    vr = vmlaq_u16(vr, va[11], vb[11]);
    vr = vmlaq_u16(vr, va[12], vb[10]);
    vr = vmlaq_u16(vr, va[13], vb[9]);
    vr = vmlaq_u16(vr, va[14], vb[8]);
    vst1q_u16(r + 32 * 6 + 16, vr);

    vr = vmulq_u16(va[9], vb[14]);
    vr = vmlaq_u16(vr, va[10], vb[13]);
    vr = vmlaq_u16(vr, va[11], vb[12]);
    vr = vmlaq_u16(vr, va[12], vb[11]);
    vr = vmlaq_u16(vr, va[13], vb[10]);
    vr = vmlaq_u16(vr, va[14], vb[9]);
    vst1q_u16(r + 32 * 7 + 16, vr);

    vr = vmulq_u16(va[10], vb[14]);
    vr = vmlaq_u16(vr, va[11], vb[13]);
    vr = vmlaq_u16(vr, va[12], vb[12]);
    vr = vmlaq_u16(vr, va[13], vb[11]);
    vr = vmlaq_u16(vr, va[14], vb[10]);
    vst1q_u16(r + 32 * 0 + 24, vr);

    vr = vmulq_u16(va[11], vb[14]);
    vr = vmlaq_u16(vr, va[12], vb[13]);
    vr = vmlaq_u16(vr, va[13], vb[12]);
    vr = vmlaq_u16(vr, va[14], vb[11]);
    vst1q_u16(r + 32 * 1 + 24, vr);

    vr = vmulq_u16(va[12], vb[14]);
    vr = vmlaq_u16(vr, va[13], vb[13]);
    vr = vmlaq_u16(vr, va[14], vb[12]);
    vst1q_u16(r + 32 * 2 + 24, vr);

    vr = vmulq_u16(va[13], vb[14]);
    vr = vmlaq_u16(vr, va[14], vb[13]);
    vst1q_u16(r + 32 * 3 + 24, vr);

    vr = vmulq_u16(va[14], vb[14]);
    vst1q_u16(r + 32 * 4 + 24, vr);

    vr = vdupq_n_u16(0);
    vst1q_u16(r + 32 * 5 + 24, vr);
    vst1q_u16(r + 32 * 6 + 24, vr);
    vst1q_u16(r + 32 * 7 + 24, vr);
}

static void toom3_toom4_k2x2_interpolate(uint16_t r[2 * L], const uint16_t a[5 * 2 * 64 * K16]) {
    uint16_t P1[2 * N];
    uint16_t Pm1[2 * N];

    uint16_t *C0 = r;
    uint16_t *C2 = r + 2 * N;
    uint16_t *C4 = r + 4 * N;

    toom4_k2x2_interpolate(C0, a + 0 * 2 * 64 * K16);
    toom4_k2x2_interpolate(P1, a + 1 * 2 * 64 * K16);
    toom4_k2x2_interpolate(Pm1, a + 2 * 2 * 64 * K16);
    toom4_k2x2_interpolate(C4, a + 4 * 2 * 64 * K16);

    size_t i;

    for (i = 0; i < 2 * N; ++i) {
        C2[i] = ((uint32_t)(P1[i] + Pm1[i])) >> 1;
        C2[i] -= C0[i] + C4[i];
        P1[i] = (P1[i] - Pm1[i]) >> 1;
    }

    /* reuse Pm1 for Pm2 */
#define Pm2 Pm1
    toom4_k2x2_interpolate(Pm2, a + 3 * 2 * 64 * K16);

    uint16_t V0, V1;

    for (i = 0; i < 2 * N; ++i) {
        V0 = P1[i];
        V1 = ((uint32_t)(C0[i] + 4 * (C2[i] + 4 * C4[i]) - Pm2[i])) >> 1;
        Pm2[i] = 43691 * ((uint32_t)(V1 - V0));
        P1[i] = V0 - Pm2[i];
    }

    for (i = 0; i < 2 * N; ++i) {
        r[1 * N + i] += P1[i];
        r[3 * N + i] += Pm2[i];
    }
#undef Pm2
}

static void toom4_k2x2_interpolate(uint16_t r[2 * N], const uint16_t a[2 * 64 * K16]) {
    size_t i;

    uint16_t P1[2 * M];
    uint16_t Pm1[2 * M];
    uint16_t P2[2 * M];
    uint16_t Pm2[2 * M];

    uint16_t *C0 = r;
    uint16_t *C2 = r + 2 * M;
    uint16_t *C4 = r + 4 * M;
    uint16_t *C6 = r + 6 * M;

    uint16_t V0, V1, V2;

    k2x2_interpolate(C0, a + 0 * 9 * 2 * K16);
    k2x2_interpolate(P1, a + 1 * 9 * 2 * K16);
    k2x2_interpolate(Pm1, a + 2 * 9 * 2 * K16);
    k2x2_interpolate(P2, a + 3 * 9 * 2 * K16);
    k2x2_interpolate(Pm2, a + 4 * 9 * 2 * K16);
    k2x2_interpolate(C6, a + 6 * 9 * 2 * K16);

    for (i = 0; i < 2 * M; i++) {
        V0 = ((uint32_t)(P1[i] + Pm1[i])) >> 1;
        V0 = V0 - C0[i] - C6[i];
        V1 = ((uint32_t)(P2[i] + Pm2[i] - 2 * C0[i] - 128 * C6[i])) >> 3;
        C4[i] = 43691 * (uint32_t)(V1 - V0);
        C2[i] = V0 - C4[i];
        P1[i] = ((uint32_t)(P1[i] - Pm1[i])) >> 1;
    }

    /* reuse Pm1 for P3 */
#define P3 Pm1
    k2x2_interpolate(P3, a + 5 * 9 * 2 * K16);

    for (i = 0; i < 2 * M; i++) {
        V0 = P1[i];
        V1 = 43691 * (((uint32_t)(P2[i] - Pm2[i]) >> 2) - V0);
        V2 = 43691 * (uint32_t)(P3[i] - C0[i] - 9 * (C2[i] + 9 * (C4[i] + 9 * C6[i])));
        V2 = ((uint32_t)(V2 - V0)) >> 3;
        V2 -= V1;
        P3[i] = 52429 * (uint32_t)V2;
        P2[i] = V1 - V2;
        P1[i] = V0 - P2[i] - P3[i];
    }

    for (i = 0; i < 2 * M; i++) {
        r[1 * M + i] += P1[i];
        r[3 * M + i] += P2[i];
        r[5 * M + i] += P3[i];
    }
#undef P3
}

static inline void k2x2_interpolate(uint16_t r[2 * M], const uint16_t a[18 * K16]) {
    size_t i;
    uint16_t tmp[4 * K];

    for (i = 0; i < 2 * K; i++) {
        r[0 * K + i] = a[0 * K16 + i];
        r[2 * K + i] = a[2 * K16 + i];
    }

    for (i = 0; i < 2 * K; i++) {
        r[1 * K + i] += a[8 * K16 + i] - a[0 * K16 + i] - a[2 * K16 + i];
    }

    for (i = 0; i < 2 * K; i++) {
        r[4 * K + i] = a[4 * K16 + i];
        r[6 * K + i] = a[6 * K16 + i];
    }

    for (i = 0; i < 2 * K; i++) {
        r[5 * K + i] += a[14 * K16 + i] - a[4 * K16 + i] - a[6 * K16 + i];
    }

    for (i = 0; i < 2 * K; i++) {
        tmp[0 * K + i] = a[12 * K16 + i];
        tmp[2 * K + i] = a[10 * K16 + i];
    }

    for (i = 0; i < 2 * K; i++) {
        tmp[K + i] += a[16 * K16 + i] - a[12 * K16 + i] - a[10 * K16 + i];
    }

    for (i = 0; i < 4 * K; i++) {
        tmp[0 * K + i] = tmp[0 * K + i] - r[0 * K + i] - r[4 * K + i];
    }

    for (i = 0; i < 4 * K; i++) {
        r[2 * K + i] += tmp[0 * K + i];
    }
}

