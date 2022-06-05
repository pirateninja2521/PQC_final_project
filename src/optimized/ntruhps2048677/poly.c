#include <arm_neon.h>
#include "poly.h"

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
    toom3_toom4_k2x2_basemul(eC + 0 * 2 * 64 * K16, tmpA, tmpB);

    toom3_toom4_k2x2_eval_p1(tmpA, a);
    toom3_toom4_k2x2_eval_p1(tmpB, b);
    toom3_toom4_k2x2_basemul(eC + 1 * 2 * 64 * K16, tmpA, tmpB);

    toom3_toom4_k2x2_eval_m1(tmpA, a);
    toom3_toom4_k2x2_eval_m1(tmpB, b);
    toom3_toom4_k2x2_basemul(eC + 2 * 2 * 64 * K16, tmpA, tmpB);

    toom3_toom4_k2x2_eval_m2(tmpA, a);
    toom3_toom4_k2x2_eval_m2(tmpB, b);
    toom3_toom4_k2x2_basemul(eC + 3 * 2 * 64 * K16, tmpA, tmpB);

    toom3_toom4_k2x2_eval_inf(tmpA, a);
    toom3_toom4_k2x2_eval_inf(tmpB, b);
    toom3_toom4_k2x2_basemul(eC + 4 * 2 * 64 * K16, tmpA, tmpB);

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
    const uint16_t *pa = a, *pb = b;
    uint16_t *pr = r;
    for (int i = 0; i != 64; i += 8) {
        // Transpose
        uint16_t aT[128], bT[128], rT[256];
        for (int j = 0; j != K; ++j) {
            for (int k = 0; k != 8 && i + k < 63; ++k) {
                aT[j << 3 | k] = pa[K * k + j];
                bT[j << 3 | k] = pb[K * k + j];
            }
        }
        schoolbook_KxK_neon(rT, aT, bT);

        for (int j = 0; j != 8; ++j) {
            for (int k = 0; k != 29; ++k) {
                pr[j * 2 * K + k] = rT[k << 3 | j];
            }
        }

        pa += 8;
        pb += 8;
        pr += 8;
    }
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

    // va[1] = vld1q_u16();
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

