#include <assert.h>
#include <stdio.h>
#include "poly.h"

/* Polynomial multiplication using     */
/* Toom-3, Toom-4 and two layers of Karatsuba. */

/* L -> L/3 -> L/12 -> L/48 */
#define PAD48(X) ((((X) + 47)/48)*48)
#define L PAD48(NTRU_N)
#define N (L/3)
#define M (L/12)
#define K (L/48)

static void toom3_toom4_k2x2_mul(uint16_t ab[2 * L], const uint16_t a[L], const uint16_t b[L]);

static void toom3_toom4_k2x2_eval_0(uint16_t r[63 * K], const uint16_t a[L]);
static void toom3_toom4_k2x2_eval_p1(uint16_t r[63 * K], const uint16_t a[L]);
static void toom3_toom4_k2x2_eval_m1(uint16_t r[63 * K], const uint16_t a[L]);
static void toom3_toom4_k2x2_eval_m2(uint16_t r[63 * K], const uint16_t a[L]);
static void toom3_toom4_k2x2_eval_inf(uint16_t r[63 * K], const uint16_t a[L]);
static inline void toom4_k2x2_eval(uint16_t r[63 * K], const uint16_t a[N]);


static void toom4_k2x2_eval_0(uint16_t r[9 * K], const uint16_t a[N]);
static void toom4_k2x2_eval_p1(uint16_t r[9 * K], const uint16_t a[N]);
static void toom4_k2x2_eval_m1(uint16_t r[9 * K], const uint16_t a[N]);
static void toom4_k2x2_eval_p2(uint16_t r[9 * K], const uint16_t a[N]);
static void toom4_k2x2_eval_m2(uint16_t r[9 * K], const uint16_t a[N]);
static void toom4_k2x2_eval_p3(uint16_t r[9 * K], const uint16_t a[N]);
static void toom4_k2x2_eval_inf(uint16_t r[9 * K], const uint16_t a[N]);
static inline void k2x2_eval(uint16_t r[9 * K]);

static void toom3_toom4_k2x2_basemul(uint16_t r[2 * 63 * K], const uint16_t a[63 * K], const uint16_t b[63 * K]);
static inline void schoolbook_KxK(uint16_t r[2 * K], const uint16_t a[K], const uint16_t b[K]);

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
    uint16_t tmpA[63 * K];
    uint16_t tmpB[63 * K];
    uint16_t eC[5 * 2 * 63 * K];

    toom3_toom4_k2x2_eval_0(tmpA, a);
    toom3_toom4_k2x2_eval_0(tmpB, b);
    toom3_toom4_k2x2_basemul(eC + 0 * 2 * 63 * K, tmpA, tmpB);

    toom3_toom4_k2x2_eval_p1(tmpA, a);
    toom3_toom4_k2x2_eval_p1(tmpB, b);
    toom3_toom4_k2x2_basemul(eC + 1 * 2 * 63 * K, tmpA, tmpB);

    toom3_toom4_k2x2_eval_m1(tmpA, a);
    toom3_toom4_k2x2_eval_m1(tmpB, b);
    toom3_toom4_k2x2_basemul(eC + 2 * 2 * 63 * K, tmpA, tmpB);

    toom3_toom4_k2x2_eval_m2(tmpA, a);
    toom3_toom4_k2x2_eval_m2(tmpB, b);
    toom3_toom4_k2x2_basemul(eC + 3 * 2 * 63 * K, tmpA, tmpB);

    toom3_toom4_k2x2_eval_inf(tmpA, a);
    toom3_toom4_k2x2_eval_inf(tmpB, b);
    toom3_toom4_k2x2_basemul(eC + 4 * 2 * 63 * K, tmpA, tmpB);

    toom3_toom4_k2x2_interpolate(ab, eC);
}

static void toom3_toom4_k2x2_eval_0(uint16_t r[63 * K], const uint16_t a[L]) {
    uint16_t tmp[N];
    for (size_t i = 0; i < N; ++i) {
        tmp[i] = a[i];
    }
    toom4_k2x2_eval(r, tmp);
}

static void toom3_toom4_k2x2_eval_p1(uint16_t r[63 * K], const uint16_t a[L]) {
    uint16_t tmp[N];
    for (size_t i = 0; i < N; ++i) {
        tmp[i]  = a[0 * N + i];
        tmp[i] += a[1 * N + i];
        tmp[i] += a[2 * N + i];
    }
    toom4_k2x2_eval(r, tmp);
}

static void toom3_toom4_k2x2_eval_m1(uint16_t r[63 * K], const uint16_t a[L]) {
    uint16_t tmp[N];
    for (size_t i = 0; i < N; ++i) {
        tmp[i]  = a[0 * N + i];
        tmp[i] -= a[1 * N + i];
        tmp[i] += a[2 * N + i];
    }
    toom4_k2x2_eval(r, tmp);
}

static void toom3_toom4_k2x2_eval_m2(uint16_t r[63 * K], const uint16_t a[L]) {
    uint16_t tmp[N];
    for (size_t i = 0; i < N; ++i) {
        tmp[i]  = a[0 * N + i];
        tmp[i] -= 2 * a[1 * N + i];
        tmp[i] += 4 * a[2 * N + i];
    }
    toom4_k2x2_eval(r, tmp);
}

static void toom3_toom4_k2x2_eval_inf(uint16_t r[63 * K], const uint16_t a[L]) {
    uint16_t tmp[N];
    for (size_t i = 0; i < N; ++i) {
        tmp[i] = a[2 * N + i];
    }
    toom4_k2x2_eval(r, tmp);
}

static inline void toom4_k2x2_eval(uint16_t r[63 * K], const uint16_t a[N]) {
    toom4_k2x2_eval_0(r + 0 * 9 * K, a);
    toom4_k2x2_eval_p1(r + 1 * 9 * K, a);
    toom4_k2x2_eval_m1(r + 2 * 9 * K, a);
    toom4_k2x2_eval_p2(r + 3 * 9 * K, a);
    toom4_k2x2_eval_m2(r + 4 * 9 * K, a);
    toom4_k2x2_eval_p3(r + 5 * 9 * K, a);
    toom4_k2x2_eval_inf(r + 6 * 9 * K, a);
}

static void toom3_toom4_k2x2_basemul(uint16_t r[2 * 63 * K], const uint16_t a[63 * K], const uint16_t b[63 * K]) {
    schoolbook_KxK(r + 0 * 2 * K, a + 0 * K, b + 0 * K);
    schoolbook_KxK(r + 1 * 2 * K, a + 1 * K, b + 1 * K);
    schoolbook_KxK(r + 2 * 2 * K, a + 2 * K, b + 2 * K);
    schoolbook_KxK(r + 3 * 2 * K, a + 3 * K, b + 3 * K);
    schoolbook_KxK(r + 4 * 2 * K, a + 4 * K, b + 4 * K);
    schoolbook_KxK(r + 5 * 2 * K, a + 5 * K, b + 5 * K);
    schoolbook_KxK(r + 6 * 2 * K, a + 6 * K, b + 6 * K);
    schoolbook_KxK(r + 7 * 2 * K, a + 7 * K, b + 7 * K);
    schoolbook_KxK(r + 8 * 2 * K, a + 8 * K, b + 8 * K);
    schoolbook_KxK(r + 9 * 2 * K, a + 9 * K, b + 9 * K);
    schoolbook_KxK(r + 10 * 2 * K, a + 10 * K, b + 10 * K);
    schoolbook_KxK(r + 11 * 2 * K, a + 11 * K, b + 11 * K);
    schoolbook_KxK(r + 12 * 2 * K, a + 12 * K, b + 12 * K);
    schoolbook_KxK(r + 13 * 2 * K, a + 13 * K, b + 13 * K);
    schoolbook_KxK(r + 14 * 2 * K, a + 14 * K, b + 14 * K);
    schoolbook_KxK(r + 15 * 2 * K, a + 15 * K, b + 15 * K);
    schoolbook_KxK(r + 16 * 2 * K, a + 16 * K, b + 16 * K);
    schoolbook_KxK(r + 17 * 2 * K, a + 17 * K, b + 17 * K);
    schoolbook_KxK(r + 18 * 2 * K, a + 18 * K, b + 18 * K);
    schoolbook_KxK(r + 19 * 2 * K, a + 19 * K, b + 19 * K);
    schoolbook_KxK(r + 20 * 2 * K, a + 20 * K, b + 20 * K);
    schoolbook_KxK(r + 21 * 2 * K, a + 21 * K, b + 21 * K);
    schoolbook_KxK(r + 22 * 2 * K, a + 22 * K, b + 22 * K);
    schoolbook_KxK(r + 23 * 2 * K, a + 23 * K, b + 23 * K);
    schoolbook_KxK(r + 24 * 2 * K, a + 24 * K, b + 24 * K);
    schoolbook_KxK(r + 25 * 2 * K, a + 25 * K, b + 25 * K);
    schoolbook_KxK(r + 26 * 2 * K, a + 26 * K, b + 26 * K);
    schoolbook_KxK(r + 27 * 2 * K, a + 27 * K, b + 27 * K);
    schoolbook_KxK(r + 28 * 2 * K, a + 28 * K, b + 28 * K);
    schoolbook_KxK(r + 29 * 2 * K, a + 29 * K, b + 29 * K);
    schoolbook_KxK(r + 30 * 2 * K, a + 30 * K, b + 30 * K);
    schoolbook_KxK(r + 31 * 2 * K, a + 31 * K, b + 31 * K);
    schoolbook_KxK(r + 32 * 2 * K, a + 32 * K, b + 32 * K);
    schoolbook_KxK(r + 33 * 2 * K, a + 33 * K, b + 33 * K);
    schoolbook_KxK(r + 34 * 2 * K, a + 34 * K, b + 34 * K);
    schoolbook_KxK(r + 35 * 2 * K, a + 35 * K, b + 35 * K);
    schoolbook_KxK(r + 36 * 2 * K, a + 36 * K, b + 36 * K);
    schoolbook_KxK(r + 37 * 2 * K, a + 37 * K, b + 37 * K);
    schoolbook_KxK(r + 38 * 2 * K, a + 38 * K, b + 38 * K);
    schoolbook_KxK(r + 39 * 2 * K, a + 39 * K, b + 39 * K);
    schoolbook_KxK(r + 40 * 2 * K, a + 40 * K, b + 40 * K);
    schoolbook_KxK(r + 41 * 2 * K, a + 41 * K, b + 41 * K);
    schoolbook_KxK(r + 42 * 2 * K, a + 42 * K, b + 42 * K);
    schoolbook_KxK(r + 43 * 2 * K, a + 43 * K, b + 43 * K);
    schoolbook_KxK(r + 44 * 2 * K, a + 44 * K, b + 44 * K);
    schoolbook_KxK(r + 45 * 2 * K, a + 45 * K, b + 45 * K);
    schoolbook_KxK(r + 46 * 2 * K, a + 46 * K, b + 46 * K);
    schoolbook_KxK(r + 47 * 2 * K, a + 47 * K, b + 47 * K);
    schoolbook_KxK(r + 48 * 2 * K, a + 48 * K, b + 48 * K);
    schoolbook_KxK(r + 49 * 2 * K, a + 49 * K, b + 49 * K);
    schoolbook_KxK(r + 50 * 2 * K, a + 50 * K, b + 50 * K);
    schoolbook_KxK(r + 51 * 2 * K, a + 51 * K, b + 51 * K);
    schoolbook_KxK(r + 52 * 2 * K, a + 52 * K, b + 52 * K);
    schoolbook_KxK(r + 53 * 2 * K, a + 53 * K, b + 53 * K);
    schoolbook_KxK(r + 54 * 2 * K, a + 54 * K, b + 54 * K);
    schoolbook_KxK(r + 55 * 2 * K, a + 55 * K, b + 55 * K);
    schoolbook_KxK(r + 56 * 2 * K, a + 56 * K, b + 56 * K);
    schoolbook_KxK(r + 57 * 2 * K, a + 57 * K, b + 57 * K);
    schoolbook_KxK(r + 58 * 2 * K, a + 58 * K, b + 58 * K);
    schoolbook_KxK(r + 59 * 2 * K, a + 59 * K, b + 59 * K);
    schoolbook_KxK(r + 60 * 2 * K, a + 60 * K, b + 60 * K);
    schoolbook_KxK(r + 61 * 2 * K, a + 61 * K, b + 61 * K);
    schoolbook_KxK(r + 62 * 2 * K, a + 62 * K, b + 62 * K);
}

static void toom4_k2x2_eval_0(uint16_t r[9 * K], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i] = a[i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_p1(uint16_t r[9 * K], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i]  = a[0 * M + i];
        r[i] += a[1 * M + i];
        r[i] += a[2 * M + i];
        r[i] += a[3 * M + i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_m1(uint16_t r[9 * K], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i]  = a[0 * M + i];
        r[i] -= a[1 * M + i];
        r[i] += a[2 * M + i];
        r[i] -= a[3 * M + i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_p2(uint16_t r[9 * K], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i]  = a[0 * M + i];
        r[i] += 2 * a[1 * M + i];
        r[i] += 4 * a[2 * M + i];
        r[i] += 8 * a[3 * M + i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_m2(uint16_t r[9 * K], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i]  = a[0 * M + i];
        r[i] -= 2 * a[1 * M + i];
        r[i] += 4 * a[2 * M + i];
        r[i] -= 8 * a[3 * M + i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_p3(uint16_t r[9 * K], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i]  = a[0 * M + i];
        r[i] += 3 * a[1 * M + i];
        r[i] += 9 * a[2 * M + i];
        r[i] += 27 * a[3 * M + i];
    }
    k2x2_eval(r);
}

static void toom4_k2x2_eval_inf(uint16_t r[9 * K], const uint16_t a[N]) {
    for (size_t i = 0; i < M; i++) {
        r[i] = a[3 * M + i];
    }
    k2x2_eval(r);
}

static inline void k2x2_eval(uint16_t r[9 * K]) {
    /* Input:  e + f.Y + g.Y^2 + h.Y^3                              */
    /* Output: [ e | f | g | h | e+f | f+h | g+e | h+g | e+f+g+h ]  */

    size_t i;
    for (i = 0; i < 4 * K; i++) {
        r[4 * K + i] = r[i];
    }
    for (i = 0; i < K; i++) {
        r[4 * K + i] += r[1 * K + i];
        r[5 * K + i] += r[3 * K + i];
        r[6 * K + i] += r[0 * K + i];
        r[7 * K + i] += r[2 * K + i];
        r[8 * K + i] = r[5 * K + i];
        r[8 * K + i] += r[6 * K + i];
    }
}

static inline void schoolbook_KxK(uint16_t r[2 * K], const uint16_t a[K], const uint16_t b[K]) {
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

static void toom3_toom4_k2x2_interpolate(uint16_t r[2 * L], const uint16_t a[5 * 2 * 63 * K]) {
    uint16_t P1[2 * N];
    uint16_t Pm1[2 * N];

    uint16_t *C0 = r;
    uint16_t *C2 = r + 2 * N;
    uint16_t *C4 = r + 4 * N;

    toom4_k2x2_interpolate(C0, a + 0 * 2 * 63 * K);
    toom4_k2x2_interpolate(P1, a + 1 * 2 * 63 * K);
    toom4_k2x2_interpolate(Pm1, a + 2 * 2 * 63 * K);
    toom4_k2x2_interpolate(C4, a + 4 * 2 * 63 * K);

    size_t i;

    for (i = 0; i < 2 * N; ++i) {
        C2[i] = ((uint32_t)(P1[i] + Pm1[i])) >> 1;
        C2[i] -= C0[i] + C4[i];
        P1[i] = (P1[i] - Pm1[i]) >> 1;
    }

    /* reuse Pm1 for Pm2 */
#define Pm2 Pm1
    toom4_k2x2_interpolate(Pm2, a + 3 * 2 * 63 * K);

    uint16_t V0, V1;

    for (i = 0; i < 2 * N; ++i) {
        V0 = P1[i];
        V1 = ((uint32_t)(C0[i] + 4 * (C2[i] + 4 * C4[i]) - Pm2[i])) >> 1;
        // printf("%ld %d %d %d %d %d %d\n", i, V1, V0, C0[i], C2[i], C4[i], Pm2[i]);
        Pm2[i] = 43691 * ((uint32_t)(V1 - V0));
        // assert(((uint32_t)(V1 - V0) >> 1) % 3 == 0);
        P1[i] = V0 - Pm2[i];
    }

    for (i = 0; i < 2 * N; ++i) {
        r[1 * N + i] += P1[i];
        r[3 * N + i] += Pm2[i];
    }
#undef Pm2
}

static void toom4_k2x2_interpolate(uint16_t r[2 * N], const uint16_t a[7 * 18 * K]) {
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

    k2x2_interpolate(C0, a + 0 * 9 * 2 * K);
    k2x2_interpolate(P1, a + 1 * 9 * 2 * K);
    k2x2_interpolate(Pm1, a + 2 * 9 * 2 * K);
    k2x2_interpolate(P2, a + 3 * 9 * 2 * K);
    k2x2_interpolate(Pm2, a + 4 * 9 * 2 * K);
    k2x2_interpolate(C6, a + 6 * 9 * 2 * K);

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
    k2x2_interpolate(P3, a + 5 * 9 * 2 * K);

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

static inline void k2x2_interpolate(uint16_t r[2 * M], const uint16_t a[18 * K]) {
    size_t i;
    uint16_t tmp[4 * K];

    for (i = 0; i < 2 * K; i++) {
        r[0 * K + i] = a[0 * K + i];
        r[2 * K + i] = a[2 * K + i];
    }

    for (i = 0; i < 2 * K; i++) {
        r[1 * K + i] += a[8 * K + i] - a[0 * K + i] - a[2 * K + i];
    }

    for (i = 0; i < 2 * K; i++) {
        r[4 * K + i] = a[4 * K + i];
        r[6 * K + i] = a[6 * K + i];
    }

    for (i = 0; i < 2 * K; i++) {
        r[5 * K + i] += a[14 * K + i] - a[4 * K + i] - a[6 * K + i];
    }

    for (i = 0; i < 2 * K; i++) {
        tmp[0 * K + i] = a[12 * K + i];
        tmp[2 * K + i] = a[10 * K + i];
    }

    for (i = 0; i < 2 * K; i++) {
        tmp[K + i] += a[16 * K + i] - a[12 * K + i] - a[10 * K + i];
    }

    for (i = 0; i < 4 * K; i++) {
        tmp[0 * K + i] = tmp[0 * K + i] - r[0 * K + i] - r[4 * K + i];
    }

    for (i = 0; i < 4 * K; i++) {
        r[2 * K + i] += tmp[0 * K + i];
    }
}

