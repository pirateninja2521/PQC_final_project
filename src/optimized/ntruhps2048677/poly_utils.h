// Ref. https://github.com/GMUCERG/PQC_NEON/blob/main/neon/ntru/neon-hps2048677/neon_matrix_transpose.c
#ifndef POLY_UTILS_H
#define POLY_UTILS_H

#include <stdint.h>

#include "params.h"

void transpose8x16(uint16_t *polys);
void transpose8x32(uint16_t *polys);

#endif