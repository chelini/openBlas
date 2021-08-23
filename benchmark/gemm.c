/***************************************************************************
Copyright (c) 2014, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

// #include "bench.h"

#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// mlir-clang gemm.c -I .. -I /home/lchelini/scratch/polygeist/llvm-project/llvm/../clang/lib/Headers -o gemm --emit-llvm

int main(int argc, char *argv[]) {

  float *a, *b;
  float *c;
  float alpha = 1.0;
  float beta = 1.0;
  char transa = 'N';
  char transb = 'N';
  int m, n, k, i, j, lda, ldb, ldc;

  m = 500;
  k = 500;
  n = 500;

  a = (float *)malloc(sizeof(float) * m * k);
  b = (float *)malloc(sizeof(float) * k * n);
  c = (float *)malloc(sizeof(float) * m * n);

  for (i = 0; i < m * k; i++) {
    a[i] = ((float)rand() / (float)RAND_MAX) - 0.5;
  }
  for (i = 0; i < k * n; i++) {
    b[i] = ((float)rand() / (float)RAND_MAX) - 0.5;
  }
  for (i = 0; i < m * n; i++) {
    c[i] = ((float)rand() / (float)RAND_MAX) - 0.5;
  }

  if (transa == 'N') {
    lda = m;
  } else {
    lda = k;
  }
  if (transb == 'N') {
    ldb = k;
  } else {
    ldb = n;
  }
  ldc = m;

  sgemm_(&transa, &transb, &m, &n, &k, &alpha, &(a[5]), &lda, b, &ldb, &beta, c, &ldc);
  fprintf(stderr, "%.6f\n", c[100]);

  return 0;
}
