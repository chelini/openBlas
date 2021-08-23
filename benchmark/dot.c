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

//#include "bench.h"
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// mlir-clang dot.c -I .. -I /home/lchelini/scratch/polygeist/llvm-project/llvm/../clang/lib/Headers -o dot --emit-llvm

int main(int argc, char *argv[]) {

  float *x, *y;
  float r;

  int inc_x = 1, inc_y = 1;
    
  int size = 500;
  x = (float *)malloc(sizeof(float) * size);
  y = (float *)malloc(sizeof(float) * size);
  

  for (int i = 0; i < size; i++) {
    x[i] = ((float)rand() / (float)RAND_MAX) - 0.5;
  }

  for (int i = 0; i < size; i++) {
    y[i] = ((float)rand() / (float)RAND_MAX) - 0.5;
  }

  #pragma plugin(sdot_, "linalg", "r += x(i) * y(i)")  
  r = sdot_(&size, x, &inc_x, y, &inc_y);
  fprintf(stderr, "%.6f\n", result);

  return 0;
}
