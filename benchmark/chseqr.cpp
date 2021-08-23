/* C++ version of the Hessenberg QR w/ Shift described here:
* http://wiki.inhuawei.com/display/VLZ/Eigenproblem-solvers+analysis
* We only focus on H real symmetric to avoid dealing with complex for now.
 */

/*
 g++ benchmark/chseqr.cpp -I /home/lchelini/scratch/openBlas/install/include/openblas/ -L /home/lchelini/scratch/openBlas/install/lib/ -lopenblas -lpthread -o chseqr
*/

#include <iostream>
#include <vector>
#include <array>
#include <lapacke.h>
#include <cblas.h>

int ssign(float v) {

  if (v < 0)
    return -1;

  if (v > 0)
    return 1;

  return 0;
}

void print_M(float *M, int m, int n, int ldm) {
  for (auto i = 0; i < m; i++) {
    for (auto j = 0; j < n; j++)
      std::cout << M[i + j * ldm] << " ";
    std::cout << std::endl;
  }
}

// HSEQR computes the eigenvalues of a Hessenberg matrix H
// and, optionally, the matrices T and Z from the Schur decomposition
// H = Z T Z**H, where T is an upper triangular matrix (the
// Schur form), and Z is the unitary matrix of Schur vectors.
// We only focus on H symmetric to avoid complex eigenvalues for now.

void test_LAPACKE_chseqr(int n, float *H, int ldh, float *wr,
                         float *Z, int ldz) {

  float eps = 1e-9;
  std::vector<std::array<float, 4>> grots;

  // Workspace for matmuls of the kind A=A*B
  auto tmp = new float[n * n];

  // Initialize Z to Identity
  LAPACKE_slaset(LAPACK_COL_MAJOR, 'x', n, n, 0., 1., Z, ldz);

  for (auto m = n - 1; m > 0; --m) {
    float nrm2;
    int ctr = 0;
    while ((nrm2 = std::abs(H[m + (m - 1) * n])) > eps) {

      // Using Wilkinson shift.
      float c, s;
      float shift;

      // float d = (H[m-1 + (m-1)*n]-H[m + m*n])/2.;

      // if (d == 0) // Need better condition?

      //     shift = H[m + m*n] - nrm2;

      // else {

      //     auto hmm1sq = H[m + (m-1)*n]*H[m + (m-1)*n];

      //     shift = H[m + m*n] - hmm1sq / ( d + float(ssign(d)) * std::sqrt(d*d
      //     + hmm1sq) );

      // }

      // Using Rayleigh quotient shift
      shift = H[m + m * n];

      // Apply shift on the diagonal
      for (auto i = 0; i <= m; ++i)
        H[i + i * n] -= shift;

      // Keep track of Givens rotations

      // LAPACK has a function for applying a series of rotations to a matrix
      // (LASR) OpenBLAS doesn't seem to export it

      grots.clear();

      // putting H in the upper-triangular form and accumulating givens
      // rotations

      for (auto i = 0; i < m; ++i) {
        // Compute Givens rotation
        float a = H[i + i * n], b = H[i + 1 + i * n];
        cblas_srotg(&a, &b, &c, &s);
        std::array<float, 4> grot{c, -s, s, c};
        grots.push_back(grot);

        // Apply Givens rotation
        blasint M = 2, N = m + 1 - i, K = 2;
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.,
                    grot.data(), M, &(H[i + i * n]), n, 0., tmp, n);
        LAPACKE_slacpy(LAPACK_COL_MAJOR, 'x', M, N, tmp, n, &(H[i + i * n]), n);
      }

      // Applying Givens rotations for completing Schur form in H and Schur
      // vectors in Z.

      for (auto i = 0; i < m; ++i) {
        auto grot = grots[i];
        blasint M = i + 2, N = 2, K = 2;

        // Need to make use of workspace + copy in both cases
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, 1.,
                    &(H[0 + i * n]), n, grot.data(), K, 0., tmp, n);
        LAPACKE_slacpy(LAPACK_COL_MAJOR, 'x', M, N, tmp, n, &(H[0 + i * n]), n);

        M = n, N = 2, K = 2;
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, M, N, K, 1.,
                    &(Z[0 + i * n]), n, grot.data(), K, 0., tmp, n);
        LAPACKE_slacpy(LAPACK_COL_MAJOR, 'x', M, N, tmp, n, &(Z[0 + i * n]), n);
      }

      // Apply shift on the diagonal

      for (auto i = 0; i <= m; ++i)
        H[i + i * n] += shift;
    }
  }

  // Storing eigenvalues in w

  for (auto i = 0; i < n; ++i)
    wr[i] = H[i + i * n];

  delete[] tmp;
}

int main(int argc, const char *argv[])

{

  int info, n, ldh, ldz;
  float *H, *Z, *wr, *wi;
  float err = 0.0;
  n = 4;
  ldh = ldz = n;

  // Simple example of Hessenberg H: a real sym tridiag matrix

  H = new float[n * n]{4., -2.4495, 0.,     0., -2.4495, 2.3333, 0.9428, 0.,
                       0., 0.9428,  4.6667, 0., 0.,      0.,     0.,     5.};
  Z = new float[n * n];
  wr = new float[n];
  wi = new float[n];

  std::cout << "Matrix H pre:" << std::endl;
  print_M(H, n, n, n);

  // LAPACK version

  // info = LAPACKE_shseqr(LAPACK_COL_MAJOR, 'S', 'I', n, 1, n, H, n, wr, wi, Z,
  // n);

  // std::cout << "\n\nInfo:" << info << (info ? " (Problems occurred)" : "
  // (Success)") << std::endl;

  // Our test implementation

  test_LAPACKE_chseqr(n, H, ldh, wr, Z, ldz);

  std::cout << "\n\nMatrix H post:" << std::endl;
  print_M(H, n, n, n);

  std::cout << "\n\nMatrix Z post:" << std::endl;
  print_M(Z, n, n, n);

  std::cout << "\n\nVector wr post:" << std::endl;
  print_M(wr, n, 1, n);

  std::cout << "\n\nVector wi post:" << std::endl;
  print_M(wi, n, 1, n);

  delete[] H;

  delete[] Z;

  delete[] wr;

  delete[] wi;

  return 0;
}
