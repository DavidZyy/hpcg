#include "ComputeWAXPBY_avx.hpp"
#include <cstdio>
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include <immintrin.h>  // Include the header for AVX

int ComputeWAXPBY_avx(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w) {
    // printf("const char *__restrict format, ...");
  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  const double * const xv = x.values;
  const double * const yv = y.values;
  double * const wv = w.values;

  local_int_t i = 0;

  // Process 4 elements at a time using AVX
  if (alpha == 1.0) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
#endif
    for (i = 0; i <= n-4; i += 4) {
      __m256d x_vec = _mm256_loadu_pd(&xv[i]);
      __m256d y_vec = _mm256_loadu_pd(&yv[i]);
      __m256d beta_vec = _mm256_set1_pd(beta);
      __m256d result = _mm256_fmadd_pd(beta_vec, y_vec, x_vec);
      _mm256_storeu_pd(&wv[i], result);
    }
  } else if (beta == 1.0) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
#endif
    for (i = 0; i <= n-4; i += 4) {
      __m256d alpha_vec = _mm256_set1_pd(alpha);
      __m256d x_vec = _mm256_loadu_pd(&xv[i]);
      __m256d y_vec = _mm256_loadu_pd(&yv[i]);
      __m256d result = _mm256_fmadd_pd(alpha_vec, x_vec, y_vec);
      _mm256_storeu_pd(&wv[i], result);
    }
  } else {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
#endif
    for (i = 0; i <= n-4; i += 4) {
      __m256d alpha_vec = _mm256_set1_pd(alpha);
      __m256d beta_vec = _mm256_set1_pd(beta);
      __m256d x_vec = _mm256_loadu_pd(&xv[i]);
      __m256d y_vec = _mm256_loadu_pd(&yv[i]);
      __m256d alpha_x = _mm256_mul_pd(alpha_vec, x_vec);
      __m256d beta_y = _mm256_mul_pd(beta_vec, y_vec);
      __m256d result = _mm256_add_pd(alpha_x, beta_y);
      _mm256_storeu_pd(&wv[i], result);
    }
  }

  // Process remaining elements
  for (; i < n; i++) {
    if (alpha == 1.0) {
      wv[i] = xv[i] + beta * yv[i];
    } else if (beta == 1.0) {
      wv[i] = alpha * xv[i] + yv[i];
    } else {
      wv[i] = alpha * xv[i] + beta * yv[i];
    }
  }

  return 0;
}