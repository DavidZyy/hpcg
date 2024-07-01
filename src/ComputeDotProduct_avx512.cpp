#include <immintrin.h> // Header for AVX512 intrinsics
#include <assert.h>

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include "ComputeDotProduct_avx512.hpp"

int ComputeDotProduct_avx512(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce) {
  assert(x.localLength >= n); // Test vector lengths
  assert(y.localLength >= n);

  double local_result = 0.0;
  double * xv = x.values;
  double * yv = y.values;

  // Use AVX-512 intrinsics for vectorized computation
  __m512d sum = _mm512_setzero_pd();

  if (yv == xv) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction(+:local_result)
#endif
    for (local_int_t i = 0; i < n; i += 8) { // AVX-512 processes 8 doubles at a time
      __m512d x_vec = _mm512_loadu_pd(&xv[i]);
      __m512d prod = _mm512_mul_pd(x_vec, x_vec);
      sum = _mm512_add_pd(sum, prod);
    }
  } else {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for reduction(+:local_result)
#endif
    for (local_int_t i = 0; i < n; i += 8) {
      __m512d x_vec = _mm512_loadu_pd(&xv[i]);
      __m512d y_vec = _mm512_loadu_pd(&yv[i]);
      __m512d prod = _mm512_mul_pd(x_vec, y_vec);
      sum = _mm512_add_pd(sum, prod);
    }
  }

  // Horizontal sum of the AVX-512 vector
  double temp[8];
  _mm512_storeu_pd(temp, sum);
  for (int i = 0; i < 8; ++i) {
    local_result += temp[i];
  }

  // Handle the remaining elements if n is not a multiple of 8
  for (local_int_t i = (n / 8) * 8; i < n; ++i) {
    if (yv == xv) {
      local_result += xv[i] * xv[i];
    } else {
      local_result += xv[i] * yv[i];
    }
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}
