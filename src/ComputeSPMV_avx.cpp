#include "ComputeSPMV_avx.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>

#include <immintrin.h>  // Include the header for AVX

// int ComputeSPMV_avx(const SparseMatrix & A, Vector & x, Vector & y) {
//   assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
//   assert(y.localLength >= A.localNumberOfRows);
// 
// #ifndef HPCG_NO_MPI
//   ExchangeHalo(A, x);
// #endif
// 
//   const double * const xv = x.values;
//   double * const yv = y.values;
//   const local_int_t nrow = A.localNumberOfRows;
// 
// #ifndef HPCG_NO_OPENMP
//   #pragma omp parallel for
// #endif
//   for (local_int_t i = 0; i < nrow; i++) {
//     double sum = 0.0;
//     const double * const cur_vals = A.matrixValues[i];
//     const local_int_t * const cur_inds = A.mtxIndL[i];
//     const int cur_nnz = A.nonzerosInRow[i];
// 
//     // Vectorize the inner loop using AVX
//     int j;
//     for (j = 0; j <= cur_nnz - 4; j += 4) {
//       __m256d vals = _mm256_loadu_pd(&cur_vals[j]);
//       __m256d xvals = _mm256_set_pd(xv[cur_inds[j+3]], xv[cur_inds[j+2]], xv[cur_inds[j+1]], xv[cur_inds[j]]);
//       __m256d prod = _mm256_mul_pd(vals, xvals);
//       sum += prod[0] + prod[1] + prod[2] + prod[3]; // Horizontal sum of AVX register
//     }
// 
//     // Handle remaining elements
//     for (; j < cur_nnz; j++) {
//       sum += cur_vals[j] * xv[cur_inds[j]];
//     }
// 
//     yv[i] = sum;
//   }
//   return 0;
// }


int ComputeSPMV_avx(const SparseMatrix & A, Vector & x, Vector & y) {
  assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif

  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;

#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i = 0; i < nrow; i++) {
    __m256d sum_vec = _mm256_setzero_pd();  // Initialize sum_vec to zero
    const double * const cur_vals = A.matrixValues[i];
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const int cur_nnz = A.nonzerosInRow[i];

    // Vectorize the inner loop using AVX
    int j;
    for (j = 0; j <= cur_nnz - 4; j += 4) {
      __m256d vals = _mm256_loadu_pd(&cur_vals[j]);
      __m256d xvals = _mm256_set_pd(xv[cur_inds[j+3]], xv[cur_inds[j+2]], xv[cur_inds[j+1]], xv[cur_inds[j]]);
      __m256d prod = _mm256_mul_pd(vals, xvals);
      sum_vec = _mm256_add_pd(sum_vec, prod);  // Accumulate product results
    }

    // Handle remaining elements
    double sum = 0.0;
    for (; j < cur_nnz; j++) {
      sum += cur_vals[j] * xv[cur_inds[j]];
    }

    // Horizontal sum of AVX register
    __m128d low = _mm256_castpd256_pd128(sum_vec);
    __m128d high = _mm256_extractf128_pd(sum_vec, 1);
    __m128d sum128 = _mm_add_pd(low, high);
    sum128 = _mm_hadd_pd(sum128, sum128);
    sum += _mm_cvtsd_f64(sum128);

    yv[i] = sum;
  }
  return 0;
}


// int ComputeSPMV_avx(const SparseMatrix & A, Vector & x, Vector & y) {
//   assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
//   assert(y.localLength >= A.localNumberOfRows);
// 
// #ifndef HPCG_NO_MPI
//   ExchangeHalo(A, x);
// #endif
// 
//   const double * const xv = x.values;
//   double * const yv = y.values;
//   const local_int_t nrow = A.localNumberOfRows;
// 
// #ifndef HPCG_NO_OPENMP
//   #pragma omp parallel for
// #endif
//   for (local_int_t i = 0; i < nrow; i++) {
//     __m256d sum_vec = _mm256_setzero_pd();  // Initialize sum_vec to zero
//     const double * const cur_vals = A.matrixValues[i];
//     const local_int_t * const cur_inds = A.mtxIndL[i];
//     const int cur_nnz = A.nonzerosInRow[i];
// 
//     // Prefetching the first cache lines
//     _mm_prefetch((const char*)&cur_vals[0], _MM_HINT_T0);
//     _mm_prefetch((const char*)&cur_inds[0], _MM_HINT_T0);
// 
//     // Vectorize the inner loop using AVX
//     int j;
//     for (j = 0; j <= cur_nnz - 4; j += 4) {
//       _mm_prefetch((const char*)&cur_vals[j + 4], _MM_HINT_T0);
//       _mm_prefetch((const char*)&cur_inds[j + 4], _MM_HINT_T0);
// 
//       __m256d vals = _mm256_loadu_pd(&cur_vals[j]);
//       __m256d xvals = _mm256_set_pd(xv[cur_inds[j + 3]], xv[cur_inds[j + 2]], xv[cur_inds[j + 1]], xv[cur_inds[j]]);
//       __m256d prod = _mm256_mul_pd(vals, xvals);
//       sum_vec = _mm256_add_pd(sum_vec, prod);  // Accumulate product results
//     }
// 
//     // Handle remaining elements
//     double sum = 0.0;
//     for (; j < cur_nnz; j++) {
//       sum += cur_vals[j] * xv[cur_inds[j]];
//     }
// 
//     // Horizontal sum of AVX register
//     __m128d low = _mm256_castpd256_pd128(sum_vec);
//     __m128d high = _mm256_extractf128_pd(sum_vec, 1);
//     __m128d sum128 = _mm_add_pd(low, high);
//     sum128 = _mm_hadd_pd(sum128, sum128);
//     sum += _mm_cvtsd_f64(sum128);
// 
//     yv[i] = sum;
//   }
//   return 0;
// }
