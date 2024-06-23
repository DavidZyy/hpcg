
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"
#include "ComputeDotProduct_cuda.hpp"
#include "ComputeDotProduct_avx.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeDotProduct should be used.
  isOptimized = false;

  // return ComputeDotProduct_ref(n, x, y, result, time_allreduce);
  // return ComputeDotProduct_cuda(n, x, y, result, time_allreduce);
  return ComputeDotProduct_avx(n, x, y, result, time_allreduce);


  // test
  // Define a tolerance value，允许有一定的误差
//   double tolerance = 1e-6;
//   double result_ref, result_cuda, result_avx;
// 
//   ComputeDotProduct_ref(n, x, y, result_ref, time_allreduce);
// 
//   // ComputeDotProduct_cuda(n, x, y, result_cuda, time_allreduce);
//   // printf("Dot product result_ref = %10f, result_cuda = %10f\n", result_ref, result_cuda);
//   // assert(fabs(result_ref - result_cuda) < tolerance);
// 
//   ComputeDotProduct_avx(n, x, y, result_avx, time_allreduce);
//   printf("Dot product result_ref = %10f, result_avx = %10f\n", result_ref, result_avx);
//   assert(fabs(result_ref - result_avx) < tolerance);
// 
//   // ComputeDotProduct_ref(n, x, y, result, time_allreduce);
//   return 0;
}
