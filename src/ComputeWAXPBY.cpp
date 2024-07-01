
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"
#include "ComputeWAXPBY_avx.hpp"
#include "Vector.hpp"
#include <cassert>
#include <cstdio>
#include <cmath>
#include <new>

#include "mytimer.hpp"
/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
double ComputeWAXPBY_time = 0;
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  isOptimized = false;
//   ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
// 
//   for (local_int_t i=0; i<n; i++) {
//     // printf("w_ref = %10f, w_avx = %10f\n", w_ref.values[i], w_avx.values[i]);
//     printf("w = %10f\n", w.values[i]);
//     // if (fabs(w_avx.values[i] - w_ref.values[i]) >= tolerance) {
//     //   printf("w_ref = %10f, w_avx = %10f\n", w_ref.values[i], w_avx.values[i]);
//     // }
//     // assert(fabs(w_avx.values[i] - w_ref.values[i]) < tolerance);
//   }
//   assert(0);
//   return 0;

  myTICK();
  int err = ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
  // int err = ComputeWAXPBY_avx(n, alpha, x, beta, y, w);
  myTOCK(ComputeWAXPBY_time);
  return err;

  // test
  // Define a tolerance value，允许有一定的误差
//   double tolerance = 1e-6;
//   Vector w_ref, w_avx;
//   InitializeVector(w_ref, n);
//   InitializeVector(w_avx, n);
// 
//   ComputeWAXPBY_ref(n, alpha, x, beta, y, w_ref);
// 
//   ComputeWAXPBY_avx(n, alpha, x, beta, y, w_avx);
// 
//   for (local_int_t i=0; i<n; i++) {
//     // printf("w_ref = %10f, w_avx = %10f\n", w_ref.values[i], w_avx.values[i]);
//     assert(fabs(w_avx.values[i] - w_ref.values[i]) < tolerance);
//     printf("w_ref = %10f, w_avx = %10f\n", w_ref.values[i], w_avx.values[i]);
//     // if (fabs(w_avx.values[i] - w_ref.values[i]) >= tolerance) {
//     //   printf("w_ref = %10f, w_avx = %10f\n", w_ref.values[i], w_avx.values[i]);
//     // }
//     // assert(fabs(w_avx.values[i] - w_ref.values[i]) < tolerance);
//   }
// 
//   // return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);
//   return 0;
}
