
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeMG_ref.hpp"
#include "ComputeMG_new.hpp"
#include "mytimer.hpp"
/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
double ComputeMG_time = 0;
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

  // This line and the next two lines should be removed and your version of ComputeSYMGS should be used.
  A.isMgOptimized = false;
  myTICK();
  // int err = ComputeMG_ref(A, r, x);
  int err = ComputeMG_new(A, r, x);
  myTOCK(ComputeMG_time);
  return err;
}
