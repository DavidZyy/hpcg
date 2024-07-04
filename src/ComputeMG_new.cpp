#include "ComputeMG_ref.hpp"
#include "ComputeSYMGS_ref.hpp"
#include "ComputeSYMGS_new.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#include <cassert>
#include <iostream>
#include <fstream>

void outputVectorToFile(const Vector_STRUCT& vector, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    outFile << "Vector Length: " << vector.localLength << std::endl;
    outFile << "Values: \n";
    for (local_int_t i = 0; i < vector.localLength; ++i) {
        outFile << vector.values[i] << "\n";
    }
    outFile << std::endl;

    outFile.close();

    std::cout << "Vector data has been written to " << filename << std::endl;
}

/*!
  多重网格法 (Multigrid Method)
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG
*/
int ComputeMG_new(const SparseMatrix & A, const Vector & r, Vector & x) {
  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS(A, r, x);
    // for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    // for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS_new(A, r, x);
    // outputVectorToFile(x, "b.txt");
    if (ierr!=0) return ierr;
    ierr = ComputeSPMV_ref(A, x, *A.mgData->Axf); if (ierr!=0) return ierr;
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_ref(A, r);  if (ierr!=0) return ierr;
    // ierr = ComputeMG_ref(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return ierr;
    ierr = ComputeMG_new(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return ierr;
    ierr = ComputeProlongation_ref(A, x);  if (ierr!=0) return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS(A, r, x);
    // for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS_ref(A, r, x);
    if (ierr!=0) return ierr;
  }
  else {
    ierr = ComputeSYMGS(A, r, x);
    // ierr = ComputeSYMGS_ref(A, r, x);
    if (ierr!=0) return ierr;
  }
  return 0;
}

