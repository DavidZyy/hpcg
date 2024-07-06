
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include "OptimizeProblem.hpp"
#include "Geometry.hpp"
#include "MGData.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "WriteProblem.hpp"
#include <cstddef>
/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/

// int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {
// 
//   // This function can be used to completely transform any part of the data structures.
//   // Right now it does nothing, so compiling with a check for unused variables results in complaints
// 
// #if defined(HPCG_USE_MULTICOLORING)
//   const local_int_t nrow = A.localNumberOfRows;
//   std::vector<local_int_t> colors(nrow, nrow); // value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
//   int totalColors = 1;
//   colors[0] = 0; // first point gets color 0
// 
//   // Finds colors in a greedy (a likely non-optimal) fashion.
// 
//   for (local_int_t i=1; i < nrow; ++i) {
//     if (colors[i] == nrow) { // if color not assigned
//       std::vector<int> assigned(totalColors, 0);
//       int currentlyAssigned = 0;
//       const local_int_t * const currentColIndices = A.mtxIndL[i];
//       const int currentNumberOfNonzeros = A.nonzerosInRow[i];
// 
//       for (int j=0; j< currentNumberOfNonzeros; j++) { // scan neighbors
//         local_int_t curCol = currentColIndices[j];
//         if (curCol < i) { // if this point has an assigned color (points beyond `i' are unassigned)
//           if (assigned[colors[curCol]] == 0)
//             currentlyAssigned += 1;
//           assigned[colors[curCol]] = 1; // this color has been used before by `curCol' point
//         } // else // could take advantage of indices being sorted
//       }
// 
//       if (currentlyAssigned < totalColors) { // if there is at least one color left to use
//         for (int j=0; j < totalColors; ++j)  // try all current colors
//           if (assigned[j] == 0) { // if no neighbor with this color
//             colors[i] = j;
//             break;
//           }
//       } else {
//         if (colors[i] == nrow) {
//           colors[i] = totalColors;
//           totalColors += 1;
//         }
//       }
//     }
//   }
// 
//   std::vector<local_int_t> counters(totalColors);
//   for (local_int_t i=0; i<nrow; ++i)
//     counters[colors[i]]++;
// 
//   // form in-place prefix scan
//   local_int_t old=counters[0], old0;
//   for (local_int_t i=1; i < totalColors; ++i) {
//     old0 = counters[i];
//     counters[i] = counters[i-1] + old;
//     old = old0;
//   }
//   counters[0] = 0;
// 
//   // translate `colors' into a permutation
//   for (local_int_t i=0; i<nrow; ++i) // for each color `c'
//     colors[i] = counters[colors[i]]++;
// #endif
// 
//   return 0;
// }

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

  return 0.0;
}

void ColorMatrix(SparseMatrix & A, Vector *bPtr, SparseMatrix * LastAptr) {
  const local_int_t nrow = A.localNumberOfRows;
  std::vector<local_int_t> colors(nrow, nrow); // value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
  int totalColors = 1;
  colors[0] = 0; // first point gets color 0

  // Finds colors in a greedy (a likely non-optimal) fashion.

  for (local_int_t i=1; i < nrow; ++i) {
    if (colors[i] == nrow) { // if color not assigned
      std::vector<int> assigned(totalColors, 0);
      int currentlyAssigned = 0;
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];

      for (int j=0; j< currentNumberOfNonzeros; j++) { // scan neighbors
        local_int_t curCol = currentColIndices[j];
        if (curCol < i) { // if this point has an assigned color (points beyond `i' are unassigned)
          if (assigned[colors[curCol]] == 0)
            currentlyAssigned += 1;
          assigned[colors[curCol]] = 1; // this color has been used before by `curCol' point
        } // else // could take advantage of indices being sorted
      }

      if (currentlyAssigned < totalColors) { // if there is at least one color left to use
        for (int j=0; j < totalColors; ++j)  // try all current colors
          if (assigned[j] == 0) { // if no neighbor with this color
            colors[i] = j;
            break;
          }
      } else {
        if (colors[i] == nrow) {
          colors[i] = totalColors;
          totalColors += 1;
        }
      }
    }
  }

  std::vector<local_int_t> counters(totalColors);
  for (local_int_t i=0; i<nrow; ++i)
    counters[colors[i]]++;

  // my codes: assign to A's fields about the colors
  A.colors = new int[nrow];
  A.colorPtr = new int[totalColors + 1];
  A.nColors = totalColors;

  int colors_ptr = 0;
  for (local_int_t i=0; i < totalColors; ++i) {
    for (local_int_t j=0; j < nrow; ++j) {
      if (colors[j] == i) {
        A.colors[colors_ptr++] = j;
      }
    }
  }

  A.colorPtr[0] = 0;
  for (local_int_t i=1; i < totalColors + 1; ++i) {
    A.colorPtr[i] = counters[i-1] + A.colorPtr[i-1];
  }


  // a test version of coloring reorder.

//////////////////////////// reorder A /////////////////////////////////
  // the following codes are refered from GenerateProblem.cpp: 75
  // which should be changed when reorder rows.
  local_int_t numberOfNonzerosPerRow = 27;
  char * nonzerosInRow_ro = new char[A.localNumberOfRows];
  local_int_t ** mtxIndL_ro = new local_int_t*[A.localNumberOfRows];
  double ** matrixValues_ro = new double*[A.localNumberOfRows];
  A.rowIdx_ro = new local_int_t[A.localNumberOfRows];

  for (local_int_t i=0; i< A.localNumberOfRows; ++i) {
    mtxIndL_ro[i] = new local_int_t[numberOfNonzerosPerRow];
    matrixValues_ro[i] = new double[numberOfNonzerosPerRow];
  }

  // reorder rows of A
  local_int_t rowPtr = 0;
  for (int i=0; i < totalColors; i++) {
    for (int j = 0; j < A.localNumberOfRows; j++) {
      if (colors[j] == i) {
        nonzerosInRow_ro[rowPtr] = A.nonzerosInRow[j];
        for (int k = 0; k < nonzerosInRow_ro[rowPtr]; k++) {
          mtxIndL_ro[rowPtr][k] = A.mtxIndL[j][k];
          matrixValues_ro[rowPtr][k] = A.matrixValues[j][k];
        }
        A.rowIdx_ro[j] = rowPtr;
        rowPtr++;
      }
    }
  }

  // reorder columns of A, only consider 1 process now, not consider the case of multiple processes.
  // should based on the reordered rows when reorder columns
  int k,l;
  local_int_t colPtr = 0;
  for (int i=0; i < totalColors; i++) { // search for all colors
    for (int j = 0; j < A.localNumberOfRows; j++) { // search for all columns Columns == Rows??
    // for (int j = 0; j < A.localNumberOfColumns; j++) {
      if (colors[j] == i) { // if column j is in color i

        // search for all column j, change it to colPtr
        // search all rows of mtxIndL_ro, if idx is j, change it to colPtr
        // for (k = 0; k < A.localNumberOfRows; k++) {
        //   for (l = 0; l < nonzerosInRow_ro[k]; l++) {
        //     if (mtxIndL_ro[k][l] == j && j != colPtr) {
        //       mtxIndL_ro[k][l] = colPtr;
        //     }
        //   }
        // }

        for (k = 0; k < A.localNumberOfRows; k++) {
          for (l = 0; l < A.nonzerosInRow[k]; l++) {
            if (A.mtxIndL[k][l] == j && j != colPtr) {
              mtxIndL_ro[A.rowIdx_ro[k]][l] = colPtr;
            }
          }
        }

        colPtr++;
      }
    }
  }

  // replace A's structure
  for (local_int_t i = 0; i< A.localNumberOfRows; ++i) {
    delete [] A.matrixValues[i];
    delete [] A.mtxIndL[i];
  }
  delete [] A.nonzerosInRow;
  delete [] A.mtxIndL;
  // delete [] mtxIndL_ro;
  delete [] A.matrixValues;

  A.nonzerosInRow = nonzerosInRow_ro;
  A.mtxIndL = mtxIndL_ro;
  A.matrixValues = matrixValues_ro;

//////////////////////////// reorder b /////////////////////////////////
  if (bPtr) {
    // for first level A of multigrid
    Vector b = *bPtr;
    double * bv = new double[b.localLength];
    local_int_t bRowPtr = 0;
    for (int i=0; i < totalColors; i++) {
      for (int j = 0; j < A.localNumberOfRows; j++) {
        if (colors[j] == i) {
          bv[bRowPtr++] = b.values[j];
        }
      }
    }

    delete [] b.values;
    // b.values = bv;
    bPtr->values = bv;
  }

//////////////////////////// reorder f2c /////////////////////////////////
  if (LastAptr) {
    // not the firt level of multigrid
    // remap LastAptr->mgData->f2cOperator
    local_int_t * f2c = LastAptr->mgData->f2cOperator;
    local_int_t * f2c_ro = new local_int_t[LastAptr->localNumberOfRows];
    for (int i=0; i<LastAptr->mgData->rc->localLength; i++) {
      // f2c_ro[LastAptr->rowIdx_ro[i]] = A.rowIdx_ro[f2c[i]]; // false
      f2c_ro[A.rowIdx_ro[i]] = LastAptr->rowIdx_ro[f2c[i]]; // true
    }

    delete [] f2c;
    LastAptr->mgData->f2cOperator = f2c_ro;
  }

  WriteA(A);
}

/**
 * only A and b need to reorder, x and xexact are not needed. 
 */
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {
  // This function can be used to completely transform any part of the data structures.
  // Right now it does nothing, so compiling with a check for unused variables results in complaints
  SparseMatrix * Aptr = &A;
  SparseMatrix * LastAptr= NULL;

  // for debug
  // std::vector<double> xVec(x.values, x.values + x.localLength);

  // coloring A and its Coarse grid matrix
  while (Aptr != 0) {
    Vector *bPtr = (Aptr == &A ? &b : NULL);
    ColorMatrix(*Aptr, bPtr, LastAptr);
    LastAptr = Aptr;
    Aptr = Aptr->Ac;
  }

  return 0;
}
