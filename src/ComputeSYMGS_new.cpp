#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include "ComputeSYMGS_ref.hpp"
#include <cassert>

#include <cstdio>

int ComputeSYMGS_new(const SparseMatrix &A, const Vector &r, Vector &x) {
  assert(x.localLength == A.localNumberOfColumns); // Make sure x contains space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A, x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  double **matrixDiagonal = A.matrixDiagonal; // An array of pointers to the diagonal entries A.matrixValues
  const double *const rv = r.values;
  double *const xv = x.values;

  // Assuming A.colorPtr and A.colors are already set up by a graph coloring algorithm
  int *colors = A.colors; // Color array
  int *colorPtr = A.colorPtr; // Pointer array for the start of each color

  const int nColors = A.nColors;

  std::vector<int> colorsVector(colors, colors + nrow);
  std::vector<int> colorPtrVector(colorPtr, colorPtr + nColors + 1);

  // Forward sweep
  for (int color = 0; color < nColors; ++color) {
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
    for (int i = colorPtr[color]; i < colorPtr[color + 1]; ++i) {
      // printf("in row!!!\n");
      const int row = colors[i];
      const double *const currentValues = A.matrixValues[row];
      const local_int_t *const currentColIndices = A.mtxIndL[row];
      const int currentNumberOfNonzeros = A.nonzerosInRow[row];
      const double currentDiagonal = matrixDiagonal[row][0]; // Current diagonal value
      double sum = rv[row]; // RHS value

      for (int j = 0; j < currentNumberOfNonzeros; ++j) {
        local_int_t curCol = currentColIndices[j];
        sum -= currentValues[j] * xv[curCol];
      }
      sum += xv[row] * currentDiagonal; // Remove diagonal contribution from previous loop

      xv[row] = sum / currentDiagonal;
    }
  }

  // printf("hellohellohellohellohello\n");
  // Backward sweep
  for (int color = nColors - 1; color >= 0; --color) {
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for
#endif
    for (int i = colorPtr[color+1]-1; i >= colorPtr[color]; --i) {
      const int row = colors[i];
      const double *const currentValues = A.matrixValues[row];
      const local_int_t *const currentColIndices = A.mtxIndL[row];
      const int currentNumberOfNonzeros = A.nonzerosInRow[row];
      const double currentDiagonal = matrixDiagonal[row][0]; // Current diagonal value
      double sum = rv[row]; // RHS value

      for (int j = 0; j < currentNumberOfNonzeros; ++j) {
        local_int_t curCol = currentColIndices[j];
        sum -= currentValues[j] * xv[curCol];
      }
      sum += xv[row] * currentDiagonal; // Remove diagonal contribution from previous loop

      xv[row] = sum / currentDiagonal;
    }
  }

  return 0;
}
