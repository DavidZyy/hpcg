
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
 @file GenerateProblem.cpp

 HPCG routine
 */

#if defined(HPCG_DEBUG) || defined(HPCG_DETAILED_DEBUG)
#include <fstream>
using std::endl;
#include "hpcg.hpp"
#endif
#include <cassert>

#include "Geometry.hpp"
#include "GenerateProblem.hpp"

#ifndef HPCG_NOMPI
#include <mpi.h>
#endif

#ifndef HPCG_NOOPENMP
#include <omp.h>
#endif

/*!
  Routine to construct a prolongation/restriction operator for a given fine grid matrix
  solution (as computed by a direct solver).

  @param[in]  A      The known system matrix
  @param[out] RP     Restriction/Prolongation matrix
  @see GenerateGeometry
*/

void GenerateProblem(const Geometry & geom, const SparseMatrix & A, SparseMatrix & RP) {

  // Make local copies of geometry information.  Use global_int_t since the RHS products in the calculations
  // below may result in global range values.
  global_int_t nx = geom.nx;
  global_int_t ny = geom.ny;
  global_int_t nz = geom.nz;
  global_int_t npx = geom.npx;
  global_int_t npy = geom.npy;
  global_int_t npz = geom.npz;
  global_int_t ipx = geom.ipx;
  global_int_t ipy = geom.ipy;
  global_int_t ipz = geom.ipz;
  global_int_t size = geom.size;
  global_int_t gnx = nx*npx;
  global_int_t gny = ny*npy;
  global_int_t gnz = nz*npz;

  global_int_t localNumberOfRows = nx*ny*nz; // This is the size of our subblock
  // If this assert fails, it most likely means that the local_int_t is set to int and should be set to long long
  assert(localNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)
  global_int_t numberOfNonzerosPerRow = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil

  global_int_t totalNumberOfRows = localNumberOfRows*geom.size; // Total number of grid points in mesh
  // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
  assert(totalNumberOfRows>0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)


  // Allocate arrays that are of length localNumberOfRows
  char * nonzerosInRow = new char[localNumberOfRows];
  global_int_t ** mtxIndG = new global_int_t*[localNumberOfRows];
  local_int_t  ** mtxIndL = new local_int_t*[localNumberOfRows];
  double ** matrixValues = new double*[localNumberOfRows];
  double ** matrixDiagonal = new double*[localNumberOfRows];

  A.localToGlobalMap.resize(localNumberOfRows);

  // Use a parallel loop to do initial assignment:
  // distributes the physical placement of arrays of pointers across the memory system
#ifndef HPCG_NOOPENMP
  #pragma omp parallel for
#endif
  for (local_int_t i=0; i< localNumberOfRows; ++i) {
    matrixValues[i] = 0;
    matrixDiagonal[i] = 0;
    mtxIndG[i] = 0;
    mtxIndL[i] = 0;
  }
  // Now allocate the arrays pointed to
  for (local_int_t i=0; i< localNumberOfRows; ++i) {
    mtxIndL[i] = new local_int_t[numberOfNonzerosPerRow];
    matrixValues[i] = new double[numberOfNonzerosPerRow];
    mtxIndG[i] = new global_int_t[numberOfNonzerosPerRow];
  }



  local_int_t localNumberOfNonzeros = 0;
  // TODO:  This triply nested loop could be flattened or use nested parallelism
#ifndef HPCG_NOOPENMP
  #pragma omp parallel for
#endif
  for (local_int_t iz=0; iz<nz; iz++) {
    global_int_t giz = ipz*nz+iz;
    for (local_int_t iy=0; iy<ny; iy++) {
      global_int_t giy = ipy*ny+iy;
      for (local_int_t ix=0; ix<nx; ix++) {
        global_int_t gix = ipx*nx+ix;
        local_int_t currentLocalRow = iz*nx*ny+iy*nx+ix;
        global_int_t currentGlobalRow = giz*gnx*gny+giy*gnx+gix;
#ifndef HPCG_NOOPENMP
// C++ std::map is not threadsafe for writing
        #pragma omp critical
#endif
        A.globalToLocalMap[currentGlobalRow] = currentLocalRow;

        A.localToGlobalMap[currentLocalRow] = currentGlobalRow;
#ifdef HPCG_DETAILED_DEBUG
        HPCG_fout << " rank, globalRow, localRow = " << geom.rank << " " << currentGlobalRow << " " << A.globalToLocalMap[currentGlobalRow] << endl;
#endif
        char numberOfNonzerosInRow = 0;
        double * currentValuePointer = matrixValues[currentLocalRow]; // Pointer to current value in current row
        global_int_t * currentIndexPointerG = mtxIndG[currentLocalRow]; // Pointer to current index in current row
        for (int sz=-1; sz<=1; sz++) {
          if (giz+sz>-1 && giz+sz<gnz) {
            for (int sy=-1; sy<=1; sy++) {
              if (giy+sy>-1 && giy+sy<gny) {
                for (int sx=-1; sx<=1; sx++) {
                  if (gix+sx>-1 && gix+sx<gnx) {
                    global_int_t curcol = currentGlobalRow+sz*gnx*gny+sy*gnx+sx;
                    if (curcol==currentGlobalRow) {
                      matrixDiagonal[currentLocalRow] = currentValuePointer;
                      *currentValuePointer++ = 26.0;
                    } else {
                      *currentValuePointer++ = -1.0;
                    }
                    *currentIndexPointerG++ = curcol;
                    numberOfNonzerosInRow++;
                  } // end x bounds test
                } // end sx loop
              } // end y bounds test
            } // end sy loop
          } // end z bounds test
        } // end sz loop
        nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;
#ifndef HPCG_NOOPENMP
        #pragma omp critical
#endif
        localNumberOfNonzeros += numberOfNonzerosInRow; // Protect this with an atomic
      } // end ix loop
    } // end iy loop
  } // end iz loop
#ifdef HPCG_DETAILED_DEBUG
  HPCG_fout     << "Process " << geom.rank << " of " << geom.size <<" has " << localNumberOfRows    << " rows."     << endl
      << "Process " << geom.rank << " of " << geom.size <<" has " << localNumberOfNonzeros<< " nonzeros." <<endl;
#endif

  global_int_t totalNumberOfNonzeros = 0;
#ifndef HPCG_NOMPI
  // Use MPI's reduce function to sum all nonzeros
#ifdef HPCG_NO_LONG_LONG
  MPI_Allreduce(&localNumberOfNonzeros, &totalNumberOfNonzeros, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
  long long lnnz = localNumberOfNonzeros, gnnz = 0; // convert to 64 bit for MPI call
  MPI_Allreduce(&lnnz, &gnnz, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
  totalNumberOfNonzeros = gnnz; // Copy back
#endif
#else
  totalNumberOfNonzeros = localNumberOfNonzeros;
#endif
  // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
  // This assert is usually the first to fail as problem size increases beyond the 32-bit integer range.
  assert(totalNumberOfNonzeros>0); // Throw an exception of the number of nonzeros is less than zero (can happen if int overflow)

  A.title = 0;
  A.totalNumberOfRows = totalNumberOfRows;
  A.totalNumberOfNonzeros = totalNumberOfNonzeros;
  A.localNumberOfRows = localNumberOfRows;
  A.localNumberOfColumns = localNumberOfRows;
  A.localNumberOfNonzeros = localNumberOfNonzeros;
  A.nonzerosInRow = nonzerosInRow;
  A.mtxIndG = mtxIndG;
  A.mtxIndL = mtxIndL;
  A.matrixValues = matrixValues;
  A.matrixDiagonal = matrixDiagonal;

  return;
}