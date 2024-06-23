#ifndef COMPUTESPMV_AVX_HPP
#define COMPUTESPMV_AVX_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"

int ComputeSPMV_avx( const SparseMatrix & A, Vector  & x, Vector & y);

#endif  // COMPUTESPMV_REF_HPP

