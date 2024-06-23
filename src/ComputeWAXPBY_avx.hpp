#ifndef COMPUTEWAXPBY_AVX_HPP
#define COMPUTEWAXPBY_AVX_HPP
#include "Vector.hpp"
int ComputeWAXPBY_avx(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w);
#endif // COMPUTEWAXPBY_REF_HPP

