#ifndef COMPUTEDOTPRODUCT_AVX_HPP
#define COMPUTEDOTPRODUCT_AVX_HPP
#include "Vector.hpp"
int ComputeDotProduct_avx(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce);

#endif // COMPUTEDOTPRODUCT_REF_HPP


