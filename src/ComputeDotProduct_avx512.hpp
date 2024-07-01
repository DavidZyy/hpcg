#ifndef COMPUTEDOTPRODUCT_AVX512_HPP
#define COMPUTEDOTPRODUCT_AVX512_HPP
#include "Vector.hpp"
int ComputeDotProduct_avx512(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce);

#endif // COMPUTEDOTPRODUCT_REF_HPP

