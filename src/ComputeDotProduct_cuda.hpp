#ifndef COMPUTEDOTPRODUCT_CUDA_HPP
#define COMPUTEDOTPRODUCT_CUDA_HPP
#include "Vector.hpp"
int ComputeDotProduct_cuda(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce);

#endif // COMPUTEDOTPRODUCT_REF_HPP

