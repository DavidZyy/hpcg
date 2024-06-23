
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

#ifndef MYTIMER_HPP
#define MYTIMER_HPP
double mytimer(void);

// add these macros to measure time of functions
#define myTICK()  double t0 = mytimer() //!< record current time in 't0'
#define myTOCK(t) t += mytimer() - t0 //!< store time difference in 't' using time in 't0'
#endif // MYTIMER_HPP
