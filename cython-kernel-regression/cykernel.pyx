# distutils: language = c++

import numpy as np
cimport numpy as np
import numpy.linalg as npla
cimport cython
from libcpp.string cimport string
from libc.math cimport exp, abs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def k_pure_python_cython(np.ndarray[double, ndim=2] x_np,
                         np.ndarray[double, ndim=2] y_np,
                         np.ndarray[double, ndim=2] x_train_np,
                         np.ndarray[double, ndim=2] y_train_np,
                         np.ndarray[double, ndim=1] s_np):
    cdef int n = x_train_np.shape[0]   # number of basis (training) points
    cdef int d = x_train_np.shape[1]   # number of independent variable dimensions
    cdef int m = x_np.shape[0]         # number of evaluation (testing) points
    cdef int p = y_np.shape[1]         # number of quantities to evaluate
    cdef double [:, :] x_train = x_train_np
    cdef double [:, :] y_train = y_train_np
    cdef double [:] s = s_np
    cdef double [:, :] x = x_np
    cdef double [:, :] y = y_np
    cdef double [:] sum_ky = np.zeros(p)
    cdef double sum_k = 0.
    cdef double hh = 0.
    cdef double u = 0.
    cdef double kj = 0.
    for i in range(m):
        sum_k = 0
        hh = s[i] * s[i]
        for j in range(n):
            u = 0
            for l in range(d):
                u += (x_train_np[j, l] - x[i, l]) ** 2
            kj = exp(-u / hh)
            sum_k += kj
            for q in range(p):
                sum_ky[q] += kj * y_train[j, q]
        for q in range(p):
            y[i, q] = sum_ky[q] / sum_k
            sum_ky[q] = 0.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def k_pure_python_cython_beta(np.ndarray[double, ndim=2] x_np,
                              np.ndarray[double, ndim=2] y_np,
                              np.ndarray[double, ndim=2] x_train_np,
                              np.ndarray[double, ndim=2] y_train_np,
                              np.ndarray[double, ndim=1] s_np,
                              float beta_coeff):
    cdef int n = x_train_np.shape[0]   # number of basis (training) points
    cdef int d = x_train_np.shape[1]   # number of independent variable dimensions
    cdef int m = x_np.shape[0]         # number of evaluation (testing) points
    cdef int p = y_np.shape[1]         # number of quantities to evaluate
    cdef double [:, :] x_train = x_train_np
    cdef double [:, :] y_train = y_train_np
    cdef double [:] s = s_np
    cdef double [:, :] x = x_np
    cdef double [:, :] y = y_np
    cdef double [:] sum_ky = np.zeros(p)
    cdef double sum_k = 0.
    cdef double hh = 0.
    cdef double u = 0.
    cdef double kj = 0.
    for i in range(m):
        sum_k = 0
        hh = s[i] * s[i]
        for j in range(n):
            u = 0
            for l in range(d):
                u += (x_train_np[j, l] - x[i, l]) ** 2
            kj = exp(-u / hh)
            sum_k += kj
            for q in range(p):
                sum_ky[q] += kj * (y_train[j, q] + beta_coeff * abs(y_train[j, q]) * (x_train[j, q]-x[i,q]))
        for q in range(p):
            y[i, q] = sum_ky[q] / sum_k
            sum_ky[q] = 0.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)            
def find_band(np.ndarray[double, ndim=1] pts1_np, np.ndarray[double, ndim=1] pts2_np): 
    cdef double band1 = -1
    cdef double band2 = -1
    cdef double mi1 = 1e305
    cdef double mi2 = 1e305
    cdef double dx
    cdef double dy
    cdef double [:] pts1 = pts1_np
    cdef double [:] pts2 = pts2_np
    
    cdef int m = pts1_np.size
    
    for i in range(m):
        mi1 = 1e305
        mi2 = 1e305
        for j in range(m):
            if i != j:
                dx = abs(pts1[i]-pts1[j])
                if dx > 1e-14:
                    mi1 = min(mi1,dx)
                dy = abs(pts2[i]-pts2[j])
                if dy > 1e-14:
                    mi2 = min(mi2,dy)
        band1 = max(mi1,band1)
        band2 = max(mi2,band2)
    return band1,band2
    
    