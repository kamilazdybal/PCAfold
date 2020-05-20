import numpy as np
from numpy.linalg import norm
from numpy.matlib import repmat
import scipy.spatial.distance as d
import matplotlib.pyplot as plt
from time import perf_counter
from numba import double
import numba as nb
from cykernel import k_pure_python_cython as cython_kernel_evaluate


def k_original(x, xi, yi, variable_sigma):
    ksum = np.zeros_like(x, dtype=np.float)
    ky = np.zeros((x.shape[0], yi.shape[1]), dtype=np.float)
    for i in range(len(xi)):
        u = norm(xi[i, :] - x, axis=1) / variable_sigma
        K = np.exp(-u * u)
        K = K.reshape(K.shape[0], -1)
        ky += repmat(K, 1, yi.shape[1]) * yi[i, :]
        ksum += K
    y = ky / ksum
    return y


xi = np.linspace(0., 1., 100000)[:, None]
yi = np.hstack([np.cos(2. * np.pi * xi), 
                np.cos(4. * np.pi * xi), 
                np.cos(6. * np.pi * xi), 
                np.cos(8. * np.pi * xi), 
                np.cos(10. * np.pi * xi)])

x = np.linspace(0.01, 0.99, 1)[:, None]

sigma_pts = 2
sigma_thresh = 1.e-16

var_sigma = np.sort(d.cdist(x, xi, 'euclidean'), axis=1)[:, sigma_pts - 1]
var_sigma[var_sigma < sigma_thresh] = sigma_thresh

nruns = 4

cput0 = perf_counter()
for i in range(nruns):
    k = k_original(x, xi, yi, var_sigma)
cpudt_original = perf_counter() - cput0
print(cpudt_original / nruns)

kc = np.zeros_like(k)
s = var_sigma.flatten()

cput0 = perf_counter()
for i in range(nruns):
    cython_kernel_evaluate(x, kc, xi, yi, s)
cpudt_cython = perf_counter() - cput0
print(cpudt_cython / nruns)

print(cpudt_original / cpudt_cython, 'x')

print(k.shape)
print(norm(k - kc))
