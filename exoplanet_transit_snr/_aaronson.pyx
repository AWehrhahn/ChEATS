#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cimport cython
cimport numpy as np

import numpy as np
from scipy.sparse import coo_array
from tqdm import tqdm

from libc.math cimport sqrt

DTYPE = np.float64
cdef double c_light = 299792.458

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Disable division by zero error
def _projection_matrix(double[:, ::1] wave_shifted, double[:, ::1] wave_to, double[:] rv):

    cdef Py_ssize_t i, j
    cdef Py_ssize_t nrv = wave_to.shape[0]
    cdef Py_ssize_t nwave = wave_to.shape[1]

    assert nrv == rv.size
    assert nrv == wave_shifted.shape[0]
    assert nwave == wave_shifted.shape[1]

    cdef double[:] rv_factor = np.empty(nrv, dtype=DTYPE)
    cdef Py_ssize_t[:] digits
    cdef Py_ssize_t d
    cdef double w, tmp, w0, w1
    cdef double[:] data = np.empty(2 * nwave, dtype=DTYPE)
    cdef Py_ssize_t[:] idx_i = np.empty(2 * nwave, dtype=int)
    cdef Py_ssize_t[:] idx_j = np.empty(2 * nwave, dtype=int)

    proj = [None for _ in range(nrv)]

    for j in tqdm(range(nrv), total=nrv):
        digits = np.digitize(wave_to[j], wave_shifted[j])
        for i in range(nwave):
            d = digits[i]
            w = wave_to[j, i]
            if (d > 0) and (d < nwave):
                w0 = wave_shifted[j, d - 1]
                w1 = wave_shifted[j, d]
                tmp = (w - w0) / (w1 - w0)
                # Fill the arrays
                data[2*i] = 1 - tmp
                idx_i[2*i] = i
                idx_j[2*i] = d - 1

                data[2*i+1] = tmp
                idx_i[2*i+1] = i
                idx_j[2*i+1] = d
            elif d >= nwave:
                w0 = wave_shifted[j, -2]
                w1 = wave_shifted[j, -1]
                tmp = (w - w0) / (w1 - w0)
                # Fill the arrays
                data[2*i] = 1 - tmp
                idx_i[2*i] = i
                idx_j[2*i] = nwave-2

                data[2*i+1] = tmp
                idx_i[2*i+1] = i
                idx_j[2*i+1] = nwave-1
            else: # d <= 0
                w0 = wave_shifted[j, 0]
                w1 = wave_shifted[j, 1]
                tmp = (w - w0) / (w1 - w0)
                # Fill the arrays
                data[2*i] = 1 - tmp
                idx_i[2*i] = i
                idx_j[2*i] = 0

                data[2*i+1] = tmp
                idx_i[2*i+1] = i
                idx_j[2*i+1] = 1

        proj[j] = coo_array((data, (idx_i, idx_j)), shape=(nwave, nwave)).tocsc(copy=True)
        # To test that proj works as exped we can use this:
        # np.all(np.isclose((proj[j] @ wave_from), wave_from.value * rv_factor[j]))
    return proj
