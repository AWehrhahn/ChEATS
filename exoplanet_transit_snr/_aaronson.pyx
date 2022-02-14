cimport cython
cimport numpy as np

import numpy as np
from scipy.sparse import lil_array

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

    proj = [lil_array((nwave, nwave)) for _ in range(nrv)]

    for j in range(nrv):
        digits = np.digitize(wave_shifted[j], wave_to[j])
        for i in range(nwave):
            d = digits[i]
            w = wave_to[j, i]
            if d >= 0 and d < nwave - 1:
                w0 = wave_shifted[j, d]
                w1 = wave_shifted[j, d+1]
                tmp = (w - w0) / (w1 - w0)
                proj[j][i, d] = 1 - tmp
                proj[j][i, d + 1] = tmp
            elif d >= nwave - 1:
                w0 = wave_shifted[j, -2]
                w1 = wave_shifted[j, -1]
                tmp = (w - w0) / (w1 - w0)
                proj[j][i, -2] = 1 - tmp
                proj[j][i, -1] = tmp
            else: # d < 0
                w0 = wave_shifted[j, 0]
                w1 = wave_shifted[j, 1]
                tmp = (w - w0) / (w1 - w0)
                proj[j][i, 0] = 1 - tmp
                proj[j][i, 1] = tmp
        proj[j] = proj[j].tocsc()
        # To test that proj works as exped we can use this:
        # np.all(np.isclose((proj[j] @ wave_from), wave_from.value * rv_factor[j]))
    return proj
