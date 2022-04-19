#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cimport cython
cimport numpy as np

import numpy as np

from libc.math cimport isnan

DTYPE = np.float64
cdef double c_light = 299792.458

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Disable division by zero error
def _forward_model(double[:] ptr_wave, double[:] ptr_flux, double[:] wave, double[:, ::1] model, double[:] v_tot, double[:] area):
    # Calculate the shifted model spectrum
    cdef Py_ssize_t nrv = v_tot.shape[0]
    cdef Py_ssize_t nptrwave = ptr_wave.shape[0]
    cdef Py_ssize_t model_x = model.shape[0]
    cdef Py_ssize_t model_y = model.shape[1]
    cdef double ptr_min = 0.0
    cdef Py_ssize_t i, j

    ptr_wave_shifted = np.empty((nrv, nptrwave), dtype=DTYPE)
    cdef double [:, ::1] ptr_wave_shifted_view = ptr_wave_shifted
    ptr_flux_new = np.empty((model_x, model_y), dtype=DTYPE)
    cdef double [:, ::1] ptr_flux_new_view = ptr_flux_new
    cdef double [:] ptr_flux_single

    for i in range(model_x):
        for j in range(model_y):
            ptr_wave_shifted_view[i, j] = ptr_wave[j] * (1 - v_tot[i] / c_light)

    for i in range(model_x):
        ptr_flux_single = np.interp(wave, ptr_wave_shifted[i], ptr_flux)
        for j in range(model_y):
            ptr_flux_new_view[i, j] = ptr_flux_single[j]

    # Use inplace operations to avoid memory usage
    for i in range(model_x):
        ptr_min = np.min(ptr_flux_new[i])
        for j in range(model_y):
            ptr_flux_new_view[i, j] -= ptr_min
            ptr_flux_new_view[i, j] *= area[i]
            ptr_flux_new_view[i, j] -= 1.0
            ptr_flux_new_view[i, j] *= -model[i, j]
    return ptr_flux_new

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Disable division by zero error
def _log_like(double [:, ::1] model, double [:, ::1] data, double scale, double sdsq):
    cdef double a = scale
    cdef Py_ssize_t m = model.shape[0]
    cdef Py_ssize_t n = model.shape[1]
    cdef Py_ssize_t i, j

    model_copy = np.copy(model)
    cdef double [:, ::1] model_view = model_copy
    cdef double tmp, tmp2

    for i in range(m):
        # tmp = nanmean(model[i])
        tmp = 0
        tmp2 = 0
        for j in range(n):
            if not isnan(model_view[i, j]):
                tmp += model_view[i, j]
                tmp2 += 1
        tmp /= tmp2
        # model -= tmp
        for j in range(n):
            model_view[i, j] -= tmp


    cdef double sm = 0
    for i in range(m):
        for j in range(n):
            sm += model_view[i, j] ** 2
    sm /= n
    sm **= 0.5

    cdef double smsq = sm ** 2
    cdef double ccf = 0
    for i in range(m):
        for j in range(n):
            if not (isnan(model_view[i, j]) | isnan(data[i, j])):
                ccf += data[i, j] * model_view[i, j]
    ccf /= n

    logL = -n / 2 * np.log(sdsq - 2 * a * ccf + a ** 2 * smsq)
    return logL
