# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from libc.stdint cimport intptr_t

cdef struct BilateralFilterData:
    double scale
    double *spatial
    double *intensity_lut

cdef int bilateral_filter_inten_lut(double *buffer, intptr_t filter_size, double *retval, void *user_data) nogil
cdef int bilateral_filter(double *buffer, intptr_t filter_size, double *retval, void *user_data) nogil
