# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from libc.stdint cimport intptr_t

cdef int vote_greater(double *buffer, intptr_t filter_size, double *retval, void *user_data) nogil
cdef int vote_lesser(double *buffer, intptr_t filter_size, double *retval, void *user_data) nogil
