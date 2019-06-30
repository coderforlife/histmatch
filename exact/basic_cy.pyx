# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from libc.stdint cimport intptr_t

cdef int vote_greater(double *buffer, intptr_t filter_size, double *retval, void *user_data) nogil:
    """
    Voting metric. When used as the filter with scipy.ndimage.generic_filter it will count the
    number of values greater than the central value. This is the 'inverse' of beta_m in [1].

    REFERENCES
      1. Eramian M and Mould D, 2005, "Histogram Equalization using Neighborhood Metrics",
         Proceedings of the Second Canadian Conference on Computer and Robot Vision.
    """
    cdef intptr_t i, count = 0
    cdef double central = buffer[filter_size // 2]
    for i in range(filter_size):
        count += central < buffer[i]
    retval[0] = <double>count
    return 1

cdef int vote_lesser(double *buffer, intptr_t filter_size, double *retval, void *user_data) nogil:
    """
    Voting metric. When used as the filter with scipy.ndimage.generic_filter it will count the
    number of values less than the central value. This is the beta_m in [1].

    REFERENCES
      1. Eramian M and Mould D, 2005, "Histogram Equalization using Neighborhood Metrics",
         Proceedings of the Second Canadian Conference on Computer and Robot Vision.
    """
    cdef intptr_t i, count = 0
    cdef double central = buffer[filter_size // 2]
    for i in range(filter_size):
        count += central > buffer[i]
    retval[0] = <double>count
    return 1
