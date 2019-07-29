# distutils: extra_compile_args=-O3
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython accelerated filters for exact.swa.bilateral to be used with scipy.ndimage.generic_filter.
The only exposes Python function is to create the necessary user data for the bl_filter function.
The other function needs to be used with scipy.LowLevelCallable.
"""

from libc.stdint cimport intptr_t
from libc.stdlib cimport malloc, free
from libc.math cimport exp, fabs

from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer

def get_user_data(double[::1] spatial, double scale, double[::1] intensity_lut=None):
    """
    Gets the user data pointer for the bilateral_filter* low-level callables. This requires a 1D
    double array for the precomputed spatial kernel and a double for the intensity scale
    (-1/(2*sigma_r^2)). This also takes an optional precomputed LUT for for the intensity kernel.

    This returns a capsule object to be given to scipy.LowLevelCallable.
    """
    cdef BilateralFilterData *user_data = <BilateralFilterData*>malloc(sizeof(BilateralFilterData))
    if user_data is NULL: raise MemoryError()
    user_data[0].scale = scale
    user_data[0].spatial = &spatial[0]
    user_data[0].intensity_lut = NULL if intensity_lut is None else &intensity_lut[0]
    return PyCapsule_New(user_data, NULL, bilateral_filter_user_data_destructor)

cdef void bilateral_filter_user_data_destructor(object obj):
    """Free the user data pointer."""
    cdef void* pointer = PyCapsule_GetPointer(obj, NULL)
    if pointer is not NULL:
        free(pointer)

cdef int bilateral_filter_inten_lut(double *buffer, intptr_t filter_size, double *retval, void *user_data) nogil:
    """
    Calculates the bilateral filter for a single pixel using a precomputed LUT for the intensity
    kernel. The exp() calculation in this filter can take over 75% of the time so precomputing it is
    very useful.
    """
    cdef BilateralFilterData *data = <BilateralFilterData*>user_data
    cdef double *spatial = data[0].spatial
    cdef double *intensity_lut = data[0].intensity_lut
    cdef double scale = data[0].scale
    cdef double central = buffer[filter_size//2]
    cdef double weight = 0.0, result = 0.0
    cdef double weight_i, value
    cdef intptr_t i
    for i in range(filter_size):
        value = buffer[i]
        weight_i = intensity_lut[int(fabs(value-central))]*spatial[i]
        weight += weight_i
        result += value * weight_i
    retval[0] = result / weight
    return 1

cdef int bilateral_filter(double *buffer, intptr_t filter_size, double *retval, void *user_data) nogil:
    """
    Calculates the bilateral filter for a single pixel with direct computation for the intensity
    kernel.
    """
    cdef BilateralFilterData *data = <BilateralFilterData*>user_data
    cdef double *spatial= data[0].spatial
    cdef double scale = data[0].scale
    cdef double central = buffer[filter_size//2]
    cdef double weight = 0.0, result = 0.0
    cdef double weight_i, value, diff
    cdef intptr_t i
    for i in range(filter_size):
        value = buffer[i]
        diff = value-central
        weight_i = exp(diff*diff*scale)*spatial[i]
        weight += weight_i
        result += value * weight_i
    retval[0] = result / weight
    return 1
