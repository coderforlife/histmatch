# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""WA Cython extern defs."""

from libc.stdint cimport intptr_t, uint8_t

from libcpp cimport bool

from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set

# C++ code to use with a std::sort to indirect sort the wavelet data.
cdef extern from "__wa.hpp":
    # Basic types defined in the code above
    ctypedef double* dbls
    ctypedef intptr_t* intps
    ctypedef uint8_t* bytes
    ctypedef unordered_map[intptr_t, unordered_set[intptr_t]] uv_map

    # These don't count failures, the 4 and 8 one are optimized for 2D and 3D images
    cdef cppclass Comp:
        Comp(int nlvls, int nfltrs, intps* coords, dbls* thetas, bytes* tw0s) nogil
    cdef cppclass Comp4:
        Comp4(int nlvls, intps* coords, dbls* thetas, bytes* tw0s) nogil
    cdef cppclass Comp8:
        Comp8(int nlvls, intps* coords, dbls* thetas, bytes* tw0s) nogil

    # These count failures, the 4 and 8 one are optimized for 2D and 3D images
    cdef cppclass CompFC:
        CompFC(int nlvls, int nfltrs, intps* coords, dbls* thetas, bytes* tw0s, uv_map& fails) nogil
    cdef cppclass Comp4FC:
        Comp4FC(int nlvls, intps* coords, dbls* thetas, bytes* tw0s, uv_map& fails) nogil
        bool operator()(intptr_t u, intptr_t v) nogil
    cdef cppclass Comp8FC:
        Comp8FC(int nlvls, intps* coords, dbls* thetas, bytes* tw0s, uv_map& fails) nogil
