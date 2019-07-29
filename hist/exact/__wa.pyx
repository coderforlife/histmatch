# distutils: language=c++
# distutils: extra_compile_args=-O3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython and C++ accelerated WA filter code for doing a fancy sort with a custom comparator. See the
wa.py file for more information. The actual C++ code is in __wa.hpp and it is imported in __wa.pxd.
"""

from libc.stdlib cimport malloc, free

from libcpp.algorithm cimport sort

from numpy cimport PyArray_DATA, PyArray_CHKFLAGS, PyArray_TYPE
from numpy cimport NPY_ARRAY_C_CONTIGUOUS, NPY_ARRAY_ALIGNED, NPY_UINT8, NPY_FLOAT64

cdef get_coords(shape):
    """
    Gets the coordinates for the decimated pixels. These allow for quick lookup using the actual
    pixel linear index to get the linear index into the coeffs or filters arrays.
    """
    from functools import reduce
    from numpy import ogrid
    coords = ogrid[tuple(slice(x) for x in shape)]
    return reduce(lambda x, y: x*(y.size//2) + y, (x>>1 for x in coords)).ravel()

cdef inline check_array(name, arr, int dtype, shape):
    """Checks that an array has the given data type, shape, and is C contiguous and aligned."""
    if not (PyArray_TYPE(arr) == dtype and arr.shape == shape and
            PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED)):
        raise ValueError('bad %s: expected %r of %s but got %r of %s'%
            (name, shape, 'float64' if dtype == NPY_FLOAT64 else 'uint8', arr.shape, arr.dtype))

cdef inline intptr_t count_fails(intptr_t[::1] idx, dict fails) except -1:
    """
    Counts the number of failures to sort in the given set of sorted indices and a dictionary of
    pairs that failed to sort (dict of set).
    """
    cdef dict fails2 = fails.copy() # create a bi-directional lookup to speed up the lookups
    for k, v in fails.items():
        for x in v:
            fails2.setdefault(x, set()).add(k)
    cdef intptr_t counter = 0, i
    for i in range(idx.shape[0]-1):
        if idx[i] in fails2 and idx[i+1] in fails2[idx[i]]:
            counter += 1
    return counter

def argsort(im_, list thetas_, list tw0s_, bool return_fails=False):
    """
    Runs the WA argsort on the data returning the indices used to sort the data. This implements
    steps 3 and 4 from Table I algorithm from [1]. Most of this function is actually just checking
    arguments and converting them for use with C++. The actual sort is performed with C++ std::sort
    with a custom comparator written in C++ above.

    This function is optimized for 2D and 3D images but works for any dimension.

    im     - original image, array of doubles
    thetas - list of the absolute value of wavelet coefficients for each pixel tw0_ already in
             sorted ordered for each pixel from largest to smallest, each element in the list is a
             level of the wavelet
    tw0s   - list of the uint8 (bool) values for each pixel with theta_uij * w_uij_u > 0, they are
             ordered to line up with theta, each element in the list is a level of the wavelet

    All arrays must be C-contiguous and aligned.

    REFERENCES
      1. Wan Y and Shi D, 2007, "Joint exact histogram specification and image enhancement through
         the wavelet transform", IEEE Transcations on Image Processing, 16(9):2245-2250.
    """
    from numpy import flatnonzero, concatenate
    cdef int nlvls = len(thetas_), nfltrs = 1 << im_.ndim, i
    cdef intptr_t start, stop, size = im_.size
    cdef list shapes = [im_.shape]
    for i in range(nlvls):
        shapes.append(tuple((x+1) // 2 for x in shapes[i]))

    # Checks
    assert nlvls == len(tw0s_), 'thetas and tw0s must have the same length'
    check_array('im', im_, NPY_FLOAT64, im_.shape)
    for i in range(nlvls):
        check_array('thetas[%d]'%i, thetas_[i], NPY_FLOAT64, shapes[i+1] + (nfltrs,))
        check_array('tw0s[%d]'%i, tw0s_[i], NPY_UINT8, shapes[i] + (nfltrs,))

    # Get the linear indices and the coordinate lookups
    cdef intptr_t[::1] idx = im_.argsort(None)
    srt_ = im_.take(idx.base)
    cdef intptr_t[::1] ranges = concatenate(([0], flatnonzero(srt_[1:] != srt_[:size-1])+1, [size]))
    del srt_
    cdef list coords_ = [get_coords(shape) for shape in shapes[:nlvls]]

    # Allocate memory for the levels of data
    cdef intps* coords = <intps*>malloc(nlvls*sizeof(intps*))
    cdef dbls* thetas = <dbls*>malloc(nlvls*sizeof(dbls))
    cdef bytes* tw0s = <bytes*>malloc(nlvls*sizeof(bytes))
    if coords is NULL or thetas is NULL or tw0s is NULL:
        free(coords)
        free(thetas)
        free(tw0s)
        raise MemoryError()

    # Get data pointers
    for i in range(nlvls):
        coords[i] = <intps>PyArray_DATA(coords_[i])
        thetas[i] = <dbls>PyArray_DATA(thetas_[i])
        tw0s[i] = <bytes>PyArray_DATA(tw0s_[i])

    # Perform the sort
    cdef uv_map fails
    with nogil:
        for i in range(ranges.shape[0] - 1):
            start = ranges[i]
            stop = ranges[i+1]
            if stop - start <= 1: continue
            if not return_fails:
                if nfltrs == 4:
                    sort(&idx[start], &idx[stop], Comp4(nlvls, coords, thetas, tw0s))
                elif nfltrs == 8:
                    sort(&idx[start], &idx[stop], Comp8(nlvls, coords, thetas, tw0s))
                else:
                    sort(&idx[start], &idx[stop], Comp(nlvls, nfltrs, coords, thetas, tw0s))
            elif nfltrs == 4:
                sort(&idx[start], &idx[stop], Comp4FC(nlvls, coords, thetas, tw0s, fails))
            elif nfltrs == 8:
                sort(&idx[start], &idx[stop], Comp8FC(nlvls, coords, thetas, tw0s, fails))
            else:
                sort(&idx[start], &idx[stop], CompFC(nlvls, nfltrs, coords, thetas, tw0s, fails))

    # Free used memory
    free(coords)
    free(thetas)
    free(tw0s)

    # Return the argsort data
    if not return_fails: return idx.base

    # Compute number of failures
    return idx.base, count_fails(idx, fails)
