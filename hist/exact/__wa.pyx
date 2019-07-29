# distutils: language=c++
# distutils: extra_compile_args=-O3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython and C++ accelerated WA filter code for doing a fancy sort with a custom comparator. See the
wa.py file for more information.
"""

from libc.stdlib cimport malloc, free
from libc.stdint cimport intptr_t, uint8_t

from libcpp cimport bool
from libcpp.algorithm cimport sort

from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set

from numpy cimport PyArray_DATA, PyArray_CHKFLAGS, PyArray_TYPE
from numpy cimport NPY_ARRAY_C_CONTIGUOUS, NPY_ARRAY_ALIGNED, NPY_UINT8, NPY_FLOAT64

# C++ code to use with a std::sort to indirect sort the wavelet data.
cdef extern from *:
    r"""
#include <type_traits>

#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

typedef double dbl __attribute__((aligned()));
typedef dbl* __restrict dbls __attribute__((aligned()));

typedef intptr_t __restrict intp __attribute__((aligned()));
typedef intp* __restrict intps __attribute__((aligned()));

typedef uint8_t __restrict byte __attribute__((aligned()));
typedef byte* __restrict bytes __attribute__((aligned()));

typedef std::unordered_map<intptr_t, std::unordered_set<intptr_t>> uv_map;

template <int NF=0, bool COUNT_FAILS=false, bool STABLE=false>
class CompN {
    const int nlvls, nfltrs;
    intps* coords; // nlvls x size (pre-dec)            u* = coords[j][u]
    dbls* thetas;  // nlvls x (size (post-dec))*nfltrs  |theta_uij| => thetas[j][u*,i]
    bytes* tw0s;   // nlvls x (size (pre-dec))*nfltrs   theta_uij*w_uij_u>0 => tw0s[j][u,i]

    // This map is only available if we are counting failures
    typedef typename std::conditional<COUNT_FAILS, uv_map&, unsigned char>::type uv_map_t;
    volatile uv_map_t fails;

    template<bool _CF=COUNT_FAILS>
    inline typename std::enable_if<_CF>::type insert_fail(intp u, intp v) const {
        fails[u].insert(v);
    }
    template<bool _CF=COUNT_FAILS>
    inline typename std::enable_if<!_CF>::type insert_fail(intp u, intp v) const {}

public:
    inline CompN(int nlvls, intps* coords, dbls* thetas, bytes* tw0s) :
        nlvls(nlvls), nfltrs(NF), coords(coords), thetas(thetas), tw0s(tw0s)
    { assert(NF != 0); }

    inline CompN(int nlvls, int nfltrs, intps* coords, dbls* thetas, bytes* tw0s) :
        nlvls(nlvls), nfltrs(nfltrs), coords(coords), thetas(thetas), tw0s(tw0s)
    { assert(NF == 0 || NF == nfltrs); }

    inline CompN(int nlvls, intps* coords, dbls* thetas, bytes* tw0s, uv_map_t fails) :
        nlvls(nlvls), nfltrs(NF), coords(coords), thetas(thetas), tw0s(tw0s), fails(fails)
    { assert(NF != 0); }

    inline CompN(int nlvls, int nfltrs, intps* coords, dbls* thetas, bytes* tw0s, uv_map_t fails) :
        nlvls(nlvls), nfltrs(nfltrs), coords(coords), thetas(thetas), tw0s(tw0s), fails(fails)
    { assert(NF == 0 || NF == nfltrs); }

    /*
     Called to compare pixel u and v. Returns true if u < v. Returns false if u > v or if they are
     not comparable. This really should be sorted as a poset using the algorithms in
     https://arxiv.org/abs/0707.1532 but that is way more complicated.
     */
    inline bool operator()(intp u, intp v) const {
        // Otherwise go through all sorted coefficients and calculate order
        int nfilters = NF ? NF : this->nfltrs; // this is solved at compile-time
        intp u_orig = u, v_orig = v;
        for (int j = 0; j < this->nlvls; j++) { // j is the filter level
            // Get the decimated and filter coordinates for the pixel at this level
            intp u_dec = this->coords[j][u], v_dec = this->coords[j][v];
            if (unlikely(u_dec == v_dec)) {
                // Not directly comparable, but if we assume that there is some |theta_vij| smaller
                // than both than all we have to do is find the first theta_uij*w_uij_u>0 that
                // aren't equal between u and v and use them to sort the values.
                bytes tw0_u = &tw0s[j][u*nfilters];
                bytes tw0_v = &tw0s[j][v*nfilters];
                for (int i = 0; i < nfilters; ++i) {
                    byte tw0_u_i = tw0_u[i], tw0_v_i = tw0_v[i];
                    if (tw0_u_i != tw0_v_i) { return tw0_u_i < tw0_v_i; }
                }
                // The above works as long as the w_uij_u used to distinguish the two is lined up
                // with a theta_uij that isn't 0. If it is then we can't find a solution still (this
                // also is because our assumption that there is a |theta_vij| smaller isn't correct
                // since there can't be any smaller values than 0).
                // No additional levels will help distinguish these since they will have the same u
                // and v at any further levels.
                insert_fail(u_orig, v_orig);
                return STABLE ? u_orig < v_orig : false;
            }

            // The |theta_uij| and |theta_vij| values for all i
            dbls theta_u = &thetas[j][u_dec*nfilters];
            dbls theta_v = &thetas[j][v_dec*nfilters];
            for (int i = 0; i < nfilters; ++i) {
                // Check for ordering
                dbl theta_u_i = theta_u[i], theta_v_i = theta_v[i];
                if (theta_u_i > theta_v_i) { return !tw0s[j][u*nfilters + i]; }
                if (theta_u_i < theta_v_i) { return  tw0s[j][v*nfilters + i]; }
            }

            // Update the coordinates
            u = u_dec;
            v = v_dec;
        }

        // Not enough data to make a decision (more levels could help distinguish these)
        insert_fail(u_orig, v_orig);
        return STABLE ? u_orig < v_orig : false;
    }
};

typedef CompN<0, false> Comp;
typedef CompN<4, false> Comp4;
typedef CompN<8, false> Comp8;
typedef CompN<0, true> CompFC;
typedef CompN<4, true> Comp4FC;
typedef CompN<8, true> Comp8FC;
"""
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
        #bool operator()(intptr_t u, intptr_t v) nogil
    cdef cppclass Comp8FC:
        Comp8FC(int nlvls, intps* coords, dbls* thetas, bytes* tw0s, uv_map& fails) nogil

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
    #counts = np.unique(im.ravel()/255, return_counts=True)[1]
    cdef list coords_ = [get_coords(shape) for shape in shapes[:nlvls]]

    # Allocate memory for the levels of data
    cdef intps* coords = <intptr_t**>malloc(nlvls*sizeof(intptr_t*))
    cdef dbls* thetas = <double**>malloc(nlvls*sizeof(double*))
    cdef bytes* tw0s = <uint8_t**>malloc(nlvls*sizeof(uint8_t*))
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
