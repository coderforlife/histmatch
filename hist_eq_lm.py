"""
Implements local means strict ordering for use with exact histogram equalization.
"""

import functools
from numpy import float64, int64, ceil, sqrt, log2
from numpy import empty, unique, ogrid

def calc_info(im, order=6):
    """
    Assign strict ordering to image pixels. Outputs an array that has the same size as the input.
    Its element entries correspond to the order of the gray level pixel in that position.

    This implements the method by Coltuc et al [2,3] which takes increasing uniform filter sizes to
    find the local means. It takes an argument for the order (called k in the paper) which must be
    at least 2 and defaults to 6.

    The idea of using local neighbors was originally proposed in [1] and if order=2 is used on 2D
    images this reproduces that concept.

    REFERENCES:
      1. Hall EL, Feb 1974, "Almost uniform distributions for computer image enhancement", IEEE
         Transcations on Computers 23(2):207–208
      1. Coltuc D and Bolon P, 1999, "Strict ordering on discrete images and applications"
      2. Coltuc D, Bolon P and Chassery J-M, 2006, "Exact histogram specification", IEEE
         Transcations on Image Processing 15(5):1143-1152
    """
    # supports 2D and 3D, isotropic only
    # returns single value per pixel in most cases, but can return stack
    # this uses scipy's 'reflect' mode [duplicated edge]

    from scipy.ndimage.filters import correlate

    # Deal with arguments
    if order < 2 or order > 6: raise ValueError('Invalid order')
    if im.dtype.kind == 'i':
        from numpy import dtype
        from .util import get_dtype_min
        dt = im.dtype
        im = im.view(dtype(dt.byteorder+'u'+str(dt.itemsize))) - get_dtype_min(dt)
    dt = im.dtype

    # Get the filters for this setup
    filters, includes_order_one = __get_filters(dt, order, im.ndim)

    if len(filters) == 1 and includes_order_one:
        # Single convolution
        return correlate(im.astype(int64), filters[0])

    # Convolve filters with the image and lexsort
    out = empty(im.shape+(len(filters)+(not includes_order_one),), float64)
    for i, fltr in enumerate(filters):
        correlate(im, fltr, out[..., i])
    if not includes_order_one:
        out[..., -1] = im.astype(out.dtype)
    return out

@functools.lru_cache()
def __get_filters(dt, order, ndim):
    """
    Get the local-means filters for an image data type and order. The returned sequence is has the
    lowest orders (most important) last for proper lexsorting. If and only if the second return
    value is True than these filters include the the central value. Otherwise it is not included
    and must be handled separately.

    For all floating point or >16-bit types this returns a sequence of boolean filters, one for each
    order. For 8-bit and 16-bit integral types the filters are combined as much as possible to make
    it so less convolutions are required. In fact for 8-bit images with order 6 or less only a
    single filter is needed.
    """
    # Create the basic filter based on distance^2 from center
    size = ceil(0.5*sqrt(8*order+1)-0.5).astype(int)-1
    slc = slice(-size, size+1) # the filter is 2*size+1 square
    raw = sum(x*x for x in ogrid[(slc,)*ndim])

    if dt.kind == 'f' or dt.itemsize > 2:
        return tuple(__trim_zeros(raw == i) for i in unique(raw)[1:order][::-1]), False

    # if dt.kind == 'u' and dt.itemsize <= 2 - pack filters tightly
    nbits = dt.itemsize*8
    vals, counts = unique(raw, return_counts=True)
    extra_bits = ceil(log2(counts[:order])).astype(int) # perfect up to any reasonable value
    order -= 1 # order 1 is at index 0
    out = []
    # Don't start a new filter for just the central pixel (that is why it is order > 0)
    while order > 0:
        fltr = (raw == vals[order]).astype(int64)
        used_bits = nbits + extra_bits[order]
        order -= 1
        # Add any additional orders possible to this filter
        while order >= 0 and used_bits + nbits + extra_bits[order] <= 63:
            fltr[raw == vals[order]] = 1 << used_bits
            used_bits += nbits + extra_bits[order]
            order -= 1
        out.append(__trim_zeros(fltr))
    return tuple(out), order < 0

def __trim_zeros(arr):
    """
    Trims rows/columns/planes of all zeros from an array. It is assumed that the array is
    symmetrical in all directions.
    """
    slices = (slice(1, -1),)*arr.ndim
    while arr.size and (arr[0] == 0).all():
        arr = arr[slices]
    return arr
