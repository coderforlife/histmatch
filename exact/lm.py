"""
Implements local means strict ordering for use with exact histogram equalization.
"""

import functools
from numpy import uint64, ceil, sqrt, log2
from numpy import empty, unique, ogrid

def calc_info(im, order=6):
    """
    Assign strict ordering to image pixels. The returned value is the same shape as the image but
    with values for each pixel that can be used for strict ordering. For some types of images or
    large orders this will return a stack of image values for lex-sorting.

    This implements the method by Coltuc et al [2,3] which takes increasing uniform filter sizes to
    find the local means. It takes an argument for the order (called k in the paper) which must be
    at least 2 and defaults to 6.

    Their method has been adapted to work in 3D, however only isotropic data is supported. The order
    parameter is based on distance so an order of 6 will include some pixels for 2D images (distance
    of 2*sqrt(2)) that are not included in 3D images (order 6 only goes up to sqrt(6) away).

    Since means have a constant factor in them and that won't change the relative order, it is
    removed from the computations and the returned values do not include them. The results are
    compacted as much as possible.

    The idea of using local neighbors was originally proposed in [1] and if order=2 is used on 2D
    images this reproduces that concept.

    REFERENCES:
      1. Hall EL, Feb 1974, "Almost uniform distributions for computer image enhancement", IEEE
         Transcations on Computers 23(2):207–208
      1. Coltuc D and Bolon P, 1999, "Strict ordering on discrete images and applications"
      2. Coltuc D, Bolon P and Chassery J-M, 2006, "Exact histogram specification", IEEE
         Transcations on Image Processing 15(5):1143-1152
    """
    # this uses scipy's 'reflect' mode (duplicated edge)

    from ..util import as_unsigned, correlate

    # Deal with arguments
    if order < 2 or order > 6: raise ValueError('Invalid order')
    im = as_unsigned(im)
    dt = im.dtype

    # Get the filters for this setup
    filters, includes_order_one = __get_filters(dt, order, im.ndim)

    if len(filters) == 1 and includes_order_one:
        # Single convolution
        return correlate(im.astype(uint64, copy=False), filters[0])

    # Convolve filters with the image and stack
    im = im.astype(float if im.dtype.kind == 'f' else uint64, copy=False)
    out = empty(im.shape+(len(filters)+(not includes_order_one),), im.dtype)
    for i, fltr in enumerate(filters):
        correlate(im, fltr, out[..., i])
    if not includes_order_one:
        out[..., -1] = im
    return out

@functools.lru_cache(maxsize=None)
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
        fltr = (raw == vals[order]).astype(uint64)
        used_bits = nbits + extra_bits[order]
        order -= 1
        # Add any additional orders possible to this filter
        while order >= 0 and used_bits + nbits + extra_bits[order] <= 64:
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