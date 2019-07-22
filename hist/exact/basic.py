"""
Implements basic strict ordering techniques for use with exact histogram equalization. Included are:

* Random
* Gaussian/Laplacian
* Mean/Laplacian
* Local contrast
* Neighborhood average and inverted neighborhood average
* Neighborhood voting and inverted neighborhood voting
"""

from numpy import empty, zeros, uint64
from ..util import as_unsigned, log2i, get_uint_dtype_fit, get_dtype_max, generate_disks, correlate

def calc_info_rand(im):
    """
    Gets a pixel order where ties are broken randomly. As stated in [1] this will introduce noise.
    Note that for continuous images (i.e. float) no randomness is added.

    REFERENCES
      1. Rosenfeld A and Kak A, 1982, "Digital Picture Processing"
    """
    from numpy.random import random
    if im.dtype.kind in 'iub':
        im = im + random(im.shape) - 0.5
    return im

def calc_info_gaussian_laplacian(im, sigmas=(0.5, 1.0), laplacian_mag=True):
    """
    Assign strict ordering to image pixels. The returned value is the same shape but with an extra
    dimension for the results of the additional filters. This stack needs to be lex-sorted.

    This calculates the extra information by taking a series of Gaussian and Laplacian of Gaussian
    convolutions with the standard deviation of the kernel increasing to use information from
    further away in the picture. The default settings result in 4 additional versions of the image,
    alternating between Gaussian and Laplacian of Gaussian with a standard deviation of 0.5 and 1.0.
    The absolute value of the Laplacian of Gaussian results are taken. The logic behind the
    alternation is one gives information about local brightness and the other one gives information
    about the edges.

    REFERENCES
      1. Coltuc D and Bolon P, 1999, "Strict ordering on discrete images and applications"
    """
    # this uses scipy's 'reflect' mode (duplicated edge)
    from numpy import abs, multiply # pylint: disable=redefined-builtin
    from scipy.ndimage import gaussian_filter, laplace
    from ..util import as_float

    # Compaction only works for 8-bit integers
    if im.dtype.kind in 'iub' and im.dtype.itemsize == 1:
        im = as_unsigned(im)
        out = empty((len(sigmas),) + im.shape, uint64)
        out[-1, ...] = im
        out[-1, ...] <<= 56 # identity in 56-63 (8 bits)
        gauss, lap = empty(im.shape), empty(im.shape)
        tmp = empty(im.shape, uint64)
        for i, sigma in enumerate(sigmas):
            gaussian_filter(im, sigma, output=gauss, truncate=2)
            laplace(gauss, output=lap)
            if laplacian_mag:
                abs(lap, lap)
            else:
                lap += 1020 # needs to be positive
            # Gaussian filter in bits 26-49 (24 bits)
            tmp[...] = multiply(gauss, 1<<16, gauss).round(out=gauss)
            tmp <<= 26
            out[-i-1, ...] |= tmp
            # Laplacian filter in bits 0-25 (26 bits)
            tmp[...] = multiply(lap, 1<<16, lap).round(out=lap)
            out[-i-1, ...] |= tmp
        return out[0] if len(sigmas) <= 1 else out

    # Non-compacted version
    im = as_float(im)
    out = empty((2*len(sigmas)+1,) + im.shape)
    out[-1, ...] = im
    for i, sigma in enumerate(sigmas):
        gauss = gaussian_filter(im, sigma, output=out[-2*i-2, ...], truncate=2)
        lap = laplace(gauss, output=out[-2*i-3, ...])
        if laplacian_mag: abs(lap, lap)
    return out

def calc_info_mean_laplacian(im, order=3, laplacian_mag=True):
    """
    Assign strict ordering to image pixels. The returned value is the same shape as the image but
    with values for each pixel that can be used for strict ordering. For some types of images or
    large orders this will return a stack of image values for lex-sorting.

    This method was proposed in [1] and is a derivative of the local means method by [2] except that
    it interleaves high-frequency filters with the local means so that it handles edges better. The
    concept is similar to [3] except for the style of filters. However, as described, it is actually
    identical to the LM method by [2] since the Laplacian filters end up producing redundant
    information with earlier filters.

    REFERENCES
      1. Jung S-W, 2014, "Exact Histogram Specification Considering the Just Noticeable Difference",
         IEIE Transactions on Smart Processing and Computing, 3(2):52-58.
      2. Coltuc D, Bolon P and Chassery J-M, 2006, "Exact histogram specification", IEEE
         Transcations on Image Processing 15(5):1143-1152.
      3. Coltuc D and Bolon P, 1999, "Strict ordering on discrete images and applications"
    """
    # this uses scipy's 'reflect' mode (duplicated edge)
    from .__compaction import non_compact, compact, scale_from_filter

    # Deal with arguments
    if order < 2: raise ValueError('order')

    # Create the filters
    filters = [None, None] * (order-1)
    filters[::2] = [__make_laplacian(fltr) for fltr in generate_disks(order, im.ndim, False)]
    filters[1::2] = generate_disks(order, im.ndim)
    filters = filters[::-1] # compact goes from most-to-least important

    # Filter the image
    if im.dtype.kind == 'f' or im.dtype.itemsize > 2:
        return non_compact(im, 2*order-2, __gen_mean_laplacian_filtered, (filters, laplacian_mag))
    scales = [scale_from_filter(fltr) for fltr in filters] if not laplacian_mag else \
        [(0, int(fltr.max() if i%2 == 1 else fltr.sum())) for i, fltr in enumerate(filters)]
    return compact(im, scales, __gen_mean_laplacian_filtered, (filters, laplacian_mag))

def __gen_mean_laplacian_filtered(im, filters, laplacian_mag=True):
    """Generator for Mean-Laplacian strict ordering."""
    from numpy import int64, abs # pylint: disable=redefined-builtin
    data = empty(im.shape, float if im.dtype.kind == 'f' else int64)
    for i, fltr in enumerate(filters):
        correlate(im, fltr, output=data)
        if laplacian_mag and i%2 == 1: abs(data, data)
        yield data

def __make_laplacian(fltr):
    """
    Take a disk and convert it into a 'Laplacian' filter with the central value equal to the number
    of other values in the disk and all other values changed to -1. The Laplacian matches those
    defined in [1].

    REFERENCES
      1. Jung S-W, 2014, "Exact Histogram Specification Considering the Just Noticeable Difference",
         IEIE Transactions on Smart Processing and Computing, 3(2):52-58.
    """
    fltr = fltr * -1
    fltr[tuple(x//2 for x in fltr.shape)] = -1-fltr.sum()
    return fltr

def calc_info_local_contrast(im, order=6):
    """
    Assign strict ordering to image pixels. The returned value is the same shape as the image but
    with values for each pixel that can be used for strict ordering. For some types of images or
    large orders this will return a stack of image values for lex-sorting.

    This calculates the extra information by taking the difference between the maximum and minimum
    of the values within increasing sized disks around the pixel as per equation 6 in [1]. The
    default order is 6.

    REFERENCES
      1. Coltuc D and Bolon P, 1999, "Strict ordering on discrete images and applications"
    """
    # this uses scipy's 'reflect' mode (duplicated edge) and thus has no effect
    from scipy.ndimage import maximum_filter, minimum_filter
    im = as_unsigned(im)
    disks = generate_disks(order, im.ndim)

    # Floating-point images cannot be compacted
    if im.dtype.type == 'f':
        out = empty((order,) + im.shape)
        min_tmp = out[-1, ...]
        for i, disk in enumerate(disks):
            maximum_filter(im, footprint=disk, output=out[i, ...])
            minimum_filter(im, footprint=disk, output=min_tmp)
            out[i, ...] -= min_tmp
        out[-1, ...] = im
        return out

    # Integral images can be compacted or at least stored in uint64 outputs
    bpp = im.dtype.itemsize*8
    dst_type = uint64 if order*bpp >= 64 else get_uint_dtype_fit(order*bpp)
    out = empty(((order*bpp+63)//64,) + im.shape, dst_type)
    max_tmp, min_tmp = empty(im.shape, dst_type), empty(im.shape, dst_type)
    layer, shift = 0, 0
    for disk in disks:
        maximum_filter(im, footprint=disk, output=max_tmp)
        minimum_filter(im, footprint=disk, output=min_tmp)
        max_tmp -= min_tmp
        max_tmp <<= shift
        out[layer, ...] |= max_tmp
        shift += bpp
        if shift >= 64:
            layer += 1
            shift = 0

    im = im.astype(dst_type)
    im <<= shift
    out[-1, ...] |= im
    return out[0, ...] if layer == 0 else out

def calc_info_neighborhood_avg(im, size=3, invert=False):
    """
    Assign strict ordering to image pixels. The returned value is the same shape as the image but
    with values for each pixel that can be used for strict ordering. For some types of images or
    large sizes this will return a stack of image values for lex-sorting.

    Uses the average of a single neighborhood of size-by-size (default 3-by-3). The size must be
    odd and greater than or equal to 3.

    If invert is True than the values are inverted so that high averages will sort lower and low
    average will sort higher.

    Since the average will have a constant factor in it and that won't change the relative order, it
    is removed from the computations. Results are compacted if possible.

    Called bar-alpha_m in [2] where m is the size or alpha_m if inverted. Their method has been
    adapted to work in 3D, however only isotropic data is supported. Additionally, when doing
    invserion, instead of subtracting the average from the central value the average is subtracted
    from the max possible value so that no negative numbers are created.

    REFERENCES
      1. Rosenfeld A and Kak A, 1982, "Digital Picture Processing"
      2. Eramian M and Mould D, 2005, "Histogram Equalization using Neighborhood Metrics",
         Proceedings of the Second Canadian Conference on Computer and Robot Vision.
    """
    # this uses scipy's 'reflect' mode (duplicated edge) ([2] says this should be constant-0)
    from numpy import ones, subtract
    from ..util import FLOAT64_NMANT

    # Deal with arguments
    if size < 3 or size % 2 != 1: raise ValueError('size')
    im = as_unsigned(im)
    dt = im.dtype
    n_neighbors = size ** im.ndim
    shift = dt.itemsize*8 + log2i(n_neighbors)
    nbits = shift + dt.itemsize*8

    if dt.kind == 'f' or nbits > FLOAT64_NMANT:
        # No compaction possible
        out = empty((2,) + im.shape)
        avg = correlate(im, ones(size), out[0, ...])
        if invert:
            subtract(avg.max() if dt.kind == 'f' else (n_neighbors * get_dtype_max(dt)), avg, avg)
        out[1, ...] = im # the original image is still part of this
    else:
        # Compact the results
        out = correlate(im, ones(size), empty(im.shape, uint64))
        if invert:
            subtract(n_neighbors * get_dtype_max(dt), out, out)
        im = im.astype(out.dtype)
        im <<= shift
        out |= im # the original image is still part of this
    return out

def calc_info_neighborhood_voting(im, size=3, invert=False):
    """
    Assign strict ordering to image pixels. The returned value is the same shape as the image but
    with values for each pixel that can be used for strict ordering. For some types of images or
    large sizes this will return a stack of image values for lex-sorting.

    Counts the number of neighbors in a size-by-size (default 3-by-3) region that are greater than
    the central pixel (default) or less than (invert=True). The size must be odd and greater than or
    equal to 3. Results are compacted if possible.

    When invert=True, called beta_m in [2] where m is the size. Their method has been adapted to
    work in 3D, however only isotropic data is supported.

    REFERENCES
      1. Eramian M and Mould D, 2005, "Histogram Equalization using Neighborhood Metrics",
         Proceedings of the Second Canadian Conference on Computer and Robot Vision.
    """
    # this uses scipy's 'reflect' mode (duplicated edge) ([2] says this should be constant-0)

    # TODO: get different results with/without compaction and with/without Cython...

    # Deal with arguments
    if size < 3 or size % 2 != 1: raise ValueError('size')
    im = as_unsigned(im)
    dt = im.dtype
    n_neighbors = size ** im.ndim
    shift = log2i(n_neighbors)
    nbits = shift + dt.itemsize*8

    if dt.kind == 'f' or nbits > 63:
        # No compaction possible
        out = zeros((2,) + im.shape, float if dt.kind == 'f' else uint64)
        __count_votes(im, out[0, ...], size, invert)
        out[1, ...] = im # the original image is still part of this
    else:
        # Compact the results
        out = zeros(im.shape, get_uint_dtype_fit(nbits))
        __count_votes(im, out, size, invert)
        im = im.astype(out.dtype)
        im <<= shift
        out |= im # the original image is still part of this

    return out

def __count_votes(im, out, size, invert):
    """
    Count the votes for each pixel in a size-by-size region around each one, saving the totals to
    out. If invert is supplied, every pixel less than the values for it, otherwise every pixel
    greater than it votes for it.

    REFERENCES
      1. Eramian M and Mould D, 2005, "Histogram Equalization using Neighborhood Metrics",
         Proceedings of the Second Canadian Conference on Computer and Robot Vision.
    """
    # Try to use the Cython functions if possible - they are ~160x faster!
    try:
        from scipy import LowLevelCallable
        import hist.exact.__basic as cy
        voting = LowLevelCallable.from_cython(cy, 'vote_lesser' if invert else 'vote_greater')
    except ImportError:
        # Fallback
        from numpy import greater, less
        compare = less if invert else greater
        tmp = empty(size ** im.ndim, bool)
        mid = tmp.size // 2
        voting = lambda x: compare(x[mid], x, tmp).sum()

    from scipy.ndimage import generic_filter
    generic_filter(im, voting, size, output=out)

    # Original solution
    # Doesn't take into image edges or size but was slightly faster than Cython
    # from numpy import greater, less
    # from ..util import get_diff_slices
    # tmp = empty(im.shape, bool)
    # if invert: greater, less = less, greater
    # for slc_pos, slc_neg in get_diff_slices(im.ndim):
    #     tmp_x = tmp[slc_neg]
    #     out[slc_pos] += greater(im[slc_pos], im[slc_neg], tmp_x)
    #     out[slc_neg] += less(im[slc_pos], im[slc_neg], tmp_x)
    # return out
