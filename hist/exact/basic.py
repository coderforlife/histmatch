"""
Implements basic strict ordering techniques for use with exact histogram equalization. Included are:

* Random
* Gaussian/Laplacian
* Mean/Laplacian
* Local contrast
* Neighborhood average and inverted neighborhood average
* Neighborhood voting and inverted neighborhood voting
"""

from numpy import uint64
from ..util import HAS_CUPY, get_array_module, get_ndimage_module, is_on_gpu
from ..util import as_unsigned, log2i, get_uint_dtype_fit, get_dtype_max, generate_disks

def calc_info_rand(im):
    """
    Gets a pixel order where ties are broken randomly. As stated in [1] this will introduce noise.
    Note that for continuous images (i.e. float) no randomness is added.

    REFERENCES
      1. Rosenfeld A and Kak A, 1982, "Digital Picture Processing"
    """
    if im.dtype.kind in 'iub':
        im = im + get_array_module(im).random.random(im.shape)
        im -= 0.5
    return im


def calc_info_gaussian_laplacian(im, sigmas=(0.5, 1.0), laplacian_mag=True, allow_compaction=True):
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
    from ..util import as_float
    xp = get_array_module(im)
    ndi = get_ndimage_module(im)

    # Compaction only works for 8-bit integers
    # On the CPU compaction is a bit slower but will sort faster later
    if allow_compaction and im.dtype.kind in 'iub' and im.dtype.itemsize == 1:
        im = as_unsigned(im)
        out = xp.zeros((len(sigmas),) + im.shape, uint64)
        out[-1, ...] = im
        out[-1, ...] <<= 56 # identity in 56-63 (8 bits)
        gauss, lap = xp.empty(im.shape), xp.empty(im.shape)
        tmp = xp.empty(im.shape, uint64)
        for i, sigma in enumerate(sigmas):
            ndi.gaussian_filter(im, sigma, output=gauss, truncate=2)
            ndi.laplace(gauss, output=lap)
            if laplacian_mag:
                xp.abs(lap, lap)
            else:
                lap += 1020 # needs to be positive
            # Gaussian filter in bits 26-49 (24 bits)
            tmp[...] = xp.multiply(gauss, 1<<16, gauss).round(out=gauss)
            tmp <<= 26
            out[-i-1, ...] |= tmp
            # Laplacian filter in bits 0-25 (26 bits)
            tmp[...] = xp.multiply(lap, 1<<16, lap).round(out=lap)
            out[-i-1, ...] |= tmp
        return out[0] if len(sigmas) <= 1 else out

    # Non-compacted version
    im = as_float(im)
    out = xp.empty((2*len(sigmas)+1,) + im.shape)
    out[-1, ...] = im
    for i, sigma in enumerate(sigmas):
        gauss = ndi.gaussian_filter(im, sigma, output=out[-2*i-2, ...], truncate=2)
        lap = ndi.laplace(gauss, output=out[-2*i-3, ...])
        if laplacian_mag: xp.abs(lap, lap)
    return out


def calc_info_mean_laplacian(im, order=3, laplacian_mag=True, allow_compaction=True):
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
    if not allow_compaction or im.dtype.kind == 'f' or im.dtype.itemsize > 2:
        return non_compact(im, 2*order-2, __gen_mean_laplacian_filtered, (filters, laplacian_mag))
    scales = [scale_from_filter(fltr) for fltr in filters] if not laplacian_mag else \
        [(0, int(fltr.max() if i%2 == 1 else fltr.sum())) for i, fltr in enumerate(filters)]
    return compact(im, scales, __gen_mean_laplacian_filtered, (filters, laplacian_mag))

def __gen_mean_laplacian_filtered(im, filters, laplacian_mag=True):
    """Generator for Mean-Laplacian strict ordering."""
    xp = get_array_module(im)
    ndi = get_ndimage_module(im)
    data = xp.empty(im.shape, float if im.dtype.kind == 'f' else xp.int64)
    for i, fltr in enumerate(filters):
        ndi.correlate(im, fltr, output=data)
        if laplacian_mag and i%2 == 1: xp.abs(data, data)
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


def calc_info_local_contrast(im, order=6, allow_compaction=True):
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
    xp = get_array_module(im)
    ndi = get_ndimage_module(im)

    if order < 1: raise ValueError('order')
    im = as_unsigned(im)
    disks = generate_disks(order, im.ndim)

    # Floating-point images cannot be compacted
    if not allow_compaction or im.dtype.type == 'f':
        out = xp.empty((order,) + im.shape)
        min_tmp = out[-1, ...]
        for i, disk in enumerate(disks):
            disk = xp.asanyarray(disk)
            ndi.maximum_filter(im, footprint=disk, output=out[i, ...])
            ndi.minimum_filter(im, footprint=disk, output=min_tmp)
            out[i, ...] -= min_tmp
        out[-1, ...] = im
        return out

    # Integral images can be compacted or at least stored in uint64 outputs
    # On the CPU compaction is a bit slower but will sort faster later
    bpp = im.dtype.itemsize*8
    dst_type = uint64 if order*bpp >= 64 else get_uint_dtype_fit(order*bpp)
    out = xp.zeros(((order*bpp+63)//64,) + im.shape, dst_type)
    max_tmp, min_tmp = xp.empty(im.shape, dst_type), xp.empty(im.shape, dst_type)
    layer, shift = 0, 0
    for disk in disks:
        disk = xp.asanyarray(disk)
        ndi.maximum_filter(im, footprint=disk, output=max_tmp)
        ndi.minimum_filter(im, footprint=disk, output=min_tmp)
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


def calc_info_neighborhood_avg(im, size=3, invert=False, allow_compaction=True):
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
    from ..util import FLOAT64_NMANT
    xp = get_array_module(im)

    # Deal with arguments
    if size < 3 or size % 2 != 1: raise ValueError('size')
    im = as_unsigned(im)
    dt = im.dtype
    n_neighbors = size ** im.ndim
    shift = dt.itemsize*8 + log2i(n_neighbors)
    nbits = shift + dt.itemsize*8

    if not allow_compaction or dt.kind == 'f' or nbits > FLOAT64_NMANT:
        # No compaction possible
        out = xp.empty((2,) + im.shape)
        avg = __correlate_uniform(im, size, out[0, ...])
        if invert:
            xp.subtract(avg.max() if dt.kind == 'f' else
                        (n_neighbors * get_dtype_max(dt)), avg, avg)
        out[1, ...] = im # the original image is still part of this
    else:
        # Compact the results
        out = __correlate_uniform(im, size, xp.empty(im.shape, uint64))
        if invert:
            xp.subtract(n_neighbors * get_dtype_max(dt), out, out)
        im = im.astype(out.dtype)
        im <<= shift
        out |= im # the original image is still part of this
    return out

def __correlate_uniform(im, size, output):
    """
    Uses repeated scipy.ndimage.filters.correlate1d() calls to compute a uniform filter. Unlike
    scipy.ndimage.filters.uniform_filter() this just uses ones(size) instead of ones(size)/size.
    """
    # TODO: smarter handling of in-place convolutions?
    ndi = get_ndimage_module(im)
    weights = get_array_module(im).ones(size)
    for axis in range(im.ndim):
        ndi.correlate1d(im, weights, axis, output)
        im = output
    return output


def calc_info_neighborhood_voting(im, size=3, invert=False, allow_compaction=True):
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
    xp = get_array_module(im)

    # Deal with arguments
    if size < 3 or size % 2 != 1: raise ValueError('size')
    im = as_unsigned(im)
    dt = im.dtype
    n_neighbors = size ** im.ndim
    shift = log2i(n_neighbors)
    nbits = shift + dt.itemsize*8

    if not allow_compaction or dt.kind == 'f' or nbits > 63:
        # No compaction possible
        out = xp.zeros((2,) + im.shape, float if dt.kind == 'f' else uint64)
        __count_votes(im, out[0, ...], size, invert)
        out[1, ...] = im # the original image is still part of this
    else:
        # Compact the results
        out = xp.zeros(im.shape, get_uint_dtype_fit(nbits))
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
    if is_on_gpu(im):
        voting = __get_count_votes_cupy_kernel(invert)

    else:
        # Try to use the Cython functions if possible - they are ~160x faster!
        try:
            from scipy import LowLevelCallable
            import hist.exact.__basic as cy
            voting = LowLevelCallable.from_cython(cy, 'vote_lesser' if invert else 'vote_greater')
        except ImportError:
            # Fallback
            from numpy import empty, greater, less
            compare = greater if invert else less
            tmp = empty(size ** im.ndim, bool)
            mid = tmp.size // 2
            voting = lambda x: compare(x[mid], x, tmp).sum()

    # Once we get the appropriate voting function we can call generic filter
    get_ndimage_module(im).generic_filter(im, voting, size, output=out)

if HAS_CUPY:
    import cupy # pylint: disable=import-error
    @cupy.util.memoize(for_each_device=True)
    def __get_count_votes_cupy_kernel(invert):
        sign = '>' if invert else '<'
        return cupy.ReductionKernel('X x', 'Y y',
                                    '_raw_x[_in_ind.size()/2]{}x'.format(sign),
                                    'a + b', 'y = a', '0', 'lt', reduce_type='int')
