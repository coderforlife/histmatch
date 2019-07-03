"""
Basic utilities for working with images.
"""

from functools import lru_cache
from numpy import dtype, sctypes, bool_, spacing, sqrt

EPS = spacing(1)
EPS_SQRT = sqrt(EPS)

BIT_TYPES = [bool_, bool] # bool8?
INT_TYPES = sctypes['int'] + [int]
UINT_TYPES = sctypes['uint']
FLOAT_TYPES = sctypes['float'] + [float]
BASIC_TYPES = BIT_TYPES + INT_TYPES + UINT_TYPES + FLOAT_TYPES


##### Image Verification #####
def is_single_channel_image(im):
    """
    Returns True if `im` is a single-channel image, basically it is a ndarray of 2 or 3 dimensions
    where the 3rd dimension length can only be 1 and the data type is a basic data type (integer
    or float but not complex). Does not check to see that the image has no zero-length dimensions.
    """
    if im.ndim in (3, 4) and im.shape[-1] == 1:
        im = im.squeeze(-1)
    return im.dtype.type in BASIC_TYPES and (im.ndim == 2 or im.ndim == 3 and im.shape[2] > 4)

def check_image_single_channel(im):
    """
    Similar to is_single_channel_image except instead of returning True/False it throws an exception
    if it isn't an image. Also, it returns a 2D image (with the 3rd dimension, if 1, removed).
    """
    if im.ndim in (3, 4) and im.shape[-1] == 1:
        im = im.squeeze(-1)
    if im.dtype.type not in BASIC_TYPES or (im.ndim != 2 and (im.ndim != 3 or im.shape[2] <= 4)):
        raise ValueError('Not single-channel image format')
    return im

def check_image_mask_single_channel(im, mask):
    """
    Checks if an image and possibly a mask are single-channel. The mask, if not None, must be bool
    and the same shape as the image. The image and mask are returned (without a 3rd dimension).
    """
    im = check_image_single_channel(im)
    if mask is not None:
        mask = check_image_single_channel(mask)
        if mask.dtype != bool or mask.shape != im.shape:
            raise ValueError('The mask must be a binary image with equal dimensions to the image')
    return im, mask


##### Min/Max for data types #####
def get_dtype_min_max(dt):
    """Gets the min and max value a dtype can take"""
    if not hasattr(get_dtype_min_max, 'mn_mx'):
        from numpy import iinfo
        mn_mx = {t:(iinfo(t).min, iinfo(t).max) for t in INT_TYPES + UINT_TYPES}
        mn_mx.update({t:(t(False), t(True)) for t in BIT_TYPES})
        mn_mx.update({t:(t('0.0'), t('1.0')) for t in FLOAT_TYPES})
        get_dtype_min_max.mn_mx = mn_mx
    return get_dtype_min_max.mn_mx[dtype(dt).type]

def get_dtype_min(dt):
    """Gets the min value a dtype can take"""
    return get_dtype_min_max(dt)[0]

def get_dtype_max(dt):
    """Gets the max value a dtype can take"""
    return get_dtype_min_max(dt)[1]

def get_im_min_max(im):
    """Gets the min and max values for an image or an image dtype."""
    from numpy import ndarray
    if not isinstance(im, ndarray):
        return get_dtype_min_max(im)
    dt = im.dtype
    if dt.kind != 'f':
        return get_dtype_min_max(dt)
    mn, mx = im.min(), im.max()
    return (mn, mx) if mn < 0.0 or mx > 1.0 else get_dtype_min_max(dt)


##### Data Type Conversions #####
def as_unsigned(im):
    """
    If the given image is a signed integer image then it is converted to an unsigned image while
    keeping the order of values correct (by moving min to 0).
    """
    if im.dtype.kind == 'i':
        dt = im.dtype
        im = im.view(dtype(dt.byteorder+'u'+str(dt.itemsize))) - get_dtype_min(dt)
    return im

def as_float(im):
    """
    If the given image is integral then it is converted to float64. Unsigned images will end up in
    the range [0.0, 1.0] while signed images will end up in the range [-1.0, 1.0). For int64 images
    this may result in a loss of precision.
    """
    if im.dtype.kind == 'i':
        return im / get_dtype_min(im.dtype)
    if im.dtype.kind == 'bu':
        return im / get_dtype_max(im.dtype)
    return im.astype(float, copy=False)


##### Other Helpers #####
@lru_cache(maxsize=None)
def get_diff_slices(ndim):
    """
    Gets a list of pairs of slices to use to get all the neighbor differences for the given number
    of dimensions.
    """
    from itertools import product
    slices = (slice(1, None), slice(None, None), slice(None, -1))
    out = []
    for item in product((0, 1, 2), repeat=ndim):
        if all(x == 1 for x in item): continue
        item_inv = tuple(slices[2-x] for x in item)
        item = tuple(slices[x] for x in item)
        if (item_inv, item) not in out:
            out.append((item, item_inv))
    return out

@lru_cache(maxsize=None)
def log2i(val):
    """Gets the log base 2 of an integer, rounded up to an integer."""
    return val.bit_length() - is_power_of_2(val)

def is_power_of_2(val):
    """Returns True if an integer is a power of 2. Only works for x > 0."""
    return not val & (val-1)

def tuple_set(base, values, inds):
    """
    Creates a new tuple with the given values put at indices and otherwise the same as base. The
    list of inds must be in sorted order.
    """
    new = base[:inds[0]]
    for i in range(len(inds)-1):
        new += (values[i],) + base[inds[i]+1:inds[i+1]]
    return new + (values[-1],) + base[inds[-1]+1:]


##### Correlate Function #####
def correlate(im, weights, output=None, mode='reflect', cval=0.0):
    """
    Improved version of scipy.ndimage.filters.correlate that checks if a multidimensional filter can
    be broken into several 1D filters and then performs several 1D correlations instead of an nD
    correlation which is much faster.

    Additionally it supports the following special weights:
       if weights is a tuple of 1D vectors it treats it as already decomposed
       if weights is a 1D ndarray it is used along every axis

    Does not support the origin argument of scipy.ndimage.correlate. The mode argument does not
    support different values for each axis.
    """
    from scipy.ndimage.filters import correlate, correlate1d # pylint: disable=redefined-outer-name
    from scipy.ndimage._ni_support import _get_output
    output = _get_output(output, im)
    if isinstance(weights, tuple):
        if len(weights) != im.ndim: raise ValueError('len(weights) != im.ndim')
        for axis, axis_weights in enumerate(weights):
            correlate1d(im, axis_weights, axis, output, mode, cval)
            im = output
        return output
    if weights.ndim == 1:
        for axis in range(im.ndim):
            correlate1d(im, weights, axis, output, mode, cval)
            im = output
        return output
    # TODO: decompose
    #if any(weights.shape == 1):
    #    ...
    return correlate(im, weights, output, mode, cval)

def __decompose_2d(kernel): # [h1,h2]
    """
    Decompose a 2D kernel into 2 1D kernels if possible. Returns None, None otherwise.
    """
    # NOTE: this always returns float arrays and may introduce negative signs and other quirks
    # pylint: disable=invalid-name
    from numpy.linalg import svd
    u, s, vh = svd(kernel)
    if sum(s > max(kernel.shape)*EPS*s.max()) != 1: return None, None
    s = sqrt(s[0])
    return u[:, 0]*s, vh[0, :]*s


##### Image as blocks #####
def block_view(im, block_size):
    """
    View an image as a series of non-overlapping blocks. If the shape of the image is not a multiple
    of the block size in a particular dimension, the extra rows/columns/planes will simply be
    dropped.

    This function causes no memory allocation instead it manipulates the view of the image. The
    returned view with have the square of the number of dimensions originally in the image with the
    last being exactly block_size.
    """
    if len(block_size) != im.ndim: raise ValueError('block_size')
    from numpy.lib.stride_tricks import as_strided
    shape = tuple(x//sz for x, sz in zip(im.shape, block_size)) + block_size
    strides = tuple(stride*sz for stride, sz in zip(im.strides, block_size)) + im.strides
    return as_strided(im, shape=shape, strides=strides)

def reduce_blocks(blocks, func, out=None):
    """
    Take a set of blocks like those given by block_view and apply func to each each axis of each
    block, eventually reducing to a single scalar for each block. That function must take a
    positional argument of axis (always given -1) and a keyword argument of out.
    """
    n = blocks.ndim // 2
    blocks = func(blocks, -1)
    for _ in range(n - 2):
        blocks = func(blocks, -1, out=blocks[..., 0])
    return func(blocks, -1, out=out)
