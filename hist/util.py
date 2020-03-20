"""
Basic utilities for working with images.

Any function here that takes arrays can take either numpy or cupy arrays.
All functions are typically faster with GPU arrays except trim_zeros().
"""

from functools import lru_cache
import numpy
import scipy.ndimage

EPS = numpy.spacing(1)
EPS_SQRT = numpy.sqrt(EPS)
FLOAT64_NMANT = numpy.finfo(float).nmant

BIT_TYPES = [numpy.bool_, bool] # bool8?
INT_TYPES = numpy.sctypes['int'] + [int]
UINT_TYPES = numpy.sctypes['uint']
FLOAT_TYPES = numpy.sctypes['float'] + [float]
BASIC_TYPES = BIT_TYPES + INT_TYPES + UINT_TYPES + FLOAT_TYPES

try:
    import cupy
    import cupyx.scipy.ndimage
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

##### Image Verification #####
def is_on_gpu(arr):
    """Checks if an array is on the GPU (i.e. it uses cupy)."""
    return HAS_CUPY and isinstance(arr, cupy.ndarray)

def get_array_module(*args):
    """
    Returns either the numpy or cupy module, cupy module is returned if any argument is on the GPU.
    """
    return cupy if any(is_on_gpu(arg) for arg in args) else numpy

def get_ndimage_module(*args):
    """
    Returns either the scipy.ndimage or cupyx.scipy.ndimage module, cupy module is returned if any
    argument is on the GPU.
    """
    return cupyx.scipy.ndimage if any(is_on_gpu(arg) for arg in args) else scipy.ndimage

def as_numpy(arr):
    """Get the given array as a numpy array."""
    return arr.get() if is_on_gpu(arr) else numpy.asanyarray(arr)

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
        if is_on_gpu(im) or is_on_gpu(mask):
            im, mask = cupy.asanyarray(im), cupy.asanyarray(mask)
    return im, mask


##### Min/Max for data types #####
def get_dtype_min_max(dt):
    """Gets the min and max value a dtype can take"""
    if not hasattr(get_dtype_min_max, 'mn_mx'):
        mn_mx = {t:(numpy.iinfo(t).min, numpy.iinfo(t).max) for t in INT_TYPES + UINT_TYPES}
        mn_mx.update({t:(t(False), t(True)) for t in BIT_TYPES})
        mn_mx.update({t:(t('0.0'), t('1.0')) for t in FLOAT_TYPES})
        get_dtype_min_max.mn_mx = mn_mx
    return get_dtype_min_max.mn_mx[numpy.dtype(dt).type]

def get_dtype_min(dt):
    """Gets the min value a dtype can take"""
    return get_dtype_min_max(dt)[0]

def get_dtype_max(dt):
    """Gets the max value a dtype can take"""
    return get_dtype_min_max(dt)[1]

def get_im_min_max(im):
    """Gets the min and max values for an image or an image dtype."""
    if not hasattr(im, 'dtype'):
        return get_dtype_min_max(im)
    dt = im.dtype
    if dt.kind != 'f':
        return get_dtype_min_max(dt)
    mn, mx = im.min(), im.max()
    return (mn, mx) if mn < 0.0 or mx > 1.0 else get_dtype_min_max(dt)

def get_uint_dtype_fit(nbits):
    """
    Gets the uint dtype string that describes the type that has at least the given number of bits.
    """
    return 'u' +str(max(1 << (log2i(nbits)-3), 1))


##### Data Type Conversions #####
def as_unsigned(im):
    """
    If the given image is a signed integer image then it is converted to an unsigned image while
    keeping the order of values correct (by moving min to 0).
    """
    if im.dtype.kind == 'i':
        dt = im.dtype
        im = im.view(numpy.dtype(dt.byteorder+'u'+str(dt.itemsize))) - get_dtype_min(dt)
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

def ci_artificial_gpu_support(calc_info):
    """
    Decorator for calc_info(im, **kwargs) functions that adds "support" for GPU-based images by
    taking the image provided and converting it to a numpy array and taking the returned values and
    converting back to a GPU array (if the original was a GPU array).
    """
    def _wrapper(im, *args, **kwargs):
        if not is_on_gpu(im):
            return calc_info(im, *args, **kwargs)
        from cupy import asanyarray # pylint: disable=import-error
        values = calc_info(im.get(), *args, **kwargs)
        return asanyarray(values) if not isinstance(values, tuple) else \
            (asanyarray(values[0]), values[1])
    _wrapper.__doc__ = calc_info.__doc__
    return _wrapper


##### Other Helpers #####
def imhist(im, nbins=256, mask=None):
    """Calculate the histogram of an image. By default it uses 256 bins (nbins)."""
    im, mask = check_image_mask_single_channel(im, mask)
    if mask is not None: im = im[mask]
    return __imhist(im, nbins)

def __imhist(im, nbins=256):
    """Core of imhist with no checks or handling of mask."""
    xp = get_array_module(im)
    mn, mx = get_im_min_max(im)
    return xp.histogram(im, xp.linspace(mn, mx, nbins+1))[0]

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

def axes_combinations(ndim):
    """
    A generator for all possible combinations of axes not including no axes. For example, for ndim=2
    this produces (0,), (1,), (0, 1) and for ndim=3 this produces (0,), (1,), (2,), (0, 1), (0, 2),
    (1, 2), (0, 1, 2).
    """
    from itertools import chain, combinations
    yield from chain.from_iterable(combinations(range(ndim), i+1) for i in range(ndim))

@lru_cache(maxsize=None)
def log2i(val):
    """Gets the log base 2 of an integer, rounded up to an integer."""
    return val.bit_length() - is_power_of_2(val)

def is_power_of_2(val):
    """Returns True if an integer is a power of 2. Only works for x > 0."""
    return not val & (val-1)

def tuple_set(base, values, indices):
    """
    Creates a new tuple with the given values put at indices and otherwise the same as base. The
    list of indices must be in sorted order.
    """
    new = base[:indices[0]]
    for i in range(len(indices)-1):
        new += (values[i],) + base[indices[i]+1:indices[i+1]]
    return new + (values[-1],) + base[indices[-1]+1:]

def prod(iterable):
    """Product of all values in an iterable, like sum() but for multiplication."""
    # NOTE: in Python 3.8 this is now available as math.prod()
    from functools import reduce
    from operator import mul
    return reduce(mul, iterable, 1)

def make_readonly(value):
    """
    Makes numpy arrays read-only. If the value is a tuple then it is searched for arrays
    recursively. Lists are changed into tuples. Anything else is just return as-is.
    """
    if isinstance(value, (tuple, list)):
        return tuple(make_readonly(elem) for elem in value)
    if isinstance(value, numpy.ndarray):
        value.flags.writeable = False
    return value

def lru_cache_array(func):
    """
    To lru_cache(maxsize=None) this adds that all returned numpy arrays are made readonly so that
    they cannot be modified and thus change the cached value. This uses make_readonly().
    """
    @lru_cache(maxsize=None)
    def _wrapped(*args, **kwargs):
        return make_readonly(func(*args, **kwargs))
    _wrapped.__doc__ = func.__doc__
    return _wrapped

@lru_cache_array
def dist2_matrix(order, ndim):
    """
    Generate a squared-distance matrix with ndim dimensions that has at least order unique distances
    in it. The distance are all relative to the middle of the matrix.
    """
    import math
    size = int(math.ceil(0.5*math.sqrt(8*order+1)-0.5))-1
    slc = slice(-size, size+1) # the filter is 2*size+1 square
    return sum(x*x for x in numpy.ogrid[(slc,)*ndim])

@lru_cache_array
def generate_disks(order, ndim, hollow=False):
    """
    Generate disk/sphere masks up to a particular order for a number of dimensions. Does not include
    order 1 (which would simply be the 1 surrounded by zeros). The order is based on number of
    unique distance values from the middle of the disk/sphere.

    Returns a tuple of boolean masks from highest to lowest order. If hollow=True is provided then
    the masks are 0 where other orders would cover the values, otherwise the disks are solid True.
    Each individual mask is made to be as small as possible by removing extraneous 0s on the
    outside.
    """
    raw = dist2_matrix(order, ndim)
    values = numpy.unique(raw)[1:order][::-1]
    return tuple(trim_zeros((raw == i) if hollow else (raw <= i)) for i in values)

def trim_zeros(arr):
    """
    Trims rows/columns/planes of all zeros from an array. It is assumed that the array is
    symmetrical in all directions.
    """
    n = (len(arr) + 1) // 2
    line = arr[(n,)*(arr.ndim-1)] != 0
    for i in range(n):
        if line[i]: return arr[(slice(i, -i),)*arr.ndim]
    return arr[(slice(0, 0),)*arr.ndim]


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
    shape = tuple(x//sz for x, sz in zip(im.shape, block_size)) + block_size
    strides = tuple(stride*sz for stride, sz in zip(im.strides, block_size)) + im.strides
    return get_array_module(im).lib.stride_tricks.as_strided(im, shape=shape, strides=strides)

def reduce_blocks(blocks, func, out=None):
    """
    Take a set of blocks like those given by block_view and apply func to each each axis of each
    block, eventually reducing to a single scalar for each block. That function must take a
    positional argument of axis (always given -1) and a keyword argument of out.

    If the blocks array passed to this function is on the GPU (i.e. a cupy.ndarray) then the
    function given should be a cupy function and the optional out array should be a cupy.ndarray as
    well. This function is typically much, much, faster when running on a GPU array.
    """
    n = blocks.ndim // 2
    blocks = func(blocks, -1)
    for _ in range(n - 2):
        blocks = func(blocks, -1, out=blocks[..., 0])
    return func(blocks, -1, out=out)
