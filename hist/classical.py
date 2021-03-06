"""
Implements classical histogram equalization / matching.

These function support images as either numpy or cupy arrays. The histeq_trans() function does
accept the histogram arguments as cupy arrays but immediately converts to numpy arrays and always
returns a numpy array.
"""

from .util import check_image_mask_single_channel, get_array_module, get_dtype_max, get_dtype_min

def histeq(im, h_dst=64, h_src=None, mask=None):
    """
    Equalize the histogram of an image. The destination histogram is either given explicitly (as a
    sequence) or a uniform distribution of a fixed number of bins (defaults to 64 bins).

    The performs an approximate histogram equalization. This is the most common technique used.
    This is faster and less memory intensive than exact histogram equalization and can be given the
    source histogram or split into two functions (thus it cannot be easily parallelized).

    Additionally, you can specify the source historgram instead of calculating it from the image
    itself (using 256 bins). This is useful if you want to use the source histogram to be from
    multiple images and so that each image will be mapped the same way (although using
    histeq_trans/histeq_apply will be more efficient for something like that).

    Supports using a mask in which only those elements will be transformed and considered in
    calcualting h_src.

    Supports integral and floating-point (from 0.0 to 1.0) image data types. To do something similar
    with bool/logical, use convert.bw. The h_dst and h_src must have at least 2 bins.

    Supports GPU array images (as cupy arrays) for a considerable speedup.
    """
    im, mask = check_image_mask_single_channel(im, mask)
    if mask is None:
        im = __histeq(im, h_dst, h_src)
    else:
        im[mask] = __histeq(im[mask], h_dst, h_src)
    return im

def __histeq(im, h_dst, h_src):
    """
    Calls histeq_trans then __histeq_apply with some minor optimizations. This is the internal
    function used by histeq which only handles checking and masking.
    """
    im, orig_dt = __as_unsigned(im)
    if h_src is None:
        from .util import __imhist
        h_src = __imhist(im)
    transform = histeq_trans(h_src, h_dst, im.dtype)
    return __restore_signed(__histeq_apply(im, transform), orig_dt)

def histeq_trans(h_src, h_dst, dt):
    """
    Calculates the histogram equalization transform. It takes a source histogram, destination
    histogram, and a data type. It returns the transform, which has len(h_src) elements of a
    data-type similar to the given data-type but unsigned. This transform can be used with
    histeq_apply. This allows you to calculate the transform just once for the same source and
    destination histograms and use it many times.

    This is really just one-half of histeq, see it for more details.

    This function never uses the GPU. If either h_src or h_dst are GPU-arrays, they are converted to
    numpy arrays before doing any computions. A numpy array is always returned.
    """
    from numbers import Integral
    from numpy import tile, vstack, dtype
    from .util import EPS_SQRT, as_numpy

    dt = dtype(dt)
    if dt.base != dt or dt.kind not in 'iuf': raise ValueError("Unsupported data-type")
    if dt.kind == 'i': dt = dtype(dt.byteorder+'u'+str(dt.itemsize))

    # Prepare the source histogram
    h_src = as_numpy(h_src)
    h_src = h_src.ravel()/h_src.sum()

    # Prepare the destination histogram
    if isinstance(h_dst, Integral):
        h_dst = int(h_dst)
        h_dst = tile(1/h_dst, h_dst)
    else:
        h_dst = as_numpy(h_dst)
        h_dst = h_dst.ravel()/h_dst.sum()

    if h_dst.size < 2 or h_src.size < 2: raise ValueError('Invalid histograms')

    # Compute the transform
    xx = vstack((h_src, h_src)) # pylint: disable=invalid-name
    xx[0, -1], xx[1, 0] = 0.0, 0.0
    tol = tile(xx.min(0)/2.0, (h_dst.size, 1))
    err = tile(h_dst.cumsum(), (h_src.size, 1)).T - tile(h_src.cumsum(), (h_dst.size, 1)) + tol
    err[err < -EPS_SQRT] = 1.0
    transform = err.argmin(0)*(get_dtype_max(dt)/(h_dst.size-1.0))
    transform = transform.round(out=transform).astype(dt, copy=False)
    return transform

def histeq_apply(im, transform, mask=None):
    """
    Apply a histogram-equalization transformation to an image. The transform can be created with
    histeq_trans. The image must have the same data-type as given to histeq_trans.

    This is really just one-half of histeq, see it for more details.

    Supports GPU array images (as cupy arrays) for a considerable speedup.
    """
    im, mask = check_image_mask_single_channel(im, mask)
    if mask is None:
        im = __histeq_apply(im, transform)
    else:
        im[mask] = __histeq_apply(im[mask], transform)
    return im

def __histeq_apply(im, transform):
    """
    Core of histeq_apply, that function only handles checking the image and mask and deals with the
    mask if necessary.
    """
    from .util import is_on_gpu
    xp = get_array_module(im)
    im, orig_dt = __as_unsigned(im)
    nlevels = get_dtype_max(im.dtype)
    transform = xp.asanyarray(transform)
    if orig_dt.kind != 'f' and nlevels == len(transform)-1:
        # perfect fit, we don't need to scale the indices
        # TODO: GPU: github.com/cupy/cupy/issues/3017
        idx = im.astype(xp.intp) if is_on_gpu(im) else im
    else:
        # scale the indices
        idx = im*(float(len(transform)-1)/nlevels)
        idx = xp.rint(idx, out=xp.empty(im.shape, dtype=xp.intp), casting='unsafe')
    return __restore_signed(transform.take(idx), orig_dt)

def __as_unsigned(im):
    """
    If the image is signed integers then it is converted to unsigned. The image and the original
    dtype are returned.
    """
    from numpy import dtype
    dt = im.dtype
    if dt.kind == 'i':
        im = im.view(dtype(dt.byteorder+'u'+str(dt.itemsize))) - get_dtype_min(dt)
    return im, dt

def __restore_signed(im, dt):
    """Restore an image data type after using __as_unsigned."""
    if dt.kind == 'i': im -= -int(get_dtype_min(dt))
    return im.view(dt)
