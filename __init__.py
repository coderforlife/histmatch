"""
Histogram equalization / matching techniques.
"""

from .util import check_image_mask_single_channel

from .hist_standard import histeq, histeq_trans, histeq_apply

# numpy >= v1.8

def imhist(im, nbins=256, mask=None):
    """Calculate the histogram of an image. By default it uses 256 bins (nbins)."""
    im, mask = check_image_mask_single_channel(im, mask)
    if mask is not None: im = im[mask]
    return __imhist(im, nbins)

def __imhist(im, nbins):
    from scipy.ndimage import histogram
    from .util import get_im_min_max
    mn, mx = get_im_min_max(im)
    return histogram(im, mn, mx, nbins)

def histeq_exact(im, h_dst=256, mask=None, print_info=False, method='VA', **kwargs):
    """
    Like histeq except the histogram of the output image is exactly as given in h_dst. This is
    acomplished by strictly ordering all pixels based on other properties of the image. There is
    currently no way to specify the source histogram or to calculate the transform to apply to
    different images. See histeq for the details of the arguments h_dst and mask.

    This method takes significantly more time than the approximate ("standard") version. See below
    for more details.

    Internally this uses either the random values, local mean ordering (LM) by Coltuc, Bolon and
    Chassery, the variational approach (VA) of Nikolova, Wen and Chan, or the wavelet approach (WA)
    by Wan and Shi to create the strict ordering of pixels. The method defaults to 'VA' which is a
    bit slower than 'LM' for 8-bit images but faster for other types. Also it is more accurate in
    general.

    ARBITRARY/QHS:
        Simpliest approach. All ties are broken in an arbitrary but consistent manner. This method
        is dubbed "Quick Histogram Specification" (QHS) in [5]. Useful for establishing a baseline
        on time and pixel ordering failure rate for using any strict ordering method.

    RAND:
        Almost the simpliest approach. All ties in the ordering are broken randomly. Unlike
        arbitrary/qhs, this will not produce consistent results. Additionally, unlike arbitrary/qhs,
        this will have an extremely low pixel failure rate.

    LM:
        This uses the gray levels of expanding mean filters to distinguish between same-valued
        pixels. Accepts a `order` argument which has a default of 6 that controls how far away
        pixels are used to help distinguish pixels from each other. An order of 1 would be using
        only the pixel itself and no neighbors (but this isn't allowed). It has been optimized for
        8-bit images which take about twice the memory and 8x the time from standard histeq. Other
        image types which can take 7x the memory and 40x the time.

    VA:
        This attempts to reconstruct the original real-valued version of the image and thus is a
        continuous-valued version of the image which could be strictly ordered. Accepts a `niters`
        argument that specifies how many minimization iterations to perform to make sure the real-
        valued image is faithly reproduced. The default is 5. If takes about twice the memory and
        10x the time from standard histeq regardless of type.

    WA:
        ...

    REFERENCES:
      1. Coltuc D and Bolon P, 1999, "Strict ordering on discrete images and applications"
      2. Coltuc D, Bolon P and Chassery J-M, 2006, "Exact histogram specification", IEEE
         Trans. on Image Processing 15(5):1143-1152
      3. Nikolova M, Wen Y-W, and Chan R, 2013, "Exact histogram specification for digital images
         using a variational approach", J of Mathematical Imaging and Vision, 46(3):309-325
      4. Nikolova M and Steidl G, 2014, "Fast ordering algorithm for exact histogram specification",
         IEEE Trans. on Image Processing, 23(12):5274-5283
      5. Wan Y and Shi D, 2007, "Joint exact histogram specification and image enhancement through
         the wavelet transform", IEEE Trans. on Image Processing, 16(9):2245-2250.
    """
    from numbers import Integral
    from numpy import tile

    # Check arguments
    im, mask = check_image_mask_single_channel(im, mask)
    shape, n, dt = im.shape, im.size, im.dtype
    if mask is not None: mask, n = mask.ravel(), mask.sum()
    h_dst = tile(n/h_dst, h_dst) if isinstance(h_dst, Integral) else h_dst.ravel()*(n/h_dst.sum()) #pylint: disable=no-member
    if len(h_dst) < 2: raise ValueError('Invalid histogram')

    ##### Create strict-orderable versions of image #####
    # These are frequently floating-point 'images' and/or images with an extra dimension giving a
    # 'tuple' of data for each pixel
    values = __ehe_calc_info(im, method, **kwargs)
    del im

    ##### Assign strict ordering #####
    idx = __ehe_sort_pixels(values, shape, mask, print_info)
    del values, mask

    ##### Create the transform that is the size of the image but with sorted histogram values #####
    # Since there could be fractional amounts, make sure they are added up and put somewhere
    transform = __ehe_calc_transform(idx, h_dst, dt, n)
    del h_dst

    ##### Create the equalized image #####
    return transform.take(idx).reshape(shape)

def __ehe_calc_info(im, method, **kwargs):
    """
    Calculate the strict-orderable version of an image. Returns a floating-point 'images' or images
    with an extra dimension giving a 'tuple' of data for each pixel.
    """
    method = method.lower()
    if method in ('arbitrary', 'qhs', None):
        calc_info = lambda x: x
    elif method in ('rand', 'random'):
        calc_info = __ehe_calc_info_rand
    elif method == 'lm':
        from .hist_eq_lm import calc_info
    elif method == 'va':
        from .hist_eq_va import calc_info
    elif method == 'wa':
        from .hist_eq_wa import calc_info
    else:
        raise ValueError('method')
    return calc_info(im, **kwargs)

def __ehe_calc_info_rand(im):
    """Gets a pixel order where ties are broken randomly."""
    from numpy.random import random
    if im.dtype.kind in 'iub':
        im = im + random(im.shape) - 0.5
    return im

def __ehe_sort_pixels(values, shape, mask=None, print_info=False):
    """
    Uses the values (pixels with extra data) to sort all of the pixels.
    Returns the indices of the sorted values.
    """
    from numpy import lexsort

    ##### Assign strict ordering #####
    if values.shape == shape:
        # Single value per pixel
        values = values.ravel()
        sort_pass1 = values.argsort()
        idx = sort_pass1.argsort()
    else:
        # Tuple of values per pixel
        assert values.shape[:len(shape)] == shape
        values = values.reshape((-1, values.shape[-1]))
        sort_pass1 = lexsort(values.T, 0)
        idx = sort_pass1.argsort()
    if print_info:
        # TODO: this doesn't take into account the mask
        values_sorted = values[sort_pass1]
        equals = values_sorted[1:] != values_sorted[:-1]
        if equals.ndim == 2: equals = equals.any(1)
        n_unique, n = equals.sum() + 1, idx.size
        del equals, values_sorted
        print('duplicates:', (n - n_unique) / n, n - n_unique)
    del sort_pass1

    ##### Handle the mask #####
    if mask is not None:
        idx[~mask] = 0 #pylint: disable=invalid-unary-operand-type
        idx[mask] = idx[mask].argsort().argsort()

    # Done
    return idx

def __ehe_calc_transform(idx, h_dst, dt, n):
    """
    Create the transform that is the size of the image but with sorted histogram values. Since
    there could be fractional amounts, make sure they are added up and put somewhere.
    """
    # pylint: disable=invalid-name
    from numpy import floor, intp, empty, repeat, linspace
    from .util import get_dtype_min_max
    H_whole = floor(h_dst).astype(intp, copy=False)
    nw = H_whole.sum()
    if n == nw:
        h_dst = H_whole
    else:
        R = (h_dst-H_whole).argpartition(-(n-nw))[-(n-nw):]
        h_dst = H_whole
        h_dst[R] += 1
        del R
    mn, mx = get_dtype_min_max(dt)
    transform = empty(idx.size, dtype=dt)
    transform[-n:] = repeat(linspace(mn, mx, len(h_dst), dtype=dt), h_dst)
    return transform
