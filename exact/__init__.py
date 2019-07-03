"""
Implements exact histogram equalization / matching. The underlying methods for establishing a strict
ordering are in other files in this module.
"""

def histeq_exact(im, h_dst=256, mask=None, return_fails=False, method='VA', **kwargs):
    """
    Like histeq except the histogram of the output image is exactly as given in h_dst. This is
    accomplished by strictly ordering all pixels based on other properties of the image. There is
    currently no way to specify the source histogram or to calculate the transform to apply to
    different images. See histeq for the details of the arguments h_dst and mask. If return_fails
    parameter is set to True then a second value is returned with the number of non-unique values
    in the strict ordering. To get a percentage, divide by the number of pixels in the image or
    the number of Trues in the mask if provided.

    This method takes significantly more time than the approximate ("classical") version.

    Internally this uses one of the following methods to create the strict ordering of pixels. The
    method defaults to 'VA' which is a bit slower than 'LM' for 8-bit images but faster for other
    types. Also it is more accurate in general.

    ARBITRARY: [1,2,4]
        Simplest approach. All ties are broken in an arbitrary but consistent manner. This method
        is dubbed "Sort-Matching" in [1] and "Quick Histogram Specification" (QHS) in [4]. Useful
        for establishing a baseline on time and pixel ordering failure rate for using any strict
        ordering method. Inherently supports 3D and/or anisotropic data.

    RAND: [2,3]
        Very simple approach. All ties in the ordering are broken randomly. Like arbitrary this
        is extremely fast and introduces noise into the data. Unlike arbitrary, this will not
        produce consistent results. Additionally, unlike arbitrary, this will have an extremely
        low pixel failure rate. Inherently supports 3D and/or anisotropic data.

    NA: Neighborhood Average [3,4]
        Uses the average of the 3-by-3 neighborhood to break ties. The size param can be used to
        change the neighborhood size. If invert=True parameter is given then higher average
        neighborhoods will lower the outcome of the pixel instead of raise it, encouraging
        enhancement of edges as described in [4]. Remaining ties are broken via arbitrary as per
        [3] where [4] instead didn't break remaining ties at all.

        Method has been adapted to support 3D data. Does not support anisotropic data.

    NV: Neighborhood Voting by Eramian and Mould [4]
        Uses the number of pixels in the 3-by-3 neighborhood that are greater than the central pixel
        to break ties. The size param can be used to change the neighborhood size. If invert=True
        parameter is given then the number of pixels less than the central pixel is used,
        encouraging enhancement of edges as described in [4]. Remaining ties are broken via
        arbitrary where [4] instead didn't break remaining ties at all.

        Method has been adapted to support 3D data. Does not support anisotropic data.

    GL: Gaussian and Lapacian Filtering by Coltuc and Bolon [5]
        Uses a series of Gaussian and Laplacian of Gaussian (LoG) convolutions with the standard
        deviation of the kernel increasing to use information from further away in the picture. The
        default creates 6 additional versions of the image, alternating between Gaussian and LoG
        with standard deviations of 0.5, 1.0, and 1.5. The absolute value of the LoG is used. The
        logic behind the alternation is one gives information about local brightness and the other
        one gives information about the edges.

        Method has been adapted to support 3D data. Does not support anisotropic data.

    LC: Local Contrast by Coltuc and Bolon [5]
        Uses the difference between the maximum and minimum values of expanding disks to distinguish
        between same-valued pixels. Accepts a `order` argument which has a default of 6 that
        controls how far away pixels are used to help distinguish pixels from each other.

        Method has been adapted to support 3D data. Does not support anisotropic data.

    LM: Local Means by Coltuc, Bolon and Chassery [5,6]
        Uses the gray levels of expanding mean filters to distinguish between same-valued pixels.
        Accepts a `order` argument which has a default of 6 that controls how far away pixels are
        used to help distinguish pixels from each other. An order of 1 would be using only the pixel
        itself and no neighbors (but this isn't allowed). It has been optimized for 8-bit images
        which take about twice the memory and 8x the time from classical histeq. Other image types
        which can take 7x the memory and 40x the time.

        Method has been adapted to support 3D data. Does not support anisotropic data.

    WA: Wavelet Approach by Wan and Shi [7]
        ...

        Method has been adapted to support 3D data. Does not support anisotropic data.

    VA: Variational Approach by Nikolova, Wen and Chan [8]
        This attempts to reconstruct the original real-valued version of the image and thus is a
        continuous-valued version of the image which can be strictly ordered. This has several
        parameters including niters, beta, alpha_1, alpha_2, and gamma to control how the
        real-valued version is computed. See hist_eq_va.calc_info for more information.

        Method has been adapted to support 3D and/or anisotropic data. Use gamma parameter to
        control for anisotropicity.

    REFERENCES:
      1. Rolland JP, Vo V, Bloss B, and Abbey CK, 2000, "Fast algorithm for histogram
         matching applications to texture synthesis", Journal of Electronic Imaging 9(1):39–45.
      2. Bevilacqua A, Azzari P, 2007, "A high performance exact histogram specification algorithm",
         14th International Conference on Image Analysis and Processing.
      3. Rosenfeld A and Kak A, 1982, "Digital Picture Processing".
      4. Eramian M and Mould D, 2005, "Histogram Equalization using Neighborhood Metrics",
         Proceedings of the Second Canadian Conference on Computer and Robot Vision.
      5. Coltuc D and Bolon P, 1999, "Strict ordering on discrete images and applications".
      6. Coltuc D, Bolon P and Chassery J-M, 2006, "Exact histogram specification", IEEE
         Transcations on Image Processing 15(5):1143-1152.
      7. Wan Y and Shi D, 2007, "Joint exact histogram specification and image enhancement through
         the wavelet transform", IEEE Transcations on Image Processing, 16(9):2245-2250.
      8. Nikolova M and Steidl G, 2014, "Fast ordering algorithm for exact histogram specification",
         IEEE Transcations on Image Processing, 23(12):5274-5283.

    Additional references for each are available with their respective calc_info functions.
    """
    from numbers import Integral
    from numpy import tile
    from ..util import check_image_mask_single_channel

    # Check arguments
    im, mask = check_image_mask_single_channel(im, mask)
    shape, n, dt = im.shape, im.size, im.dtype
    if mask is not None: mask, n = mask.ravel(), mask.sum()
    h_dst = tile(n/h_dst, h_dst) if isinstance(h_dst, Integral) else h_dst.ravel()*(n/h_dst.sum()) #pylint: disable=no-member
    if len(h_dst) < 2: raise ValueError('h_dst')

    ##### Create strict-orderable versions of image #####
    # These are frequently floating-point 'images' and/or images with an extra dimension giving a
    # 'tuple' of data for each pixel
    values = __calc_info(im, method, **kwargs)
    del im

    ##### Assign strict ordering #####
    idx, fails = __sort_pixels(values, shape, mask, return_fails)
    del values, mask

    ##### Create the transform that is the size of the image but with sorted histogram values #####
    # Since there could be fractional amounts, make sure they are added up and put somewhere
    transform = __calc_transform(idx, h_dst, dt, n)
    del h_dst

    ##### Create the equalized image #####
    out = transform.take(idx).reshape(shape)

    # Done
    return (out, fails) if return_fails else out

def __calc_info(im, method, **kwargs):
    """
    Calculate the strict-orderable version of an image. Returns a floating-point 'images' or images
    with an extra dimension giving a 'tuple' of data for each pixel.
    """
    method = method.lower()
    if method in ('arbitrary', None):
        calc_info = lambda x: x
    elif method in ('rand', 'random'):
        from .basic import calc_info_rand as calc_info
    elif method == 'na':
        from .basic import calc_info_neighborhood_avg as calc_info
    elif method == 'nv':
        from .basic import calc_info_neighborhood_voting as calc_info
    elif method == 'gl':
        from .basic import calc_info_gaussian_laplacian as calc_info
    elif method == 'lc':
        from .basic import calc_info_local_contrast as calc_info
    elif method == 'lm':
        from .lm import calc_info
    elif method == 'wa':
        from .wa import calc_info
    elif method == 'va':
        from .va import calc_info
    else:
        raise ValueError('method')
    return calc_info(im, **kwargs)

def __sort_pixels(values, shape, mask=None, return_fails=False):
    """
    Uses the values (pixels with extra data) to sort all of the pixels.
    Returns the indices of the sorted values and the number of fails (or None if not requested).
    """
    ##### Assign strict ordering #####
    if values.shape == shape:
        # Single value per pixel
        values = values.ravel()
        sort_pass1 = values.argsort()
    else:
        # Tuple of values per pixel - need lexsort
        from numpy import lexsort
        assert values.shape[:len(shape)] == shape
        values = values.reshape((-1, values.shape[-1]))
        sort_pass1 = lexsort(values.T, 0)

    # Calculate the number of sort failures
    fails = None
    if return_fails:
        values_sorted = values[sort_pass1]
        not_equals = values_sorted[1:] != values_sorted[:-1]
        del values_sorted
        if not_equals.ndim == 2: not_equals = not_equals.any(1) # lexsorted
        if mask is not None: not_equals = not_equals[mask[:-1]]
        fails = not_equals.size - not_equals.sum()
        del not_equals

    # Final sort to establish strict ordering
    idx = sort_pass1.argsort()
    del sort_pass1

    ##### Handle the mask #####
    if mask is not None:
        idx[~mask] = 0 #pylint: disable=invalid-unary-operand-type
        idx[mask] = idx[mask].argsort().argsort()

    # Done
    return idx, fails

def __calc_transform(idx, h_dst, dt, n):
    """
    Create the transform that is the size of the image but with sorted histogram values. Since
    there could be fractional amounts, make sure they are added up and put somewhere.
    """
    # pylint: disable=invalid-name
    from numpy import floor, intp, empty, repeat, linspace
    from ..util import get_dtype_min_max
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
