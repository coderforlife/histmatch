"""
Implements exact histogram equalization / matching. The underlying methods for establishing a strict
ordering are in other files in this module.

These function support images as either numpy or cupy arrays. If the chosen method doesn't support
cupy arrays then the array is convert to a numpy array before calling calc_info() and the resulting
values returned are converted back into a cupy array.
"""

def histeq_exact(im, h_dst=256, mask=None, method='VA', return_fails=False, stable=False, **kwargs): #pylint: disable=too-many-arguments
    """
    Like histeq except the histogram of the output image is exactly as given in h_dst. This is
    accomplished by strictly ordering all pixels based on other properties of the image. There is
    currently no way to specify the source histogram or to calculate the transform to apply to
    different images. See histeq for the details of the arguments h_dst and mask. If return_fails
    parameter is set to True then a second value is returned with the number of non-unique values
    in the strict ordering. To get a percentage, divide by the number of pixels in the image or
    the number of Trues in the mask if provided.

    This histogram equalization method takes significantly more time than the approximate
    ("classical") histogram equalization method.

    This function supports images that are stored on the GPU (as cupy arrays) for a considerable
    speedup. However, not all methods listed below support using the GPU for their computations and
    thus you may not get the full speedup.

    Internally this uses one of the following methods to create the strict ordering of pixels. The
    method defaults to 'VA' which is a bit slower than 'LM' for 8-bit images but faster then other
    types. Also it is more accurate in general.

    ARBITRARY: [1,2,4]
        Simplest approach. All ties are broken in an arbitrary but consistent manner. This method
        is dubbed "Sort-Matching" in [1] and "Quick Histogram Specification" (QHS) in [4]. Useful
        for establishing a baseline on time and pixel ordering failure rate for using any strict
        ordering method. Inherently supports 3D and/or anisotropic data. Supports using GPU.

    RAND: [2,3]
        Very simple approach. All ties in the ordering are broken randomly. Like arbitrary this
        is extremely fast and introduces noise into the data. Unlike arbitrary, this will not
        produce consistent results. Additionally, unlike arbitrary, this will have an extremely
        low pixel failure rate. Inherently supports 3D and/or anisotropic data. Supports using GPU.

    NA: Neighborhood Average [3,4]
        Uses the average of the 3-by-3 neighborhood to break ties. The size param can be used to
        change the neighborhood size. If invert=True parameter is given then higher average
        neighborhoods will lower the outcome of the pixel instead of raise it, encouraging
        enhancement of edges as described in [4]. Remaining ties are broken via arbitrary as per
        [3] where [4] instead didn't break remaining ties at all.

        Method adapted to support 3D data. Does not support anisotropic data. Supports using GPU.

    NV: Neighborhood Voting by Eramian and Mould [4]
        Uses the number of pixels in the 3-by-3 neighborhood that are greater than the central pixel
        to break ties. The size param can be used to change the neighborhood size. If invert=True
        parameter is given then the number of pixels less than the central pixel is used,
        encouraging enhancement of edges as described in [4]. Remaining ties are broken via
        arbitrary where [4] instead didn't break remaining ties at all.

        Method has been adapted to support 3D data. Does not support anisotropic data.

    GL: Gaussian and Laplacian Filtering by Coltuc and Bolon [5]
        Uses a series of Gaussian and Laplacian of Gaussian (LoG) convolutions with the standard
        deviation of the kernel increasing to use information from further away in the picture. The
        default creates 4 additional versions of the image, alternating between Gaussian and LoG
        with standard deviations of 0.5 and 1.0. The absolute value of the LoG is used (this can be
        turned off with the laplacian_mag=False parameter). The logic behind the alternation is one
        gives information about local brightness and the other one gives information about the
        edges.

        Method adapted to support 3D data. Does not support anisotropic data. Supports using GPU.

    ML: Mean and Laplacian Filtering by Jung [7]
        Uses a series of local mean and Laplacian convolutions with the order increasing to use
        information from further away in the picture. Attempts to be a combination between GL and LM
        methods however the Laplacians end up being redundant with the mean filters and thus this is
        equivalent to the LM method but much less efficient.

        Method has been adapted to support 3D data. Does not support anisotropic data.

    LC: Local Contrast by Coltuc and Bolon [5]
        Uses the difference between the maximum and minimum values of expanding disks to distinguish
        between same-valued pixels. Accepts a `order` argument which has a default of 6 that
        controls how far away pixels are used to help distinguish pixels from each other.

        Method adapted to support 3D data. Does not support anisotropic data. Supports using GPU.

    LM: Local Means by Coltuc, Bolon and Chassery [5,6]
        Uses the gray levels of expanding mean filters to distinguish between same-valued pixels.
        Accepts a `order` argument which has a default of 6 that controls how far away pixels are
        used to help distinguish pixels from each other. An order of 1 would be using only the pixel
        itself and no neighbors (but this isn't allowed). It has been optimized for 8-bit images
        which take about twice the memory and 8x the time from classical histeq. Other image types
        which can take 7x the memory and 40x the time.

        Method adapted to support 3D data. Does not support anisotropic data. Supports using GPU.

    WA: Wavelet Approach by Wan and Shi [8]
        Uses the traditional (decimating) wavelet transform and complex sorting rules to distinguish
        between same-valued pixels. Even though this method is referenced in many papers and used as
        a baseline to demonstrate improvements, it is unlikely to have been implemented correctly in
        any of them due to the complex sorting required. In some papers this is obvious by the way
        they reference the data (for example saying that k=9 and a standard strict ordering is
        performed).

        Due to the complexity of this method and the various confusing parts of the original paper,
        this is a best effort to do it correctly but may still be incorrect.

        Method has been adapted to support 3D data. Does not support anisotropic data.

        This requires pyWavelets along with Cython to use the C++ std:sort for efficiency.

    SWA: Stationary Wavelet Approach
        This is a derivation of the WA that uses the stationary wavelet transform and does not
        have the additional sorting steps. This is likely (or similar to) the method used by most of
        the papers referencing [8] such as [10].

        To reproduce [9] the image must have a bilateral filter applied first with a size of 3 and a
        sigma_r and sigma_d of 1.

        Can also set detail_magnitude=False to disable getting magnitude of edges which might be
        more accurate to various papers but further from [8]. Can adjust the number of levels with
        nlevels (defaults to 2) and the kernel used (defaults to 'haar').

        Method has been adapted to support 3D data. Does not support anisotropic data.

    VA: Variational Approach by Nikolova, Wen and Chan [10]
        This attempts to reconstruct the original real-valued version of the image and thus is a
        continuous-valued version of the image which can be strictly ordered. This has several
        parameters including niters, beta, alpha_1, alpha_2, and gamma to control how the
        real-valued version is computed. See hist_eq_va.calc_info for more information.

        Method has been adapted to support 3D and/or anisotropic data. Use gamma parameter to
        control for anisotropicity.

    OPTIMUM: Optimum Approach by Balado [11]
        It turns out that it is equivalent to arbitrary with stable sorting except during
        reconstruction (when passing reconstruction=True) in which case minor changes are made to
        the order to be optimal. Inherently supports 3D and/or anisotropic data. Supports using GPU.

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
      7. Jung S-W, 2014, "Exact Histogram Specification Considering the Just Noticeable Difference",
         IEIE Transactions on Smart Processing and Computing, 3(2):52-58.
      8. Wan Y and Shi D, 2007, "Joint exact histogram specification and image enhancement through
         the wavelet transform", IEEE Transcations on Image Processing, 16(9):2245-2250.
      9. R A and Wilscu M, 2008, "Enhancing Contrast in Color Images Using Bilateral Filter and
         Histogram Equalization Using Wavelet Coefficients", 2008 Second International Conference on
         Future Generation Communication and Networking Symposia.
     10. Nikolova M and Steidl G, 2014, "Fast ordering algorithm for exact histogram specification",
         IEEE Transcations on Image Processing, 23(12):5274-5283.
     11. Balado, Félix, 2018, "Optimum Exact Histogram Specification", IEEE International Conference
         on Acoustics, Speech and Signal Processing.

    Additional references for each are available with their respective calc_info functions.
    """
    from ..util import check_image_mask_single_channel

    # Check arguments
    im, mask = check_image_mask_single_channel(im, mask)
    n = im.size if mask is None else int(mask.sum())
    h_dst = __check_h_dst(h_dst, n)

    ##### Create strict-orderable versions of image #####
    # These are frequently floating-point 'images' and/or images with an extra dimension giving a
    # 'tuple' of data for each pixel
    method = method.lower()
    if method == 'optimum': kwargs['h_dst'] = h_dst # optimum needs the h_dst data
    if method in ('wa', 'optimum'): # these directly compute failures
        kwargs['return_fails'] = return_fails
    values = __calc_info(im, method, **kwargs)

    ##### Assign strict ordering #####
    idx, fails = __sort_pixels(values, im.shape, mask, return_fails, stable)
    del values

    ##### Create the transform that is the size of the image but with sorted histogram values #####
    transform = __calc_transform(h_dst, im.dtype, n, idx)
    del h_dst

    ##### Create the equalized image #####
    out = __apply_transform(idx, transform, im.shape, mask)

    # Done
    return (out, fails) if return_fails else out

def __check_h_dst(h_dst, n):
    """
    Check the h_dst argument and return it converted to a fleshed-out histogram for the given number
    of pixels. Accepts integers or arrays. Always returns a numpy array and never a GPU array.
    """
    # pylint: disable=invalid-name
    from numbers import Integral
    from numpy import tile, floor, intp
    from ..util import as_numpy
    if isinstance(h_dst, Integral):
        h_dst = int(h_dst)
        h_dst = tile(n/h_dst, h_dst)
    else:
        h_dst = as_numpy(h_dst)
        h_dst = h_dst.ravel()*(n/h_dst.sum())
    if len(h_dst) < 2: raise ValueError('h_dst')
    H_whole = floor(h_dst).astype(intp, copy=False)
    nw = H_whole.sum()
    if n == nw:
        h_dst = H_whole
    else:
        # Add up the fractional amounts and put somewhere
        R = (h_dst-H_whole).argpartition(-(n-nw))[-(n-nw):]
        h_dst = H_whole
        h_dst[R] += 1
        del R
    return h_dst

def __calc_info(im, method, **kwargs):
    """
    Calculate the strict-orderable version of an image. Returns a floating-point 'images' or images
    with an extra dimension giving a 'tuple' of data for each pixel.

    If the method does not support GPU arrays but im is a GPU array, it will be copied off of the
    GPU to call the calc_info() function and then the resulting values will be copied back to the
    GPU at the end. Thus if given a GPU array this always returns a GPU array.
    """
    # pylint: disable=too-many-branches
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
    elif method == 'ml':
        from .basic import calc_info_mean_laplacian as calc_info
    elif method == 'lc':
        from .basic import calc_info_local_contrast as calc_info
    elif method == 'lm':
        from .lm import calc_info
    elif method == 'wa':
        from .wa import calc_info
    elif method == 'swa':
        from .swa import calc_info
    elif method == 'va':
        from .va import calc_info
    elif method == 'optimum':
        from .optimum import calc_info
    else:
        raise ValueError('method')
    return calc_info(im, **kwargs)

def __sort_pixels(values, shape, mask=None, return_fails=False, stable=False):
    """
    Uses the values (pixels with extra data) to sort all of the pixels. If stable is True than a
    stable sort is performed, defaulting to False. However, if values represent 'tuples' of data
    per pixel than lexsort is used which is always stable. Additionally, 1D data is assumed to be
    already sorted in which case this simply calculates fails if requested (and likely to be 0) and
    applies the mask if necessary.

    Returns the indices of the sorted values and the number of fails (or None if not requested).
    """
    from ..util import is_on_gpu

    ##### Check if the values already contain the failures #####
    fails = None
    if return_fails and isinstance(values, tuple):
        values, fails = values

    ##### Assign strict ordering #####
    if values.ndim == 1:
        # Already sorted
        from ..util import prod
        assert values.size == prod(shape)
        if mask is not None: values = values[mask.ravel()]
        idx = values
    elif values.shape == shape:
        # Single value per pixel
        values = values.ravel() if mask is None else values[mask]
        # Numpy now uses radix sort for integers which is much faster than quicksort
        kind = 'stable' if stable or values.dtype.kind not in 'fc' else 'quicksort'
        idx = values.argsort() if is_on_gpu(values) else values.argsort(kind=kind)
    else:
        # Tuple of values per pixel - need lexsort
        assert values.shape[1:] == shape
        values = values.reshape(values.shape[0], -1) if mask is None else values[:, mask]
        if is_on_gpu(values):
            from cupy import lexsort # pylint: disable=import-error
            idx = lexsort(values)
        else:
            from numpy import lexsort
            idx = lexsort(values, 0)

    # Done if not calculating failures
    if not return_fails or fails is not None: return idx, fails

    # Calculate the number of sort failures
    values = values.T # for lexsorted values
    values_sorted = values[idx]
    not_equals = values_sorted[1:] != values_sorted[:-1]
    del values_sorted
    if not_equals.ndim == 2: not_equals = not_equals.any(1) # for lexsorted values
    return idx, int(not_equals.size - not_equals.sum())

def __calc_transform(h_dst, dt, n_mask, idx):
    """Create the transform that is the size of the image but with sorted histogram values."""
    from ..util import get_dtype_min_max, get_array_module, is_on_gpu

    xp = get_array_module(idx)
    mn, mx = get_dtype_min_max(dt)
    values = xp.linspace(mn, mx, len(h_dst), dtype=dt)
    transform = xp.zeros(idx.size, dtype=dt) # start with all zeros

    if is_on_gpu(idx):
        # Even though .repeat() is supported in cupy, it is very slow - this is WAY faster
        trans_kernel = __get_calc_transform_kernel()
        trans_kernel(xp.asarray(h_dst).cumsum(), values, transform[-n_mask:])
    else:
        transform[-n_mask:] = values.repeat(h_dst)
    return transform

try:
    import cupy
    @cupy.memoize()
    def __get_calc_transform_kernel():
        return cupy.ElementwiseKernel(
            'raw int64 h_dst_cs, raw T values', 'T transform',
            '''int left = 0, right = h_dst_cs.shape()[0];
while (left < right) {
    int j = left + (right - left)/2; // (left + right) / 2
    if (h_dst_cs[j] <= i) { left = j + 1; }
    else { right = j; }
}
transform = values[left];''', 'calc_transform')
except ImportError:
    pass

def __apply_transform(idx, transform, shape, mask=None):
    """Apply a transform with the strict ordering in idx."""
    from ..util import get_array_module
    xp = get_array_module(idx)
    if mask is not None:
        mask_idx = xp.empty(idx.size, transform.dtype)
        mask_idx.put(idx, transform)
        out = xp.zeros(shape, transform.dtype)
        xp.place(out, mask, mask_idx)
    else:
        out = xp.empty(shape, transform.dtype)
        out.put(idx, transform)
    return out
