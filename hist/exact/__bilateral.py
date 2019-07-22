"""Implements the bilateral filter for images."""

from numpy import ceil, exp, dot, ogrid, arange

def bilateral_filter(im, size=None, sigma_r=None, sigma_d=1, **kwargs):
    """
    Bilaterally filter an image. Uses Gaussian kernels for the spatial and intensity filters.

    im is the image to filter, must be grayscale but can be any dimension
    size is the kernel size, must be odd and >=3, defaults to int(max(5, 2*ceil(3*sigma_d)+1)).
    sigma_r is the range/intensity standard deviation, defaults to image standard deviation.
    sigma_d is the domain/spatial standard deviation, default to 1.
    other keyword arguments are passed to scipy.ndimage.generic_filter.

    This attempts to use a Cython optimized function if possible. Additionally in common cases many
    parts are computed to greatly speed up the function.

    REFERENCES
     1. Tomasi C and Manduchi R, 1998, "Bilateral filtering for gray and color images". Sixth
        International Conference on Computer Vision. pp. 839–846.
     2. R A and Wilscu M, 2008, "Enhancing Contrast in Color Images Using Bilateral Filter and
        Histogram Equalization Using Wavelet Coefficients", 2008 Second International Conference on
        Future Generation Communication and Networking Symposia.
    """
    from scipy.ndimage import generic_filter

    if sigma_r is None: sigma_r = im.std()
    if size is None:
        size = int(max(5, 2*ceil(3*sigma_d)+1))
    elif size < 3 or size%2 != 1:
        raise ValueError(size)

    # Calculate the kernels
    spatial, scale, inten_lut = __bilateral_kernels(im.dtype, im.ndim, size, sigma_r, sigma_d)

    try:
        # Try to import Cython optimized code - 20 to 75x faster
        from scipy import LowLevelCallable
        import hist.exact.__bilateral_cy as cy
        _bilateral_filter = LowLevelCallable.from_cython(
            cy, 'bilateral_filter' if inten_lut is None else 'bilateral_filter_inten_lut',
            cy.get_user_data(spatial, scale, inten_lut)) # pylint: disable=c-extension-no-member
    except ImportError:
        # Fallback to pure Python function
        # Note: it seems the pure Python function actually gets slower with the intensity LUT
        def _bilateral_filter(data):
            diff = data - data[data.size // 2]
            weight = exp(diff*diff*scale) * spatial
            return dot(data, weight) / weight.sum()

    return generic_filter(im, _bilateral_filter, size, **kwargs)

def __bilateral_kernels(dt, ndim, size, sigma_r, sigma_d):
    """
    Computes the spatial kernel and the intensity kernel scale. Also computes the intensity LUT if
    it makes sense. If not None is returned in its place.
    """
    from ..util import get_dtype_min_max

    # Calculate the fixed spatial kernel
    scale = -1/(2*sigma_d*sigma_d)
    dist2 = sum(x*x for x in ogrid[(slice(-(size//2), size//2+1),)*ndim])
    spatial = (dist2*scale).ravel()
    exp(spatial, spatial)
    spatial /= spatial.sum()

    # Calculate the complete intensity LUT kernel if it makes sense
    # Don't do this for 32-bit+ integral images or floating-point images
    scale = -1/(2*sigma_r*sigma_r)
    intensity_lut = None
    if dt.kind in 'uib' and dt.itemsize <= 2:
        mn, mx = get_dtype_min_max(dt)
        intensity_lut = arange(0, mx-mn+1)
        intensity_lut *= intensity_lut
        intensity_lut = intensity_lut * scale
        exp(intensity_lut, intensity_lut)

    return spatial, scale, intensity_lut
