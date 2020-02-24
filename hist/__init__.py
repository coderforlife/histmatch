"""
Histogram calculation and equalization/matching techniques.
"""

from .classical import histeq, histeq_trans, histeq_apply
from .exact import histeq_exact
from .metrics import (contrast_per_pixel, distortion, count_differences, psnr, ssim,
                      enhancement_measurement)

def imhist(im, nbins=256, mask=None):
    """Calculate the histogram of an image. By default it uses 256 bins (nbins)."""
    from .util import check_image_mask_single_channel
    im, mask = check_image_mask_single_channel(im, mask)
    if mask is not None: im = im[mask]
    return __imhist(im, nbins)

def __imhist(im, nbins):
    """Core of imhist with no checks or handling of mask."""
    from .util import get_im_min_max, is_on_gpu
    if is_on_gpu(im):
        from cupy import linspace, histogram # pylint: disable=import-error
    else:
        from numpy import linspace, histogram
    mn, mx = get_im_min_max(im)
    return histogram(im, linspace(mn, mx, nbins+1))[0]
