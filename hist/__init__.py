"""
Histogram calculation and equalization/matching techniques.
"""

from .util import imhist
from .classical import histeq, histeq_trans, histeq_apply
from .exact import histeq_exact
from .metrics import (contrast_per_pixel, distortion, count_differences, psnr, ssim,
                      enhancement_measurement)
