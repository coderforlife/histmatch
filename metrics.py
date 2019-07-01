"""
Various image metrics from a variety of papers.
"""

def contrast_per_pixel(im):
    """
    Contrast-per-pixel measure, normalized.

    Calculated as the average difference in greylevel between adjacent pixels. Slightly modified
    from [1] to support 3D images and so that the result is normalized based on the size of the
    neighborhood (8 for 2D images and 26 for 3D images). Additionally, differences are not taken
    with values outside the image ([1] said the outside of the image was assumed to be 0).

    REFERENCES
      1. Eramian M and Mould D, 2005, "Histogram Equalization using Neighborhood Metrics",
         Proceedings of the Second Canadian Conference on Computer and Robot Vision.
    """
    # TODO: since I am not using pixels outside the image this will cause problems with the
    # division at the end assuming every pixel has the same number of neighbors
    from numpy import empty, subtract, abs # pylint: disable=redefined-builtin
    from .util import get_diff_slices
    im = im.astype(float, copy=False)
    tmp = empty(im.shape)
    total = 0
    for slc_pos, slc_neg in get_diff_slices(im.ndim):
        tmp_x = tmp[slc_neg]
        abs(subtract(im[slc_pos], im[slc_neg], tmp_x), tmp_x)
        total += 2*tmp_x.sum()
    return total / (im.size * (3**im.ndim-1))

def distortion(im1, im2):
    """
    Measure of distortion between two images. Also called dissimilarity between the images.

    This is calculated as the variance of the ratios of pixel grey levels pairwise in the two
    images. Can be thought of as the variance of local change in contrast.

    Note that [1] calls this standard deviation, but their formula describes variance.

    REFERENCES
      1. Eramian M and Mould D, 2005, "Histogram Equalization using Neighborhood Metrics",
         Proceedings of the Second Canadian Conference on Computer and Robot Vision.
    """
    return (im1 / im2).var()
