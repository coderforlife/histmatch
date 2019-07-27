"""
Implements wavelet approach strict ordering for use with exact histogram equalization.
"""

from itertools import product

def calc_info(im, nlevels=2, wavelet='haar', return_fails=False):
    """
    Perform exact histogram equalization using the wavelet approach as defined by [1]. Unlike the
    other exact histogram equalization methods this returns already sorted indices as a 1D array
    meaning the sorting step in histeq_exact is skipped. If return_fails=True then this also returns
    the number of sorting failures (since that information is no longer available with pre-sorted
    indices.

    This calculates the extra information using the traditional (decimating) wavelet transform and
    then uses complex sorting rules to determine the sorted order of the pixels. Even though this
    method is referenced in many papers and used as a baseline to demonstrate improvements, it is
    unlikely to have been implemented correctly in any of them due to the complex sorting required.
    In some papers this is obvious by the way they reference the data (for example saying that k=9
    and a standard strict ordering is performed).

    Due to the complexity of this method and the various confusing parts of the original paper, this
    is a best effort to do it correctly but may still be incorrect.

    This implementation supports 3D data, however only isotropic data is supported. This requires
    pyWavelets along with Cython to use the C++ std:sort for efficiency.

    REFERENCES
      1. Wan Y and Shi D, 2007, "Joint exact histogram specification and image enhancement through
         the wavelet transform", IEEE Transcations on Image Processing, 16(9):2245-2250.
    """
    # this uses pywt's 'symmetric' mode (duplicated edge)
    from pywt import Wavelet
    from ..util import get_dtype_max, as_unsigned
    from .__wa import argsort # pylint: disable=no-name-in-module

    # Normalize image data to 0 to 1
    im = as_unsigned(im)
    im = im / float(get_dtype_max(im.dtype))

    # Compute the values needed to sort
    thetas, tw0s = __compute_data(im, Wavelet(wavelet), nlevels)

    # Perform the sorting
    return argsort(im, thetas, tw0s, return_fails)

def __compute_data(im, wavelet, nlevels=2):
    """
    Compute all of the subband coefficient data for the image using the given wavelet. This returns
    two lists one with the sorted |theta_uij| values and the other with the theta_uij*w_uij_u > 0
    values in the same order. The lists are indexed with j (the level) which gives an array of the
    values at that level.

    Each level still contains the approximate coefficients and thus this doesn't use pywt.wavedec2.
    """
    from numpy import abs # pylint: disable=redefined-builtin
    filters, scales = __compute_filters(im.ndim, wavelet.filter_bank)

    thetas = [] # |theta_uij| accessed like thetas[j][u*,i]
    tw0s = []   # theta_uij*w_uij_u > 0 accessed like tw0s[j][u,i]
    for _ in range(nlevels):
        # Compute the coefficients
        theta = __compute_coeffs(im, wavelet)
        theta /= scales

        # Compute the theta*w > 0 conditions
        tw0 = __compute_positives(im.shape, theta, filters)

        # The next subband level will use the current approximate result
        im = theta[..., -1]

        # Get the magnitude of the coefficients (except the approximate values)
        abs(theta[..., :-1], theta[..., :-1])

        # Sort the data
        theta, tw0 = __sort_level(theta, tw0)

        # Save
        thetas.append(theta)
        tw0s.append(tw0)

    return thetas, tw0s

def __compute_filters(ndim, filter_bank):
    """
    Compute the n-dimensional filters from the wavelet filter bank. The filter bank must start with
    the 1D approximate and detail filters. The stack uses the first index for which filter, from
    fully detail down to fully approximate. Also returns the scaling factor for each filter in an
    array in the same order.
    """
    from functools import reduce
    from numpy import stack, multiply, array
    f_approx = filter_bank[0]
    f_detail = filter_bank[1]
    filters = stack([reduce(multiply.outer, f) for f in product((f_detail, f_approx), repeat=ndim)])
    scales = array([max(-fltr.clip(None, 0).sum(), fltr.clip(0, None).sum()) for fltr in filters])
    return filters, scales

def __compute_coeffs(im, wavelet):
    """
    Compute the wavelet coefficients for an image (called theta_uij in the paper). Returns a stack
    of values where the last index is which set of coefficients starting at fully detail down to
    fully approximate.
    """
    from numpy import stack
    from pywt import dwtn
    theta = dwtn(im, wavelet)
    return stack([theta[''.join(axes)] for axes in product('da', repeat=im.ndim)], axis=-1)

def __compute_positives(shape, theta, filters):
    """
    Compute the positive indicators (called theta_uij*w_uij_u > 0 in the paper). Returns a stack of
    values where the last index is which set of coefficients starting at fully detail down to fully
    approximate.
    """
    from numpy import empty, greater, uint8
    tw0 = empty(shape + (len(filters),), bool)
    for offset in product((0, 1), repeat=len(shape)):
        f_slcs = (slice(None),) + offset
        tw_slcs = tuple(slice(x, None, 2) for x in offset)
        greater(theta * filters[f_slcs], 0, tw0[tw_slcs])
    return tw0.view(uint8)

def __sort_level(theta, tw0):
    """
    Sort all of the data on a subband level. This sorts theta and then sorts tw0 in the same order
    (but tw0 is twice as large in each dimension). Return the sorted theta and tw0.

    NOTE: returned tw0 is actually the given tw0 (modified in-place), theta is a copy however.
    """
    from numpy import take_along_axis
    order = theta.argsort(-1)[..., ::-1] # descending order
    theta = take_along_axis(theta, order, -1)
    for slc in product((slice(0, None, 2), slice(1, None, 2)), repeat=tw0.ndim-1):
        tw0[slc] = take_along_axis(tw0[slc], order, -1)
    return theta, tw0
