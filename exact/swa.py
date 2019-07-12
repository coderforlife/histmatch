"""
Implements stationary/redundant wavelet approach with and without bilateral filtering to strict
ordering for use with exact histogram equalization. When used with bilateral filtering this is the
algorithm by R and Wilscy. Without bilateral filtering this is likely the algorithm used by
Nikolova et al when trying to implement the algorithm by Wan and Shi.
"""

def calc_info(im, bilateral_filter=False, detail_magnitude=True, nlevels=2, kernel='haar'):
    """
    Assign strict ordering to image pixels. The returned value is the same shape but with an extra
    dimension for the results of the additional filters. This stack needs to be lex-sorted.

    This calculates the extra information using the stationary wavelet transform. The first piece of
    data is the image gray values. And then, for each wavelet level, the approximate coefficients
    followed by the various detail coefficients for that level. Increasing levels use the algorithme
    à trous to generate the new kernels/filters. By default this creates 2 levels which is 8
    additional versions of the image for 2D and 16 for 3D.

    This is a derivative of the method described in [1] however that method used the standard,
    non-stationary wavelet transform and had an additional sorting step. For that method, see wa.
    All subsequent papers likely used method much more similar to this one, and this is obvious in
    some papers such as [2,3] and other papers are not clear [4,5].

    Besides adjusting the number of levels, the wavelet kernel can be set as well. It defaults to
    'haar' as used in [1,2] and likely in all other papers even if not explicit. As stated in [1,3]
    the default number of levels is 2. Other papers are not explicit about this but likely used 1 or
    2 levels. Any kernel supported by pywt is allowed.

    The detail coefficients can optionally be used as magnitudes so that a strong edge in any
    direction sorts higher. The default is to not do this as no paper makes this explicit although
    it makes sense and may reproduce the extra sorting information in [1] partially.

    To reproduce [2] this can also apply a bilateral filtering to the image before calculating the
    coefficients. The parameter bilateral_filter can be set to True to use the default options for
    bilaterial_filter or it can be a 3-element tuple to set the size, sigma_r, and sigma_d
    parameters of bilateral_filter.

    This implementation supports 3D data, however only isotropic data is supported.

    REFERENCES
      1. Wan Y and Shi D, 2007, "Joint exact histogram specification and image enhancement through
         the wavelet transform", IEEE Transcations on Image Processing, 16(9):2245-2250.
      2. R A and Wilscy M, 2008, "Enhancing Contrast in Color Images Using Bilateral Filter and
         Histogram Equalization Using Wavelet Coefficients", Second International Conference on
         Future Generation Communication and Networking Symposia.
      3. Nikolova M and Steidl G, 2014, "Fast ordering algorithm for exact histogram specification",
         IEEE Transcations on Image Processing, 23(12):5274-5283.
      4. Avanaki A, 2009, "Exact Global Histogram Specification Optimized for Structural
         Similarity", Optical Review, 16(6):613:621.
      5. Jung S-W, 2014, "Exact Histogram Specification Considering the Just Noticeable Difference",
         IEIE Transactions on Smart Processing and Computing, 3(2):52-58.
    """
    # returns stack always
    # this uses pywt's 'symmetric' mode (duplicated edge)

    from itertools import product
    from numpy import divide, empty, copyto, abs #pylint: disable=redefined-builtin
    from pywt import swt2, swtn
    from ..util import get_im_min_max

    if bilateral_filter:
        args = (None, None, 1) if bilateral_filter is True else bilateral_filter
        from .__bilateral import bilateral_filter
        im = bilateral_filter(im.astype(float, copy=False), *args)

    # Change the copy method if we are using detail magnitudes
    copy = lambda x, y: copyto(y, x)
    detail_copy = abs if detail_magnitude else copy

    coeffs_per_level = 2**im.ndim
    out = empty(im.shape + (1+nlevels*coeffs_per_level,))

    # Normalize image data to -0.5 to 0.5
    # This means all coefficients have the same range (-1 to 1 for 2D using haar)
    im = divide(im, float(max(get_im_min_max(im))), out[..., -1])
    im -= 0.5

    # Calculate coefficients
    if im.ndim == 2:
        # 2D is easy
        coeffs = swt2(im, kernel, nlevels) # level 0 is at postion -1, nlevels-1 is at position 0
        for i, (c_approx, (c_horiz, c_vert, c_diag)) in enumerate(coeffs):
            detail_copy(c_diag, out[..., i*4 + 0])
            detail_copy(c_vert, out[..., i*4 + 1])
            detail_copy(c_horiz, out[..., i*4 + 2])
            copy(c_approx, out[..., i*4 + 3])
    else:
        # 3D is more complicated to get all the data into the right places
        coeffs = swtn(im, kernel, nlevels)
        # coeffs is a list of dictionaries with the keys like ddd, dda, ..., aaa
        # where the a stands for approximate and d stands for detail
        coeff_names = [''.join(x) for x in product('da', repeat=im.ndim)]
        coeff_approx = coeff_names.pop()
        for i, data in enumerate(coeffs):
            for j, name in enumerate(coeff_names[:-1]):
                detail_copy(data[name], out[..., i*coeffs_per_level + j])
            copy(data[coeff_approx], out[..., i*coeffs_per_level + len(coeff_names)])

    return out

# TODO: compact using something like the following:
#ndi.correlate(im.astype(int), [[1,1],[1,1]], origin=(-1,-1)) == np.around(c_approx*2).astype(int)
#ndi.correlate(im.astype(int), [[1,1],[-1,-1]], origin=(-1,-1)) == np.around(c_horiz*2).astype(int)
#      still has negative values
