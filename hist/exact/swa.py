"""
Implements stationary/redundant wavelet approach with and without bilateral filtering to strict
ordering for use with exact histogram equalization. When used with bilateral filtering this is the
algorithm by R and Wilscy. Without bilateral filtering this is likely the algorithm used by
Nikolova et al when trying to implement the algorithm by Wan and Shi.
"""

from itertools import product

from numpy import asarray, empty

from ..util import ci_artificial_gpu_support

@ci_artificial_gpu_support
def calc_info(im, detail_magnitude=True, nlevels=2, kernel='haar', allow_compaction=True):
    """
    Assign strict ordering to image pixels. The returned value is the same shape but with an extra
    dimension for the results of the additional filters. This stack needs to be lex-sorted. The
    results are compacted when possible (only when using Haar kernel and image is 8- or 16-bit
    integral type) to increase speed of the future lexsort but they can never be fully compacted to
    avoid the need of lexsort entirely.

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
    2 levels. Any kernel supported by pyWavelet is allowed.

    The detail coefficients can optionally be used as magnitudes so that a strong edge in any
    direction sorts higher. The default is to not do this as no paper makes this explicit although
    it makes sense and may reproduce the extra sorting information in [1] partially.

    To reproduce [2] the image must first have a bilateral filtering applied with a size of 3 and a
    sigma_r and sigma_d of 1.

    This implementation supports 3D data, however only isotropic data is supported.

    Note that this implementation doesn't use pyWavelet's swt2/swtn functions since those functions:
     * Restrict the image shapes to be multiples of 2**nlevels
     * Are slower (about half as fast when compacting, +10% when not compacting)
    This still uses pyWavelet for the determining the filter banks when not using the default Haar
    kernel (which itself is slightly modified to reduce numerical instability).

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
    # this uses scipy's wrap mode (equivalent to pywt's 'periodization' mode)
    from .__compaction import non_compact, compact

    # Look for optimized filter banks
    filter_bank = None
    if isinstance(kernel, str):
        kernel = kernel.lower()
        filter_bank = __FILTER_BANKS.get(kernel, None)

    if not allow_compaction or im.dtype.kind == 'f' or im.dtype.itemsize > 2 or filter_bank is None:
        # Get any filter
        if filter_bank is None:
            from pywt import Wavelet
            filter_bank = Wavelet(kernel).filter_bank[:2] # pylint: disable=unsubscriptable-object
        # Generate non-compacted results
        return non_compact(im, nlevels * (1 << im.ndim), __generate_swts,
                           (filter_bank, nlevels, detail_magnitude))

    # Generate compacted results
    scales = list(__generate_scales(im.ndim, filter_bank, nlevels, detail_magnitude))
    return compact(im, scales, __generate_swts, (filter_bank, nlevels, detail_magnitude))

# Filter banks that have non-integer scaling factors removed so they can be used with compacting.
# Filter banks are given as approximate and detail. Other filter banks will be taken from pywt.
__FILTER_BANKS = {
    'haar': ((1, 1), (1, -1)),
}

def __generate_swts(im, filter_bank, nlevels=2, detail_magnitude=False):
    """
    Generate the SWTs for an image and a given set of filters across the given number of levels.
    They are generated with all-approximate first and the A-D, D-A, D-D (for 2D) and in the order of
    the levels. This yields the filtered image which is an ndarray that cannot be saved as it will
    be reused to generate the next sample. This also means that the results of the generator cannot
    simply be put in a list.
    """
    # pylint: disable=too-many-locals
    from itertools import islice
    from numpy import int64, abs # pylint: disable=redefined-builtin
    from scipy.ndimage import correlate1d

    temps = [im] + \
            [empty(im.shape, float if im.dtype.kind == 'f' else int64) for _ in range(im.ndim)]
    filter_bank = tuple(asarray(f, float) for f in filter_bank)

    for level in range(nlevels):
        last_filters = (None,) # the last set of filters used to generate a sample
        has_detail = False # first iteration below never has a detail filter
        for filters in product(filter_bank, repeat=im.ndim):
            # Generate the filtered image but only for the axes that are different
            start = __get_first_mismatch(filters, last_filters)
            for i, fltr in islice(enumerate(filters), start, None):
                correlate1d(temps[i], fltr, i, origin=-1, mode='wrap', output=temps[i+1])
            last_filters = filters # save which filters we just used
            if has_detail and detail_magnitude:
                abs(temps[-1], temps[-1])
            if not has_detail and level + 1 != nlevels:
                # Complete approximate image is used for the next level
                next_im = temps[-1].copy()
            # Generate the SWT sample
            yield temps[-1]
            has_detail = True # remaining samples in this level has at least one detail filter

        if level + 1 != nlevels:
            # Upsample the filters and set the full-approximate filtered image as the base
            filter_bank = __upsample_filters(filter_bank)
            temps[0] = next_im

def __generate_scales(ndim, filter_bank, nlevels=2, detail_magnitude=True):
    """
    Generates tuples of the scales that occur for a series of SWT samples. The tuples have the
    negative and positive scales caused by the transforms. It is assumed that the filters are
    integral.
    """
    from numpy import multiply
    from .__compaction import scale_from_filter
    filter_bank = tuple(asarray(f, int) for f in filter_bank)
    level_scale = 1 # the scaling caused just be the current level
    for level in range(nlevels):
        has_detail = False # first iteration below never has a detail filter
        for filters in product(filter_bank, repeat=ndim):
            # Generate the "full" filter from the 1D filters
            full = 1
            for fltr in filters:
                full = multiply.outer(full, fltr)
            # Generate the scales for the negative and positive parts of the current filter
            neg, pos = scale_from_filter(full)
            if has_detail and detail_magnitude: neg, pos = 0, max(pos, neg)
            yield (level_scale*neg, level_scale*pos)
            # The all-approximate filter is used for determining the scale of the next level
            if not has_detail: next_level_scale = pos-neg
            has_detail = True # remaining samples in this level has at least one detail filter
        if level + 1 != nlevels:
            filter_bank = __upsample_filters(filter_bank)
            level_scale = next_level_scale

def __upsample_filters(filter_bank):
    """Upsample a filter bank using algorithme à trous and thus inserting zeros."""
    from numpy import zeros
    upsampled_filter_bank = []
    for fltr in filter_bank:
        upsampled = zeros(2*len(fltr)-1, fltr.dtype)
        for i, value in enumerate(fltr):
            upsampled[2*i] = value
        upsampled_filter_bank.append(upsampled)
    return upsampled_filter_bank

def __get_first_mismatch(itr1, itr2):
    """
    Gets the index of the first mismatch (using 'is' not '==') between itr1 and itr2. Returns -1 i
    not found. Extra elements from the longer iterator are ignored.
    """
    return next((i for i, (a, b) in enumerate(zip(itr1, itr2)) if a is not b), -1)
