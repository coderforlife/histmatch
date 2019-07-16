"""
Various image metrics from a variety of papers.
"""

from functools import lru_cache
from collections.abc import Sequence

from numpy import asarray, arange, empty, exp, log, log10, subtract, stack
from scipy.ndimage import gaussian_filter

from .util import (get_dtype_min_max, check_image_single_channel, check_image_mask_single_channel,
                   axes_combinations)

def contrast_per_pixel(im):
    """
    Contrast-per-pixel measure, normalized.

    Calculated as the average difference in greylevel between adjacent pixels. Slightly modified
    from [1] to support 3D images and so that the result is normalized based on the size of the
    neighborhood (8 for 2D images and 26 for 3D images). Additionally, differences are not taken
    with values outside the image ([1] said the outside of the image was assumed to be 0).

    Higher indicates more contrast and thus is better.

    REFERENCES
     1. Eramian M and Mould D, 2005, "Histogram Equalization using Neighborhood Metrics",
        Proceedings of the Second Canadian Conference on Computer and Robot Vision.
    """
    from numpy import abs # pylint: disable=redefined-builtin
    from .util import get_diff_slices
    im = im.astype(float, copy=False)
    tmp = empty(im.shape)
    total = 0
    for slc_pos, slc_neg in get_diff_slices(im.ndim):
        tmp_x = tmp[slc_neg]
        abs(subtract(im[slc_pos], im[slc_neg], tmp_x), tmp_x)
        total += 2*tmp_x.sum()

    # Compute the scale
    # In the original paper this was im.size
    # This value is essentially im.size * (3**nim.dim-1) to account for number of neighbors
    scale = __get_total_neighbors(im.shape)
    return total / scale

def __get_total_neighbors(shape):
    """
    Gets the total number of neighbors over all pixels in an image assuming full connectivity. Doing
    num_px * (3**ndim-1) would get close but doesn't account for pixels along the corners/edges that
    have less neighbors. For small images this can make a big difference, for larger images not so
    much.

    For example, for a 16x16 image the true value is 9.2% lower than the estimated value but for a
    256x256 image this value is only 0.6% smaller.

    This assumes every dimension is at least 3 long.
    """
    from .util import prod

    ndim = len(shape)

    # Count the bulk of the pixels in the core
    core_n_pixels = prod(x-2 for x in shape)
    core_n_neighbors = 3**ndim-1
    count = core_n_pixels * core_n_neighbors

    # Go through pixels that are along planes/edges/corners
    # The number of neighbors is missing n_axes+1 axes
    n_axes = arange(ndim)
    n_neighbors = core_n_neighbors - ((1<<n_axes) * 3**(ndim-n_axes-1)).cumsum()
    for inds in axes_combinations(ndim):
        n_pixels = core_n_pixels // prod(shape[i]-2 for i in inds)
        count += (1<<len(inds)) * n_pixels * n_neighbors[len(inds)-1]

    return count

def distortion(im1, im2, mask=None):
    """
    Measure of distortion between two images. Also called dissimilarity between the images.

    This is calculated as the variance of the ratios of pixel grey levels pairwise in the two
    images. Can be thought of as the variance of local change in contrast.

    Note that [1] calls this standard deviation, but their formula describes variance.

    Higher indicates more distortion and thus is worse.

    REFERENCES
     1. Eramian M and Mould D, 2005, "Histogram Equalization using Neighborhood Metrics",
        Proceedings of the Second Canadian Conference on Computer and Robot Vision.
    """
    im1, mask = check_image_mask_single_channel(im1, mask)
    im2 = check_image_single_channel(im2)
    if im1.shape != im2.shape: raise ValueError('im1 and im2 must be the same shape')
    if mask is not None: im1, im2 = im1[mask], im2[mask]
    # Need to avoid divide-by-zero
    mask = im2 != 0
    im1, im2 = im1[mask], im2[mask]
    return (im1 / im2).var()

def count_differences(im1, im2, mask=None):
    """
    Returns the number of differences between two images. This can be used to calculate the percent
    error between two images if the result is divided by im1.size or im2.size.

    Higher indicates more differences and thus is worse.

    Called the errorneous pixel percentage in [1] and the reconstruction error rate, nu, in [x].

    REFERENCES
     1. Coltuc D and Bolon P, 1998, "An inverse problem: Histogram equalization", 9th European
        Signal Processing Conference (EUSIPCO), 2:861–864.
    """
    im1, mask = check_image_mask_single_channel(im1, mask)
    im2 = check_image_single_channel(im2)
    if im1.shape != im2.shape: raise ValueError('im1 and im2 must be the same shape')
    if mask is not None: im1, im2 = im1[mask], im2[mask]
    return (im1 != im2).sum()

def psnr(im1, im2, mask=None):
    """
    Calculates the peak signal-to-noise ratio between two images. The returned value is in dB.

    Higher indicates a stronger signal and thus is better (in fact a perfect match will result in
    infinity).

    Computed as:
        PSNR = 10*log10((MAX-MIN)^2/MSE)
    where:
        MAX is the maximum value possible for an image type
        MIN is the minimum value possible for an image type
        MSE is the mean squared error, defined as:
            MSE = 1/n * sum((F-G)^2)
        F and G are the images
        n is the number of pixels in each image

    REFERENCES
     1. Coltuc D and Bolon P, 1998, "An inverse problem: Histogram equalization", 9th European
        Signal Processing Conference (EUSIPCO), 2:861–864.
     2. Nikolova M and Steidl G, 2014, "Fast Ordering Algorithm for Exact Histogram Specification"
        IEEE Trans. on Image Processing, 23(12):5274-5283
    """
    im1, mask = check_image_mask_single_channel(im1, mask)
    im2 = check_image_single_channel(im2)
    if im1.shape != im2.shape: raise ValueError('im1 and im2 must be the same shape')
    if im1.dtype != im2.dtype: raise ValueError('im1 and im2 must be the same dtype')
    mn, mx = get_dtype_min_max(im1.dtype)
    if mask is not None: im1, im2 = im1[mask], im2[mask]
    diff = im1.astype(float, copy=False) - im2
    diff *= diff
    mse = diff.sum()/im1.size
    scale = mx - mn
    return 10 * log10(scale*scale/mse)

def enhancement_measurement(im, block_size=3, metric='sdme', alpha=0.75):
    """
    Compute a Measure of Enhancement by Panetta et al [4]. There are several related measures that
    all take the following form:

        1/n * sum(20 * ln(metric))

    for standard measures and the following for entropic measures:

        1/n * sum(alpha*metric^alpha * ln(alpha*metric^alpha))    ?

    where the metric is computed over n non-overlapping blocks in the image. The sum operates over
    all blocks. The metrics are defined in terms of the maximum and minimum values in the blocks:
    B_max and B_min respectively. Any block for which the metric is 0 or undefined (divide by 0)
    will not be counted.

    The following metrics are defined:

        EME - Measure of Enhancement - Weber-law-based [1,2]
        metric = B_max / B_min

        AME - Advanced Measure of Enhancement - Michelson-law-based [2]
        metric = (B_max+B_min) / (B_max-B_min)

        SDME - Secord-Derivative-Like Measure of Enhancement [4]
        metric = |(B_max+B_min+2*B_center)/(B_max+B_min-2*B_center)|

    where B_center is the intensity of the pixel in the center of the block for SDME. Any of these
    metrics can be given as the metric parameter. Appending an "E" to the name will use the entropic
    measure. The size of the blocks is given by block_size (which can be a scalar or sequence) and
    defaults to 3x3. The entropic measures require an alpha parameter which defaults to 0.75. The
    value of alpha can also be a sequence in which case a sequence of values is returned for the
    alphas. This is significantly faster than calling this function with separate values of alpha.

    All of these have higher values to indicate "enhancement" and this higher is better.

    Currently the logAME/logAMEE (Logarithmic AME / Integrated PLIP operators from [3]) are not
    included.

    REFERENCES
     1. Agaian S S, Panetta K, and Grigoryan A M, Mar 2001, "Transform-based image enhancement
        algorithms with performance measure", IEEE Transactions on Image Processing, 10(3):367–382.
     2. Agaian S S, Silver B, and Panetta K A, Mar 2007, "Transform coefficient histogram-based
        image enhancement algorithms using contrast entropy", IEEE Transactions on Image Processing,
        16(3):741–758.
     3. Panetta K A, Wharton E J, and Agaian S S, Feb 2008, "Human visual system-based image
        enhancement and logarithmic contrast measure", IEEE Transactions on Systems, Man,
        Cybernetics, 38(1):174–188.
     4. Panetta K, Zhou Y, Agaian S, Jia H, Nov 2011, "Nonlinear unsharp masking for mammogram
        enhancement", IEEE Transactions on IT in Biomedicine, 15(6):918-928.
    """
    # TODO: test, get response, and also just implement using convolutions like in measures.m?
    # Also, EME has a divide-by-0 issue? their code picks the second-minimum
    from numpy import abs, isfinite # pylint: disable=redefined-builtin
    im = check_image_single_channel(im)
    metric = metric.lower()
    if metric not in ('eme', 'emee', 'ame', 'amee', 'logame', 'logamee', 'sdme', 'sdmee'):
        raise ValueError('metric')
    alpha = asarray(alpha if isinstance(alpha, Sequence) else [alpha])
    if alpha.ndim != 1 or alpha.size == 0 or ((alpha <= 0) | (alpha > 1)).any():
        raise ValueError('alpha')

    if metric[:4] == 'sdme':
        maxes, mins, centers = __calc_block_min_max(im, block_size, True)
    else:
        maxes, mins = __calc_block_min_max(im, block_size)
    maxes = maxes.astype(float, copy=False)

    # Calculate the numerator and denominator for each metric
    entropic = metric[-2:] == 'ee'
    if entropic: metric = metric[:-1]
    if metric == 'eme':
        numer, denom = maxes, mins
    elif metric == 'ame':
        numer, denom = maxes+mins, maxes-mins
    elif metric == 'logame':
        raise NotImplementedError('logame')
    elif metric == 'sdme':
        maxes += mins
        numer, denom = maxes+2*centers, maxes-2*centers

    # Adjust metric values for undefined, infinities, 0 values, and negatives and divide
    #denom += 0.0001
    mask = numer != 0
    mask &= denom != 0
    mask &= isfinite(numer)
    mask &= isfinite(denom)
    metric = numer[mask] / denom[mask]
    abs(metric, metric)

    # Measure of Enhancement (standard)
    # In theory we could do log(metric.prod()) which would be faster but that overflows
    if not entropic: return 20/metric.size * log10(metric, metric).sum()

    # Measure of Enhancement by Entropy (with many alphas)
    scale = alpha/metric.size
    alpha, metric = alpha[None, :], metric[:, None]
    return scale * (metric**alpha * log10(metric)).sum()

def __calc_block_min_max(im, block_size, compute_centers=False): # pylint: disable=too-many-locals
    """
    Calculate the minimum and maximum (and possibly the center values) of all blocks in an image.
    This includes all of the blocks that don't quite fit along the edges.
    """
    from numpy import amin, amax, copyto, fromiter
    from .util import block_view, reduce_blocks, tuple_set

    block_size = tuple(block_size) if isinstance(block_size, Sequence) else ((block_size,)*im.ndim)
    if len(block_size) != im.ndim or any(bs < 1 for bs in block_size):
        raise ValueError('block_size')

    # Shape of the data after the blocking
    # up means round up, down means round down
    shape_up = tuple((x+sz-1)//sz for x, sz in zip(im.shape, block_size))
    #shape_down = tuple(x//sz for x, sz in zip(im.shape, block_size))
    slice_down = tuple(slice(x//sz) for x, sz in zip(im.shape, block_size))
    slice_all = (slice(None),)*im.ndim

    # Get the blocks that cover the majority of the image
    blocks = block_view(im, block_size)

    # Get the bulk of the values
    maxes = empty(shape_up, im.dtype)
    reduce_blocks(blocks, amax, maxes[slice_down])
    mins = empty(shape_up, im.dtype)
    reduce_blocks(blocks, amin, mins[slice_down])
    if compute_centers:
        centers = empty(shape_up, im.dtype)
        copyto(centers[slice_down], blocks[slice_all[:im.ndim]+tuple(sz//2 for sz in block_size)])

    # Compute the values for remainders
    # We need to do this for every combination of axes
    remainder = fromiter((x%sz for x, sz in zip(im.shape, block_size)), int)
    for inds in axes_combinations(im.ndim):
        inds = list(inds)
        if (remainder[inds] == 0).any(): continue # no remainder for this combination

        # Compute the slices and sizes of the remainders for this combination of axes
        im_slc_ = tuple_set(slice_all, [slice(im.shape[i]-remainder[i], None) for i in inds], inds)
        block_size_ = tuple_set(block_size, remainder[inds], inds)
        slice_ = tuple_set(slice_down, (slice(-1, None),)*len(inds), inds)

        # Get the remainder blocks and then compute the values
        blocks = block_view(im[im_slc_], block_size_)
        reduce_blocks(blocks, amax, maxes[slice_])
        reduce_blocks(blocks, amin, mins[slice_])
        if compute_centers:
            copyto(centers[slice_], blocks[slice_all[:im.ndim]+tuple(sz//2 for sz in block_size_)])

    # Done
    return (maxes, mins, centers) if compute_centers else (maxes, mins)

def ssim(im1, im2, block_size=None, sigma=1.5, k1=0.01, k2=0.03, remove_edges=False, mask=None): # pylint: disable=too-many-arguments, invalid-name
    """
    Calculates the mean SSIM image as the average of all:
        SSIM(x,y) = (2*mu_x*mu_y+C1)*(2*sig_xy+C2)/((mu_x^2+mu_y^2+C1)*(sig_x^2+sig_y^2+C2))
    where:
        x and y are blocks in the im1 and im2 images
        C1 = (k1*L)^2; C2 = (k2*L)^2; values close to 0 to prevent division-by-zero
        L = range of image data type
        mu_x = G (*) x
        sig_x^2 = G (*) x^2 - mu_x^2
        sig_xy = G (*) xy - mu_x*mu_y
        (*) is a convolution
        G is a Gaussian kernel

    This is equation 13 in [1] and 1 in [2]. This always uses a Gaussian filter although it is
    reasonable to use other filters like a uniform filter. Also always uses population covariance
    instead of sample covariance.

    Perfect results will return 1 and terrible results will be close to 0 thus higher is better.

    By default the results will match the specifications in [1] and [2]. Changing sigma will
    automatically change block_size unless a block size is specified to override it. If
    remove_edges=True is given then the edges will be removed after calculating the SSIM image to
    reduce edge effects.

    REFERENCES
     1. Wang Z, Bovik A C, Sheikh H R, and Simoncelli E P, 2004, "Image quality assessment: From
        error visibility to structural similarity", IEEE Trans. on Image Processing, 13:600-612.
     2. Avanaki, A N, 2009, "Exact global histogram specification optimized for structural
        similarity", Optical Review, 16:613-621.
    """
    # Check arguments
    im1, mask = check_image_mask_single_channel(im1, mask)
    im2 = check_image_single_channel(im2)
    if im1.shape != im2.shape: raise ValueError('im1 and im2 must be the same shape')
    if im1.dtype != im2.dtype: raise ValueError('im1 and im2 must be the same dtype')
    if k1 < 0: raise ValueError('k1')
    if k2 < 0: raise ValueError('k2')
    if sigma < 0: raise ValueError('sigma')
    if block_size is None: block_size = 2*int(3.5*sigma+0.5)+1
    if block_size < 0 or block_size%2 != 1: raise ValueError('block_size')

    # Compute SSIM Image
    ssim_im = __ssim_im(im1, im2, block_size, k1, k2, sigma)

    # Compute mean of SSIM
    if remove_edges:
        # Avoid edge effects by ignoring filter radius at edges
        radius = (block_size - 1) // 2
        ssim_im = ssim_im[(slice(radius, -radius),)*im1.ndim]
        if mask is not None: mask = mask[(slice(radius, -radius),)*im1.ndim]
    if mask is not None: ssim_im = ssim_im[mask]
    return ssim_im.mean()

def __ssim_im(im1, im2, block_size, k1, k2, sigma): # pylint: disable=too-many-arguments, too-many-locals, invalid-name
    """
    Calculates the SSIM image from the formula:
        SSIM(x,y) = (2*mu_x*mu_y+C1)*(2*sig_xy+C2)/((mu_x^2+mu_y^2+C1)*(sig_x^2+sig_y^2+C2))
    where:
        x and y are blocks in the im1 and im2 images
        C1 = (k1*L)^2; C2 = (k2*L)^2; values close to 0 to prevent division-by-zero
        L = range of image data type
        mu_x = G (*) x
        sig_x^2 = G (*) x^2 - mu_x^2
        sig_xy = G (*) xy - mu_x*mu_y
        (*) is a convolution
        G is a Gaussian kernel

    This is equation 13 in [1] and 1 in [2]. This always uses a Gaussian filter although it is
    reasonable to use other filters like a uniform filter. Also always uses population covariance
    instead of sample covariance.

    REFERENCES
     1. Wang Z, Bovik A C, Sheikh H R, and Simoncelli E P, 2004, "Image quality assessment: From
        error visibility to structural similarity", IEEE Trans. on Image Processing, 13:600-612.
     2. Avanaki, A N, 2009, "Exact global histogram specification optimized for structural
        similarity", Optical Review, 16:613-621.
    """
    # pylint: disable=invalid-name
    from numpy import multiply, add, divide

    # Calculate constants
    mn, mx = get_dtype_min_max(im1.dtype)
    L = mx - mn
    C1 = k1*k1*L*L
    C2 = k2*k2*L*L
    truncate = (block_size-1)//2 / sigma

    # Convert to doubles since gaussian_filter will do that anyways
    x = im1.astype(float)
    y = im2.astype(float)

    # Compute means
    mu_x = gaussian_filter(x, sigma, truncate=truncate)
    mu_y = gaussian_filter(y, sigma, truncate=truncate)
    mu_xy = mu_x*mu_y
    mu_xx = multiply(mu_x, mu_x, mu_x)
    mu_yy = multiply(mu_y, mu_y, mu_y)

    # Compute variances and covariances
    sig_xy = gaussian_filter(x*y, sigma, truncate=truncate)
    sig_xy -= mu_xy
    sig_xx = gaussian_filter(multiply(x, x, x), sigma, truncate=truncate)
    sig_xx -= mu_xx
    sig_yy = gaussian_filter(multiply(y, y, y), sigma, truncate=truncate)
    sig_yy -= mu_yy

    # Compute SSIM
    A1 = add(multiply(mu_xy, 2, mu_xy), C1, mu_xy)    # A1 = 2*mu_xy + C1
    A2 = add(multiply(sig_xy, 2, sig_xy), C2, sig_xy) # A2 = 2*sig_xy + C2
    numer = multiply(A1, A2, A1) # numerator
    B1 = add(add(mu_xx, mu_yy, mu_xx), C1, mu_xx)     # B1 = mu_xx + mu_yy + C1
    B2 = add(add(sig_xx, sig_yy, sig_xx), C2, sig_xx) # B2 = sig_xx + sig_yy + C2
    denom = multiply(B1, B2, B1) # denominator
    return divide(numer, denom, numer)

def contrast_enhancement(im1, im2, mask=None, p=5, freqs=None, M=3): # pylint: disable=too-many-arguments
    """
    Computes the contrast enhancement between the original image (im1) and the enhanced image (im2).
    This measure of contrast enhancement is based on the center-surround effect in the retina. As
    described in [1] in eq 16-25, 31-32 for luminance (grayscale) images, this requires applying a
    large series of band-pass filters to the images to detect different frequencies. The frequencies
    (𝜈 in eq 23) default to 24 recommended values 72π/80, 69π/80, ..., 6π/80, 3π/80 in [1]. This
    means that 24*4=96 convolutions are required making the computation of this metric extremely
    slow. The M parameter defaults to the recommended value of 3 from [1] and controls width of the
    band passes.

    The p parameter is the most 'critical' parameter and controls the sensitivity to spatial
    frequencies. Low values, like 1 and 2, cause the results to be more sensitive spatial
    frequencies within the bands while high values like 15 and infinity (np.inf) cause all spatial
    frequencies to be considered more equally. The default is set to 5 which is in-between the
    extremes since there is no recommended value in the paper. The paper recommends that you try
    several values for p. To save re-computing the convolutions for each one, this function supports
    passing p as a sequence and all will be computed and returned.

    NOTE: This does not currently take into account the requirement that χ_he ≈ χ_ho from eq 33-35.

    When im1 and im2 are the same, 1.0 is returned. Larger numbers indicate an increase in contrast
    while smaller numbers indicate a loss of contrast from im1 to im2.

    REFERENCES:
     1. Sen D and Pal S K, May 2011, "Automatic Exact Histogram Specification for Contrast
        Enhancement and Visual System Based Quantitative Evaluation", IEEE Transactions on Image
        Processing, 20(5):1211-1220.
    """
    # pylint: disable=invalid-name
    from numpy import pi

    im1, mask = check_image_mask_single_channel(im1, mask)
    im2 = check_image_single_channel(im2)
    if im1.shape != im2.shape: raise ValueError('im1 and im2 must be the same shape')
    if im1.dtype != im2.dtype: raise ValueError('im1 and im2 must be the same dtype')
    p = asarray(p if isinstance(p, Sequence) else [p])
    if (p < 0).any(): raise ValueError('p')
    freqs = (arange(72, 0, -3) * (pi / 80)) if freqs is None else asarray(freqs)
    if ((freqs < 0) | (freqs > pi)).any(): raise ValueError('freqs')
    if M <= 1: raise ValueError('M')

    chi = __compute_chi(im1, im2, p, freqs, M)
    if mask is not None: chi = chi[mask]
    if len(p) == 1: return chi.mean()
    return chi.reshape(-1, len(p)).mean(0)

    # Calculating the increase in contrast in heterogeneous and homogeneous regions
    # Sigma=0.455
    # if Sigma <= 0 or Sigma >= 1: raise ValueError('Sigma')
    # if mask is not None: contrasts_im1 = contrasts_im1[mask, :]
    # contrasts_im1 /= contrasts_im1.reshape(-1, len(p)).max(0)
    # contrasts_im1 -= 1
    # f = -1/(2*Sigma*Sigma)
    # alpha = exp(contrasts_im1*contrasts_im1*f)
    # chi_heterogeneous = alpha * chi
    # chi_homogeneous = (1 + exp(f) - alpha) * chi

def __compute_chi(im1, im2, p, freqs, M):
    """
    Computes the contrast enhancement metric χ from [1] defined as:

        χ(x,y) = (mean_c_L + c_L_E(x,y)) / (mean_c_L + c_L(x,y))    (eq 31 and 32)

    where c_L is the luminance (i.e. grayscale) contrast for a pixel in the image (im1) and c_L_E
    is it for a pixel in the enhanced image (im2). The mean_c_L is the average of the c_L values
    across the entire image. The c_L values are computed as:

        c_L(x,y) = || 1/P * c_σ_g₁(x,y) ||ₚ         (eq 24)

    where ||...||ₚ is the ℓₚ-norm for a given positive value p and:

        P = exp(-f) - exp(-f*M²)                    (eq 25)
        f = log(1/M²)/(1-M²)
        M>1 is an arbitrary value

    The c_σ_g₁(x,y) values are computed in __compute_contrasts and that is where the frequencies
    (𝜈) come into play.

    REFERENCES:
     1. Sen D and Pal S K, May 2011, "Automatic Exact Histogram Specification for Contrast
        Enhancement and Visual System Based Quantitative Evaluation", IEEE Transactions on Image
        Processing, 20(5):1211-1220.
    """
    #pylint: disable=invalid-name
    from numpy.linalg import norm

    # Precompute kernels
    f = log(1/(M*M)) / (1-M*M)
    sigma_g1s = 2 * f / (freqs*freqs)
    #sigma_g1s = sigma_g1s[sigma_g1s > 1/(6*M)] # remove completely unresolvable filters

    # Compute the contrasts for each frequency
    contrasts_im1 = __compute_contrasts(im1, sigma_g1s, M*sigma_g1s)
    contrasts_im2 = __compute_contrasts(im2, sigma_g1s, M*sigma_g1s)

    # Compute the l-p norm of the contrasts (along with scaling)
    P_inv = 1/(exp(-f)-exp(-f*M*M))
    contrasts_im1 *= P_inv
    contrasts_im2 *= P_inv
    if len(p) == 1:
        contrasts_im1 = norm(contrasts_im1, p, -1, True)
        contrasts_im2 = norm(contrasts_im2, p, -1, True)
    else:
        contrasts_im1 = stack([norm(contrasts_im1, p, -1) for p in p], -1)
        contrasts_im2 = stack([norm(contrasts_im2, p, -1) for p in p], -1)

    # Compute the mean
    c_I_mean = contrasts_im1.reshape(-1, len(p)).mean(0)

    # Compute chi
    return (c_I_mean + contrasts_im2) / (c_I_mean + contrasts_im1)

def __compute_contrasts(im, sigma_g1s, sigma_g2s):
    """
    Compute the contrasts for measuring contrast enhancement. This computes c_σ_g₁ from [1] for each
    pixel across all of the frequencies, which is defined as:

        c_σ_g₁(x,y) = O(x,y) / S(x,y) for each σ_g1 (eq 22)
        O(x,y) = C(x,y) - S(x,y)                    (eq 19)
        C(x,y) = im ⊗ g₁(x,y)                       (eq 20)
        S(x,y) = im ⊗ g₂(x,y)                       (eq 21)
        σ_g₁ = 2 log(1/M²)/(1-M²) /𝜈² for 𝜈∈[0,π]   (eq 23)
            𝜈 = 72π/80, 69π/80, ..., 6π/80, 3π/80
        g(x,y) = exp(-1/(2σ²(x²+y²))) - a Gaussian with σ_g1 given above or σ_g₂=M*σ_g₁

    where ⊗ is a convolution. All of these are fairly easily expanded to any number of dimensions by
    using a higher-dimensional Gaussian kernel (actually a 1D Gaussian kernel is applied along each
    dimension).

    The g₁ and g₂ kernels are given for all σs along with the image.

    REFERENCES:
     1. Sen D and Pal S K, May 2011, "Automatic Exact Histogram Specification for Contrast
        Enhancement and Visual System Based Quantitative Evaluation", IEEE Transactions on Image
        Processing, 20(5):1211-1220.
    """
    from scipy.ndimage import fourier_gaussian

    contrasts = empty(im.shape + (len(sigma_g1s),))

    rfftn, irfftn, empty_aligned = __get_rfft()
    im = rfftn(im.astype(float, copy=False))
    temp = empty_aligned(im.shape, im.dtype)

    for i, (sigma_g1, sigma_g2) in enumerate(zip(sigma_g1s, sigma_g2s)):
        # Using real-space correlations  (also need to remove rfftn() above but keep the astype)
        # The real-space correlations produce lower values
        #from .util import correlate, EPS
        #from scipy.ndimage.filters import _gaussian_kernel1d as gaussian1d # could cache
        #g_1 = gaussian1d(sigma_g1, 0, int(3 * sigma_g1 + 0.5))
        #g_2 = gaussian1d(sigma_g2, 0, int(3 * sigma_g2 + 0.5))
        #center = im if g_1.size == 1 else correlate(im, g_1)
        #surround = correlate(im, g_2, output=temp)
        #surround[surround == 0] = EPS

        center = irfftn(fourier_gaussian(im, sigma_g1, im.shape[-1], output=temp))
        surround = irfftn(fourier_gaussian(im, sigma_g2, im.shape[-1], output=temp))

        contrast = contrasts[..., i]
        subtract(center, surround, contrast)
        contrast /= surround
        #abs(contrast, contrast) # norm will automatically take the absolute value for us
    return contrasts

@lru_cache(maxsize=None)
def __get_rfft():
    """
    Gets the rfftn, irfftn, and empty_aligned functions either from pyfftw or numpy depending on
    what is available. For pyfftw, also sets up lots of options to increase its speed.
    """
    try:
        import pyfftw
        import os
        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(5)
        if pyfftw.config.NUM_THREADS == 1:
            pyfftw.config.NUM_THREADS = \
                len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
        pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

        def rfftn(*args, **kwargs):
            return pyfftw.interfaces.numpy_fft.rfftn(*args, overwrite_input=True, **kwargs)
        def irfftn(*args, **kwargs):
            return pyfftw.interfaces.numpy_fft.irfftn(*args, overwrite_input=True, **kwargs)
        empty_aligned = pyfftw.empty_aligned

    except ImportError:
        from numpy.fft import rfftn, irfftn
        empty_aligned = empty

    return rfftn, irfftn, empty_aligned
