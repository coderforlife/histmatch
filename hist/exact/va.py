"""
Implements variation approach strict ordering for use with exact histogram equalization.
"""

from numpy import asarray, empty, sqrt, subtract, nonzero

from ..util import ci_artificial_gpu_support, lru_cache_array

SQRT2I = 1/sqrt(2)
SQRT3I = 1/sqrt(3)

# Baus et al 2013 eq 4 and fig 3
CONNECTIVITY_N4 = asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) # __create_gamma_min(2)
CONNECTIVITY_N8 = asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) # __create_gamma_full(2)
CONNECTIVITY_N8_DIST = asarray( # __create_gamma_dist(2)
    [[SQRT2I, 1, SQRT2I], [1, 0, 1], [SQRT2I, 1, SQRT2I]])

# Also support 3D connectivity
CONNECTIVITY3_N6 = asarray([ # __create_gamma_min(3)
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ])
CONNECTIVITY3_N18 = asarray([
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    ])
CONNECTIVITY3_N18_DIST = asarray([
    [[0, SQRT2I, 0], [SQRT2I, 1, SQRT2I], [0, SQRT2I, 0]],
    [[SQRT2I, 1, SQRT2I], [1, 0, 1], [SQRT2I, 1, SQRT2I]],
    [[0, SQRT2I, 0], [SQRT2I, 1, SQRT2I], [0, SQRT2I, 0]]
    ])
CONNECTIVITY3_N26 = asarray([ # __create_gamma_full(3)
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ])
CONNECTIVITY3_N26_DIST = asarray([ # __create_gamma_dist(3)
    [[SQRT3I, SQRT2I, SQRT3I], [SQRT2I, 1, SQRT2I], [SQRT3I, SQRT2I, SQRT3I]],
    [[SQRT2I, 1, SQRT2I], [1, 0, 1], [SQRT2I, 1, SQRT2I]],
    [[SQRT3I, SQRT2I, SQRT3I], [SQRT2I, 1, SQRT2I], [SQRT3I, SQRT2I, SQRT3I]]
    ])
del SQRT2I, SQRT3I

# pylint: disable=invalid-name

@ci_artificial_gpu_support
def calc_info(im, niters=5, beta=None, alpha=None, gamma=None):
    """
    Assign strict ordering to image pixels. The returned value is the same shape as the image but
    with values for each pixel that can be used for strict ordering.

    This implements the method by Nikolova et al which attempts to reconstruct the original real-
    valued version of the image using a very fast minimization method. It takes an argument for the
    number of iterations to perform, defaulting to 5.

    Always uses theta = |t| - alpha * log(1+|t|/alpha). That is Theta_2 in [1 table 1],
    f3 in [2 table 1], and theta_2 in [3 table 1]:

        theta(t)   = |t| - alpha * log(1+|t|/alpha)
        theta'(t)  = t / (alpha + |t|)
        theta''(t) = alpha / (alpha + |t|)^2
        xi(t)      = (theta')^(-1) (t) = alpha * y / (1 - |y|)
        xi'(t)     = alpha / (1 - |y|)^2

    beta, alpha_1, and alpha_2 default to 0.4/eta, 0.1/eta, and 0.1/eta respectively as chosen in
    [3] but generalized to other neighborhoods. See [2] section III, for choosing the parameter
    values. In general keeping alphas small is important. The values are dependent on eta (the
    connectivity) and the image contents in attempt to make c in (1-1e-5, 1). Various other
    functions in this module can be used to help calculate the parameter values based on the image,
    connectivity, and other values.

    If only one alpha is provided, it is used for both alpha_1 and alpha_2. If a sequence is given
    it is used for both of the alphas.

    Their method has been adapted to work in 3D, including anisotropic data by adjusting gamma.

    The value of gamma determines the connectivity. It can be a CONNECTIVITY* constant or None to
    generate a minimal neighborhood (the default), 'full' for a fully connected neighborhood, 'dist'
    for a fully connected neighborhood weighted by distance, or a custom n-D array with each
    dimension equal to 3, a zero in the middle, no negative values, and fully symmetrical.

    REFERENCES:
     1. Baus F, Nikolova M, Steidl G, 2013, "Fully smoothed l1−TV models: bounds for the minimizers
        and parameter choice", Tech. report, v3, http://hal.archives-ouvertes.fr/hal-00722743
     2. Nikolova M, Wen Y-W, and Chan R, 2013, "Exact histogram specification for digital images
        using a variational approach", J of Mathematical Imaging and Vision, 46(3):309-325
     3. Nikolova M and Steidl G, 2014, "Fast Ordering Algorithm for Exact Histogram Specification"
        IEEE Trans. on Image Processing, 23(12):5274-5283
    """
    from numpy import abs, divide #pylint: disable=redefined-builtin
    # this does not use pixels outside of the image at all

    # NOTE: This does not use the *_theta_* functions but instead has it solved out. If those
    # functions are changed this function will also need to be updated.

    gamma, _, beta, alpha_1, alpha_2 = __check_args(gamma, beta, alpha, im.ndim)
    if niters <= 0: raise ValueError('niters') # niters is R in [3]

    # Allocate temporaries
    u, t = im.astype(float), empty(im.shape)
    d_phi, denom = empty(im.shape), empty(im.shape)

    # Slices to use for each difference in the gradient matrix G
    G_info = __get_g_info(gamma)

    # Minimization method from [3]
    for _ in range(niters):
        ## Calculate Phi' = sum(sum_eta(phi'(u[i]-u[j])) = G^T phi'(G u) [3 eq 9] ##
        __calculate_d_Phi(u, t, G_info, alpha_2, d_phi, denom)

        ## Calculate u = f - xi(beta*Phi') [3 eq 9] ##
        t *= beta
        # -xi(t) = alpha_1*t/(|t|-1)
        abs(t, u)
        u -= 1
        divide(alpha_1, u, u)
        u *= t
        # u = f + (-xi); f == im
        u += im

        # This, in general, is slower than just doing more iterations
        #if k >= 1 and is_unique(f.ravel()): break
    return u

# @lru_cache_array # gamma is unhashable...
def __get_g_info(gamma):
    """
    Gets a list of entries from the G gradient matrix for the differences that need to be performed.
    Each element in the list has a gamma value, the positive slice, and the negative slice. The list
    does not include reflected or 0 entries.
    """
    slices = (slice(1, None), slice(None), slice(None, -1)) # 0 -> 1:, 1 -> :, and 2 -> :-1
    return [(gamma[coord], tuple(slices[x] for x in coord), tuple(slices[2-x] for x in coord))
            for coord in zip(*nonzero(gamma)) if not __is_reflection_coord(coord)]

@lru_cache_array # doesn't return arrays, but doesn't hurt to use this one
def __is_reflection_coord(coord):
    """Reflection coordinates have a 2 followed by 0 or more trailing 1s."""
    i = len(coord)-1
    while coord[i] == 1:
        i -= 1
    return coord[i] == 2

def __calculate_d_Phi(u, t, G_info, alpha_2, d_phi, denom): # pylint: disable=too-many-arguments
    """
    Phi' = sum(sum_eta(phi'(u[i]-u[j])) = G^T phi'(G u) [3 eq 9]

    Efficiently evaluates
       phi'(gamma * (u[slc_pos] - u[slc_neg], alpha_2)) +
       phi'(gamma * (u[slc_neg] - u[slc_pos], alpha_2))
    and adds the results onto t for each element in G_info.

    u and t are the input and output images (same shape)
    G_info contains information about the gammas and slices to use for differences, see __get_g_info
    alpha_2 is a constant scalar
    d_phi and denom are temporaries that are the same shape as u and t
    """
    from numpy import abs #pylint: disable=redefined-builtin
    t.fill(0)
    for (gamma, slc_pos, slc_neg) in G_info:
        d_phi_x, denom_x = d_phi[slc_neg], denom[slc_neg]
        subtract(u[slc_pos], u[slc_neg], d_phi_x)
        if gamma != 1.0: d_phi_x *= gamma
        abs(d_phi_x, denom_x)
        denom_x += alpha_2
        d_phi_x /= denom_x
        t[slc_pos] += d_phi_x
        t[slc_neg] -= d_phi_x

# @lru_cache_array # gamma is unhashable...
def __check_gamma(gamma):
    """
    Checks the gamma matrix to make sure that it is a 3x3 or 3x3x3 matrix with a 0 in the middle,
    no negative values, and symmetrical in all directions. Returns gamma, converted to an array if
    necessary. This corresponds to the N^2 neighborhoods in [1] and the N neighborhoods in [2].

    REFERENCES:
     1. Baus F, Nikolova M, Steidl G, 2013, "Fully smoothed l1−TV models: bounds for the minimizers
        and parameter choice", Tech. report, v3, http://hal.archives-ouvertes.fr/hal-00722743
     2. Nikolova M, Wen Y-W, and Chan R, 2013, "Exact histogram specification for digital images
        using a variational approach", J of Mathematical Imaging and Vision, 46(3):309-325
    """
    gamma = asarray(gamma)
    # Check that all dimension lengths are 3, there are no negative values, and the middle is 0
    ndim = gamma.ndim
    if any(x != 3 for x in gamma.shape) or (gamma < 0).any() or gamma[(1,)*ndim] != 0:
        raise ValueError('gamma')
    # Check for symmetry along each axis
    slc_none = slice(None)
    slc_start = slice(2) # first half of the matrix (rounded up)
    slc_end = slice(2, 0, -1) # reversed second half of the matrix (rounded up)
    if any((gamma[tuple(slc_none if i != j else slc_start for j in range(ndim))] !=
            gamma[tuple(slc_none if i != j else slc_end for j in range(ndim))]).any()
           for i in range(ndim)):
        raise ValueError('gamma')
    return gamma

# @lru_cache_array # gamma is unhashable...
def __get_gamma(gamma, ndim):
    """
    Gets and checks the value for gamma (either the one given or a default one of the given
    dimension if None). Also ensures that gamma has the same dimension as ndim.
    """
    if gamma is None: return __create_gamma_min(ndim)
    if isinstance(gamma, str):
        if gamma == 'min': return __create_gamma_min(ndim)
        if gamma == 'full': return __create_gamma_full(ndim)
        if gamma == 'dist': return __create_gamma_dist(ndim)
        raise ValueError('gamma')
    if gamma.ndim != ndim: raise ValueError('gamma and im must have the same dimension')
    return __check_gamma(gamma)

@lru_cache_array
def __create_gamma_min(ndim):
    """Creates gamma that represents connected to a minimal set of neighbors."""
    from numpy import ogrid
    return (sum([x*x for x in ogrid[(slice(-1, 2),)*ndim]]) == 1).astype(float)

@lru_cache_array
def __create_gamma_full(ndim):
    """
    Creates gamma that represents fully connected to all neighbors (1s except the middle value).
    """
    from numpy import ones
    gamma = ones((3,)*ndim)
    gamma[(1,)*ndim] = 0
    return gamma

@lru_cache_array
def __create_gamma_dist(ndim):
    """Creates gamma that represents connected to all neighbors weighted by their distance."""
    from numpy import ogrid
    gamma = sqrt(sum([x*x for x in ogrid[(slice(-1, 2),)*ndim]]))
    gamma[(1,)*ndim] = 1
    gamma = 1/gamma
    gamma[(1,)*ndim] = 0
    return gamma

def __get_beta(beta, eta):
    """
    Get and check the beta argument. The argument can be None (which then uses 0.4/eta) or a single
    value. The value must be positive and less than 1/eta.
    """
    if beta is None: return 0.4/eta
    if beta <= 0 or beta >= 1/eta: raise ValueError('beta')
    return beta

def __check_args(gamma, beta, alpha, ndim):
    """
    Get and check the gamma, beta, and alpha arguments returning gamma, eta, beta, alpha_1, and
    alpha_2.

    The gamma argument is checked with __get_gamma(). eta is computed as sum(gamma). The beta
    argument is checked with __get_beta().

    The alpha argument becomes the alpha_1 and alpha_2 values. The argument can be None (which then
    uses 0.1/eta for both), a single value (which uses the same value for both), or a sequence of 2
    values. Both values must be positive.
    """
    # Check gamma and compute eta
    gamma = __get_gamma(gamma, ndim)
    eta = gamma.sum()

    # Check alpha
    from collections.abc import Sequence
    if alpha is None:
        alpha_1 = alpha_2 = 0.1/eta
    elif isinstance(alpha, Sequence):
        if alpha <= 0: raise ValueError('alpha')
        alpha_1 = alpha_2 = alpha
    else:
        alpha_1, alpha_2 = alpha
    if alpha_1 <= 0 or alpha_2 <= 0: raise ValueError('alpha')

    return gamma, eta, __get_beta(beta, eta), alpha_1, alpha_2


##### Everything else is auxiliary and not needed except to help with choosing the parameters #####

def theta(t, alpha):
    """
    theta(t) = |t| - alpha * log(1+|t|/alpha)
    Also called phi or psi.
    Baus et al 2013, table 1, Theta_2.
    Nikolova et al 2013, table 1, f3.
    Nikolova et al 2014, table 1, theta_2.
    """
    from numpy import log
    assert alpha > 0
    t_abs = abs(t)
    return t_abs - alpha * log(1+t_abs/alpha)

def d_theta(t, alpha):
    """
    theta'(t) = t / (alpha + |t|)
    Also called phi' or psi'.
    Baus et al 2013, table 1, Theta_2.
    Nikolova et al 2013, table 1, f3.
    Nikolova et al 2014, table 1, theta_2.
    """
    assert alpha > 0
    return t / (abs(t) + alpha)

def d2_theta(t, alpha):
    """
    theta''(t) = alpha / (alpha + |t|)^2
    Also called phi'' or psi''.
    Nikolova et al 2013, table 1, f3.
    """
    assert alpha > 0
    denom = alpha + abs(t)
    return alpha / (denom*denom)

def d_theta_inv(y, alpha):
    """
    (theta')^(-1) (y) = alpha * y / (1 - |y|)
    Alternatives:
        In Baus et al 2013      b(beta) = (theta')^(-1) (beta*eta) [eq 12]
        In Nikolova et al 2013  b(y)    = (theta')^(-1) (y)        [eq 12]
        In Nikolova et al 2014  xi(t)   = (theta')^(-1) (t)        [eq 4]
    Baus et al 2013, table 1, Theta_2 with b(beta) given in table 2.
    Nikolova et al 2014, table 1, theta_2.
    """
    assert -1 < y < 1 and alpha > 0
    return alpha * y / (1 - abs(y))

def d_d_theta_inv(y, alpha):
    """
    xi'(y) = 1/theta''(xi(y)) > 0
           = alpha / (1 - |y|)^2
    Nikolova et al 2014, table 1, theta_2 and eq 5.
    """
    assert -1 < y < 1 and alpha > 0
    denom = 1 - abs(y)
    return alpha / (denom*denom)

def compute_v(im, gamma=None):
    """
    Calculates the largest minimal neighbor differences in the image.

    v = max(min_N(gamma_ij*|f[i]-f[j]|))      [1 eq 14]

    where i is restricted to pixels that are local minima or maxima.

    gamma is the neighbor weights, (i.e. a CONNECTIVITY constant) or one of the options allowable
    for calc_info (i.e. None, 'full', or 'dist'). Default is None which generates a minimal
    neighborhood.

    REFERENCES:
     1. Baus F, Nikolova M, Steidl G, 2013, "Fully smoothed l1−TV models: bounds for the minimizers
        and parameter choice", Tech. report, v3, http://hal.archives-ouvertes.fr/hal-00722743
    """

    # Non-matching v_f's to the paper:
    #   * couple - 41.61, I get 179.61; their image is clearly cropped, my prediction is that they
    #     crop it to about 481x481. When I try doing this lining up the image as best as possible
    #     and I get 36.06. Their number is actually impossible, closest is 59/sqrt(2)=41.72.
    #   * clock - 51.52, I get 44.55; their value is impossible, closest is 73/sqrt(2)=51.62.
    #   * tree - 54; it's a color image so I have to guess how they converted to grayscale,
    #     but none of the methods I tried got exactly 54, I get:
    #       50 for mean of R,G,B
    #       50.91 for max of R,G,B
    #       52 for min of R,G,B
    #       86.27 for (max-min)/2 of R,G,B
    #       43.13 for R
    #       56 for G
    #       52 for B
    #       50 for 0.3*R+0.59*G+0.11*B, 0.2989*R+0.5870*G+0.1140*B, or 0.299*R+0.587*G+0.114*B
    #       52 for 0.2126*R+0.7152*G+0.0722*B
    #     they are however mostly in the correct range (with the exception of the (max-min)/2 and R
    #     channel)
    #   * stream and tank are each off by 0.01, just a rounding error

    from numpy import abs, greater, less # pylint: disable=redefined-builtin

    # Perform checks
    gamma = __get_gamma(gamma, im.ndim)

    # Will work with everything as floats
    im = im.astype(float, copy=False)

    # Compute all differences for all non-zero gammas
    internal = im[(slice(1, -1),)*im.ndim]
    gamma_nz_pos = nonzero(gamma)
    diffs = empty((len(gamma_nz_pos[0]),) + internal.shape)
    for i, pos in enumerate(zip(*gamma_nz_pos)):
        subtract(internal, im[tuple(slice(x, x+n-2) for x, n in zip(pos, im.shape))], diffs[i])

    # Only use pixels where the pixel is lower or greater than all neighbors
    tmp = empty(diffs.shape, bool)
    local_max_min = greater(diffs, 0, out=tmp).all(0)
    local_max_min |= less(diffs, 0, out=tmp).all(0, out=tmp[0])
    diffs = diffs[:, local_max_min]
    if diffs.size == 0: return 0 # no local minimums or maximums?

    # Compute gamma*|diffs|
    abs(diffs, diffs)
    diffs *= gamma[gamma != 0][:, None]

    # Compute the max of all of the mins of each remaining pixel
    return diffs.min(0, out=diffs[0]).max()

def compute_c(im, beta=None, alpha=None, gamma=None):
    """
    Computes c = phi'(z, alpha_2)    [1 eq 15 & 22; 2 eq 14]
    where:
        z = v(im, gamma) - 2*b(beta*eta, alpha_1) > 0
        v(im, gamma) is the largest minimal neighbor differences in the image    [1 eq 14]
        eta = sum(gamma)   [1 eq 10]
        gamma is the neighbor weights, (i.e. a CONNECTIVITY constant), default is CONNECTIVITY_N4
            for 2D images and CONNECTIVITY3_N6 for 3D images
        b(y) is the inverse of theta'(t)    [2 eq 12]
            In [1] b(y) = (theta')^(-1)(y*eta)    [1 eq 12]
        alpha_2 equals alpha_1 if one alpha provided, provide a sequence to have different values
    The c value must be <1.0 but should be very close to 1.0 (between 1.0-1e-5 and 1.0) for a good
    set of parameters for a given image.

    Instead of an image, the first argument may also be a tuple of (v_f and ndim) for an image.

    REFERENCES:
     1. Baus F, Nikolova M, Steidl G, 2013, "Fully smoothed l1−TV models: bounds for the minimizers
        and parameter choice", Tech. report, v3, http://hal.archives-ouvertes.fr/hal-00722743
     2. Nikolova M, Wen Y-W, and Chan R, 2013, "Exact histogram specification for digital images
        using a variational approach", J of Mathematical Imaging and Vision, 46(3):309-325
    """
    from collections.abc import Sequence
    v_f, ndim = im if isinstance(im, Sequence) else (None, im.ndim)
    gamma, eta, beta, alpha_1, alpha_2 = __check_args(gamma, beta, alpha, ndim)
    if v_f is None: v_f = compute_v(im, gamma)
    z = v_f - 2*d_theta_inv(beta*eta, alpha_1)
    if z <= 0: raise ValueError('z')
    return d_theta(z, alpha_2)

def compute_optimal_alpha_1(beta=None, gamma=CONNECTIVITY_N4, delta=0.5):
    """
    Computes the optimal alpha_1 value for a given beta and gamma as given by the solution of:
        (theta')^(-1) (beta*eta, alpha_1) = delta  [1 eq 21]
    Use delta=0.5 (default) to ensure the pixel value order is kept. eta = sum(gamma)* [1 eq 10].

    The default beta is 0.4/eta. The default gamma is CONNECTIVITY_N4 which is only appropriate for
    2D images.

    REFERENCES:
     1. Baus F, Nikolova M, Steidl G, 2013, "Fully smoothed l1−TV models: bounds for the minimizers
        and parameter choice", Tech. report, v3, http://hal.archives-ouvertes.fr/hal-00722743
    """
    # NOTE: This does not use the d_theta_inv function but instead has it solved out. If that
    # function is changed this function will also need to be updated. See [1 table 2].
    eta = __check_gamma(gamma).sum()
    beta = __get_beta(beta, eta)
    if delta <= 0: raise ValueError('delta')
    return delta * (1 / (beta*eta) - 1)

def check_convergence(beta=None, alpha=None, gamma=CONNECTIVITY_N4, return_value=False):
    """
    Check convergence of the fixed point algorithm, the following condition must be true:
        2*eta2*beta*xi'(beta*eta, alpha_1)*theta''(0, alpha_2) < 1    [3 eq 10]
    where:
        eta = sum(gamma)   [1 eq 10]
        eta2 = sum(gamma^2)
        gamma is the neighbor weights, (i.e. a CONNECTIVITY constant)
        alpha_2 equals alpha_1 if one alpha provided, provide a sequence to have different values

    The default beta is 0.4/eta. The deafult alphas are 0.1/eta. The default gamma is
    CONNECTIVITY_N4 which is only appropriate for 2D images.

    If return_value is given as True, the value that is computed which must be < 1 for convergence
    is returned instead of a True or False value.

    This check is significantly restrictive as it makes several assumptions including that the image
    is a constant graylevel so values slightly larger than 1 will likely work. See [3 Remark 1].

    REFERENCES:
     3. Nikolova M and Steidl G, 2014, "Fast Ordering Algorithm for Exact Histogram Specification"
        IEEE Trans. on Image Processing, 23(12):5274-5283
    """
    gamma, eta, beta, alpha_1, alpha_2 = __check_args(gamma, beta, alpha, gamma.ndim)
    eta2 = (gamma*gamma).sum()
    value = 2*eta2*beta*d_d_theta_inv(beta*eta, alpha_1)*d2_theta(0, alpha_2)
    return value if return_value else (value < 1)

def compute_upper_beta(gamma=CONNECTIVITY_N4, alpha_1_2_ratio=1):
    """
    Computes the upper value of beta using:
        2*eta2*beta*xi'(beta*eta, alpha_1)*theta''(0, alpha_2) < 1    [3 eq 10]
    which simplifies to:
        alpha_1/alpha_2 * 2*eta2*beta/(1-beta*eta)^2 < 1
    for the current theta where:
        eta = sum(gamma)   [1 eq 10]
        eta2 = sum(gamma^2)
        gamma is the neighbor weights, (i.e. a CONNECTIVITY constant), default is CONNECTIVITY_N4
          which is only appropriate for 2D images

    The default assumes alpha_1 == alpha_2. If not, specify alpha_1/alpha_2 as alpha_1_2_ratio.

    This upper bound is significantly restrictive as it makes several assumptions including that the
    image is a constant graylevel so values slightly larger than the computed value will likely
    work. See [3 Remark 1].

    REFERENCES:
     3. Nikolova M and Steidl G, 2014, "Fast Ordering Algorithm for Exact Histogram Specification"
        IEEE Trans. on Image Processing, 23(12):5274-5283
    """
    # NOTE: This does not use the theta functions but instead has it solved out. If those functions
    # are changed this function will also need to be updated. See [3 table after eq 10] but need to
    # change the 8 to 2*eta2 and the 4 to eta set < 1 and then solve to beta.
    gamma = __check_gamma(gamma)
    if alpha_1_2_ratio <= 0: raise ValueError('alpha_1_2_ratio')
    eta, eta2 = gamma.sum(), (gamma*gamma).sum()
    f = alpha_1_2_ratio * eta2
    return (eta + f - sqrt(f*(f+2*eta))) / (eta*eta)
