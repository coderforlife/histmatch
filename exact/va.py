"""
Implements variation approach strict ordering for use with exact histogram equalization.
"""

from functools import lru_cache

from numpy import asarray, empty, sqrt, log, divide, subtract, nonzero

SQRT2I = 1/sqrt(2)
SQRT3I = 1/sqrt(3)

# Baus et al 2013 eq 4 and fig 3
CONNECTIVITY_N4 = asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
CONNECTIVITY_N8 = asarray([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
CONNECTIVITY_N8_DIST = asarray([[SQRT2I, 1, SQRT2I], [1, 0, 1], [SQRT2I, 1, SQRT2I]])

# Also have 3D connectivity
CONNECTIVITY3_N6 = asarray([
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
CONNECTIVITY3_N26 = asarray([
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ])
CONNECTIVITY3_N26_DIST = asarray([
    [[SQRT3I, SQRT2I, SQRT3I], [SQRT2I, 1, SQRT2I], [SQRT3I, SQRT2I, SQRT3I]],
    [[SQRT2I, 1, SQRT2I], [1, 0, 1], [SQRT2I, 1, SQRT2I]],
    [[SQRT3I, SQRT2I, SQRT3I], [SQRT2I, 1, SQRT2I], [SQRT3I, SQRT2I, SQRT3I]]
    ])
del SQRT2I, SQRT3I

# pylint: disable=invalid-name

def calc_info(im, niters=5, beta=0.1, alpha_1=0.05, alpha_2=None, gamma=None):
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

    beta and alpha_1 default to 0.1 and 0.05 respectively as chosen in [3]. alpha_2 defaults to
    alpha_1. See the 2013 paper, section III, for choosing the parameter values. In general keeping
    alpha_1 and alpha_2 small is important. The values are dependent on eta (the connectivity) and
    the image contents in attempt to make c in (1-1e-5, 1). Various other functions in this module
    can be used to help calculate the parameter values based on the image, connectivity, and other
    values.

    Their method has been adapted to work in 3D, including anisotropic data by adjusting gamma.

    The value of gamma determines the connectivity. It defaults to CONNECTIVITY_N4 for 2D images and
    CONNECTIVITY3_N6 for 3D images.

    REFERENCES:
     1. Baus F, Nikolova M, Steidl G, 2013, "Fully smoothed l1−TV models: bounds for the minimizers
        and parameter choice", Tech. report, v3, http://hal.archives-ouvertes.fr/hal-00722743
     2. Nikolova M, Wen Y-W, and Chan R, 2013, "Exact histogram specification for digital images
        using a variational approach", J of Mathematical Imaging and Vision, 46(3):309-325
     3. Nikolova M and Steidl G, 2014, "Fast Ordering Algorithm for Exact Histogram Specification"
        IEEE Trans. on Image Processing, 23(12):5274-5283
    """
    from numpy import abs #pylint: disable=redefined-builtin
    # this does not use pixels outside of the image at all

    # NOTE: This does not use the *_theta_* functions but instead has it solved out. If those
    # functions are changed this function will also need to be updated.

    if gamma is None: gamma = CONNECTIVITY3_N6 if im.ndim == 3 else CONNECTIVITY_N4
    gamma = __check_gamma(gamma)
    eta = gamma.sum()
    if niters <= 0: raise ValueError('niters') # niters is R in [3]
    if beta <= 0 or beta >= 1/eta: raise ValueError('beta')
    if alpha_1 <= 0: raise ValueError('alpha_1')
    if alpha_2 is None:
        alpha_2 = alpha_1
    elif alpha_2 <= 0: raise ValueError('alpha_2')
    if gamma.ndim != im.ndim: raise ValueError('gamma and im must have the same dimension')

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

# @lru_cache(maxsize=None) # gamma is unhashable...
def __get_g_info(gamma):
    """
    Gets a list of entries from the G gradient matrix for the differences that need to be performed.
    Each element in the list has a gamma value, the positive slice, and the negative slice. The list
    does not include reflected or 0 entries.
    """
    slices = (slice(1, None), slice(None), slice(None, -1)) # 0 -> 1:, 1 -> :, and 2 -> :-1
    return [(gamma[coord], tuple(slices[x] for x in coord), tuple(slices[2-x] for x in coord))
            for coord in zip(*nonzero(gamma)) if not __is_reflection_coord(coord)]

@lru_cache(maxsize=None)
def __is_reflection_coord(coord):
    """Reflection coordinates have a 2 followed by 0 or more trailing 1s."""
    while coord[-1] == 1:
        coord = coord[:-1]
    return coord[-1] == 2

def __calculate_d_Phi(u, t, G_info, alpha_2, d_phi, denom):
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

# @lru_cache(maxsize=None) # gamma is unhashable...
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
    if gamma.ndim == 2:
        if not __check_gamma_2D(gamma): raise ValueError('gamma')
    else: # gamma.ndim == 3:
        if (gamma.shape != (3, 3, 3) or (gamma < 0).any() or not __check_gamma_2D(gamma[1]) or
                (gamma[0] != gamma[2, ::-1, ::-1]).any()):
            raise ValueError('gamma')
    return gamma

# @lru_cache(maxsize=None) # gamma is unhashable...
def __check_gamma_2D(gamma):
    """
    Simplified version of __check_gamma for only 3x3 matrices. Argument must already be array.
    Returns True/False instead of raising exception.
    """
    return (gamma.shape == (3, 3) and (gamma >= 0).all() and gamma[1, 1] == 0 and
            gamma[0, 1] == gamma[2, 1] and (gamma[:, 0] == gamma[::-1, 2]).all())


##### Everything else is auxiliary and not needed except to help with choosing the parameters #####

def theta(t, alpha):
    """
    theta(t) = |t| - alpha * log(1+|t|/alpha)
    Also called phi or psi.
    Baus et al 2013, table 1, Theta_2.
    Nikolova et al 2013, table 1, f3.
    Nikolova et al 2014, table 1, theta_2.
    """
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

    gamma is the neighbor weights, (i.e. a CONNECTIVITY constant), default is CONNECTIVITY_N4 for
    2D images and CONNECTIVITY3_N6 for 3D images.

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
    if gamma is None: gamma = CONNECTIVITY3_N6 if im.ndim == 3 else CONNECTIVITY_N4
    gamma = __check_gamma(gamma)
    if gamma.ndim != im.ndim: raise ValueError('gamma and im must have the same dimension')

    # Work with non-zero gamma values
    gamma_nz = gamma[gamma != 0]
    if gamma_nz.size == 0: return 0

    # Will work with everything as floats
    im = im.astype(float, copy=False)

    # Compute all differences for all non-zero gammas
    diffs = __compute_v_diffs_2(im, gamma) if im.ndim == 2 else __compute_v_diffs_3(im, gamma)

    # Only use pixels where the pixel is lower or greater than all neighbors
    tmp = empty(diffs.shape, bool)
    local_max_min = greater(diffs, 0, out=tmp).all(0)
    local_max_min |= less(diffs, 0, out=tmp).all(0, out=tmp[0])
    diffs = diffs[:, local_max_min]

    # Compute gamma*|diffs|
    abs(diffs, diffs)
    diffs *= gamma_nz[:, None]

    # Compute the max of all of the mins of each remaining pixel
    return diffs.min(0, out=diffs[0]).max()

def __compute_v_diffs_2(im, gamma):
    """Compute diffs for compute_v for 2D images"""
    h, w = im.shape
    internal = im[1:-1, 1:-1]
    gamma_nz_pos = nonzero(gamma)
    diffs = empty((len(gamma_nz_pos[0]),) + internal.shape)
    for i, (row, col) in enumerate(zip(*gamma_nz_pos)):
        subtract(internal, im[row:h+row-2, col:w+col-2], diffs[i])
    return diffs

def __compute_v_diffs_3(im, gamma):
    """Compute diffs for compute_v for 3D images"""
    h, w, d = im.shape
    internal = im[1:-1, 1:-1, 1:-1]
    gamma_nz_pos = nonzero(gamma)
    diffs = empty((len(gamma_nz_pos[0]),) + internal.shape)
    for i, (row, col, depth) in enumerate(zip(*gamma_nz_pos)):
        subtract(internal, im[row:h+row-2, col:w+col-2, depth:d+depth-2], diffs[i])

def compute_c(im, beta, alpha_1, alpha_2=None, gamma=None):
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
        alpha_2 defaults to equal to alpha_1
    The c value must be <1.0 but should be very close to 1.0 (between 1.0-1e-5 and 1.0) for a good
    set of parameters for a given image.

    REFERENCES:
     1. Baus F, Nikolova M, Steidl G, 2013, "Fully smoothed l1−TV models: bounds for the minimizers
        and parameter choice", Tech. report, v3, http://hal.archives-ouvertes.fr/hal-00722743
     2. Nikolova M, Wen Y-W, and Chan R, 2013, "Exact histogram specification for digital images
        using a variational approach", J of Mathematical Imaging and Vision, 46(3):309-325
    """
    if gamma is None: gamma = CONNECTIVITY3_N6 if im.ndim == 3 else CONNECTIVITY_N4
    gamma = __check_gamma(gamma)
    eta = gamma.sum()
    if beta <= 0 or beta >= 1/eta: raise ValueError('beta')
    if alpha_1 <= 0: raise ValueError('alpha_1')
    if alpha_2 is None:
        alpha_2 = alpha_1
    elif alpha_2 <= 0: raise ValueError('alpha_2')
    z = compute_v(im, gamma) - 2*d_theta_inv(beta*eta, alpha_1)
    if z <= 0: raise ValueError('z')
    return d_theta(z, alpha_2)

def compute_optimal_alpha_1(beta, gamma=CONNECTIVITY_N4, delta=0.5):
    """
    Computes the optimal alpha_1 value for a given beta and gamma as given by the solution of:
        (theta')^(-1) (beta*eta, alpha_1) = delta  [1 eq 21]
    Use delta=0.5 (default) to ensure the pixel value order is kept. eta = sum(gamma)* [1 eq 10].

    The default gamma is CONNECTIVITY_N4 which is only appropriate for 2D images.

    REFERENCES:
     1. Baus F, Nikolova M, Steidl G, 2013, "Fully smoothed l1−TV models: bounds for the minimizers
        and parameter choice", Tech. report, v3, http://hal.archives-ouvertes.fr/hal-00722743
    """
    # NOTE: This does not use the d_theta_inv function but instead has it solved out. If that
    # function is changed this function will also need to be updated. See [1 table 2].
    eta = __check_gamma(gamma).sum()
    if beta <= 0 or beta >= 1/eta: raise ValueError('beta')
    if delta <= 0: raise ValueError('delta')
    return delta * (1 - abs(beta*eta)) / beta*eta

def check_convergence(beta, alpha_1, alpha_2=None, gamma=CONNECTIVITY_N4):
    """
    Check convergence of the fixed point algorithm, the following condition must be true:
        2*eta2*beta*xi'(beta*eta, alpha_1)*theta''(0, alpha_2) < 1    [3 eq 10]
    where:
        eta = sum(gamma)   [1 eq 10]
        eta2 = sum(gamma^2)
        gamma is the neighbor weights, (i.e. a CONNECTIVITY constant), default is CONNECTIVITY_N4
          which is only appropriate for 2D images
        alpha_2 defaults to equal to alpha_1

    REFERENCES:
     3. Nikolova M and Steidl G, 2014, "Fast Ordering Algorithm for Exact Histogram Specification"
        IEEE Trans. on Image Processing, 23(12):5274-5283
    """
    gamma = __check_gamma(gamma)
    eta, eta2 = gamma.sum(), (gamma*gamma).sum()
    if beta <= 0 or beta >= 1/eta: raise ValueError('beta')
    if alpha_1 <= 0: raise ValueError('alpha_1')
    if alpha_2 is None:
        alpha_2 = alpha_1
    elif alpha_2 <= 0: raise ValueError('alpha_2')
    return 2*eta2*beta*d_d_theta_inv(beta*eta, alpha_1)*d2_theta(0, alpha_2) < 1

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
