"""The optimum exact histogram equalization according to minimizing the MSE."""

def ehe(im, h_dst, mask=None, return_fails=False, reconstruction=False):
    """
    Perform exact histogram equalization using the 'optimum' method as defined by [1]. This is
    optimal in the sense of reducing MSE during reconstruction.

    Unlike the other exact histogram equalization methods this does not create a strict ordering of
    pixel values but instead directly computes the new enhanced image.

    This function should not be called directly but is instead designed to be called by histeq_exact
    which performs checking and preprocessing on the arguments.

    REFERENCES:
     1. Balado, Félix, 2018, "Optimum Exact Histogram Specification", IEEE International Conference
        on Acoustics, Speech and Signal Processing.
    """

    # NOTE: when reconstruction=False (default) this is the same as arbitrary with stable sorting
    # but is faster. The real benfit of their system is the reconstruction.

    from numpy import zeros
    from . import calc_transform
    n = im.size if mask is None else mask.sum()
    # In the paper: im is z, h_dst is h^x, h_src is h^z, and index_z is analagous to Π_σ_z

    # Find closest equalization using minimum distance decoder
    #pylint: disable=invalid-name
    index_z = (im.ravel() if mask is None else im[mask]).argsort(kind='stable') # Πz
    if reconstruction:
        # When reconstructing we need to reverse various indices so that it goes optimally

        # Within each set of equal z we need to reverse the values
        from numpy import bincount
        h_src = bincount(im.ravel())
        for a, b in __bracket_iter(h_src):
            index_z[a:b] = index_z[a:b][::-1] # reverse

        # Fix ties of z, formally the way to recover stable sorting
        for a, b in __bracket_iter(h_dst):
            index_z[a:b].sort()

    # Complete the process and generate the image
    x = calc_transform(h_dst, im.dtype, n, index_z.size)
    y = zeros(im.shape, im.dtype)
    y.put(index_z, x) # y=(Πz')x
    return (y, -1) if return_fails else y

def __bracket_iter(iterable):
    "s -> (0,s0), (s0,s0+s1), (s0+s1, s0+s1+s2), ..."
    counter = 0
    for value in iterable:
        yield counter, counter+value
        counter += value
