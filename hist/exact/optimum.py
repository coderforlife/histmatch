"""The optimum exact histogram equalization according to minimizing the MSE."""

from ..util import is_on_gpu

def calc_info(im, h_dst, reconstruction=False, return_fails=False):
    """
    Perform exact histogram equalization using the 'optimum' method as defined by [1]. This is
    optimal in the sense of reducing MSE during reconstruction.

    Unlike the other exact histogram equalization methods this returns already sorted indices as a
    1D array meaning the sorting step in histeq_exact is skipped. If return_fails=True then this
    also returns the number of sorting failures (since that information is no longer available with
    pre-sorted indices)

    REFERENCES:
     1. Balado, Félix, 2018, "Optimum Exact Histogram Specification", IEEE International Conference
        on Acoustics, Speech and Signal Processing.
    """

    # NOTE: when reconstruction=False (default) this is the same as arbitrary with stable sorting.
    # The real benfit of their system is the reconstruction.

    # In the paper: im is z, h_dst is h^x, h_src is h^z, and index_z is analogous to Π_σ_z
    # In the hist.exact.histeq_exact method, transform is x and out is y.
    on_gpu = is_on_gpu(im)

    # Find closest equalization using minimum distance decoder
    im = im.ravel()
    index_z = im.argsort() if on_gpu else im.argsort(kind='stable')

    # When reconstructing we need to reverse various indices so that it goes optimally
    if reconstruction:
        if on_gpu:
            __reconstruction_on_gpu(im, h_dst, index_z)
        else:
            __reconstruction(im, h_dst, index_z)

    if return_fails:
        # This is the same as done in hist.exact.__sort_pixels but since we are returning the
        # sorted data we need to do this here.
        values_sorted = im[index_z]
        not_equals = values_sorted[1:] != values_sorted[:-1]
        return index_z, int(not_equals.size - not_equals.sum())
    return index_z

def __reconstruction(im, h_dst, index_z):
    # Within each set of equal z we need to reverse the values
    from numpy import bincount
    h_src = bincount(im)
    for i, j in __bracket_iter(h_src):
        index_z[i:j] = index_z[i:j][::-1] # reverse

    # Fix ties of z, formally the way to recover stable sorting
    for i, j in __bracket_iter(h_dst):
        if i+1 < j:
            index_z[i:j].sort()

def __reconstruction_on_gpu(im, h_dst, index_z):
    # pylint: disable=import-error
    # Within each set of equal z we need to reverse the values
    from cupy import bincount
    from cupy.cuda.stream import Stream
    h_src = bincount(im).get()
    for i, j in __bracket_iter(h_src):
        index_z[i:j] = index_z[i:j][::-1] # reverse

    # Fix ties of z, formally the way to recover stable sorting
    if is_on_gpu(h_dst): h_dst = h_dst.get()
    streams = [Stream() for _ in range(len(h_dst))]
    for stream, (i, j) in zip(streams, __bracket_iter(h_dst)):
        if i+1 < j:
            with stream:
                index_z[i:j].sort()
    Stream.null.synchronize()
    del streams

def __bracket_iter(iterable):
    "s -> (0,s0), (s0,s0+s1), (s0+s1, s0+s1+s2), ..."
    counter = 0
    for value in iterable:
        yield counter, counter+value
        counter += value
