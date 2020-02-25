"""
Generalized code used for compacting arbitrary sets of filters.
"""

from numpy import uint64
from ..util import get_array_module, get_dtype_max, as_unsigned

def compact(im, scales, filtered_gen, args=()):
    """
    Run filter compaction for an arbitrary set of filters. The image must be an integral type of
    either 8 or 16 bits and scales is a list of tuples of the maximum multiplication of the negative
    and positive values from the original images that can be generated.

    filtered_gen is a function that takes an image and any arguments in args and returns a generator
    that produces im-shaped filtered images. The returned values must be uint64 or int64 arrays and
    will be modified in this function but can be modified and reused in the generator. The images
    must be generated starting with the most important to the least important. The identity filtered
    image is not included.
    """
    xp = get_array_module(im)

    # Make sure data is unsigned
    im = as_unsigned(im)
    filtered_gen = filtered_gen(im, *args)

    # Compact the results
    mx, bpp = get_dtype_max(im.dtype), im.dtype.itemsize * 8
    extra_bits = __compute_extra_bits_from_scales(scales)
    nlayers = __compute_num_layers(bpp, extra_bits)
    out = xp.zeros((nlayers,) + im.shape, uint64)

    # Save the original image
    layer, shift = nlayers-1, 64-bpp
    out[layer, ...] = im
    out[layer, ...] <<= shift

    # Perform filtering
    for (neg, _), extra_bits, filtered in zip(scales, extra_bits, filtered_gen):
        # Make sure all values are positive
        if neg: filtered += neg*mx
        filtered = filtered.view(uint64)

        # Adjust the save location
        shift -= bpp + extra_bits
        if shift < 0:
            layer -= 1
            shift = 64 - bpp - extra_bits

        # Save the filtered image
        filtered <<= shift
        out[layer, ...] |= filtered

    return out[0, ...] if nlayers == 1 else out

def non_compact(im, n, filtered_gen, args=()):
    """
    Generates a non-compacted filter set using a very similar interface to the compact function so
    that the same generator could be used for both easily. The argument n is the number of filtered
    images to be generated. The filtered images generated should be floats.
    """
    xp = get_array_module(im)
    out = xp.empty((n + 1,) + im.shape)
    out[-1, ...] = im
    filtered_gen = filtered_gen(out[-1, ...], *args)
    for i, data in enumerate(filtered_gen):
        out[-i-2, ...] = data
    return out

def __compute_extra_bits_from_scales(scales):
    """
    Compute the number of extra bits needed to store values with the given scales. Each scale is a
    tuple of the maximum multiplication possible in the negative and positive directions.
    """
    from ..util import log2i
    return [log2i(pos + neg) for neg, pos in scales]

def __compute_num_layers(bpp, extra_bits, layer_bits=64):
    """
    Computes the number of 'layers' required to store the original image with bpp bits-per-pixel
    along with the results of len(extra_bits) filtered versions of the image, each requiring
    bpp + extra_bits[i] bits. The filters are done in that order and each layer can have at most
    layer_bits bits (defaults to 64).
    """
    nlayers, shift = 1, layer_bits-bpp
    for ex_bits in extra_bits:
        shift -= bpp + ex_bits
        if shift < 0:
            nlayers += 1
            shift = layer_bits - bpp - ex_bits
    return nlayers

def scale_from_filter(fltr):
    """Get the scale from a convolution/correlation filter."""
    return (-int(fltr.clip(None, 0).sum()), int(fltr.clip(0, None).sum()))
