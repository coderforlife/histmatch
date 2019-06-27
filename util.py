"""
Basic utilities for working with images.
"""

from numpy import sctypes, bool_

BIT_TYPES = [bool_, bool] # bool8?
INT_TYPES = sctypes['int'] + [int]
UINT_TYPES = sctypes['uint']
FLOAT_TYPES = sctypes['float'] + [float]
BASIC_TYPES = BIT_TYPES + INT_TYPES + UINT_TYPES + FLOAT_TYPES


##### Image Verification #####
def is_single_channel_image(im):
    """
    Returns True if `im` is a single-channel image, basically it is a ndarray of 2 or 3 dimensions
    where the 3rd dimension length can only be 1 and the data type is a basic data type (integer
    or float but not complex). Does not check to see that the image has no zero-length dimensions.
    """
    if im.ndim in (3, 4) and im.shape[-1] == 1:
        im = im.squeeze(-1)
    return im.dtype.type in BASIC_TYPES and (im.ndim == 2 or im.ndim == 3 and im.shape[2] > 4)

def check_image_single_channel(im):
    """
    Similar to is_single_channel_image except instead of returning True/False it throws an exception
    if it isn't an image. Also, it returns a 2D image (with the 3rd dimension, if 1, removed).
    """
    if im.ndim in (3, 4) and im.shape[-1] == 1:
        im = im.squeeze(-1)
    if im.dtype.type not in BASIC_TYPES or (im.ndim != 2 and (im.ndim != 3 or im.shape[2] <= 4)):
        raise ValueError('Not single-channel image format')
    return im

def check_image_mask_single_channel(im, mask):
    """
    Checks if an image and possibly a mask are single-channel. The mask, if not None, must be bool
    and the same shape as the image. The image and mask are returned (without a 3rd dimension).
    """
    im = check_image_single_channel(im)
    if mask is not None:
        mask = check_image_single_channel(mask)
        if mask.dtype != bool or mask.shape != im.shape:
            raise ValueError('The mask must be a binary image with equal dimensions to the image')
    return im, mask


##### Min/Max for data types #####
def get_dtype_min_max(dt):
    """Gets the min and max value a dtype can take"""
    from numpy import dtype
    if not hasattr(get_dtype_min_max, 'mn_mx'):
        from numpy import iinfo
        mn_mx = {t:(iinfo(t).min, iinfo(t).max) for t in INT_TYPES + UINT_TYPES}
        mn_mx.update({t:(t(False), t(True)) for t in BIT_TYPES})
        mn_mx.update({t:(t('0.0'), t('1.0')) for t in FLOAT_TYPES})
        get_dtype_min_max.mn_mx = mn_mx
    return get_dtype_min_max.mn_mx[dtype(dt).type]

def get_dtype_min(dt):
    """Gets the min value a dtype can take"""
    return get_dtype_min_max(dt)[0]

def get_dtype_max(dt):
    """Gets the max value a dtype can take"""
    return get_dtype_min_max(dt)[1]

def get_im_min_max(im):
    """Gets the min and max values for an image or an image dtype."""
    from numpy import ndarray
    if not isinstance(im, ndarray):
        return get_dtype_min_max(im)
    dt = im.dtype
    if dt.kind != 'f':
        return get_dtype_min_max(dt)
    mn, mx = im.min(), im.max()
    return (mn, mx) if mn < 0.0 or mx > 1.0 else get_dtype_min_max(dt)
