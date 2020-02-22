"""Utilities for the command line programs."""

from numpy import nan, inf
import hist.exact.va as va

METHODS = ('classic', 'arbitrary', 'rand', 'na', 'nv', 'gl', 'ml', 'lc', 'lm', 'wa', 'swa', 'va',
           'optimum')

def add_method_arg(parser):
    """Add the method argument to an argument parser object."""
    parser.add_argument('method', choices=METHODS, help='method of histogram equalization')

def add_kwargs_arg(parser):
    """Add the kwargs arg to an argument parser object which accepts a series of k=v arguments."""
    parser.add_argument('kwargs', type=__kwargs_arg, nargs='*', help='any special keyword '+
                        'arguments to pass to the method, formated as key=value with value being '+
                        'a valid Python literal or one of the special values nan, inf, -inf, N4, '+
                        'N8, N8_DIST, N6, N18, N18_DIST, N26, N26_DIST')

def add_input_image(parser):
    """
    Add a required input argument and an optional --float argument. Use the open_input_image
    function to read the image. This supports filenames with glob wildcards or directories to read a
    series of images in as a 3D image.
    """
    parser.add_argument('input', help='input image file (including .npy, .npy.gz, and directories/wildcard names for 3D images)')
    parser.add_argument('--float', action='store_true', help='convert image to float')
    parser.add_argument('--gpu', action='store_true', help='utilize the GPU when able')

def open_input_image(args_or_filename, conv_to_float=False, use_gpu=False):
    """
    Opens the args.input image, converting to float is args.float is True. If the image filename
    ends with .npy or .npy.gz then it is directly loaded. Otherwise imageio is used to load the
    image and if it is color the mean of the color channels is used. If the filename does not
    exist and includes wildcard characters (? * []) then it is assumed to be a glob pattern to load
    a 3D image from. You can use add_input_image to setup the parser arguments for this function. 
    """
    import os
    from glob import glob
    from numpy import stack
    if isinstance(args_or_filename, str):
        filename = args_or_filename
    else:
        filename = args_or_filename.input
        conv_to_float = conv_to_float or args_or_filename.float
        use_gpu = use_gpu or args_or_filename.gpu
    if os.path.isdir(filename):
        filenames = [os.path.join(filename, image) for image in os.listdir(filename)]
    elif not os.path.exists(filename) and ('?' in filename or '*' in filename or
                                           ('[' in filename and ']' in filename)):
        filenames = glob(filename)
    else:
        return __load_image(filename, conv_to_float, use_gpu)
    filenames.sort()
    ims = [__load_image(filename, conv_to_float, use_gpu) for filename in filenames]
    return stack(ims)

def __load_image(filename, conv_to_float=False, use_gpu=False):
    """
    Loads a single image from the filename taking care of color data and conversion to float and/or
    loading onto the GPU.
    """
    import sys
    import gzip
    import imageio
    from numpy import load
    from hist.util import as_float
    if filename.endswith('.npy.gz'):
        with gzip.GzipFile(filename, 'rb') as f:
            im = load(f)
    elif filename.endswith('.npy'):
        im = load(filename)
    else:
        im = imageio.imread(filename)
        if im.ndim != 2: im = im.mean(2)
    if conv_to_float: im = as_float(im)
    if use_gpu:
        try:
            from cupy import asanyarray
        except ImportError:
            print("To utilize the GPU you must install the cupy package", file=sys.stderr)
            sys.exit(1)
        im = asanyarray(im)
    return im

__CONVERSIONS = {
    'nan': nan,
    'inf': inf,
    '-inf': -inf,
    'N4': va.CONNECTIVITY_N4,
    'N8': va.CONNECTIVITY_N8,
    'N8_DIST': va.CONNECTIVITY_N8_DIST,
    'N6': va.CONNECTIVITY3_N6,
    'N18': va.CONNECTIVITY3_N18,
    'N18_DIST': va.CONNECTIVITY3_N18_DIST,
    'N26': va.CONNECTIVITY3_N26,
    'N26_DIST': va.CONNECTIVITY3_N26_DIST,
}

def __kwargs_arg(value):
    import ast
    key, val = value.split('=', 1)
    return key, __CONVERSIONS[val] if val in __CONVERSIONS else ast.literal_eval(val)
