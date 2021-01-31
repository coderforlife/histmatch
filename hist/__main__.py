"""
Simple main program to perform histogram equalization.
"""

from . import histeq, histeq_exact

def main():
    """Main function that runs histogram equalization on an image."""
    import argparse
    import os.path
    import re
    import hist._cmd_line_util as cui

    # Extra imports to make sure everything is available now
    import numpy, scipy.ndimage # pylint: disable=unused-import, multiple-imports

    parser = argparse.ArgumentParser(prog='python3 -m hist',
                                     description='Perform histogram equalization on an image')
    cui.add_input_image(parser)
    parser.add_argument('output', nargs='?',
                        help='output image file, defaults to input file name with _out before '
                        'the extension')
    cui.add_method_arg(parser)
    cui.add_kwargs_arg(parser)
    parser.add_argument('--nbins', '-n', type=int, default=256, metavar='N',
                        help='number of bins in the intermediate histogram, default is 256')
    args = parser.parse_args()

    # Load image
    im = cui.open_input_image(args)

    # Run HE
    if args.method == 'classic':
        out = histeq(im, args.nbins, **dict(args.kwargs))
    else:
        out = histeq_exact(im, args.nbins, method=args.method, **dict(args.kwargs))

    # Save (if not testing)
    filename = args.output
    if filename == '': return # hidden feature for "testing" mode, no saving
    if filename is None:
        if os.path.isdir(args.input):
            filename = args.input + '_out'
        elif os.path.exists(args.input) and ('?' in args.input or '*' in args.input or
                                             ('[' in args.input and ']' in args.input)):
            filename = re.sub(r'\*|\?|\[.+\]', '#', args.input)
        elif args.input.lower().endswith('.npy.gz'):
            filename = args.input[:-7] + '_out' + args.input[-7:]
        else:
            filename = '_out'.join(os.path.splitext(args.input))
    __save(filename, out)

def __save(filename, out):
    import gzip
    import numpy
    import imageio
    from .util import is_on_gpu
    if is_on_gpu(out):
        out = out.get()
    if filename.lower().endswith('.npy'):
        numpy.save(filename, out)
    elif filename.lower().endswith('.npy.gz'):
        with gzip.GzipFile(filename, 'wb') as file:
            numpy.save(filename, file)
    elif out.ndim == 3 and '#' in filename: 
        start = filename.index('#')
        end = start + 1
        while end < len(filename) and filename[end] == '#':
            end += 1
        num_str, fmt_str = filename[start:end], '%0'+str(end-start)+'d'
        for i in range(out.shape[0]):
            imageio.imwrite(filename.replace(num_str, fmt_str % i), out[i, :, :])
    else:
        imageio.imwrite(filename, out)


if __name__ == "__main__":
    main()
