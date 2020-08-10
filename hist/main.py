"""
Simple main program to perform histogram equalization.
"""

from . import histeq, histeq_exact

def main():
    """Main function that runs histogram equalization on an image."""
    import argparse
    import hist._cmd_line_util as cui

    # Extra imports to make sure everything is available now
    import numpy, scipy.ndimage # pylint: disable=unused-import, multiple-imports

    parser = argparse.ArgumentParser(description='Perform histogram equalization on an image')
    cui.add_input_image(parser)
    parser.add_argument('output', help='output image file')
    cui.add_method_arg(parser)
    cui.add_kwargs_arg(parser)
    args = parser.parse_args()

    # Load image
    im = cui.open_input_image(args)

    # Run HE
    if args.method == 'classic':
        out = histeq(im, 256, **dict(args.kwargs))
    else:
        out = histeq_exact(im, method=args.method, **dict(args.kwargs))

    # Save (if not testing)
    if args.output != '':
        __save(args, out)

def __save(args, out):
    import gzip
    import numpy
    import imageio
    from .util import is_on_gpu
    if is_on_gpu(out):
        out = out.get()
    if args.output.endswith('.npy'):
        numpy.save(args.output, out)
    elif args.output.endswith('.npy.gz'):
        with gzip.GzipFile(args.output, 'wb') as file:
            numpy.save(args.output, file)
    elif out.ndim == 3 and '#' in args.output:
        start = args.output.index('#')
        end = start + 1
        while end < len(args.output) and args.output[end] == '#':
            end += 1
        num_str, fmt_str = args.output[start:end], '%0'+str(end-start)+'d'
        for i in range(out.shape[0]):
            imageio.imwrite(args.output.replace(num_str, fmt_str % i), out[i, :, :])
    else:
        imageio.imwrite(args.output, out)


if __name__ == "__main__":
    main()
