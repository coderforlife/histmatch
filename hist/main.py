"""
Simple main program to perform histogram equalization.
"""

from . import histeq, histeq_exact

def main():
    """Main function that runs histogram equalization on an image."""
    import argparse
    import imageio
    import hist._cmd_line_util as cui

    parser = argparse.ArgumentParser(description='Perform histogram equalization on an image')
    parser.add_argument('input', help='input image file')
    parser.add_argument('output', help='output image file')
    cui.add_method_arg(parser)
    cui.add_kwargs_arg(parser)

    args = parser.parse_args()

    im = imageio.imread(args.input)
    if im.ndim != 2: im = im.mean(2)
    if args.method == 'classic':
        out = histeq(im, 256, **dict(args.kwargs))
    else:
        out = histeq_exact(im, method=args.method, **dict(args.kwargs))

    imageio.imwrite(args.output, out)

if __name__ == "__main__":
    main()
