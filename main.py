"""
Simple main program to perform histogram equalization.
"""

import argparse

import imageio

from . import histeq, histeq_exact

def main():
    parser = argparse.ArgumentParser(description='Perform histogram equalization on an image')
    parser.add_argument('input', help='input image file')
    parser.add_argument('output', help='output image file')
    parser.add_argument('method', choices=['classic', 'arbitrary', 'rand', 'lm', 'wa', 'va'],
                        help='method of histogram equalization')

    args = parser.parse_args()

    im = imageio.imread(args.input)
    if im.ndim != 2: im = im.mean(2)
    if args.method == 'classic':
        out = histeq(im, 256)
    else:
        out = histeq_exact(im, print_info=True, method=args.method)

    """
    import matplotlib.pylab as plt
    plt.gray()
    plt.subplot(2, 2, 1)
    plt.imshow(im)
    plt.subplot(2, 2, 2)
    plt.hist(im.ravel(), 256)
    plt.subplot(2, 2, 3)
    plt.imshow(out)
    plt.subplot(2, 2, 4)
    plt.hist(out.ravel(), 256)
    plt.show()
    """

    imageio.imwrite(args.output, out)

if __name__ == "__main__":
    main()
