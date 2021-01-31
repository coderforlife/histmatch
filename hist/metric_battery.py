"""
Simple program to run all the metrics on a histogram equalization method and an image.
"""

from . import imhist, histeq, histeq_exact
from .metrics import (contrast_per_pixel, enhancement_measurement, distortion,
                      contrast_enhancement, count_differences, psnr, ssim)

def metric_battery(original, method, nbins=256, csv=False, plot=False, **kwargs):
    """
    Runs a battery of metrics while historgram equalizing the original image and then attempting to
    reconstruct the original image using the given method (either 'classic' or one of the methods
    supported by histeq_exact). If csv is True then data is printed as CSV data with no header. If
    plot is True plots of the images and their histograms is shown. All kwargs are passed onto the
    histeq method*. All results are printed out.

    * If there is a reconstruction kwarg provided its value is ignored and it is set to False during
    the equalization and True during the reconstruction.
    """
    # pylint: disable=too-many-locals, too-many-statements
    hist_orig = imhist(original)
    if method == 'classic':
        enhanced = histeq(original, nbins, **kwargs)
        recon = histeq(enhanced, hist_orig, **kwargs)
        fails_forward = fails_reverse = -1
    else:
        kwargs['method'] = method
        kwargs['return_fails'] = True
        if 'reconstruction' in kwargs: kwargs['reconstruction'] = False
        enhanced, fails_forward = histeq_exact(original, nbins, **kwargs)
        if 'reconstruction' in kwargs: kwargs['reconstruction'] = True
        recon, fails_reverse = histeq_exact(enhanced, hist_orig, **kwargs)

    cpp_orig = contrast_per_pixel(original)
    cpp_enh = contrast_per_pixel(enhanced)
    cpp_ratio = cpp_enh / cpp_orig

    eme_orig = enhancement_measurement(original, block_size=5, metric='eme')
    eme_enh = enhancement_measurement(enhanced, block_size=5, metric='eme')
    eme_ratio = eme_enh / eme_orig

    sdme_orig = enhancement_measurement(original, block_size=5, metric='sdme')
    sdme_enh = enhancement_measurement(enhanced, block_size=5, metric='sdme')
    sdme_ratio = sdme_enh / sdme_orig

    distortion_oe = distortion(original, enhanced)
    distortion_or = distortion(original, recon)

    contrast_enhancement_ = contrast_enhancement(original, enhanced)

    n_diffs = count_differences(original, recon)

    psnr_oe = psnr(original, enhanced)
    psnr_or = psnr(original, recon)

    ssim_oe = ssim(original, enhanced)
    ssim_or = ssim(original, recon)

    if csv:
        print(cpp_orig, cpp_enh, eme_orig, eme_enh, sdme_orig, sdme_enh,
              fails_forward, fails_reverse,
              contrast_enhancement_, distortion_oe, psnr_oe, ssim_oe,
              n_diffs, distortion_or, psnr_or, ssim_or, sep=',')
    else:
        print('Image shape: %s'%(original.shape,))
        print('Original and enhanced image:')
        print('  Contrast-per-pixel: %5.2f %5.2f (%6.2f%%)'%(cpp_orig, cpp_enh, cpp_ratio*100))
        print('  EME:                %5.2f %5.2f (%6.2f%%)'%(eme_orig, eme_enh, eme_ratio*100))
        print('  SDME:               %5.2f %5.2f (%6.2f%%)'%(sdme_orig, sdme_enh, sdme_ratio*100))
        if method != 'classic':
            print('During enhancement and reconstruction there were %d and %d fails'%
                  (fails_forward, fails_reverse))
        print('Enhancement:')
        print('  Contrast:   %.2f'%contrast_enhancement_)
        print('  Distortion: %.5f'%distortion_oe)
        print('  PSNR:       %.2f dB'%psnr_oe)
        print('  SSIM:       %.5f'%ssim_oe)
        print('Reconstruction:')
        print('  Num Diffs:  %.2f%%'%(n_diffs/original.size))
        print('  Distortion: %.5f'%distortion_or)
        print('  PSNR:       %.2f dB'%psnr_or)
        print('  SSIM:       %.5f'%ssim_or)

    if plot:
        import matplotlib.pylab as plt # pylint: disable=import-error
        plt.gray()
        __plot(0, original)
        __plot(1, enhanced)
        __plot(2, recon)
        plt.show()

def __plot(idx, im):
    """Show an image and its histogram"""
    import matplotlib.pylab as plt # pylint: disable=import-error
    plt.subplot(3, 2, 2*idx+1)
    plt.imshow(im)
    plt.subplot(3, 2, 2*idx+2)
    plt.hist(im.ravel(), 256, (0, 255))
    plt.xlim(0, 255)

def main():
    """Main function that runs histogram equalization metric battery on an image."""
    import argparse
    import hist._cmd_line_util as cui

    parser = argparse.ArgumentParser(
        description='Calculate a series of metrics on the histogram equalization on an image')
    cui.add_input_image(parser)
    cui.add_method_arg(parser)
    parser.add_argument('--csv', action='store_true', help='output data as CSV with no header')
    parser.add_argument('--plot', action='store_true',
                        help='plot original, enhanced, and reconstructed images with histograms')
    parser.add_argument('--nbins', '-n', type=int, default=256, metavar='N',
                        help='number of bins in the intermediate histogram, default is 256 '
                        '(reverse direction always uses a full histogram)')
    cui.add_kwargs_arg(parser)
    args = parser.parse_args()

    # Load image
    im = cui.open_input_image(args)

    # Run
    sep_end = ',' if args.csv else '\n'
    print(args.input, args.method, sep=sep_end, end=sep_end)
    metric_battery(im, args.method, csv=args.csv, plot=args.plot, **dict(args.kwargs))

if __name__ == "__main__":
    main()
