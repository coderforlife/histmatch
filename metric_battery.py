"""
Simple program to run all the metrics on a histogram equalization method and an image.
"""

from . import imhist, histeq, histeq_exact
from .metrics import (contrast_per_pixel, enhancement_measurement, distortion,
                      contrast_enhancement, count_differences, psnr, ssim)

def metric_battery(original, method, plot=False, **kwargs):
    """
    Runs a battery of metrics while historgram equalizing the original image and then attempting to
    reconstruct the original image using the given method (either 'classic' or one of the methods
    supported by histeq_exact). If plot is True plots of the images and their histograms is shown.
    All kwargs are passed onto the histeq method. All results are printed out.
    """
    # pylint: disable=too-many-locals
    hist_orig = imhist(original)
    if method == 'classic':
        enhanced = histeq(original, 256, **kwargs)
        recon = histeq(enhanced, hist_orig, **kwargs)
        fails_forward = fails_reverse = -1
    else:
        kwargs['method'] = method
        kwargs['return_fails'] = True
        enhanced, fails_forward = histeq_exact(original, 256, **kwargs)
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
    psnr_ = psnr(original, recon)
    ssim_ = ssim(original, recon)

    print('Image shape: %s'%(original.shape,))
    print('Original and enhanced image:')
    print('  Contrast-per-pixel: %.2f %.2f (%.2f%%)'%(cpp_orig, cpp_enh, cpp_ratio*100))
    print('  EME:                %.2f %.2f (%.2f%%)'%(eme_orig, eme_enh, eme_ratio*100))
    print('  SDME:               %.2f %.2f (%.2f%%)'%(sdme_orig, sdme_enh, sdme_ratio*100))
    if method != 'classic':
        print('During enhancement and reconstruction there were %d and %d fails'%
              (fails_forward, fails_reverse))
    print('Contrast Enhancement of: %.2f'%contrast_enhancement_)
    print('Enhancement caused a distortion of: %.2f'%distortion_oe)
    print('Reconstruction:')
    print('  Distortion: %.5f'%distortion_or)
    print('  Num Diffs:  %.2f%%'%(n_diffs/original.size))
    print('  PSNR:       %.2f dB'%psnr_)
    print('  SSIM:       %.5f'%ssim_)

    if plot:
        import matplotlib.pylab as plt
        plt.gray()
        __plot(0, original)
        __plot(1, enhanced)
        __plot(2, recon)
        plt.show()

def __plot(idx, im):
    """Show an image and its histogram"""
    import matplotlib.pylab as plt
    plt.subplot(3, 2, 2*idx+1)
    plt.imshow(im)
    plt.subplot(3, 2, 2*idx+2)
    plt.hist(im.ravel(), 256, (0, 255))
    plt.xlim(0, 255)

def main():
    """Main function that runs histogram equalization metric battery on an image."""
    import argparse
    import imageio
    import hist._cmd_line_util as cui

    parser = argparse.ArgumentParser(description='Perform histogram equalization on an image')
    parser.add_argument('input', help='input image file')
    cui.add_method_arg(parser)
    parser.add_argument('--plot', action='store_true',
                        help='plot original, enhanced, and reconstructed images with histograms')

    args = parser.parse_args()

    im = imageio.imread(args.input)
    if im.ndim != 2: im = im.mean(2)
    metric_battery(im, args.method, plot=args.plot)

if __name__ == "__main__":
    main()
