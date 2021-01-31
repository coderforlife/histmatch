Histogram Equalization
======================

This library implements several forms of histogram equalization/selection/matching. This includes classical and several forms of exact histogram equalization. Additionally, several measurements of contrast enhancement are provided.

Most functions support:
 * numpy or cupy (i.e. GPU-based) arrays
 * nd images (not just typical 2D images)
 * image masks for only working on subsets of images if they take an image
 * a number of bins or a complete histogram if they take either

Usage
-----
```python
import hist
h = hist.imhist(im) # get the histogram of an image
out = hist.histeq(im, 256) # applies classical histogram equalization using 256 bins (deafult is 64 to match MATLAB)
out = hist.histeq_exact(im) # applies exact histogram equalization using 256 bins (default) with VA method
out = hist.histeq_exact(im, method='LM') # applies exact histogram equalization using 256 bins (default) with LM method
```
See the documentation for `histeq_exact` for which methods are supported, references to the relevant papers, and additional parameters.

The `histeq` and `histeq_exact` methods will be GPU-accelerated if passed a cupy array:
```python
im_gpu = cupy.asarray(im)
h_gpu = hist.imhist(im_gpu)
out_gpu = hist.histeq(im_gpu, 256)
out_gpu = hist.histeq_exact(im_gpu)
```

The code can be run as a standalone program:
```sh
python3 -m hist input.png output.png
python3 -m hist input # converts a 3d image stored in a folder
```
See `python3 -m hist --help` for more information.

References
----------
References for individual exact strict-ordering and metric algorithms is in function documentation.

This is the code is the implementation of the algorithms discussed and enhanced in the following papers:
 * ... (coming soon)

