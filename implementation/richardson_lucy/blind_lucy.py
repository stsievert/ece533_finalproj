"""
Used a wikipedia article[1] as source. As of 2015-11-27, this script does
deconvolution when the convolution parameter is known.

Plan of attach:
x implement 1D, h known
x change notation used (x/y/h, not u/p/d as per wiki)
* implement 1D, x and h unknown (apply function twice each iteration)
* implement 2D, x and h unknown

NOTES:
* this only seems to work when `h`, the convolution operator, is of odd length

[1]:https://en.wikipedia.org/wiki/Richardson–Lucy_deconvolution
"""

__author__ = {'Scott Sievert': ['stsieveert@wisc.edu', 'scottsievert.com']}

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import seaborn as sns; sns.set()
from numpy.fft import ifft, fft, ifft2, fft2
from numpy.linalg import norm
np.random.seed(42)
import helper

def update_g(F, G_arg, C, its=1, eps=1e-9):
    G = G_arg.copy()
    f_rev = np.rot90(ifft2(F), k=2)
    for k in range(int(its)):
        G = (C / (G*F + eps)) * fft2(f_rev) * G
    return G, ifft2(G)

def update_f(F_arg, G, C, its=1, eps=1e-9):
    F = F_arg.copy()
    g_rev = np.rot90(ifft2(G), k=2)
    for k in range(int(its)):
        F = (C / (F*G + eps)) * fft2(g_rev) * F
    return F, ifft2(F)

def richardson_lucy(image, psf, iterations=50, clip=True):
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]
    for _ in range(iterations):
        relative_blur = image / fftconvolve(im_deconv, psf, 'same')
        im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv

from skimage.restoration import wiener
import skimage.io as io
from helper import gaussian
#outputs = helper.blur_and_noise(name='cross', sigma=1e-6, noise_sigma=0e-6)
#(f, F), (c, C), (g, G) = outputs
f = io.imread('../data/cross.png').mean(axis=2)
f /= f.max()

g = helper.gaussian(sigma=1, N=3)
g_k = helper.gaussian(sigma=2, N=5)

c = fftconvolve(f, g, mode='same')

f_k = f + 0.0*np.random.randn(*f.shape)
#g_k = g + 0.0*np.random.randn(*g.shape)

max_its, its = 8, 5
#max_its, its = 5, 80
for k in range(int(max_its)):
    #f_k = wiener(f_k, g_k, 0.10)
    f_k = richardson_lucy(f_k, g_k, iterations=int(its), clip=True)
    g_k = richardson_lucy(g_k, f_k, iterations=int(its), clip=True)

    print("on {}, f.max() = {:0.3e}, g.max() = {:0.3e}".format(k, np.abs(f_k.max()),
                                                     np.abs(g_k.max())))

f_k, g_k = np.abs(f_k), np.abs(g_k)
#print("max |g - g_k| = {}".format(np.max(np.abs(g - g_k))))
helper.show_images({'estimation':f_k, 'original':f, 'observations':c,
                    'g':g, 'g_k':g_k})#, colorbar=True)
