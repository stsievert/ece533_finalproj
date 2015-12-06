"""
Used a wikipedia article[1] as source. As of 2015-11-27, this script does
deconvolution when the convolution parameter is known.

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

def blind_lucy_wrapper(image, max_its=8, its=5, N_filter=3, weiner=False,
                       estimation_noise=0, filter_estimation=1,
                       observation_noise=0):
    f = io.imread('../data/'+image+'.png', dtype=float)
    if len(f.shape) == 3: f = f.mean(axis=2)
    f /= f.max()
    print(f.shape)

    g = helper.gaussian(sigma=N_filter/3, N=N_filter)
    g_k = helper.gaussian(sigma=N_filter/3 * filter_estimation, N=N_filter)
    g_0 = g_k.copy()

    c = fftconvolve(f, g, mode='same')
    #c += observation_noise*np.random.randn(*c.shape)

    f_k = f + estimation_noise*np.random.randn(*f.shape)
    #f_k = c.copy()

    for k in range(int(max_its)):
        g_k = richardson_lucy(g_k, f_k, iterations=int(its), clip=True)
        if weiner: f_k = wiener(f_k, g_k, 1e-5)
        else: f_k = richardson_lucy(f_k, g_k, iterations=int(its), clip=True)

        print("on {}, f.max() = {:0.3e}, g.max() = {:0.3e}".format(k, np.abs(f_k.max()),
                                                         np.abs(g_k.max())))

    f_k, g_k = np.abs(f_k), np.abs(g_k)
    helper.show_images({'estimation':f_k, 'original':f, 'observations':c})
                        #'original kernel':g, 'kernel estimation':g_k,
                        #'kernel initial estimate':g_0}, colorbar=True)

# Positive result -- keep it!
blind_lucy_wrapper('cross', max_its=1, its=9, N_filter=15,
                   estimation_noise=0e-1, filter_estimation=1.1, observation_noise=10)
#blind_lucy_wrapper('cross', max_its=2, its=50, N_filter=15,
                   #estimation_noise=0e-1, filter_estimation=1.1, observation_noise=10)

#blind_lucy_wrapper('cross', max_its=8, its=2, N_filter=15,
                   #estimation_noise=0e-2, filter_estimation=1.0, weiner=True)

#blind_lucy_wrapper('cross', max_its=19, its=2, N_filter=2,
                   #estimation_noise=0e-2, filter_estimation=1.1)
#blind_lucy_wrapper('barbara', max_its=7, its=3, N_filter=20,
                   #estimation_noise=0e-2, filter_estimation=1.1, weiner=False)
