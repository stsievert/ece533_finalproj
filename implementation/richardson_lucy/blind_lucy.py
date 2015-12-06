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
from numpy.fft import ifft, fft, ifft2, fft2, fftshift, ifftshift
from numpy.linalg import norm
from skimage.restoration import wiener
import skimage.io as io

np.random.seed(42)
import helper

def update_g(F, G_arg, C, iterations=1, eps=1e-9):
    G = G_arg.copy()
    g = ifft2(G)
    f_rev = np.rot90(ifft2(F), k=2)
    for k in range(int(iterations)):
        blur = (C / (G*F + eps)) * fft2(f_rev)
        G = fftconvolve(G, blur, 'same')
        g = ifft2(blur) * g
        G = fft(g)
    return G, ifft2(G)

def update_f(F, G, C, iterations=1, eps=1e-9, clip=False):
    f, g, c = ifft2(F), ifft2(G), ifft2(C)
    G_mirror = ifft2(g[::-1, ::-1])
    image, psf, im_deconv = c, g, f

    #im_deconv = 0.5*np.ones_like(f)
    psf_mirror = psf[::-1, ::-1]
    for _ in range(iterations):
        RB = C / (G*F + eps)
        #relative_blur = image / fftconvolve(im_deconv, psf, 'same')
        im_deconv *= ifft2(RB * G_mirror)
        F = fft2(im_deconv)
        #im_deconv *= fftconvolve(relative_blur, psf_mirror, 'same')

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return fft2(im_deconv), im_deconv

f = helper.get_image('cameraman64')
N, _ = f.shape
f /= f.sum()
F = fft2(f)

g = helper.gaussian(sigma=1, N=N)
g /= g.sum()
G = fft2(g)

C = F*G
c = ifft2(C)

max_its = 80
for k in range(int(max_its)):
    F, f_k = update_f(F, G, C, eps=0, iterations=1)
    #G, g_k = update_g(F, G, C, eps=0, iterations=2)

    print("iteration {}, f.max = {:0.2e}, g.max = {:0.2e}".format(k, np.abs(f).max(),
                                                    np.abs(g).max()))

G, g_k = update_g(F, G, C, eps=0, iterations=2)
helper.show_images({'kernel':np.abs(g_k), 'estimate':np.abs(f_k),
                    'observations':np.abs(c)})
