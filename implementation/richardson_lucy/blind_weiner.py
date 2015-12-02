import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import seaborn as sns; sns.set()
from numpy.fft import fft, ifft, fft2, ifft2, fftshift
np.random.seed(42)
import drawnow
import time

import helper

def ell2(H, Y, tau=0.1, G='eye'):
    r"""
    :input H: The 2D Fourier representation of a convolution matrix.
    :input Y: The 2D Fourier representation of the input signal.
    :input tau: The regularization parameter
    :input G: The matrix in the regularization parameter.
    :returns f_hat: The spatial representation of the estimate. This estimate is obtained by

        \widehat{f} = \arg \min_f ||y - Hf||_2^2 + \tau ||Gf||_2^2
                    = F^\dagger (D_h^2 + \tau D_g^2)^{-1} D_h^\dagger F y

        where F represents the Fourier transform and
        is complex and an orthogonal matrix.
    """
    if G=='eye': G = np.eye(H.shape[0])
    W = H.T.conj() / (H**2 + tau*G**2 + 1e-9)
    F_hat = Y * W
    f_hat = ifft2(F_hat)
    return F_hat, fftshift(abs(f_hat))

def show_imgs():
    helper.show_images({'x':x, 'h':h, 'x_k':x_k, 'h_k':np.abs(h_k)},
            colorbar=True, drawnow=True, cmap='gray')

(x, X), (y, Y), (h, H) = helper.blur_and_noise('cameraman', sigma=5)

max_its = 8e0
H_k = H + 0e-1*np.random.randn(*H.shape)
h_k = abs(ifft2(H_k))

plt.figure()
for k in range(int(max_its)):
    X_k, x_k = ell2(H_k, Y, tau=1000)
    H_k, h_k = ell2(X_k, Y, tau=1000)
    #H_k = Y / (X_k + 1e-1)
    #h_k = ifft2(H_k)
    print('k = {}'.format(k))
    print('   abs(x_k).max(), abs(h_k).max() = {:0.2e}, {:0.2e}'.format(
              abs(X_k).max() / 98e3, abs(H_k).max()))
    if np.isnan(abs(x_k).max()):
        break
    print('Here!')
    drawnow.drawnow(show_imgs, confirm=False, show_once=True)





