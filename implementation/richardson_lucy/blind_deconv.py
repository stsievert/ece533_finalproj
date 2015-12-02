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
np.random.seed(42)


def richardson_lucy(obs, x, h, mode='x'):
    if mode=='x':
        coeff = obs / (fftconvolve(x, h, mode='same'))
        x_hat = x * fftconvolve(coeff, h[::-1].copy(), mode='same')
        return x_hat

    if mode=='h':
        den = fftconvolve(x, h, mode='full')
        n_diff = den.shape[0] - obs.shape[0]
        den = den[n_diff/2 : -n_diff/2]
        coeff = obs / den
        print(h.shape, coeff.shape, x.shape)
        x_hat = x * fftconvolve(coeff, h[::-1].copy(), mode='same')
        return x_hat

    n_pad = x.shape[0] - h.shape[0]
    if n_pad > 0:
        h_new = np.zeros_like(x)
        h_new[:h.shape[0]] = h
        h = h_new.copy()
    if n_pad < 0:
        x_new = np.zeros_like(h)
        x_new[:x.shape[0]] = x
        x = x_new.copy()

    assert x.shape == h.shape
    X = np.fft.fft(x)
    H = np.fft.fft(h)
    H_flip = np.fft.fft(h[::-1])
    Y = np.fft.fft(obs)
    coeff = Y / (X*H) * H_flip
    X_hat = X * coeff
    x_hat = np.fft.ifft(X_hat)
    assert np.max(x_hat.imag) < 1e-6
    return np.abs(x_hat)

BLIND = True
N, k, a, sigma = 1e2, 11, 1e-1, 1e-2
max_its, delta = 5e2, 1e-3
max_its_x, max_its_h = 1e1, 1e1

# true signal
t = np.linspace(0, 2*np.pi, num=N)
x = 10 + sum([np.sin(f*t) for f in [0, 1, 3, 4, 6, 7]])
# convolution array
n = np.arange(k)
h = np.exp(-a*n)
#h = np.concatenate((h, np.zeros(int(N-k))))
h /= sum(h)

h_k = h.copy()
if BLIND:
    h_k *= (1 + np.random.rand(*h.shape))

# observations
y = fftconvolve(x, h_k, mode='same')
if not BLIND:
    y += 1e-2*np.random.randn(*y.shape)

x_k = x + 10*np.random.randn(*x.shape)
#x_k = np.random.rand(*x.shape)
for k in range(int(max_its)):
    for _ in range(int(max_its_x)):
        x_k1 = richardson_lucy(y, x_k, h_k, mode='x')
    if BLIND:
        for _ in range(int(max_its_h)):
            h_k1 = richardson_lucy(y, h_k, x_k, mode='h')
    #if np.linalg.norm(x_k1 - x_k) < delta: break
    x_k = x_k1
    if BLIND:
        h_k = h_k1

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(t, x, label='ground truth')
plt.plot(t, y, label='observations')
plt.plot(t, x_k, label='estimate')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(h, label='true h')
plt.plot(h_k, label='observed h')
plt.legend(loc='best')
plt.show()
