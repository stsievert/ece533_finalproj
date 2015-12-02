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

N, k, sigma = 1e3, 5, 1e-2
max_its, delta = 1e3, 1e-3
t = np.linspace(0, 2*np.pi, num=N)

# true signal
x = sum([np.sin(f*t) for f in [0, 1, 3, 4, 6, 7]])
# convolution array
n = np.arange(15)
h = np.exp(-n)
h /= sum(h)
# observations
y = fftconvolve(x, h, mode='same')
y += sigma*np.random.randn(*y.shape)

def richardson_lucy(obs, x, h):
    coeff = obs / (fftconvolve(x, h, mode='same'))
    x_hat = x * fftconvolve(coeff, h[::-1].copy(), mode='same')
    return x_hat

x_0 = x + 10*np.random.randn(*x.shape)
x_k = np.random.rand(*x.shape)
for k in range(int(max_its)):
    x_k1 = richardson_lucy(y, x_k, h)
    #if np.linalg.norm(x_k1 - x_k) < delta: break
    x_k = x_k1

plt.figure()
plt.plot(t, x, label='ground truth')
plt.plot(t, y, label='observations')
plt.plot(t, x_k, label='estimate')
plt.legend(loc='best')
plt.show()
