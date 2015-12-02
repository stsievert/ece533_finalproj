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
from numpy.fft import ifft, fft
from numpy.linalg import norm
np.random.seed(42)

def update_g(F, G, C):
    F_rev = fft(ifft(F)[::-1])
    G_k1 = (C / (G*F)) * F_rev * G
    return ifft(G_k1)

def update_f(F, G, C):
    G_rev = fft(ifft(G)[::-1])
    F_k1 = (C / (F*G)) * G_rev * F
    return ifft(F_k1)

BLIND = True
N, k, a, sigma = 1e2, 1e2, 5.8, 1e-2
max_its = 8e1

# true signal
t = np.linspace(0, 2*np.pi, num=N)
f = 4 + sum([np.sin(f*t) for f in [0, 1, 3, 4, 6, 7]])
# convolution array
n = np.arange(k)
g = np.exp(-a*(n-k//2)**2)
#h = np.concatenate((h, np.zeros(int(N-k))))
g /= sum(g)
G = fft(g)

g_k = g.copy()
g_k *= (1 + 0.1*np.random.rand(*g.shape))

g_k = np.concatenate((g, np.zeros(f.shape[0] - g.shape[0])))
F, G_k = fft(f), fft(g_k)

# observations
C = (F * G_k) + 0.0*fft(np.random.rand(*F.shape))
F_k = F + 0.0*np.random.rand(*F.shape)
for k in range(int(max_its)):
    G_k1 = update_g(F_k, G_k, C)
    F_k1 = update_f(F_k, G_k, C)
    print("MSE_f = {}".format(norm(F - F_k) / norm(F)))
    #print("MSE_g = {}".format(norm(G - G_k) / norm(G)))

    F_k = F_k1
    G_k = G_k1

c, f_k, g_k = ifft(C), ifft(ifft(F_k)), ifft(G_k)

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(t, f, label='Ground truth')
plt.plot(t, ifft(C), label='Observations')
plt.legend(loc='best')

plt.subplot(2, 2, 2)
plt.plot(t, np.abs(f_k), label='Estimation')
plt.legend(loc='best')

plt.subplot(2, 2, 3)
plt.plot(g, label='Ground truth')
plt.legend(loc='best')

plt.subplot(2, 2, 4)
plt.plot(np.abs(g_k), label='Estimation')
plt.legend(loc='best')
plt.show()
