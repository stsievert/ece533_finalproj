import numpy as np
from numpy.fft import ifft2, fft2, fftshift
import helper

output = helper.blur_and_noise(sigma=6, noise_sigma=1e-5)
(x, X), (y, Y), (h, H) = output

H_k = H * 5# + np.random.randn(*H.shape) * 0.1

iterations = 20
for k in range(iterations):
    X_k = Y / (H_k + 1e-9)
    H_k = Y / (X_k + 1e-9)

x_k = fftshift(ifft2(X_k))

helper.show_images({'original':x, 'estimate':np.abs(x_k), 'observations':y})
