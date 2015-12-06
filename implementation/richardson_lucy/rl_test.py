
import numpy as np
from skimage import color, data, restoration
import numpy as np
import numpy.random as npr
from scipy.signal import convolve2d
import helper

camera = color.rgb2gray(data.camera())
camera = helper.get_image('cameraman')
from scipy.signal import convolve2d
psf = np.ones((5, 5)) / 25
camera = convolve2d(camera, psf, 'same')
camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
deconvolved = restoration.richardson_lucy(camera, psf, 5)

helper.show_images({'deconv':deconvolved})
