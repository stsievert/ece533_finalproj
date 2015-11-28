
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import data

def get_image(image_of, as_gray=True, small=False, extension='.png'):
    DATA_FOLDER = '../data/'
    FILENAME = DATA_FOLDER+image_of
    if small:
        FILENAME += '_small'
    FILENAME += extension
    x = data.imread(FILENAME)
    if as_gray:
        x = skimage.color.rgb2gray(x)
    x = np.asarray(x, dtype=float)
    x /= x.max()
    return x

def get_noisy_image(image_of, *args, sigma=0.1, **kwargs):
    x = get_image(image_of, *args, **kwargs)
    y = x + np.random.normal(0, sigma, size=x.shape)
    return x, y, x.shape

def gaussian(sigma=30, N=10):
    from scipy.signal import get_window as fspecial
    h = fspecial(('gaussian', sigma), N)
    h1, h2 = np.meshgrid(h, h)
    return h1*h2

def blur_and_noise(name='cameraman', sigma=1, noise_sigma=0):
    x = get_image(name)
    n = x.shape[0]
    X = np.fft.fft2(x)

    h = gaussian(N=x.shape[0], sigma=sigma)
    h /= np.sum(h) # so it integrates to 1
    H = np.fft.fft2(h)

    N = noise_sigma*np.random.randn(n, n)
    Y = X*np.abs(H) + N
    y = np.fft.ifft2(Y)

    y = np.abs(y)
    return (x, X), (y, Y), (h, H)

def plot_gaussian_and_boxcar():
    # implementation of a boxcar and gaussian to show the difference.
    N = 100
    x = np.zeros(N)
    x[N/2-N/4:N/2+N/4] = 1
    x /= np.sqrt(sum(x**2))

    y = np.zeros(N)
    y[N/2] = 1
    y = gaussian_filter1d(y, sigma=15)
    y /= np.sqrt(sum(y**2))

    plt.figure()
    plt.plot(x, label='Box car')
    plt.plot(y, label='Gaussian')
    plt.legend(loc='best')

    plt.show()

def show_images(imgs_and_titles, figsize=None, cmaps=None, cmap='gray', colorbar=False, **kwargs):
    N = len(imgs_and_titles.keys())
    if N == 0: print("yes")
    if not cmaps and (not cmap): cmaps = [None] * N
    if not cmaps: cmaps = [cmap] * N
    m = 1 if N<6 else 2
    n = N if m == 1 else N/2
    plt.figure(figsize=figsize)
    for i, (title, img) in enumerate(imgs_and_titles.items()):
        plt.subplot(m, n, i+1)
        plt.imshow(img, cmap=cmaps[i], **kwargs)
        plt.title(title)
        plt.axis('off')
        if colorbar: plt.colorbar()
    plt.show()

def plot_computational_adv(k, m, m_brute, m_box, m_better):
    import seaborn as sns; sns.set()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k, m_brute, label='Brute force')
    plt.plot(k, m_box, label='Box method')
    plt.plot(k, m_better, label='Better method')
    plt.title('Number of operations')
    plt.xlabel('k (kernel k x k)')
    plt.ylabel('Number of computations required \nfor 512 x 512 image')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(k, m_brute / m_box, label='Box exact')
    plt.plot(k, m_brute / m_better, label='Better exact')
    plt.plot(k, k, 'o--',   markevery=5, label ='Box approximation')
    plt.plot(k, 2*k, 'o--', markevery=5, label ='Better approximation')
    plt.title('Computational advantage over brute force')
    plt.ylabel('Computational advantage')
    plt.xlabel('k (kernel k x k)')
    plt.legend(loc='best')
    plt.show()
import inspect
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter, LatexFormatter
from IPython.display import display, HTML, Markdown, Latex
def pretty_print(f):
    s = inspect.getsource(f)
    print(s)
    return None
    return display(HTML(highlight(string, PythonLexer(), LatexFormatter(full=True))))
