import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_float, random_noise, img_as_ubyte
from skimage.restoration import estimate_sigma
from PIL import Image, ImageFilter
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
from numpy import pi, sin, exp


# %% definition of functions


# 显示两幅图像的函数
def display_image(image_list, title_list):
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(image_list[i], 'gray')
        plt.title(title_list[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


# %% read image
lena = io.imread("lena.bmp", as_gray=True)
lena = img_as_float(lena)

# %% add Gaussian noise
lenaGaussianNoise = random_noise(lena, mode='gaussian', mean=0.1, var=0.04)
images = [lena, lenaGaussianNoise]
titles = ['lena', 'Gaussian noise mean=0.1, var=0.04']
display_image(images, titles)

# Estimate the average noise standard deviation.
sigma_est = estimate_sigma(lenaGaussianNoise, channel_axis=None, average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the specified sigma.
print(f'Estimated Gaussian noise standard deviation = {sigma_est}')

# %% denoise Gaussian: Gaussian spatial filter
lenaGaussianNoiseCV = img_as_ubyte(lenaGaussianNoise)
lenaGaussian = cv2.GaussianBlur(lenaGaussianNoiseCV, ksize=(11, 11), sigmaX=1.5)
images = [lenaGaussianNoiseCV, lenaGaussian]
titles = ['lena Gaussian noise', 'filtered by Gaussian']
display_image(images, titles)

# %% denoise Gaussian: median filter
lenaMedian = cv2.medianBlur(lenaGaussianNoiseCV, ksize=5)
images = [lenaGaussianNoiseCV, lenaMedian]
titles = ['lena Gaussian noise', 'filtered by median filter']
display_image(images, titles)

# %% denoise Gaussian: arithmetic mean filter
lenaMean = cv2.blur(lenaGaussianNoiseCV, ksize=(5, 5))
images = [lenaGaussianNoiseCV, lenaMean]
titles = ['lena Gaussian noise', 'filtered by arithmetic mean filter']
display_image(images, titles)

# %% denoise Gaussian: geometric mean filter
CVF = lenaGaussianNoiseCV.astype(float)
prob_tmp = np.where(CVF > 1.0e-10, CVF, 1.0e-10)
result = np.where(CVF > 1.0e-10, np.log10(prob_tmp), -1000)
lenaGeometricMean = np.uint8(np.exp(cv2.boxFilter(result, ddepth=-1, ksize=(3, 3))))
images = [lenaGaussianNoiseCV, lenaGeometricMean]
titles = ['lena Gaussian noise', 'filtered by geometric mean filter']
display_image(images, titles)

# %% another way of geometric filter
img = lenaGaussianNoiseCV.astype(float)
rows, cols = img.shape[:2]
ksize = 5
padsize = int((ksize - 1) / 2)
pad_img = cv2.copyMakeBorder(img, *[padsize] * 4, cv2.BORDER_DEFAULT)
geomean = np.zeros_like(img)
for r in range(rows):
    for c in range(cols):
        geomean[r, c] = np.prod(pad_img[r: r + ksize, c: c + ksize]) ** (1 / (ksize ** 2))
geomean = np.uint8(geomean)

images = [img, geomean]
titles = ['lena Gaussian noise', 'filtered by geometric mean filter']
display_image(images, titles)

# %% add salt-and-pepper noise
lenaSPNoise = random_noise(lena, mode='s&p', amount=0.2, salt_vs_pepper=0.5)
images = [lena, lenaSPNoise]
titles = ['lena', 'salt and pepper noise']
display_image(images, titles)

# %% denoise S&P: arithmetic mean filter
lenaSPNoiseCV = img_as_ubyte(lenaSPNoise)
lenaSPMean = cv2.blur(lenaSPNoiseCV, ksize=(5, 5))
images = [lenaSPNoiseCV, lenaSPMean]
titles = ['lena S&P noise', 'filtered by arithmetic mean filter']
display_image(images, titles)

# %% denoise S&P: median filter
lenaSPMedian = cv2.medianBlur(lenaSPNoiseCV, ksize=3)
images = [lenaSPNoiseCV, lenaSPMedian]
titles = ['lena S&P noise', 'filtered by median filter']
display_image(images, titles)

# %% denoise S&P: max filter
lenaSPNoisePIL = Image.fromarray(lenaSPNoiseCV)
lenaSPMax = lenaSPNoisePIL.filter(ImageFilter.MaxFilter(size=3))
images = [lenaSPNoiseCV, lenaSPMax]
titles = ['lena S&P noise', 'filtered by max filter']
display_image(images, titles)

# %% denoise S&P: min filter
lenaSPMin = lenaSPNoisePIL.filter(ImageFilter.MinFilter(size=3))
images = [lenaSPNoiseCV, lenaSPMin]
titles = ['lena S&P noise', 'filtered by min filter']
display_image(images, titles)


# %% denoise S&P: contra-harmonic mean filter
def contra_harmonic_mean(image, size, Q):
    numerator = np.power(image, Q + 1)
    denominator = np.power(image, Q)
    kernel = np.full(size, 1.0)
    ret = cv2.filter2D(numerator, -1, kernel) / cv2.filter2D(denominator, -1, kernel)
    return ret


lenaSPContra1 = contra_harmonic_mean(lenaSPNoise, size=(3, 3), Q=0.5)
lenaSPContra2 = contra_harmonic_mean(lenaSPNoise, size=(3, 3), Q=0)
lenaSPContra3 = contra_harmonic_mean(lenaSPNoise, size=(3, 3), Q=-0.5)
images = [lenaSPNoiseCV, lenaSPContra1, lenaSPContra2, lenaSPContra3]
titles = ['lena S&P noise', 'contra-harmonic filter Q=0.5',
          'contra-harmonic filter Q=0', 'contra-harmonic filter Q=-0.5']
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


# %% degradation function
def my_blur_function(img, t, a, b):
    base_space = np.zeros(img.shape[:2], dtype='complex_')
    rows, columns = img.shape[:2]
    for u in range(rows):
        for v in range(columns):
            x = u - rows / 2
            y = v - columns / 2
            val = pi * (x * a + y * b)
            if val == 0:
                val = 1
            base_space[u, v] = (t / val) * sin(val) * exp(-1j * val)
    centered = fftshift(fft2(img))
    pass_center = centered * base_space
    passed = ifft2(ifftshift(pass_center)).real
    return passed


lena = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
lenaBlur = my_blur_function(lena, t=1, a=0.1, b=0.1)
images = [lena, lenaBlur]
titles = ['lena', 'blurred lena']
display_image(images, titles)

# %% add Gaussian noise to the blurred image
lenaBlur = ((lenaBlur - lenaBlur.min()) * (1/(lenaBlur.max() - lenaBlur.min()) * 255)).astype('uint8')
lenaGB = random_noise(lenaBlur, mode='gaussian', mean=0, var=0.01)
images = [lenaBlur, lenaGB]
titles = ['blurred lena', 'add Gaussian noise to blurred']
display_image(images, titles)

lenaGB = img_as_ubyte(lenaGB)


# %% wiener filter
def my_wiener_filter(img, t, a, b, k):
    h = np.zeros(img.shape[:2], dtype='complex_')
    rows, columns = img.shape[:2]
    for u in range(rows):
        for v in range(columns):
            x = u - rows / 2
            y = v - columns / 2
            val = pi * (x * a + y * b)
            if val == 0:
                val = 1
            h[u, v] = (t / val) * sin(val) * exp(-1j * val)
    centered = fftshift(fft2(img))
    pass_center = np.zeros(img.shape[:2], dtype='complex_')
    for u in range(rows):
        for v in range(columns):
            pass_center[u, v] = ((1 / h[u, v]) * (abs(h[u, v])) ** 2 / ((abs(h[u, v])) ** 2 + k)) * centered[u, v]
    passed = ifft2(ifftshift(pass_center)).real
    passed = ((passed - passed.min()) * (1 / (passed.max() - passed.min()) * 255)).astype('uint8')
    return passed


lenaWiener = my_wiener_filter(lenaGB, t=1, a=0.1, b=0.1, k=0.05)
images = [lenaGB, lenaWiener]
titles = ['blurred+Gaussian noise', 'filtered by Wiener']
display_image(images, titles)


# %% constrained least squares filter
def zero_pad(image, shape, position='corner'):
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes ""have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img


def psf2otf(psf, shape):
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


LAPLACIAN = np.array([[0, -1,  0], [-1,  4, -1], [0, -1,  0]])


def my_constrained_least_squares_filter(img, t, a, b, g):
    h = np.zeros(img.shape[:2], dtype='complex_')
    rows, columns = img.shape[:2]
    for u in range(rows):
        for v in range(columns):
            x = u - rows / 2
            y = v - columns / 2
            val = pi * (x * a + y * b)
            if val == 0:
                val = 1
            h[u, v] = (t / val) * sin(val) * exp(-1j * val)
    centered = fftshift(fft2(img))
    pass_center = np.zeros(img.shape[:2], dtype='complex_')
    laplace = psf2otf(LAPLACIAN, (rows, columns))
    for u in range(rows):
        for v in range(columns):
            pass_center[u, v] = (h[u, v].conj() / ((abs(h[u, v])) ** 2 + g * abs(laplace[u, v]) ** 2)) * centered[u, v]
    passed = ifft2(ifftshift(pass_center)).real
    passed = ((passed - passed.min()) * (1 / (passed.max() - passed.min()) * 255)).astype('uint8')
    return passed


lenaLSF = my_constrained_least_squares_filter(lenaGB, t=1, a=0.1, b=0.1, g=0.001)
images = [lenaGB, lenaLSF]
titles = ['blurred+Gaussian noise', 'constrained least squares filter']
display_image(images, titles)



