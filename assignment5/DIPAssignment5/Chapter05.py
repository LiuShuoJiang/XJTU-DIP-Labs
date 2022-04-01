import cv2
from skimage import io
import numpy as np
from numpy.linalg import norm  # norm计算二范数
from scipy.signal.windows import gaussian
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from math import exp, dist  # dist返回两点间欧几里得距离
import matplotlib.pyplot as plt


# %% 函数定义

# 计算功率谱比
def calc_spectral_ratio(fft_shifted, fft_filtered):
    return (norm(fft_filtered) ** 2) / (norm(fft_shifted) ** 2)


# 高斯滤波
def gaussian_frequency_filter(img, d0, is_lowpass=True):
    base_space = np.zeros(img.shape[:2])
    rows, columns = img.shape[:2]
    center_point = (rows / 2, columns / 2)

    for x in range(columns):
        for y in range(rows):
            if is_lowpass is True:
                base_space[y, x] = exp(((-dist((y, x), center_point) ** 2) / (2 * (d0 ** 2))))
            else:
                base_space[y, x] = 1 - exp(((-dist((y, x), center_point) ** 2) / (2 * (d0 ** 2))))

    centered = fftshift(fft2(img))
    pass_center = centered * base_space
    passed = ifft2(ifftshift(pass_center)).real
    ratio = calc_spectral_ratio(centered, pass_center)
    return passed, ratio


# 另一种定义的高斯低通滤波函数（实验报告中并未呈现）
def alternative_gaussian_filter(img, std):
    kernel = np.outer(gaussian(img.shape[0], std), gaussian(img.shape[1], std))
    freq = fft2(img)
    freq_kernel = fft2(ifftshift(kernel))
    convolution = freq * freq_kernel

    output_img = ifft2(convolution).real
    output_img = output_img * 255 / np.max(output_img)
    return output_img


# 巴特沃斯滤波
def butterworth_frequency_filter(img, d0, order, is_lowpass=True):
    base_space = np.zeros(img.shape[:2])
    rows, columns = img.shape[:2]
    center_point = (rows / 2, columns / 2)
    for x in range(columns):
        for y in range(rows):
            if is_lowpass is True:
                base_space[y, x] = 1 / (1 + (dist((y, x), center_point) / d0) ** (2 * order))
            else:
                base_space[y, x] = 1 - (1 / (1 + (dist((y, x), center_point) / d0) ** (2 * order)))

    centered = fftshift(fft2(img))
    pass_center = centered * base_space
    passed = ifft2(ifftshift(pass_center)).real
    ratio = calc_spectral_ratio(centered, pass_center)
    return passed, ratio


# 拉普拉斯高通滤波
def laplacian_frequency_filter(img):
    base_space = np.zeros(img.shape[:2])
    rows, columns = img.shape[:2]
    center_point = (rows / 2, columns / 2)
    for x in range(columns):
        for y in range(rows):
            base_space[y, x] = -4 * (np.pi ** 2) * (dist((y, x), center_point) ** 2)

    centered = fftshift(fft2(img))
    pass_center = centered * base_space
    passed = ifft2(ifftshift(pass_center)).real
    return passed


# unsharp滤波（使用Gaussian高通滤波器）
def unsharp_frequency_filter(img, d0, k1, k2):
    base_space = np.zeros(img.shape[:2])
    rows, columns = img.shape[:2]
    center_point = (rows / 2, columns / 2)
    for x in range(columns):
        for y in range(rows):
            base_space[y, x] = k1 + k2 * (1 - exp(((-dist((y, x), center_point) ** 2) / (2 * (d0 ** 2)))))

    centered = fftshift(fft2(img))
    pass_center = centered * base_space
    passed = ifft2(ifftshift(pass_center)).real
    return passed


# 显示两幅图像的函数
def display_image(image_list, title_list):
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(image_list[i], 'gray')
        plt.title(title_list[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


# 显示频谱的函数
def magnitude_spectrum(img):
    img_dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(img_dft)
    spectrum = 20 * np.log10(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)  # +1是为了显示更为清晰
    return spectrum


# %% test1 Gaussian Lowpass Filter
test1 = io.imread("Resource/test1.pgm", as_gray=True)
test1GaussianLowpass, ratioGaussianLowpassTest1 = gaussian_frequency_filter(test1, d0=50, is_lowpass=True)
print(ratioGaussianLowpassTest1)
images = [test1, test1GaussianLowpass]
titles = ['Test1', 'Gaussian Lowpass Test1']
display_image(images, titles)

# %% plot spectrum of test1 and Gaussian Lowpass
plt.subplot(121), plt.imshow(magnitude_spectrum(test1)), plt.title('test1 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test1GaussianLowpass)), plt.title('test1 Gaussian Lowpass spectrum')
plt.show()

# %% test1 Butterworth Lowpass Filter
test1ButterworthLowpass, ratioButterLowpassTest1 = butterworth_frequency_filter(test1, d0=50, order=6, is_lowpass=True)
print(ratioButterLowpassTest1)
images = [test1, test1ButterworthLowpass]
titles = ['Test1', 'Butterworth Lowpass Test1']
display_image(images, titles)

# %% plot spectrum of test1 and Butterworth Lowpass
plt.subplot(121), plt.imshow(magnitude_spectrum(test1)), plt.title('test1 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test1ButterworthLowpass)), plt.title('test1 Butterworth Lowpass spectrum')
plt.show()

# %% test2 Gaussian Lowpass Filter
test2 = io.imread("Resource/test2.tif", as_gray=True)
test2GaussianLowpass, ratioGaussianLowpassTest2 = gaussian_frequency_filter(test2, d0=50, is_lowpass=True)
print(ratioGaussianLowpassTest2)
images = [test2, test2GaussianLowpass]
titles = ['Test2', 'Gaussian Lowpass Test2']
display_image(images, titles)

# %% plot spectrum of test2 and Gaussian Lowpass
plt.subplot(121), plt.imshow(magnitude_spectrum(test2)), plt.title('test2 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test2GaussianLowpass)), plt.title('test2 Gaussian Lowpass spectrum')
plt.show()

# %% test2 Butterworth Lowpass Filter
test2ButterworthLowpass, ratioButterLowpassTest2 = butterworth_frequency_filter(test2, d0=40, order=6, is_lowpass=True)
print(ratioButterLowpassTest2)
images = [test2, test2ButterworthLowpass]
titles = ['Test2', 'Butterworth Lowpass Test2']
display_image(images, titles)

# %% plot spectrum of test2 and Butterworth Lowpass
plt.subplot(121), plt.imshow(magnitude_spectrum(test2)), plt.title('test2 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test2ButterworthLowpass)), plt.title('test2 Butterworth Lowpass spectrum')
plt.show()


# %% test3 Gaussian Highpass Filter
test3 = io.imread("Resource/test3_corrupt.pgm", as_gray=True)
test3GaussianHighpass, ratioGaussianHighPassTest3 = gaussian_frequency_filter(test3, d0=50, is_lowpass=False)
print(ratioGaussianHighPassTest3)
images = [test3, test3GaussianHighpass]
titles = ['Test3', 'Gaussian Highpass Test3']
display_image(images, titles)

# %% plot spectrum of test3 and Gaussian Highpass
plt.subplot(121), plt.imshow(magnitude_spectrum(test3)), plt.title('test3 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test3GaussianHighpass)), plt.title('test3 Gaussian Highpass Spectrum')
plt.show()

# %% test3 Butterworth Highpass Filter
test3ButterworthHighpass, ratioButterHighpassTest3 = butterworth_frequency_filter(test3, d0=50, order=5, is_lowpass=False)
print(ratioButterHighpassTest3)
images = [test3, test3ButterworthHighpass]
titles = ['Test3', 'Butterworth Highpass Test3']
display_image(images, titles)

# %% plot spectrum of test3 and Butterworth Highpass
plt.subplot(121), plt.imshow(magnitude_spectrum(test3)), plt.title('test3 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test3ButterworthHighpass)), plt.title('test3 Butterworth Highpass Spectrum')
plt.show()

# %% test4 Gaussian Highpass Filter
test4 = io.imread("Resource/test4 copy.bmp", as_gray=True)
test4GaussianHighpass, ratioGaussianHighPassTest4 = gaussian_frequency_filter(test4, d0=50, is_lowpass=False)
print(ratioGaussianHighPassTest4)
images = [test4, test4GaussianHighpass]
titles = ['Test4', 'Gaussian Highpass Test4']
display_image(images, titles)

# %% plot spectrum of test4 and Gaussian Highpass
plt.subplot(121), plt.imshow(magnitude_spectrum(test4)), plt.title('test4 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test4GaussianHighpass)), plt.title('test4 Gaussian Highpass Spectrum')
plt.show()

# %% test4 Butterworth Highpass Filter
test4ButterworthHighpass, ratioButterHighpassTest4 = butterworth_frequency_filter(test4, d0=50, order=5, is_lowpass=False)
print(ratioButterHighpassTest4)
images = [test4, test4ButterworthHighpass]
titles = ['Test4', 'Butterworth Highpass Test4']
display_image(images, titles)

# %% plot spectrum of test4 and Butterworth Highpass
plt.subplot(121), plt.imshow(magnitude_spectrum(test4)), plt.title('test4 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test4ButterworthHighpass)), plt.title('test4 Butterworth Highpass Spectrum')
plt.show()


# %% test3 Laplacian Highpass Filter
test3Laplacian = laplacian_frequency_filter(test3)
images = [test3, test3Laplacian]
titles = ['Test3', 'Laplacian Filter Test3']
display_image(images, titles)

# %% plot spectrum of test3 and Laplacian
plt.subplot(121), plt.imshow(magnitude_spectrum(test3)), plt.title('test3 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test3Laplacian)), plt.title('test3 Laplacian Spectrum')
plt.show()

# %% test4 Laplacian Highpass Filter
test4Laplacian = laplacian_frequency_filter(test4)
images = [test4, test4Laplacian]
titles = ['Test4', 'Laplacian Filter Test4']
display_image(images, titles)

# %% plot spectrum of test4 and Laplacian
plt.subplot(121), plt.imshow(magnitude_spectrum(test4)), plt.title('test4 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test4Laplacian)), plt.title('test4 Laplacian Spectrum')
plt.show()


# %% test3 Unsharp Filter
test3Unsharp = unsharp_frequency_filter(test3, d0=50, k1=1, k2=1)
images = [test3, test3Unsharp]
titles = ['Test3', 'Unsharp Filter Test3']
display_image(images, titles)

# %% plot spectrum of test3 and Unsharp
plt.subplot(121), plt.imshow(magnitude_spectrum(test3)), plt.title('test3 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test3Unsharp)), plt.title('test3 Unsharp Spectrum')
plt.show()

# %% test4 Unsharp Filter
test4Unsharp = unsharp_frequency_filter(test4, d0=50, k1=1, k2=1)
images = [test4, test4Unsharp]
titles = ['Test4', 'Unsharp Filter Test4']
display_image(images, titles)

# %% plot spectrum of test4 and Unsharp
plt.subplot(121), plt.imshow(magnitude_spectrum(test4)), plt.title('test4 spectrum')
plt.subplot(122), plt.imshow(magnitude_spectrum(test4Unsharp)), plt.title('test4 Unsharp Spectrum')
plt.show()
