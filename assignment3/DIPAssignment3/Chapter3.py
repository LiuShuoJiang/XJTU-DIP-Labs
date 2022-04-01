# =======================================Coded by 刘朔江  All rights reserved.===========================================
import cv2
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt

# %% ==============================================读取所有图形文件=======================================================
citywall = cv2.cvtColor(cv2.imread("Resource/citywall.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)
citywall1 = cv2.cvtColor(cv2.imread("Resource/citywall1.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)
citywall2 = cv2.cvtColor(cv2.imread("Resource/citywall2.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)

elain = cv2.cvtColor(cv2.imread("Resource/elain.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)
elain1 = cv2.cvtColor(cv2.imread("Resource/elain1.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)
elain2 = cv2.cvtColor(cv2.imread("Resource/elain2.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)
elain3 = cv2.cvtColor(cv2.imread("Resource/elain3.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)

lena = cv2.cvtColor(cv2.imread("Resource/lena.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)
lena1 = cv2.cvtColor(cv2.imread("Resource/lena1.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)
lena2 = cv2.cvtColor(cv2.imread("Resource/lena2.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)
lena4 = cv2.cvtColor(cv2.imread("Resource/lena4.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)

woman = cv2.cvtColor(cv2.imread("Resource/woman.BMP", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)
woman1 = cv2.cvtColor(cv2.imread("Resource/woman1.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)
woman2 = cv2.cvtColor(cv2.imread("Resource/woman2.bmp", flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2GRAY)

# ==================================================绘制直方图===========================================================
# %% citywall
plt.subplot(121), plt.imshow(citywall, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("citywall")
plt.subplot(122), plt.hist(citywall.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% citywall1
plt.subplot(121), plt.imshow(citywall1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("citywall1")
plt.subplot(122), plt.hist(citywall1.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% citywall2
plt.subplot(121), plt.imshow(citywall2, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("citywall2")
plt.subplot(122), plt.hist(citywall2.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% elain
plt.subplot(121), plt.imshow(elain, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("elain")
plt.subplot(122), plt.hist(elain.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% elain1
plt.subplot(121), plt.imshow(elain1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("elain1")
plt.subplot(122), plt.hist(elain1.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% elain2
plt.subplot(121), plt.imshow(elain2, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("elain2")
plt.subplot(122), plt.hist(elain2.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% elain3
plt.subplot(121), plt.imshow(elain3, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("elain3")
plt.subplot(122), plt.hist(elain3.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% lena
plt.subplot(121), plt.imshow(lena, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("lena")
plt.subplot(122), plt.hist(lena.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% lena1
plt.subplot(121), plt.imshow(lena1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("lena1")
plt.subplot(122), plt.hist(lena1.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% lena2
plt.subplot(121), plt.imshow(lena2, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("lena2")
plt.subplot(122), plt.hist(lena2.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% lena4
plt.subplot(121), plt.imshow(lena4, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("lena4")
plt.subplot(122), plt.hist(lena4.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% woman
plt.subplot(121), plt.imshow(woman, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("woman")
plt.subplot(122), plt.hist(woman.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% woman1
plt.subplot(121), plt.imshow(woman1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("woman1")
plt.subplot(122), plt.hist(woman1.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# %% woman2
plt.subplot(121), plt.imshow(woman2, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("woman2")
plt.subplot(122), plt.hist(woman2.ravel(), bins=256, range=(0, 255)), plt.title("histogram")
plt.show()

# ================================histogram equalization(直方图均衡)=====================================================
# %% citywall
plt.subplot(121), plt.imshow(citywall, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("citywall")
plt.subplot(122), plt.imshow(cv2.equalizeHist(citywall), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% citywall1
plt.subplot(121), plt.imshow(citywall1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("citywall1")
plt.subplot(122), plt.imshow(cv2.equalizeHist(citywall1), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% citywall2
plt.subplot(121), plt.imshow(citywall2, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("citywall2")
plt.subplot(122), plt.imshow(cv2.equalizeHist(citywall2), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% elain
plt.subplot(121), plt.imshow(elain, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("elain")
plt.subplot(122), plt.imshow(cv2.equalizeHist(elain), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% elain1
plt.subplot(121), plt.imshow(elain1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("elain1")
plt.subplot(122), plt.imshow(cv2.equalizeHist(elain1), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% elain2
plt.subplot(121), plt.imshow(elain2, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("elain2")
plt.subplot(122), plt.imshow(cv2.equalizeHist(elain2), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% elain3
plt.subplot(121), plt.imshow(elain3, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("elain3")
plt.subplot(122), plt.imshow(cv2.equalizeHist(elain3), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% lena
plt.subplot(121), plt.imshow(lena, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("lena")
plt.subplot(122), plt.imshow(cv2.equalizeHist(lena), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% lena1
plt.subplot(121), plt.imshow(lena1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("lena1")
plt.subplot(122), plt.imshow(cv2.equalizeHist(lena1), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% lena2
plt.subplot(121), plt.imshow(lena2, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("lena2")
plt.subplot(122), plt.imshow(cv2.equalizeHist(lena2), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% lena4
plt.subplot(121), plt.imshow(lena4, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("lena4")
plt.subplot(122), plt.imshow(cv2.equalizeHist(lena4), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% woman
plt.subplot(121), plt.imshow(woman, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("woman")
plt.subplot(122), plt.imshow(cv2.equalizeHist(woman), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% woman1
plt.subplot(121), plt.imshow(woman1, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("woman1")
plt.subplot(122), plt.imshow(cv2.equalizeHist(woman1), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()

# %% woman2
plt.subplot(121), plt.imshow(woman2, cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title("woman2")
plt.subplot(122), plt.imshow(cv2.equalizeHist(woman2), cmap='gray'), plt.xticks([]), plt.yticks([]), plt.title(
    "equalization")
plt.show()


# ==========================================histogram matching(直方图匹配)===============================================
# %% 图像显示函数
def display_image(images, titles):
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


# %% citywall1匹配citywall
matchCitywall1WithCitywall = match_histograms(image=citywall1, reference=citywall, multichannel=False)
images = [citywall1, matchCitywall1WithCitywall]
titles = ['input:citywall1', 'matched(ref:citywall)']
display_image(images, titles)

# %% citywall2匹配citywall
matchCitywall2WithCitywall = match_histograms(image=citywall2, reference=citywall, multichannel=False)
images = [citywall2, matchCitywall2WithCitywall]
titles = ['input:citywall2', 'matched(ref:citywall)']
display_image(images, titles)

# %% elain1匹配elain
matchElain1WithElain = match_histograms(image=elain1, reference=elain, multichannel=False)
images = [elain1, matchElain1WithElain]
titles = ['input:elain1', 'matched(ref:elain)']
display_image(images, titles)

# %% elain2匹配elain
matchElain2WithElain = match_histograms(image=elain2, reference=elain, multichannel=False)
images = [elain2, matchElain2WithElain]
titles = ['input:elain2', 'matched(ref:elain)']
display_image(images, titles)

# %% elain3匹配elain
matchElain3WithElain = match_histograms(image=elain3, reference=elain, multichannel=False)
images = [elain3, matchElain3WithElain]
titles = ['input:elain3', 'matched(ref:elain)']
display_image(images, titles)

# %% lena1匹配lena
matchLena1WithLena = match_histograms(image=lena1, reference=lena, multichannel=False)
images = [lena1, matchLena1WithLena]
titles = ['input:lena1', 'matched(ref:lena)']
display_image(images, titles)

# %% lena2匹配lena
matchLena2WithLena = match_histograms(image=lena2, reference=lena, multichannel=False)
images = [lena2, matchLena2WithLena]
titles = ['input:lena2', 'matched(ref:lena)']
display_image(images, titles)

# %% lena4匹配lena
matchLena4WithLena = match_histograms(image=lena4, reference=lena, multichannel=False)
images = [lena4, matchLena4WithLena]
titles = ['input:lena4', 'matched(ref:lena)']
display_image(images, titles)

# %% woman1匹配woman
matchWoman1WithWoman = match_histograms(image=woman1, reference=woman, multichannel=False)
images = [woman1, matchWoman1WithWoman]
titles = ['input:woman1', 'matched(ref:woman)']
display_image(images, titles)

# %% woman2匹配woman
matchWoman2WithWoman = match_histograms(image=woman2, reference=woman, multichannel=False)
images = [woman2, matchWoman2WithWoman]
titles = ['input:woman2', 'matched(ref:woman)']
display_image(images, titles)

# ==================================================局部直方图增强========================================================
# %% 局部直方图增强elain
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(7, 7))
claneElain = clahe.apply(elain)
images = [elain, claneElain]
titles = ['elain', 'elain after local histogram enhancement']
display_image(images, titles)

# %% 局部直方图增强lena
claneLena = clahe.apply(lena)
images = [lena, claneLena]
titles = ['lena', 'lena after local histogram enhancement']
display_image(images, titles)

# =======================================================直方图分割======================================================
# %% elain分割（尝试）
segElain, ax = try_all_threshold(elain, figsize=(10, 8), verbose=True)
plt.show()

# %% woman分割（尝试）
segWoman, ax = try_all_threshold(woman, figsize=(10, 8), verbose=True)
plt.show()

# %% elain分割
thresholdElain = threshold_otsu(elain)
segElainBinary = elain > thresholdElain
images = [elain, segElainBinary]
titles = ['elain', 'elain after segmentation']
display_image(images, titles)
# %% woman分割
thresholdWoman = threshold_otsu(woman)
segWomanBinary = woman > thresholdWoman
images = [woman, segWomanBinary]
titles = ['woman', 'woman after segmentation']
display_image(images, titles)
