import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import color_to_gray
from filters import Filters

img = cv2.imread("assets/image1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(121)

plt.imshow(img)
plt.title('Color Image')
plt.axis('off')

print("Converting Color Image to Grayscale..")
img = color_to_gray(img)

plt.subplot(122)
plt.imshow(img, cmap='gray')
plt.title('Gray Scale Image')
plt.axis('off')
plt.show()

# ============================================================

plt.subplot(131)
plt.imshow(img,cmap='gray')
plt.title("Original Image")
plt.axis('off')

print("Applying 3X3 Mean and Gaussian Filters")

plt.subplot(132)
mf = Filters.MeanFilter(img, kernel_size=3)
plt.imshow(mf.mean_filtered_image, cmap='gray')
plt.title(f'Mean Filter - {mf.kernel_size}X{mf.kernel_size}')
plt.axis('off')

plt.subplot(133)
gf = Filters.GaussianFilter(img, kernel_size=3)
plt.imshow(gf.gaussian_filtered_img, cmap='gray')
plt.title(f'Gaussian Filter - {gf.kernel_size}X{gf.kernel_size}')
plt.axis('off')
plt.show()

# ============================================================

print("Applying 3X3, 5X5, 7X7 and 9X9 kernels of mean filters...")

plt.subplot(141)
mf = Filters.MeanFilter(img, kernel_size=3)
plt.imshow(mf.mean_filtered_image, cmap='gray')
plt.title(f'Mean Filter - {mf.kernel_size}X{mf.kernel_size}')
plt.axis('off')

plt.subplot(142)
mf = Filters.MeanFilter(img, kernel_size=5)
plt.imshow(mf.mean_filtered_image, cmap='gray')
plt.title(f'Mean Filter - {mf.kernel_size}X{mf.kernel_size}')
plt.axis('off')

plt.subplot(143)
mf = Filters.MeanFilter(img, kernel_size=7)
plt.imshow(mf.mean_filtered_image, cmap='gray')
plt.title(f'Mean Filter - {mf.kernel_size}X{mf.kernel_size}')
plt.axis('off')

plt.subplot(144)
mf = Filters.MeanFilter(img, kernel_size=9)
plt.imshow(mf.mean_filtered_image, cmap='gray')
plt.title(f'Mean Filter - {mf.kernel_size}X{mf.kernel_size}')
plt.axis('off')

plt.show()

# ==========================================================

print("Applying 3x3 Sharpening filter..")
# img =color_to_gray( cv2.imread('assets/image2.jpg'))

sf = Filters.SharpeningFliter(img, kernel_size=3)
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(122)
plt.imshow(sf.sharpened_image, cmap='gray')
plt.title("Sharpened Image")
plt.axis('off')

plt.show()