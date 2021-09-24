from utils import addPadding, downSample, ft, getImageFromMagPhase, ift, getMagnitudeSpectrum, magnitude, phase, plot, readImage, showImages, resize, upSample
from filters import Filters

# ============================= Question 1 ============================================

img = readImage("assets/image1.png", 0)
plot(img, title='Original Image', axis=False, subplot=231)

print("Converting Image to Frequency Domain")
image_fourier = ft(img)
mag_spectrum_image = getMagnitudeSpectrum(image_fourier)
plot(mag_spectrum_image, title="FT of Image", axis=False, subplot=234)

gf = Filters.GaussianFilter(img, kernel_size=3)
gaussian_filter = gf.kernel
k = gf.kernel_size
plot(gaussian_filter, title=f"{k}X{k} Gaussian Filter", axis=False, subplot=232)

gaussian_filter = addPadding(gaussian_filter, img.shape[0])
print("Converting Image to Frequency Domain")
gaussian_filter_fourier = ft(gaussian_filter)
mag_spectrum_filter = getMagnitudeSpectrum(gaussian_filter_fourier)
plot(mag_spectrum_filter, title=f" FT of {k}X{k} Gaussian Filter", axis=False, subplot=235)

print("Multiplying Image with Filter ( in frequency domain )")
gaussian_filtered_image_fourier = image_fourier*gaussian_filter_fourier

mag_spectrum_result = getMagnitudeSpectrum(gaussian_filtered_image_fourier)
plot(mag_spectrum_result, title=" FT of Gaussian Filtered Image", axis=False, subplot=236)

print("Applying Inverse Fourier Transform to resukting image")
gaussian_filtered_image_spatial = ift(gaussian_filtered_image_fourier, shifted = True)
plot(gaussian_filtered_image_spatial, title="Gaussian Filtered Image", axis = False, subplot=233)

showImages()

# ============================= Question 2 ============================================ 

plot(gaussian_filtered_image_spatial, title="Previous Result", axis=False, subplot=231)

gaussian_filtered_image_fourier = ft(gaussian_filtered_image_spatial)
mag_spectrum_result = getMagnitudeSpectrum(gaussian_filtered_image_fourier)
plot(mag_spectrum_result, title="Inverse FT of Gaussian Image", axis=False, subplot=234)

sobel_vertical_filter = Filters.SobelFilter.vertical_3by3()
plot(sobel_vertical_filter, title="Sobel Vertical Filter", axis = False, subplot=232)

sobel_vertical_filter = addPadding(sobel_vertical_filter, gaussian_filtered_image_spatial.shape[0])
sobel_vertical_fourier = ft(sobel_vertical_filter)
mag_spectrum_sobel_vertical = getMagnitudeSpectrum(sobel_vertical_fourier)
plot(mag_spectrum_sobel_vertical, title="FT of Sobel Vertical Filter", axis = False, subplot=235)

sobel_vert_fil_image_fourier = gaussian_filtered_image_fourier * sobel_vertical_fourier
mag_spectrum_result = getMagnitudeSpectrum(sobel_vert_fil_image_fourier)
plot(mag_spectrum_result, title="FT of Sobel Vertial Filtered Image", axis=False, subplot=236)

sobel_vert_fil_image_spatial = ift(sobel_vert_fil_image_fourier, shifted=True)
plot(sobel_vert_fil_image_spatial, title="Sobel Vertical Filtered Image", axis=False, subplot=233)

showImages()


plot(gaussian_filtered_image_spatial, title="Previous Result", axis=False, subplot=231)

gaussian_filtered_image_fourier = ft(gaussian_filtered_image_spatial)
mag_spectrum_result = getMagnitudeSpectrum(gaussian_filtered_image_fourier)
plot(mag_spectrum_result, title="Inverse FT of Gaussian Image", axis=False, subplot=234)

sobel_hor_filter = Filters.SobelFilter.horizontal_3by3()
plot(sobel_hor_filter, title="Sobel Horizontal Filter", axis = False, subplot=232)

sobel_hor_filter = addPadding(sobel_hor_filter, gaussian_filtered_image_spatial.shape[0])
sobel_hor_fourier = ft(sobel_hor_filter)
mag_spectrum_sobel_hor = getMagnitudeSpectrum(sobel_hor_fourier)
plot(mag_spectrum_sobel_hor, title="FT of Sobel Horizontal Filter", axis = False, subplot=235)

sobel_hor_fil_image_fourier = gaussian_filtered_image_fourier * sobel_hor_fourier
mag_spectrum_result = getMagnitudeSpectrum(sobel_hor_fil_image_fourier)
plot(mag_spectrum_result, title="FT of Sobel Horizontal Filtered Image", axis=False, subplot=236)

sobel_hor_fil_image_spatial = ift(sobel_hor_fil_image_fourier, shifted=True)
plot(sobel_hor_fil_image_spatial, title="Sobel Horizontal Filtered Image", axis=False, subplot=233)

showImages()

# ============================= Question 3============================================ 
img1 = resize(readImage('assets/dog.jpg', 0), (256,256))
img2 = resize(readImage('assets/cat.jpg', 0), (256,256))

plot(img1, axis = False, subplot=221)
plot(img2, axis = False, subplot=223)

img1_fourier = ft(img1)
img2_fourier = ft(img2)



img1_mag = magnitude(img1_fourier)
img1_phase = phase(img1_fourier)

img2_mag = magnitude(img2_fourier)
img2_phase = phase(img2_fourier)

hybrid1_fourier = getImageFromMagPhase(img1_mag, img2_phase)
hybrid2_fourier = getImageFromMagPhase(img2_mag, img1_phase)

hybrid1 = ift(hybrid1_fourier)
hybrid2 = ift(hybrid2_fourier)

plot(hybrid1, axis=False, subplot=224, title="Hybrid 2 (M2 + P1)")
plot(hybrid2, axis=False, subplot=222, title="Hybrid 1 (M1 + P2)")

showImages()

# ============================= Question 4============================================ 

# Apply Gaussian Filter to image
image_fourier = ft(img)
gf = Filters.GaussianFilter(img, kernel_size=3)
gaussian_filter = gf.kernel
k = gf.kernel_size

gaussian_filter = addPadding(gaussian_filter, img.shape[0])
gaussian_filter_fourier = ft(gaussian_filter)
gaussian_filtered_image_fourier = image_fourier*gaussian_filter_fourier
gaussian_filtered_image_spatial = ift(gaussian_filtered_image_fourier, shifted = True)

plot(gaussian_filtered_image_spatial, title = '1. Original (Gaussian Filtered)', subplot=141)

#Down Sample Image by factor of 8 times
down = downSample(gaussian_filtered_image_spatial, 8)
plot(down, title = "2. Down sample to 1/8 of (1)", subplot=142)

#Up Sample Image by factor of 8
up = upSample(down, 8)
plot(up, title="3. Up scale (2) to previous size", subplot=143)

# Difference between original and Upscaled Image 
plot(gaussian_filtered_image_spatial - up, title="4. (1) - (3)", subplot=144)
showImages()