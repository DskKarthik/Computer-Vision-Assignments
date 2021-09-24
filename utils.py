from matplotlib import image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def readImage(path, read_value=0):
    '''
    read_value = 0 for grayscale
    read_value = 1 for color
    '''
    return cv2.imread(path, read_value)

def resize(img, shape):
    return cv2.resize(img, shape)

def plot(img, title = None, cmap = "gray", axis = True, subplot=None):
    if subplot is not None:
        plt.subplot(subplot)
    plt.imshow(img, cmap= cmap)
    if title is not None:
        plt.title(title)
    if not axis:
        plt.axis('off')

def showImages():
    plt.show()

def color_to_gray(img):
    h = img.shape[0]
    w = img.shape[1]
    img_gray = np.empty([h,w])

    for i in range(h):
        for j in range(w):
            img_gray[i][j] = int(sum(img[i][j])/3)
    
    return img_gray

def addPadding(img, desired_size):
    padded_image = np.zeros((desired_size, desired_size), dtype=complex)
    k = img.shape[0]
    a = (desired_size - k)//2

    padded_image[a:a+k , a:a+k] = img
    return padded_image

def ft(img):
    '''
    Transforms image to frequency domain
    and shifts origin better vizualization
    '''
    f = np.fft.fft2(img)
    fourier = np.fft.fftshift(f)
    return fourier

def getMagnitudeSpectrum(img, factor = 1):
    return factor*np.log(1+np.abs(img))

def magnitude(img):
    return np.sqrt(np.square(np.real(img)) + np.square(np.imag(img)))
    return np.abs(img)

def phase(img):
    #return np.arctan(np.imag(img) / np.real(img))
    return np.angle(img)

def getImageFromMagPhase(mg, phase):
    return np.multiply(mg, np.exp(1j*phase))

def ift(img, shifted = False):
    spatial = np.fft.ifft2(img)
    if shifted:
        spatial = np.fft.ifftshift(spatial)
    spatial = np.abs(spatial).astype('uint8')
    return spatial

def downSample(img, factor):
    # return cv2.resize(img, (k,k))
    k = int(img.shape[0]/factor)

    down_sampled_img = np.zeros((k,k))
    n = img.shape[0]
    for i in range(0, n, factor):
        for j in range(0, n, factor):
                down_sampled_img[i//factor][j//factor] = img[i][j]
    return down_sampled_img

def upSample(img, factor):

    import scipy.ndimage as ni
    return ni.zoom(img, 8, order=0)

    n = img.shape[0]
    k = n*factor
    upsampled_img = np.zeros((k,k))

    for i in range(0,k,factor):
        for j in range(0,k,factor):
            upsampled_img[i][j] = img[i//factor][j//factor]
    
    for i in range(1, k-factor, factor):
        for j in range(k-factor):
            upsampled_img[i:i+factor-1][j] = upsampled_img[i-1,j]
    
    return upsampled_img

