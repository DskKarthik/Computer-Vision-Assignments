import math
import numpy as np

class Filters:

    @classmethod
    def pad_image(cls, img, kernel_size):
        k = kernel_size//2

        img_h = img.shape[0]
        img_w = img.shape[1]

        img_padded = np.empty( (img_h + 2*k, img_w + 2*k) )
        img_padded[k:-k,k:-k] = img

        return img_padded

    @classmethod
    def convolution(cls,kernel, img):
        
        img_h = img.shape[0]
        img_w = img.shape[1]

        k = kernel.shape[0]

        img_padded = cls.pad_image(img, k)

        filtered_img = np.empty(img.shape)
        for h in range(img_h):
            for w in range(img_w):
                local_img = img_padded[h : h+k, w : w+k]
                filtered_img[h,w] = np.sum(local_img*kernel)
        
        return filtered_img

    class MeanFilter:
        
        def __init__(self,img,kernel_size = 3):
            self.kernel_size = kernel_size
            self.img = img
            
            kernel = self.create_kernel()
            #print(kernel)
            self.mean_filtered_image = Filters.convolution(kernel, self.img)

        def create_kernel(self):
            k = self.kernel_size
            kernel = (1/(k**2))*np.ones((k,k))
            return kernel

    class GaussianFilter:
        
        def __init__(self, img, kernel_size=3, sigma=1):
            self.img = img
            self.kernel_size = kernel_size
            if sigma == 1:
                self.sigma = math.sqrt(kernel_size)
            else:
                self.sigma = sigma

            kernel = self.create_kernel()
            #print(kernel)
            self.gaussian_filtered_img = Filters.convolution(kernel, self.img)
        
        def gaussian(self, x, y, sigma):
            return 1/(2*np.pi*sigma) * np.exp(-(np.square(x) + np.square(y))/(2*np.square(sigma)))

        def create_kernel(self):
            
            ax = np.linspace(-1,1, self.kernel_size)
            x,y = np.meshgrid(ax,ax)

            self.kernel = self.gaussian(x,y,self.sigma)
            return self.kernel

    class SharpeningFliter:
        
        def __init__(self, img, kernel_size=3) -> None:
            
            self.kernel_size = kernel_size
            self.img = img
            
            kernel = self.create_kernel()

            self.sharpened_image = Filters.convolution(kernel, self.img)

            oldMin = self.sharpened_image.min()
            oldMax = self.sharpened_image.max()

            print(oldMin, oldMax)

            newMin = 0
            newMax = 255

            oldRange = (oldMax - oldMin)  
            newRange = (newMax - newMin)

            mapped_img = np.empty(self.sharpened_image.shape)

            for i in range(self.sharpened_image.shape[0]):
                for j in range(self.sharpened_image.shape[1]):  
                    mapped_img[i,j] = int((((self.sharpened_image[i,j] - oldMin) * newRange) / oldRange) + newMin)

            self.sharpened_image = mapped_img.copy()
            print(self.sharpened_image.min(), self.sharpened_image.max())

        
        def create_kernel(self):

            a = np.zeros((self.kernel_size, self.kernel_size))
            a[int(self.kernel_size/2), int(self.kernel_size/2)] = 5
            mean_filter = (4/(self.kernel_size**2))*np.ones((self.kernel_size, self.kernel_size))
            kernel = np.subtract(a, mean_filter)
            return kernel
    
    class SobelFilter:

        def vertical_3by3():
            kernel = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ])

            return kernel
        

        def horizontal_3by3():
            kernel = np.array([
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ])

            return kernel
        
        @classmethod
        def getHorVert(cls, gx, gy):
            return np.log(1+np.abs(np.square(np.square(gx) + np.square(gy))))