import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.filters import gaussian
from scipy.ndimage.filters import convolve
from skimage import io
from skimage.filters import median
from skimage.restoration import denoise_nl_means, estimate_sigma
from PIL import Image
from PIL import ImageEnhance

#reading a image from computer and taking dimensions
# img = cv2.imread('pic2.jpeg')
img = cv2.imread('pic3.png')
# img = cv2.imread('https://pbs.twimg.com/media/CdGOqAiW0AAShOM?format=jpg&name=small')
rows, cols = img.shape[:2]


image = Image.open('pic3.png')
# img = cv2.imread('https://pbs.twimg.com/media/CdGOqAiW0AAShOM?format=jpg&name=small')

  
# shows image in image viewer
# image.show()  
# Enhance Brightness


#Kernel Blurring using filter2D()
# kernel_25 = np.ones((25,25), np.float32) / 625.0
# output_kernel = cv2.filter2D(img, -1, kernel_25)

#Boxfilter and blur function blurring
# output_blur = cv2.blur(img, (25,25))
# output_box = cv2.boxFilter(img, -1, (5,5), normalize=False)

#gaussian Blur 
# output_gaus = cv2.GaussianBlur(img, (5,5), 0)

# #median Bur (reduction of noise)
# output_med = cv2.medianBlur(img, 5)

# #Bilateral filtering (Reduction of noise + Preserving of edges)
# output_bil = cv2.bilateralFilter(img, 5, 6, 6)

# # cv2.imshow('kernel blur', output_kernel)
# # cv2.imshow('Blur() output', output_blur)
# # cv2.imshow('Box filter', output_box)
# cv2.imshow('Gaussian', output_gaus)
# cv2.imshow('Bilateral', output_bil)
# cv2.imshow('Median Blur', output_med)
# cv2.imshow('Original', img)

# gaussian_kernel = np.array([[1/16, 1/8, 1/16],   #3x3 kernel
                # [1/8, 1/4, 1/8],
#                 [1/16, 1/8, 1/16]])

# conv_using_cv2 = cv2.filter2D(img, -1, gaussian_kernel, borderType=cv2.BORDER_CONSTANT) 
# when ddepth=-1, the output image will have the same depth as the source
#example, if input is float64 then output will also be float64
# BORDER_CONSTANT - Pad the image with a constant value (i.e. black or 0)
#BORDER_REPLICATE: The row or column at the very edge of the original is replicated to the extra border.


curr_bri = ImageEnhance.Color(image)
new_bri = 2.5
  
# Brightness enhanced by a factor of 2.5
img_brightened = curr_bri.enhance(new_bri)
img_brightened.show()  
img_brightened = img.astype(np.uint8)
gaussian_using_cv2 = cv2.GaussianBlur(img_brightened, (3,3), 0, borderType=cv2.BORDER_CONSTANT)

# gaussian_using_skimage = gaussian(img, sigma=1, mode='constant', cval=0.0)
#sigma defines the std dev of the gaussian kernel. SLightly different than 
#how we define in cv2


# sigma_est = np.mean(estimate_sigma(img, multichannel=True))
# #sigma_est = 0.1

# denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
#                                patch_size=5, patch_distance=3, multichannel=False)

cv2.imshow("Original", img)
# cv2.imshow("cv2 filter", conv_using_cv2)
cv2.imshow("Using cv2 gaussian", gaussian_using_cv2)
# cv2.imshow("Using skimage", gaussian_using_skimage)
# cv2.imshow("NLM Filtered", denoise_img)
#cv2.imshow("Using scipy2", conv_using_scipy2)

cv2.waitKey(0)          
cv2.destroyAllWindows()

  
# Opens the image file

  
  

  
# shows updated image in image viewer
# 
# from pgmagick import Image
 
# img = Image('noisy.jpg')
 
# # sharpening image
# img.sharpen(2)
# img.write('sharp_koala.jpg')