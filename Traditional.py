import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import colour
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

from scipy.signal import convolve2d
from skimage.color import rgb2yuv, yuv2rgb
from skimage import img_as_ubyte

#########################################################
#########################################################
#########################################################
#                   SCRIPT FUNCTION

#NOTICE: CAMERA image size currently set : 512x348
### WHITE BALANCING
"""
    White balance blaances the color temperature of the image.
"""
### DEOMSAIC
"""
    digital image process used to reconstruct a full color image from the 
    incomplete color samples output from an image sensor overlaid with a 
    color filter array.
"""
### DENOISE, SHARPEN
"""
    Denoising is done to remove unwanted noise from image to analyze it in better form. 
    Denoising an image also causes the image to lose some sharpness.
    
    image sharpening consists of adding to the original image a signal that is proportional 
    to a high-pass filtered version of the original image. 
    https://nptel.ac.in/content/storage2/courses/117104069/chapter_8/8_32.html
    https://towardsdatascience.com/image-processing-with-python-blurring-and-sharpening-for-beginners-3bcebec0583a
"""
### COLOR SPACE CONVERSION
"""
    Color space conversion is the translation of the representation of a color from one basis to another
"""
### GAMMA CORRECTION
"""
    Gamma correction or gamma is a nonlinear operation used to encode and 
    decode luminance or tristimulus values. 
    https://en.wikipedia.org/wiki/Gamma_correction
"""

#########################################################
#########################################################
#########################################################
#                   HELPING FUNCTIONS

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def multi_convolver(image, kernel, iterations):
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary = 'fill',
                           fillvalue = 0)
    return image

def convolver_rgb(image, kernel, iterations = 1):
        img_yuv = rgb2yuv(image)
        img_yuv[:,:,0] = multi_convolver(img_yuv[:,:,0], kernel, 
                                     iterations)
        final_image = yuv2rgb(img_yuv)
        return final_image

def view_image(image,changed_image):      
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    
    ax[1].imshow(changed_image)
    ax[1].set_title('Modified image')
    plt.show()
    return


#########################################################
#########################################################
#########################################################
#                   PROCESSING

#-----------------------
#White balancing:
White_balance = True
#Gray World Algorithm (True) or Ground Truth Algorithm (False)
white_gray_alg = False

#Required for Ground Truth Algorithm 
from_row =  240
from_column = 150
row_width =  100
column_width = 100

#Current good values : 240 150 100 100

#-----------------------
#Demosaic:
Demosaic = True 

demo_filter = 'BGGR'
#Use one of the following methods (Mark True)

#Bilinear
"""
Bilinear takes each of the three colour images (R,G,B) independently and uses bilinear interpolation on each image.
Taking the average of each of the neighbouring pixel to estimate the missing value. 
https://smartech.gatech.edu/bitstream/handle/1853/22542/appia_vikram_v_200805_mast.pdf
"""
Bilinear = False

#Malvar
"""
Derived as a modification of bilinear interpolation (to complex to add here please see link)
https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf 
"""

Malvar = True

#Menon
"""
http://elynxsdk.free.fr/ext-docs/Demosaicing/todo/Menon_Andriani_IEEE_T_IP_2007.pdf
"""
Menon = False

#Current good values : Malvar

#-----------------------
#Denoise:
Denoise = True 

denoise_h = 7
denoise_hcolor = 10
Templatewindowsize = 7
Searchwindowsize = 21

#Current good values : 5 10 7 21

#-----------------------
#Sharpen
Sharpen = True 

# Sharpen
sharpen_array = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])

sharpen_iteration = 1
#-----------------------
#Color_space_conversion:
Color_space_conversion = False


#-----------------------
#Gamma correction:
Gamma_correction = True  

#gamma = 1 doesn't do anything
#gamma = 0 not allowed will give error
gamma = 1.2

#Current good value : 1.5 for high light images, 4 for really low light
#-----------------------
#AOI
AOI = False


#########################################################
#########################################################
#########################################################
#                   FUNCTIONS

            

def _white_balancing(image):
    print('White balancing')
    #2 implemended methods
    #Gray World Algorithm
    if(white_gray_alg):
        white_balanced_image = ((image * image.mean() / image.mean(axis=(0,1)))).clip(0,1)
    
    #Ground Truth Algorithm
    else:
        image_patch = image[from_row:from_row+row_width, from_column:from_column+column_width]
        white_balanced_image = ((image*1.0) / image_patch.max(axis=(0,1))).clip(0,1)
    
    return white_balanced_image     
        
def _Demosaic(image):
    print('Demosaicing')
    CFA = mosaicing_CFA_Bayer(image, demo_filter)
    
    if(Bilinear):
        Demosaic_image = colour.cctf_encoding(demosaicing_CFA_Bayer_bilinear(CFA)).clip(0,1)
        
    if(Malvar):
        Demosaic_image = colour.cctf_encoding(demosaicing_CFA_Bayer_Malvar2004(CFA)).clip(0,1)
        
    if(Menon):
        Demosaic_image = colour.cctf_encoding(demosaicing_CFA_Bayer_Menon2007(CFA)).clip(0,1)
    
    if(not (Menon or Malvar or Bilinear)):
        Demosaic_image = image
         
    return Demosaic_image

def _Denoise(image):
    print('Denoising')
    Denoised_image = cv2.fastNlMeansDenoisingColored(image, None, denoise_h, denoise_hcolor, Templatewindowsize, Searchwindowsize)
    return Denoised_image

def _Sharpen(image):
    print('Sharpening')
    Sharpened_image = convolver_rgb(image, sharpen_array, sharpen_iteration)
    return Sharpened_image

def _Color_space_conversion(image):
    print('Color_space_switching')
    image = image.astype('float32')
    CSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).clip(0,1)
    return CSV_image

def _Gamma_correction(image):
    print('Gamma corrected')
    gamma_corrected_image = adjust_gamma(image, gamma=gamma)
    return gamma_corrected_image


def _AOI(image):
    edges = cv2.Canny(image,1,200)
    return edges

### Directory of images
DIR = r'C:\Users\tomas\OneDrive\Documents\HR\CV\Project\Dummy_set'


def Process():
    ##Jump around images
    
    for sub_folder in os.listdir(DIR):
        Folder = DIR + "/" + sub_folder
        for file in os.listdir(Folder):
            start_time = time.time()
            print('### New image ###')
            image = cv2.imread(Folder + '/' + file)
            hT,wT,cT = image.shape

            #APPLY GAMMA CORRECTION
            if(Gamma_correction):
                gamma_corrected_image = _Gamma_correction(image)
            else:
                gamma_corrected_image = image 

            #APPLY DENOISE
            if(Denoise):
                Denoised_image = _Denoise(gamma_corrected_image)
            else:
                Denoised_image = gamma_corrected_image 

            #APPLY WHITE BALANCE
            if(White_balance):
                white_balanced_image = _white_balancing(Denoised_image)
            else:
                white_balanced_image = Denoised_image 

            #APPLY SHARPENING
            if(Sharpen):
                Sharpened_image = _Sharpen(white_balanced_image)
            else:
                Sharpened_image = white_balanced_image
  
            #APPLY DEMOSAIC
            if(Demosaic):
                Demosaic_image = _Demosaic(Sharpened_image)
                Demosaic_image = np.asarray(Demosaic_image)
            else:
                Demosaic_image = Sharpened_image
                Demosaic_image = np.asarray(Demosaic_image)

            #APPLY COLOR_SPACE_CONVERSION
            if(Color_space_conversion):
                CSV_image =  _Color_space_conversion(Demosaic_image)
            else:
                CSV_image = Demosaic_image

            #APPLY AREA OF INTEREST
            if(AOI):
                AOI_image =  _AOI(image)
            else:
                AOI_image = CSV_image               

            #Uncomment to view original vs modified image (change second parameter after which modification you want the view)
            print("--- %s seconds ---" % (time.time() - start_time))
            view_image(image,AOI_image)
            print(' ')
            


if __name__ == '__main__':
    Process()