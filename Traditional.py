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



#########################################################
#########################################################
#########################################################
#                   SCRIPT FUNCTION

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
#                   PROCESSING

#-----------------------
#White balancing:
White_balance = True
#Gray World Algorithm (True) or Ground Truth Algorithm (False)
white_gray_alg = False

#Required for Ground Truth Algorithm 
from_row =  200
from_column = 300
row_width =  75
column_width = 75

#-----------------------
#Demosaic:
Demosaic = True 

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

Malvar = False

#Menon
"""

"""
Menon = True

#-----------------------
#Denoise:
Denoise = True 

#-----------------------
#Sharpen
Sharpen = True 

#-----------------------
#Color_space_conversion:
Color_space_conversion = True

#-----------------------
#Gamma correction:
Gamma_correction = True  
#-----------------------

#########################################################
#########################################################
#########################################################
#                   FUNCTIONS

def view_image(image,changed_image):      
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    
    ax[1].imshow(changed_image)
    ax[1].set_title('Modified image')
    plt.show()
    return
            


def _white_balancing(image,height,size):
    #2 implemended methods
    
    #Gray World Algorithm
    if(white_gray_alg):
        white_balanced_image = ((image * image.mean() / image.mean(axis=(0,1))).clip(0, 255).astype(int))
    
    #Ground Truth Algorithm
    else:
        image_patch = image[from_row:from_row+row_width, from_column:from_column+column_width]
        white_balanced_image = ((image*1.0) / image_patch.max(axis=(0,1))).clip(0,1)
        
    return white_balanced_image     
        
def _Demosaic(image):
    CFA = mosaicing_CFA_Bayer(image)
    
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
    return Denoised_image

def _Color_space_conversion(image):
    
    return CSV_image
def _Gamma_correction(image):
    
    return gamma_corrected_image



### Directory of images
DIR = r'C:\Users\tomas\OneDrive\Documents\HR\CV\Project\Dummy_set'


def Process():
    ##Jump around images
    
    for sub_folder in os.listdir(DIR):

        Folder = DIR + "/" + sub_folder
        for file in os.listdir(Folder):
            
            image = cv2.imread(Folder + '/' + file)
            hT,wT,cT = image.shape
            
            #APPLY WHITE BALANCE
            if(White_balance):
                white_balanced_image = _white_balancing(image,hT,wT)
            else:
                white_balanced_image = image
            
            #APPLY DEMOSAIC
            if(Demosaic):
                Demosaic_image = _Demosaic(white_balanced_image)
            else:
                Demosaic_image = white_balanced_image
            
            #APPLY DENOISE
            if(Denoise):
                ...
                
            #APPLY COLOR_SPACE_CONVERSION
            if(Color_space_conversion):
                ...
                
            #APPLY GAMMA CORRECTION
            if(Gamma_correction):
                ...
                
            #Uncomment to view original vs modified image (change second parameter after which modification you want the view)
            view_image(image,Demosaic_image)
            







if __name__ == '__main__':
    Process()