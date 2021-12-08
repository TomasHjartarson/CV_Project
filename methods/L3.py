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
#                   HELPING FUNCTIONS

#Pixel CAT
def getsamples(img):
    x, y, z = img.shape
    samples = np.empty([x * y, z])
    index = 0
    for i in range(x):
        for j in range(y):
            samples[index] = img[i, j]
            index += 1
    return samples


"""
    Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. "Maximum likelihood
    from incomplete data via the EM algorithm." Journal of the royal statistical
    society. Series B (methodological) (1977): 1-38.
"""
def EMSegmentation(img, no_of_clusters=2):
    output = img.copy() # BLUE           #RED
    colors = np.array([[0, 11, 111], [233, 16, 16]])
    samples = getsamples(img)
    em = cv2.ml.EM_create()
    em.setClustersNumber(no_of_clusters)
    em.trainEM(samples)
    means = em.getMeans()
    covs = em.getCovs() 
    x, y, z = img.shape
    distance = [0] * no_of_clusters
    for i in range(x):
        for j in range(y):
            for k in range(no_of_clusters):
                diff = img[i, j] - means[k]
                distance[k] = abs(np.dot(np.dot(diff, covs[k]), diff.T))
            output[i][j] = colors[distance.index(max(distance))]
    
    Object_col = [0] * no_of_clusters
    for i in range(348):
        for j in range(200,400):
            if(output[i][j][0] == colors[0][0]):
                Object_col[0] += 1
            else:
                Object_col[1] += 1
    
    Colour_obj_ind = Object_col.index(max(Object_col))
    lower = np.array(colors[Colour_obj_ind] - 1)   
    upper = np.array(colors[Colour_obj_ind] + 1)
    mask = cv2.inRange(output, lower, upper)
    masked_image = cv2.bitwise_and(img, img, mask=mask)        
    
    
    if(Erosion):
        img_erosion = cv2.erode(masked_image, kernel, Erosion_it=1)
    else:
        img_erosion = masked_image
        
    if(Dilusion):
        img_dilation = cv2.dilate(masked_image, kernel, Dilusion_it=1)
    else:
        img_dilation = masked_image
                       
    return img_dilation


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
#                   SCRIPT FUNCTION

#NOTICE: CAMERA image size currently set : 512x348

### PIXEL CATEGORIZATION
"""
    Each pixel is classified into one of many classes, c, based on its local features: the
    identity of the pixel, the mean response level of the local patch,
    the spatial variance of the neighborhood, and pixel saturation
    status, etc.
    https://arxiv.org/pdf/1605.09336.pdf
    https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
"""
### LEARNING LINEAR TRANSFORMATIONS
"""

"""
### WEIGHTED SUMMATION
"""
"""

#########################################################
#########################################################
#########################################################
#                   PROCESSING

#-----------------------
#PIXEL CATEGORIZATION:
Pixel_cat = True

NO_Clusters = 2

#Erosion and Dilusion (And kernel)
kernel = np.ones((2,2), np.uint8)

Erosion = False
Erosion_it = 1

Dilusion = False
Dilusion_it = 1

#-----------------------
#LEARNING LINEAR TRANSFORMATIONS:
LLT = True 

#-----------------------
#WEIGHTED SUMMATION:
Weighted_sum = True 



#########################################################
#########################################################
#########################################################
#                   FUNCTIONS

def _Pixel_cat(image):
    print('Pixel categorization')
    Segmented_image = EMSegmentation(image, NO_Clusters)
    return Segmented_image
    
def _LLT(image):
    print('Learning linear transformations')
    return image
    
def _Weighted_sum(image):
    print('Weighted sum')
    return image

### Directory of images
DIR = r'C:\Users\tomas\OneDrive\Documents\HR\CV\Project\Dummy_set'


def Process():
    ##Jump around images
    for sub_folder in os.listdir(DIR):
        Folder = DIR + "/" + sub_folder
        for file in os.listdir(Folder):
            ### DO SOMETHING TO FILE
            start_time = time.time()
            print('### New image ###')
            image = cv2.imread(Folder + '/' + file)
            
            
            if(Pixel_cat):
                Pixel_cat_image = _Pixel_cat(image)
            else:
                Pixel_cat_image = image 
                
            if(LLT):
                LLT_image = _LLT(Pixel_cat_image)
            else:
                LLT_image = Pixel_cat_image 
                
            if(Weighted_sum):
                Weighted_sum_image = _Weighted_sum(LLT_image)
            else:
                Weighted_sum_image = LLT_image 
                
                
            #Uncomment to view original vs modified image (change second parameter after which modification you want the view)
            print("--- %s seconds ---" % (time.time() - start_time))
            view_image(image,Weighted_sum_image)



if __name__ == '__main__':
    Process()
