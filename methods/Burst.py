import os
import time
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib import colors


import colour
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

from scipy.signal import convolve2d
from skimage.color import rgb2yuv, yuv2rgb
from skimage.io import imread, imshow
from skimage import img_as_ubyte

#########################################################
#########################################################
#########################################################
#                   SCRIPT FUNCTION

#NOTICE: CAMERA image size currently set : 512x348


#https://ieeexplore.ieee.org/document/8702220
#https://www.photometrics.com/learn/imaging-topics

### ALIGN & MERGE
"""
    Exposure Fusion
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.64.7616&rep=rep1&type=pdf
"""
### WHITE BALANCE, DEOMSAIC, CHROMA, DENOISE
"""
    White balance balances the color temperature of the image.
    
    Demosaic digital image process used to reconstruct a full color image from the 
    incomplete color samples output from an image sensor overlaid with a 
    color filter array.
    
    Chroma 
    https://medium.com/hd-pro/understanding-chroma-and-luminance-in-digital-imaging-f0b4d97ee157
    https://towardsdatascience.com/image-processing-with-python-using-rg-chromaticity-c585e7905818
    
    Denoising is done to remove unwanted noise from image to analyze it in better form. 
    Denoising an image also causes the image to lose some sharpness.
    #https://www.photometrics.com/learn/imaging-topics/dark-current

    
"""
### LOCAL TONE MAP
""" 
    LTM is used to widen the dynamic range of images
    The dynamic range is defined as the ratio of the intensity of
    the brightest point to the intensity of the darkest point in a scene or image.
    For natural scenes, this ratio can reach up to the order of millions.
    https://jieyang1987.github.io/files/el2018.pdf
"""
### DEHAZE, GLOBAL TONE MAP 
"""
    Dehaze removing haze can significantly increase
    the visibility of the scene and correct the color shift caused
    by the airlight. In general, the haze-free image is more visually pleasing
    https://arxiv.org/pdf/1601.07661.pdf
    https://www.researchgate.net/publication/224212806_Real-Time_Dehazing_for_Image_and_Video
    https://ieeexplore.ieee.org/document/5206515
    https://github.com/He-Zhang/image_dehaze/blob/master/dehaze.py
    
    Global Tone Map
""" 
### SHARPEN, HUE & SATURATION
"""
    image sharpening consists of adding to the original image a signal that is proportional 
    to a high-pass filtered version of the original image. 
    https://nptel.ac.in/content/storage2/courses/117104069/chapter_8/8_32.html
    https://towardsdatascience.com/image-processing-with-python-blurring-and-sharpening-for-beginners-3bcebec0583a
    
    Hue
    
    
    Saturation
"""

#########################################################
#########################################################
#########################################################
#                   HELPING FUNCTIONS

#the convolution method for SHARPENING (iteration function)
def multi_convolver(image, kernel, iterations):
    for i in range(iterations):
        image = convolve2d(image, kernel, 'same', boundary = 'fill',
                           fillvalue = 0)
    return image

#SHARPENING convolution method
def convolver_rgb(image, kernel, iterations = 1):
        img_yuv = rgb2yuv(image)
        img_yuv[:,:,0] = multi_convolver(img_yuv[:,:,0], kernel, 
                                     iterations)
        final_image = yuv2rgb(img_yuv)
        return final_image

#Views original image to current modified image
def view_image(image,changed_image):      
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    
    ax[1].imshow(changed_image)
    ax[1].set_title('Modified image')
    plt.show()
    return

#Displays red, green, blue channels (CHROMA)
def rgb_splitter(image):
    rgb_list = ['Reds','Greens','Blues']
    fig, ax = plt.subplots(1, 3, figsize=(17,7), sharey = True)
    for i in range(3):
        ax[i].imshow(image[:,:,i], cmap = rgb_list[i])
        ax[i].set_title(rgb_list[i], fontsize = 22)
        ax[i].axis('off')
    fig.tight_layout()

#Displays Red, green chromacity image (CHROMA)
def RG_Chroma_plotter(red,green):
    p_color = [(r, g, 1-r-g) for r,g in 
               zip(red.flatten(),green.flatten())]
    norm = colors.Normalize(vmin=0,vmax=1.)
    norm.autoscale(p_color)
    p_color = norm(p_color).tolist()
    fig = plt.figure(figsize=(10, 7), dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(red.flatten(), 
                green.flatten(), 
                c = p_color, alpha = 0.40)
    ax.set_xlabel('Red Channel', fontsize = 20)
    ax.set_ylabel('Green Channel', fontsize = 20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.show()


def gaussian(p,mean,std): #(CHROMA)
    return np.exp(-(p-mean)**2/(2*std**2))*(1/(std*((2*np.pi)**0.5)))

#CHROMA
def rg_chroma_patch(image, patch_coor, mean = 1, std = 1):
    patch = image[patch_coor[0]:patch_coor[1],
                  patch_coor[2]:patch_coor[3]]
    
    image_r = image[:,:,0] /image.sum(axis=2)
    image_g = image[:,:,1] /image.sum(axis=2)
    
    patch_r = patch[:,:,0] / patch.sum(axis=2)
    patch_g = patch[:,:,1] / patch.sum(axis=2)
    
    std_patch_r = np.std(patch_r.flatten())
    mean_patch_r = np.mean(patch_r.flatten())
    std_patch_g = np.std(patch_g.flatten())
    mean_patch_g = np.mean(patch_g.flatten())
    masked_image_r = gaussian(image_r, mean_patch_r, std_patch_r)
    masked_image_g = gaussian(image_g, mean_patch_g, std_patch_g)
    final_mask = masked_image_r * masked_image_g
    fig, ax = plt.subplots(1,2, figsize=(15,7))
    ax[0].imshow(image)
    ax[0].add_patch(Rectangle((patch_coor[2], patch_coor[0]), 
                               patch_coor[1] - patch_coor[0], 
                               patch_coor[3] - patch_coor[2], 
                               linewidth=2,
                               edgecolor='b', facecolor='none'))
    ax[0].set_title('Original Image with Patch', fontsize = 22)
    ax[0].set_axis_off()
    
    #clean the mask using area_opening
    ax[1].imshow(final_mask, cmap = 'hot')
    ax[1].set_title('Mask', fontsize = 22)
    ax[1].set_axis_off()
    fig.tight_layout()
    
    return final_mask

#CHROMA
def apply_mask(image,mask):
    yuv_image = rgb2yuv(image)
    yuv_image[:,:,0] = yuv_image[:,:,0] * mask 
    
    masked_image = yuv2rgb(yuv_image)
    
    fig, ax = plt.subplots(1,2, figsize=(15,7))
    ax[0].imshow(image)
    ax[0].set_title('Original Image', fontsize = 22)
    ax[0].set_axis_off()
    
    ax[1].imshow(masked_image)
    ax[1].set_title('Masked Image', fontsize = 22)
    ax[1].set_axis_off()
    fig.tight_layout()
    
    return masked_image


#DARK CHANNEL DEHAZE
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

#Atmospheric light DEHAZE
def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

#DEHAZE
def TransmissionEstimate(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape,im.dtype)

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission

#DEHAZE
def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

#Dehaze
def TransmissionRefine(im,et):
    im = im.astype('float32').clip(0,1)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)

    return t

#Dehaze
def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

#Local Tone map
def reinhard_tonemap(irradiance_map, gamma=1/1.2, alpha=0.25):
  C = 3 # num of channels
  
  # tone mapping parameters
  E_map = np.empty_like(irradiance_map)

  # normalize irradiance map
  for ch in range(C):
    map_channel = irradiance_map[:,:,ch]
    E_min = map_channel.min()
    E_max = map_channel.max()
    E_map[:,:,ch] = (map_channel - E_min) / (E_max - E_min)

  # gamma correction
  E_map = E_map**gamma

  # convert to grayscale and apply Reinhart Tone Mapping
  L = cv2.cvtColor(E_map.astype('float32'), cv2.COLOR_RGB2GRAY)
  print(L)
  L_avg = np.exp(np.mean(np.log(L+1))) 
  T = alpha / L_avg * (L+1)
  L_tone = T * (1 + (T / (T.max())**2)) / (1 + T)
  M = L_tone / (L+1)
  
  # apply scaling to each channel
  tonemapped_img = np.empty_like(E_map)
  for ch in range(C):
    tonemapped_img[:,:,ch] = E_map[:,:,ch] * M
    	
  return np.clip(tonemapped_img, 0.0, 1.0)

#########################################################
#########################################################
#########################################################
#                   PROCESSING

#-----------------------
#Align & Merge:
Align_Merge = True


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
#Chroma: How to make this work?
Chroma = False 
#-----------------------
#Denoise:
Denoise = True 

denoise_h = 5
denoise_hcolor = 10
Templatewindowsize = 7
Searchwindowsize = 21

#Current good values : 5 10 7 21

#-----------------------
#Dehaze:
Local_Tone_Map = True

#-----------------------
#Dehaze:
Dehaze = True 

#-----------------------
#Global_tone_map:
Global_tone_map = True  

#-----------------------
#Sharpen:
Sharpen = True 

# Sharpen
sharpen_array = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])

sharpen_iteration = 1

#-----------------------
#AOI
AOI = False

#########################################################
#########################################################
#########################################################
#                   FUNCTIONS

def _alignMerge(images):
    #http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.64.7616&rep=rep1&type=pdf
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)
    
    mergeMertens = cv2.createMergeMertens()
    exposureFusion = mergeMertens.process(images)

    exposureFusion = img_as_ubyte(exposureFusion)
    
    #cv2.imshow("exposureFusion", exposureFusion)
    #cv2.waitKey(0)
    
    return exposureFusion

def _Translation(image1,image2):
    #http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/PAMI_2008.pdf
    im1_gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    im3_gray = cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)
    
    sz = image1.shape
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    number_of_iterations = 5000
    termination_eps = 1e-10

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
             number_of_iterations,  termination_eps)

    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,
                                              warp_matrix, warp_mode, criteria)


    if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
        final_image = cv2.warpPerspective (image2, warp_matrix, (sz[1],sz[0]), 
                                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
    # Use warpAffine for Translation, Euclidean and Affine
        final_image = cv2.warpAffine(image2, warp_matrix, (sz[1],sz[0]), 
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

 

    # Show final results
    cv2.imshow("Image 1", image1)
    cv2.imshow("Image 2", image2)
    cv2.imshow("Aligned Image 2", final_image)
    cv2.waitKey(0)

    return final_image
    

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


def _Chroma(image):
    print('Chroma')
    rgb_splitter(image)

    image_r = image[:,:,0] / image.sum(axis=2)
    image_g = image[:,:,1] / image.sum(axis=2)
    
    one_matrix = np.ones_like(float,shape=image_r.shape)
    image_b = one_matrix- (image_r +image_g)
    
    patch = image[240:300,150:200]
    imshow(patch)
    patch_r = patch[:,:,0] /patch.sum(axis=2)
    patch_g = patch[:,:,1] /patch.sum(axis=2)
    RG_Chroma_plotter(patch_r,patch_g)
    
    patched_mask = rg_chroma_patch(image, patch_coor, mean = 1, std = 1)
    binarized_mask = patched_mask > patched_mask.mean()
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    imshow(binarized_mask)

    masked_image = apply_mask(image,binarized_mask)
    
    return masked_image


def _Denoise(image):
    print('Denoising')
    Denoised_image = cv2.fastNlMeansDenoisingColored(image, None, denoise_h, denoise_hcolor, Templatewindowsize, Searchwindowsize)
    return Denoised_image


def _Dehaze(image):
    I = image.astype('float64')/255
    print('Dehazing')
    dark = DarkChannel(I,15)
    A = AtmLight(I,dark)
    te = TransmissionEstimate(I,A,15)
    t = TransmissionRefine(image,te)
    J = Recover(I,t,A,0.1)
    
    return J*255


def _Local_Tone_Map(image):
    print('Local tone map')
    LTM = reinhard_tonemap(image, gamma=1/2.2, alpha=0.25)
    return LTM

def _Sharpen(image):
    print('Sharpening')
    Sharpened_image = convolver_rgb(image, sharpen_array, sharpen_iteration)
    return Sharpened_image


def _AOI(image):
    ...

### Directory of images
DIR = r'C:/Users/arnig/OneDrive/Pictures/Hr/Tolvusjon/dataset/trash'

images = []
def Process():
    ##Jump around images
    for sub_folder in os.listdir(DIR):
        Folder = DIR + "/" #+ sub_folder
        X = os.listdir(Folder)
        for k in range(0,len(X),3):
            start_time = time.time()
            images.clear()
            print('### New image ###')
            images.append(cv2.imread(Folder + '/' + X[k]))
            images.append(cv2.imread(Folder + '/' + X[k+1]))
            images.append(cv2.imread(Folder + '/' + X[k+2]))

            alignMerged_image = _alignMerge(images)
            #APPLY DENOISE
            if(Denoise):
                Denoised_image = _Denoise(alignMerged_image)
            else:
                Denoised_image = alignMerged_image

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

            if(Local_Tone_Map):
                Local_toned_image = _Local_Tone_Map(Demosaic_image)
            else:
                Local_toned_image = Demosaic_image
           
            if(Dehaze):
                Dehazed_image = _Dehaze(Local_toned_image)
            else:
                Dehazed_image = Local_toned_image
            
            #APPLY CHROMA
            if(Chroma):
                Chroma_image = _Chroma(Demosaic_image)
                Chroma_image = np.asarray(Chroma_image,dtype=np.float32)
            else:
                Chroma_image = Demosaic_image
                
            #Uncomment to view original vs modified image (change second parameter after which modification you want the view)
            print("--- %s seconds ---" % (time.time() - start_time))
            view_image(alignMerged_image,Dehazed_image.clip(0,1))
            
            Dehazed_image = cv2.cvtColor(Dehazed_image.astype(np.float32), cv2.COLOR_RGB2BGR)
            frame_normed = 255 * (Dehazed_image - Dehazed_image.min()) / (Dehazed_image.max() - Dehazed_image.min())
            frame_normed = np.array(frame_normed, np.int32)
            mod_dir = 'C:/Users/arnig/OneDrive/Pictures/Hr/Tolvusjon/modified/burst'
            cv2.imwrite(f"{mod_dir}/burst1.jpg",frame_normed)
            print(' ')





if __name__ == '__main__':
    Process()