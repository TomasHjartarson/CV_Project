from fastai.vision.all import *
from pathlib import Path
import skimage
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet34,vgg19_bn
from scipy.ndimage.filters import median_filter
import depthai as dai
import cv2
import pandas as pd
import numpy as np
import time
import re
import methods.Traditional as td
import methods.Burst as burst



'''
#initialize sorter
sorter = Learner

#initialize camera
pipeline = dai.Pipeline()
camRgb = pipeline.createColorCamera()
videoEnc = pipeline.createVideoEncoder()
xoutJpeg = pipeline.createXLinkOut()
xoutRgb = pipeline.createXLinkOut()
controlIn = pipeline.createXLinkIn()

xoutJpeg.setStreamName("jpeg")
controlIn.setStreamName('control')
xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(960, 540)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
videoEnc.setDefaultProfilePreset(camRgb.getVideoSize(), camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)

# Linking
camRgb.preview.link(xoutRgb.input)
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xoutJpeg.input)
controlIn.out.link(camRgb.inputControl)

'''



def model_init():
    #function for creating the model
    def _get_labels(file_path):
        pattern = re.compile("([a-z]+)[0-9]+")
        label = pattern.search(str(file_path)).group(1)
        return label # adapt index

    path = 'C:/Users/arnig/OneDrive/Pictures/Hr/Tolvusjon/dataset/dummy'
    #create datablock
    trah_sorter = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=GrandparentSplitter(),
                 get_y=_get_labels, 
                 item_tfms=Resize(460),
                 batch_tfms=[*aug_transforms(do_flip = False,                    
                 size=224, min_scale=0.85), 
                 Normalize.from_stats(*imagenet_stats)]
                )
    #dataloader
    dls = trah_sorter.dataloaders(Path(path))
    #Setup Neural Net as resent34 neural network
    model_path = 'C:/Users/arnig/OneDrive/Documents/HR/Tolvusjon/CV_Project/model'
    learn = cnn_learner(dls, vgg19_bn, pretrained=False, loss_func = CrossEntropyLossFlat(), metrics=error_rate, model_dir=model_path)
    global sorter 
    #load trained model to model
    sorter = learn.load('cv_model_vgg19_aug')

def unsharp(image, sigma, strength):
    # Median filtering
    image_mf = median_filter(image, sigma)
    # Calculate the Laplacian
    lap = cv2.Laplacian(image_mf,cv2.CV_64F)
    # Calculate the sharpened image
    sharp = image-strength*lap
    # Saturate the pixels in either direction
    sharp[sharp>255] = 255
    sharp[sharp<0] = 0
    
    return sharp
   
def processing_traditional(image):
    White_balance_image = td._white_balancing(image)
    #cv2.imshow("white_balance",White_balance_image)
    Demosaic_image = td._Demosaic(White_balance_image)
    Demosaic_image = np.asarray(Demosaic_image) *255
    #print(img.max())
    sharp=np.zeros_like(image)
    for i in range(3):
        sharp[:,:,i] = unsharp(Demosaic_image[:,:,i].astype(np.uint8), 3, 0.7)
    Denoise_image = td._Denoise(sharp)
    #convert to HSV space for gamma correction
    img_rbg = cv2.cvtColor(Denoise_image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    gamma_corrected_image = td._Gamma_correction(img_rbg.astype(np.uint8))
    frame_normed = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_HSV2BGR)
    
    return frame_normed

def processing_burst(images):
    
    alignMerged_image = burst._alignMerge(images)
    Denoised_image = burst._Denoise(alignMerged_image)
    white_balanced_image = burst._white_balancing(Denoised_image)
    Sharpened_image = burst._Sharpen(white_balanced_image)
    Demosaic_image = burst._Demosaic(Sharpened_image)
    Demosaic_image = np.asarray(Demosaic_image)
    Local_toned_image = burst._Local_Tone_Map(Demosaic_image)
    Dehazed_image = burst._Dehaze(Local_toned_image)
    #Chroma_image = burst._Chroma(Demosaic_image)
    #Chroma_image = np.asarray(Chroma_image,dtype=np.float32)
    #scale 
    Dehazed_image = cv2.cvtColor(Dehazed_image.astype(np.float32), cv2.COLOR_RGB2BGR)
    frame_normed = 255 * (Dehazed_image - Dehazed_image.min()) / (Dehazed_image.max() - Dehazed_image.min())
    frame_normed = np.array(frame_normed, np.int32)
    
    return frame_normed                                         
    
def take_picture(burst,imgNum = 1):
    
    with dai.Device(pipeline) as device:

    # Output queue will be used to get the rgb frames from the output defined above
        #qJpeg = device.getOutputQueue(name="jpeg", maxSize=30, blocking=True)
        qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)
        controlQueue = device.getInputQueue('control')
       
        # Make sure the destination path is present before starting to store the examples
        dirName = './rgb_data'
        Path(dirName).mkdir(parents=True, exist_ok=True)
        cnt = 0
        ctrl = dai.CameraControl()
        ctrl.setManualFocus(255)
        controlQueue.send(ctrl)
        if burst:
            print(f"Taking picture {imgNum}")
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(imgNum*11000, 1600)
            controlQueue.send(ctrl)
            
        while True:
            #get frames from camera 
            
            for encFrame in qRgb.tryGetAll():
                cnt+=1
                #delay for the camera to get focus
                if cnt > 20:
                    #with open(f"{dirName}/sample{imgNum}.jpeg", "wb") as f:
                    #    f.write(bytearray(encFrame.getData()))
                    img = encFrame.getCvFrame()
                    
            if cnt > 20:
                break
            '''
            inJpeg = qJpeg.tryGet()
            if inJpeg is not None:
                img = inJpeg.getCvFrame()
                break
            '''
        cv2.imwrite(f"{dirName}/sample1.jpeg",img.astype('uint8'))    
        #read picture 
        if burst and imgNum == 3:
            images = []
            for i in range(1,4,1):
                #img = cv2.imread(f"{dirName}/sample{i}.jpeg")       
                #scale picture to right size for model
                img_scaled = cv2.resize(img, (512, 348))
                images.append(img_scaled)

            img_modified = processing_burst(images)
            cv2.imwrite(f"{dirName}/sample_scaled.jpeg",img_modified)
        
        elif not burst:
            #img = cv2.imread(f"{dirName}/sample{imgNum}.jpeg", 1) 
            img_scaled = cv2.resize(img, (512, 348))      
            #scale picture to right size for model
            img_modified = processing_traditional(img_scaled).astype('uint8')
            #img_modified = cv2.bilateralFilter(img_modified,9,50,50,cv2.BORDER_REFLECT_101)
            cv2.imwrite(f"{dirName}/sample_scaled.jpeg",img_modified.astype('uint8'))
        


def predict_pictue():
    #class oreder: bottle, paper, plastic, trash
    print(20*'**')
    predict,tensor,predlist = sorter.predict('./rgb_data/sample_scaled.jpeg')
    print(predlist)
    
    #get predicted classes in decendig order
    
    x = np.sort(predlist.numpy())[::-1]
    classes = {'bottles' : 0, 'paper': 1, 'plastic':2 , 'trash':3} 
    #set mimimum treshold for correct prediction
    if (((x[0] - x[1]) < 0.10) | (x[0]< 0.15)):
        return 3
    #else:
    return classes[predict]
    

     
def init():
    #model_init()
    ...
def main():
	init()
	DIR = r'C:/Users/arnig/OneDrive/Pictures/Hr/Tolvusjon/dataset'
	mod_dir = 'C:/Users/arnig/OneDrive/Pictures/Hr/Tolvusjon/'
	try:
		for sub_folder in os.listdir(DIR):
			Folder = os.path.join( DIR,sub_folder)
			if not os.path.isdir(f'{mod_dir}/modified/dataset2/{sub_folder}'):
					os.mkdir(f'{mod_dir}/modified/dataset2/{sub_folder}')
			for file in os.listdir(Folder):
				
				image = cv2.imread(Folder + '/' + file)
				mod_image = processing_traditional(image)
				cv2.imwrite(f"{mod_dir}/modified/dataset2/{sub_folder}/{file}",mod_image)
		while(1):
			key = 2 #int(input("run: "))
			#class oreder: bottle, paper, plastic, trash
			burst = True
			if key > 0 :
				start = time.time()
				image = cv2.imread('./rgb_data/sample1.jpeg')
				image = cv2.resize(image, (512, 348))  
			processing_traditional(image)
			'''

			print("Taking picture")
			if key == 2:
				for i in range(1,4,1):
					take_picture(1, i)
			else: 
					take_picture(0, 1)  
			print("Predicting picture")
			classes = {'bottles' : 0, 'paper': 1, 'plastic':2 , 'trash':3} 
			predClass = predict_pictue()
			'''
			end = time.time()
			print(classes)
			print(f'Predicting class is : {list(classes.keys())[list(classes.values()).index(predClass)]}')
			print(f'Time taken: {end-start} s')


	except KeyboardInterrupt: 
		cv2.destroyAllWindows()  


if __name__ == "__main__":
	main()