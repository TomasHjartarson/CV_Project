from fastai.vision.all import *
from pathlib import Path
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet34
import depthai as dai
import cv2
import pandas as pd
import numpy as np
import time
import os
import shutil
import re
import Traditional as td




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





def model_init():
    #function for creating the model
    def _get_labels(file_path):
        pattern = re.compile("([a-z]+)[0-9]+")
        label = pattern.search(str(file_path)).group(1)
        return label # adapt index

    path = 'C:/Users/arnig/OneDrive/Pictures/Hr/Tolvusjon/dataset'
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
    learn = cnn_learner(dls, resnet34, pretrained=False, loss_func = CrossEntropyLossFlat(), metrics=error_rate, model_dir=model_path)
    global sorter 
    #load trained model to model
    sorter = learn.load("cv_demo1")

   
def image_processing(image):
    gamma_corrected_image = td._Gamma_correction(image)
    Denoised_image = td._Denoise(gamma_corrected_image)
    white_balanced_image = td._white_balancing(Denoised_image)
    Sharpened_image = td._Sharpen(white_balanced_image)
    Demosaic_image = td._Demosaic(Sharpened_image)
    Demosaic_image = np.asarray(Demosaic_image)

    AOI_image = cv2.cvtColor(Demosaic_image.astype(np.float32), cv2.COLOR_RGB2BGR)
    frame_normed = 255 * (AOI_image - AOI_image.min()) / (AOI_image.max() - AOI_image.min())
    frame_normed = np.array(frame_normed, np.int32)
    
    return frame_normed
    
def take_picture():
    with dai.Device(pipeline) as device:

    # Output queue will be used to get the rgb frames from the output defined above
        qJpeg = device.getOutputQueue(name="jpeg", maxSize=30, blocking=True)
        controlQueue = device.getInputQueue('control')
       
        # Make sure the destination path is present before starting to store the examples
        dirName = './rgb_data'
        Path(dirName).mkdir(parents=True, exist_ok=True)
        cnt = 0
        ctrl = dai.CameraControl()
        ctrl.setManualFocus(255)
        controlQueue.send(ctrl)

        while True:
    
            #get prames from camera    
            for encFrame in qJpeg.tryGetAll():
                cnt+=1
                #delay for the camera to get focus
                if cnt > 20:
                    with open(f"{dirName}/sample.jpeg", "wb") as f:
                        f.write(bytearray(encFrame.getData()))
                
            if cnt > 20:
                break

            cv2.waitKey(1)
        #read picture 
        img = cv2.imread(f"{dirName}/sample.jpeg", 1)        

        #scale picture to right size for model
        img_scaled = cv2.resize(img, (512, 348))
        img_modified = image_processing(img_scaled)
        cv2.imwrite(f"{dirName}/sample_scaled.jpeg",img_modified)    

  

def predict_pictue():
    #class oreder: bottle, paper, plastic, trash
    print(20*'**')
    predict,tensor,predlist = sorter.predict('./rgb_data/sample_scaled.jpeg')
    print(predlist)
    #X = predlist.sort()
    #print(X)
    classes = {'bottles' : 0, 'paper': 1, 'plastic':2 , 'trash':3} 
     
    #set mimimum treshold for correct prediction
    #if ((X[0] - X[1]) < 0.10):
    #    return 3
    #else:
    return classes[predict]
    

     
def init():
    model_init()
    #send init value to arduion

def main():
    init()

    try:
        
        while(1):
            '''1
            inRgb = qRgb.tryGet()
            if inRgb is not None:
                img = inRgb.getCvFrame()
                cv2.imshow("rgb", img)
            
            key = cv2.waitKey(1)
            '''
            key = int(input("run: "))
            #class oreder: bottle, paper, plastic, trash
            
            if key > 0 :
                start = time.time()
                print("Taking picture")
                take_picture()
                print("Predicting picture")
                classes = {'bottles' : 0, 'paper': 1, 'plastic':2 , 'trash':3} 
                predClass = predict_pictue()
                end = time.time()
                #print(classes)
                print(f'Predicting class is : {list(classes.keys())[list(classes.values()).index(predClass)]}')
                print(f'Time taken: {end-start} s')
                

    except KeyboardInterrupt: 
       cv2.destroyAllWindows()  


if __name__ == "__main__":
    main()