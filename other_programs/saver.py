#program to create a database for a trash sorter
#using Oak-D camera from luxinos using detpthAi

#!/usr/bin/env python3

import time
from pathlib import Path
import cv2
import os
import depthai as dai
from fastai.vision import *



def clamp(num, v0, v1):
    return max(v0, min(num, v1))

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.createColorCamera()
videoEnc = pipeline.createVideoEncoder()
xoutJpeg = pipeline.createXLinkOut()
xoutRgb = pipeline.createXLinkOut()
controlIn = pipeline.createXLinkIn()

xoutJpeg.setStreamName("jpeg")
xoutRgb.setStreamName("rgb")
controlIn.setStreamName('control')

# Properties
camRgb.setPreviewSize(512,348)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
videoEnc.setDefaultProfilePreset(camRgb.getVideoSize(), camRgb.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)


# Linking
#camRgb.video.link(xoutRgb.input)
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xoutJpeg.input)
controlIn.out.link(camRgb.inputControl)
camRgb.preview.link(xoutRgb.input)

# Defaults and limits for manual focus/exposure controls
lensPos = 150
lensMin = 0
lensMax = 255

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=30, blocking=False)
    qJpeg = device.getOutputQueue(name="jpeg", maxSize=30, blocking=True)

    controlQueue = device.getInputQueue('control')
   

    # Make sure the destination path is present before starting to store the examples
    ctrl = dai.CameraControl()
    ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_PICTURE)
    ctrl.setAutoFocusTrigger()
    controlQueue.send(ctrl)
    ctrl = dai.CameraControl()
    ctrl.setManualFocus(255)
    controlQueue.send(ctrl)
    path = 'C:/Users/arnig/OneDrive/Pictures/Hr/Tolvusjon/dataset/'
    bottleCnt = 115
    paperCnt = 115
    plasticCnt = 78
    trahCnt = 86
    pic =  0
    trigger = 0
    while True:
       
        inRgb = qRgb.tryGet()  # Non-blocking call, will return a new data that has arrived or None otherwise
        
        
        if inRgb is not None:
            img = inRgb.getCvFrame()
            can = cv2.Canny(img,50,100)
            cv2.imshow("rgb", img)
            #cv2.imshow("canny", can)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        for encFrame in qJpeg.tryGetAll():
            if type(encFrame) != 'NoneType':
                frame = encFrame.getData()
                break
               
    
                 
        
        if key == ord('b'):
            with open(f"sample.jpeg", "wb") as f:
                f.write(bytearray(frame))
            img = cv2.imread("sample.jpeg", 1)  
            img_scaled = cv2.resize(img, (512, 348))
           
            cv2.imwrite(f"{path}/bottles/bottles{bottleCnt}.jpg",img_scaled)
            bottleCnt +=1
            
        if key == ord('t'):
            with open(f"sample.jpeg", "wb") as f:
                f.write(bytearray(frame))
            img = cv2.imread("sample.jpeg", 1)  
            img_scaled = cv2.resize(img, (512, 348))
           
            cv2.imwrite(f"{path}/trash/trash{trahCnt}.jpg",img_scaled)
            trahCnt +=1
            
            
        '''
        if pic == 1 and trigger:  
            trigger = 0
            with open(f"sample.jpeg", "wb") as f:
                f.write(bytearray(frame))
            img = cv2.imread("sample.jpeg", 1)  
            img_scaled = cv2.resize(img, (512, 384))
            cv2.imwrite(f"{path}/trash/trash{trahCnt}.jpg",img_scaled)
            trahCnt +=1
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(20000, 1600)
            controlQueue.send(ctrl)
            
            print("pic 1")
        elif pic == 2 and trigger:
            trigger = 0
            with open(f"sample.jpeg", "wb") as f:
                f.write(bytearray(frame))
            img = cv2.imread("sample.jpeg", 1)  
            img_scaled = cv2.resize(img, (512, 348))
            cv2.imwrite(f"{path}/trash/trash{trahCnt}.jpg",img_scaled)
            trahCnt +=1
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(10000, 1600)
            controlQueue.send(ctrl)

            print("pic 2")
        elif  pic == 3 and trigger:
            trigger = 0
            with open(f"sample.jpeg", "wb") as f:
                f.write(bytearray(frame))
            img = cv2.imread("sample.jpeg", 1)  
            img_scaled = cv2.resize(img, (512, 348))
            cv2.imwrite(f"{path}/trash/trash{trahCnt}.jpg",img_scaled)
            trahCnt +=1
            print("Autoexposure enable")
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            controlQueue.send(ctrl)
            pic = 0
            '''
            
            
        if key == ord('p'):
            with open(f"sample.jpeg", "wb") as f:
                f.write(bytearray(frame))
            img = cv2.imread("sample.jpeg", 1)  
            img_scaled = cv2.resize(img, (512, 384))
            cv2.imwrite(f"{path}/paper/paper{paperCnt}.jpg",img_scaled)
            paperCnt +=1
            
        if key == ord('l'): 
            with open(f"sample.jpeg", "wb") as f:
                f.write(bytearray(frame))
            img = cv2.imread("sample.jpeg", 1)  
            img_scaled = cv2.resize(img, (512, 348))
            cv2.imwrite(f"{path}/plastic/plastic{plasticCnt}.jpg",img_scaled)
            plasticCnt +=1
            
            
    

cv2.waitKey(0)
cv2.destroyAllWindows()