from numpy.lib.function_base import angle
import numpy as np
from tqdm import tqdm
import os
import cv2 
from skimage.transform import rotate
import random 

def rotation(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def fill(img, h, w):
    img = cv2.resize(img, (w, h), cv2.INTER_CUBIC)
    return img

def zoom(img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img

def augmentor():
    path = 'C:/Users/arnig/OneDrive/Pictures/Hr/Tolvusjon/modified/dataset/'
    type = 'paper'
    images = os.listdir(path + type)
    data = []

    for i,name in tqdm(enumerate(images)):
        data.append(cv2.imread(f"{path}{type}/{name}", 1))

    train_x = np.array(data)

    final_train_data = []

        
    for i in tqdm(range(train_x.shape[0])):
        zoomVal= np.random.randint(55,85)/100.0
        angle = np.random.randint(-145,145)
        final_train_data.append(train_x[i])
        final_train_data.append(rotation(train_x[i],angle))
        final_train_data.append(np.fliplr(train_x[i]))
        final_train_data.append(np.flipud(train_x[i]))
        final_train_data.append(zoom(train_x[i],zoomVal))


    print('Save Images')
    save_path = 'C:/Users/arnig/OneDrive/Pictures/Hr/Tolvusjon/augmented'
    for i,image in tqdm(enumerate(final_train_data)):
        cv2.imwrite(f'{save_path}/{type}/{type}{i+1}.jpg',image)

if __name__ == '__main__':
    augmentor()
    