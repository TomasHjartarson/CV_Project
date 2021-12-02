import os
import time

### Script to do the following

### WHITE BALANCING

### DEOMSAIC

### DENOISE, SHARPEN

### COLOR SPACE CONVERSION

### GAMMA CORRECTION




### Processing

White_balance = True
Demosaic = True 
Denoise = True 
Color_space_conversion = True
Gamma_correction = True  



### Directory of images
DIR = r'DIRECTORY DESTINATION'


def Process():
    ##Jump around images
    for sub_folder in os.listdir(DIR):

        Folder = DIR + "/" + sub_folder
        for file in os.listdir(Folder):
            ### DO SOMETHING TO FILE
            print(file)


if __name__ == '__main__':
    Process()