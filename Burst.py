### Script to do the following

### ALIGN & MERGE

### WHITE BALANCE, DEOMSAIC, CHROMA, DENOISE

### LOCAL TONE MAP

### DEHAZE, GLOBAL TONE MAP 

### SHARPEN, HUE & SATURATION


### Processing

Align_Merge = True
White_balance = True 
Demosaic = True 
Chroma = True 
Denoise = True
Dehaze = True 
Global_tone_map = True  
Sharpen = True
Hue = True 
Saturation = True 


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