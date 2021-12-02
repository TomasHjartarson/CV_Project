### Script to do the following

### PIXEL CATEGORIZATION

### LEARNING LINEAR TRANSFORMATIONS

### WEIGHTED SUMMATION


### Processing

Pixel_cat = True
LLT = True 
Weighted_sum = True 

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
