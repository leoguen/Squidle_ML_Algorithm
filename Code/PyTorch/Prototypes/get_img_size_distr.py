import os
from os import listdir
from os.path import isfile, join
from fnmatch import fnmatch
from PIL import Image
import pandas as pd

root = '/pvol/Ecklonia_Database/Original_images/All_Images'
pattern = "*.jpg"
'''
img_list = []
COUNTER = 0
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            #print(os.path.join(path, name))
            img = Image.open(os.path.join(path, name))
            dimensions = [img.width, img.height]
            img_list.append(dimensions)
            COUNTER +=1
            img.close()
            print(COUNTER)

# open file in write mode
with open(r'/home/ubuntu/IMAS/Code/PyTorch/Prototypes/size_distr.txt', 'w') as fp:
    for item in img_list:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')

'''
# empty list to read list from a file
size_distr = []

# open file and read the content in a list
with open(r'/home/ubuntu/IMAS/Code/PyTorch/Prototypes/size_distr.txt', 'r') as fp:
    for line in fp:
        # remove linebreak from a current name
        # linebreak is the last character of each line
        x = line[:-1]
        # add current item to the list
        size_distr.append(x)

size_distr_df = pd.DataFrame(size_distr)

size_distr_df = size_distr_df.pivot_table(index = 0, aggfunc ='size')
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(size_distr_df)